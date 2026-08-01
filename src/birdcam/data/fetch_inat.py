"""iNaturalist fetcher -- the primary source of sex and life-stage annotations.

Why this source matters most: iNaturalist exposes *annotations* via controlled
terms, including Sex and Life Stage. Those labels are the scarce resource in
this project. Photographs of sunbirds are abundant; photographs of *known
female* sunbirds are not.

Three things here are load-bearing:

1. **Controlled-term IDs are looked up, not assumed.** `/v1/controlled_terms` is
   fetched at startup and checked against the expected values in train.yaml. A
   disagreement is fatal -- it means the vocabulary was revised and the
   annotation mapping in taxonomy.yaml needs review before any label is trusted.

2. **Pagination uses `id_above`, not `page`.** The API caps offset pagination at
   10,000 results and Tier A species exceed that. `id_above` walks the full set.

3. **Annotated observations are fetched first.** Roughly 5-30% of observations
   carry a Sex annotation (measured 2026-07-31), so uniform sampling wastes the
   budget on unlabelled males. The `sexed` variant pulls the annotated subset
   directly, then `general` fills the remainder.

Licence filtering is strict: CC0, CC-BY and CC-BY-NC only. Anything with no
licence stated, or any ND / SA-incompatible variant, is dropped -- and the drop
is counted, not silent.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import requests

from birdcam.config import Config, load_config
from birdcam.data.manifest import ImageRecord, Manifest, sha256_file
from birdcam.data.taxa import ZA_PLACE_ID, get_resolved

logger = logging.getLogger(__name__)

SOURCE = "inaturalist"


@dataclass
class FetchStats:
    seen: int = 0
    kept: int = 0
    rejected_license: int = 0
    rejected_no_photo: int = 0
    downloaded: int = 0
    download_failed: int = 0

    def merge(self, other: FetchStats) -> None:
        for f in (
            "seen",
            "kept",
            "rejected_license",
            "rejected_no_photo",
            "downloaded",
            "download_failed",
        ):
            setattr(self, f, getattr(self, f) + getattr(other, f))


class INatFetcher:
    def __init__(self, cfg: Config, manifest: Manifest) -> None:
        self.cfg = cfg
        self.m = manifest
        self.fc = cfg.train_cfg["fetch"]
        self.ic = self.fc["inaturalist"]
        self.base = self.ic["base_url"]

        self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg.user_agent()

        self.allowed_licenses = {s.lower() for s in self.fc["allowed_licenses"]}
        self._min_interval = 1.0 / self.ic["rate_limit_rps"]
        self._last_call = 0.0

        self.terms = self._load_controlled_terms()

    # -- HTTP ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

        url = f"{self.base}/{path}"
        retries = self.fc["max_retries"]
        backoff = self.fc["backoff_base_s"]
        for attempt in range(retries):
            try:
                r = self.session.get(url, params=params, timeout=self.fc["request_timeout_s"])
                if r.status_code == 429:
                    wait = backoff * (2**attempt)
                    logger.warning("429; backing off %.1fs", wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == retries - 1:
                    raise
                wait = backoff * (2**attempt)
                logger.warning("request failed (%s); retry in %.1fs", exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"exhausted retries for {url}")

    # -- controlled terms ------------------------------------------------------

    def _load_controlled_terms(self) -> dict[str, dict[str, Any]]:
        """Fetch the live vocabulary and verify it against config.

        Hardcoding these IDs is exactly the failure mode the brief warns about,
        so they are fetched. But a silent *change* is just as dangerous as a
        wrong guess, hence the assertion against expected values.
        """
        data = self._get("controlled_terms", {})
        by_label = {t["label"]: t for t in data.get("results", [])}

        resolved: dict[str, dict[str, Any]] = {}
        for key, expected in self.ic["expected_terms"].items():
            label = {"sex": "Sex", "life_stage": "Life Stage"}[key]
            term = by_label.get(label)
            if term is None:
                raise RuntimeError(
                    f"iNaturalist controlled term {label!r} not found. Available: "
                    f"{sorted(by_label)}. The vocabulary has changed; review "
                    "config/taxonomy.yaml annotation_mapping before fetching."
                )
            if term["id"] != expected["id"]:
                raise RuntimeError(
                    f"controlled term {label!r} has id={term['id']}, config expects "
                    f"{expected['id']}. Vocabulary revised -- review the annotation "
                    "mapping before trusting any label."
                )
            values = {v["label"]: v["id"] for v in term.get("values", [])}
            for vlabel, vid in expected["values"].items():
                if values.get(vlabel) != vid:
                    raise RuntimeError(
                        f"controlled value {label}/{vlabel} has id={values.get(vlabel)}, "
                        f"config expects {vid}. Review annotation mapping."
                    )
            resolved[key] = {"id": term["id"], "values": values}
            logger.info(
                "verified controlled term %s: id=%d, %d values", label, term["id"], len(values)
            )
        return resolved

    # -- observation -> records ------------------------------------------------

    def _photo_url(self, photo: dict[str, Any]) -> str | None:
        """Build a URL at the configured size.

        iNat photo URLs embed the size as a path segment (`square.jpg`,
        `large.jpg`, ...). We request `large` (<=1024px), never `original`:
        everything is downsized to 256px anyway, and originals are ~10x the
        bytes for no benefit. See DECISIONS.md D7.
        """
        url = photo.get("url")
        if not url:
            return None
        size = self.ic["photo_size"]
        for known in ("square", "thumb", "small", "medium", "large", "original"):
            if f"/{known}." in url:
                return url.replace(f"/{known}.", f"/{size}.")
        return url

    def _annotations(self, obs: dict[str, Any]) -> tuple[str | None, str | None]:
        """Extract raw Sex and Life Stage values.

        Returns the source's own labels (e.g. "Male"), NOT our schema's labels.
        Mapping to the label space happens in Phase 3 via taxonomy.yaml, so the
        manifest stays a faithful record of what the source actually said.

        Annotations with a negative vote score are ignored: the community has
        voted them down.
        """
        sex = life = None
        sex_id = self.terms["sex"]["id"]
        life_id = self.terms["life_stage"]["id"]
        sex_by_id = {v: k for k, v in self.terms["sex"]["values"].items()}
        life_by_id = {v: k for k, v in self.terms["life_stage"]["values"].items()}

        for a in obs.get("annotations") or []:
            if (a.get("vote_score") or 0) < 0:
                continue
            attr, val = a.get("controlled_attribute_id"), a.get("controlled_value_id")
            if attr == sex_id and val in sex_by_id:
                sex = sex_by_id[val]
            elif attr == life_id and val in life_by_id:
                life = life_by_id[val]
        return sex, life

    def _records_from_obs(
        self, obs: dict[str, Any], taxon, tier: str, stats: FetchStats
    ) -> list[ImageRecord]:
        photos = obs.get("photos") or []
        if not photos:
            stats.rejected_no_photo += 1
            return []

        sex, life = self._annotations(obs)
        user = obs.get("user") or {}
        loc = (obs.get("location") or "").split(",")
        lat = lon = None
        if len(loc) == 2:
            try:
                lat, lon = float(loc[0]), float(loc[1])
            except ValueError:
                pass

        out: list[ImageRecord] = []
        for p in photos:
            lic = (p.get("license_code") or "").lower()
            # No licence stated is a rejection, not a default.
            if lic not in self.allowed_licenses:
                stats.rejected_license += 1
                continue
            url = self._photo_url(p)
            if not url:
                stats.rejected_no_photo += 1
                continue
            dims = p.get("original_dimensions") or {}
            out.append(
                ImageRecord(
                    image_id=f"{SOURCE}:{p['id']}",
                    source=SOURCE,
                    source_url=url,
                    page_url=obs.get("uri"),
                    scientific_name=taxon.scientific_name,
                    common_name=taxon.common_name,
                    resolved_taxon_key=taxon.gbif_usage_key,
                    inat_taxon_id=taxon.inat_taxon_id,
                    tier=tier,
                    observation_id=str(obs.get("id")),
                    observer_id=str(user.get("id")) if user.get("id") else None,
                    sex_annotation=sex,
                    life_stage_annotation=life,
                    annotation_source=SOURCE if (sex or life) else None,
                    license=lic,
                    rights_holder=p.get("attribution") and _rights_holder(p["attribution"]),
                    attribution_text=p.get("attribution"),
                    lat=lat,
                    lon=lon,
                    observed_date=obs.get("observed_on"),
                    quality_grade=obs.get("quality_grade"),
                    width=dims.get("width"),
                    height=dims.get("height"),
                )
            )
            stats.kept += 1
        return out

    # -- paging ----------------------------------------------------------------

    def fetch_species(self, taxon, tier: str, scope: str, limit: int, variant: str) -> FetchStats:
        """Walk one species with id_above pagination, checkpointing every page."""
        stats = FetchStats()
        state = self.m.get_state(SOURCE, taxon.scientific_name, variant)
        if state and state["exhausted"]:
            logger.info(
                "  %s/%s already exhausted (%d fetched)",
                taxon.scientific_name,
                variant,
                state["fetched"],
            )
            return stats

        cursor = state["cursor"] if state else None
        already = state["fetched"] if state else 0

        params: dict[str, Any] = {
            "taxon_id": taxon.inat_taxon_id,
            "quality_grade": self.ic["quality_grade"],
            "photos": "true",
            "per_page": self.ic["per_page"],
            "order_by": "id",
            "order": "asc",
        }
        if scope == "za":
            params["place_id"] = ZA_PLACE_ID
        if variant == "sexed":
            # Only observations carrying a Sex annotation. This is the scarce
            # resource, so it is fetched first and separately.
            params["term_id"] = self.terms["sex"]["id"]

        fetched = already
        while fetched < limit:
            if cursor:
                params["id_above"] = cursor
            data = self._get("observations", params)
            results = data.get("results") or []
            if not results:
                self.m.set_state(SOURCE, taxon.scientific_name, cursor, fetched, True, variant)
                break

            batch: list[ImageRecord] = []
            for obs in results:
                stats.seen += 1
                batch.extend(self._records_from_obs(obs, taxon, tier, stats))
                cursor = str(obs["id"])

            self.m.upsert_images(batch)
            fetched += len(batch)
            # Checkpoint after EVERY page. This runs for hours; a killed run
            # must resume, not restart.
            self.m.set_state(SOURCE, taxon.scientific_name, cursor, fetched, False, variant)
            logger.info(
                "  %s/%s: +%d images (%d/%d) cursor=%s",
                taxon.scientific_name,
                variant,
                len(batch),
                fetched,
                limit,
                cursor,
            )
            if len(results) < self.ic["per_page"]:
                self.m.set_state(SOURCE, taxon.scientific_name, cursor, fetched, True, variant)
                break
        return stats


def _rights_holder(attribution: str) -> str:
    """Pull the name out of an iNat attribution string.

    Format: "(c) Marion Maclean, some rights reserved (CC BY-NC)".
    Falls back to the whole string rather than losing the attribution.
    """
    s = attribution.removeprefix("(c) ").removeprefix("© ")
    return s.split(",")[0].strip() if "," in s else attribution


# -- downloading ---------------------------------------------------------------


def download_pending(
    cfg: Config, m: Manifest, workers: int = 4, limit: int | None = None
) -> FetchStats:
    """Download image bytes for pending rows.

    Runs concurrently: image bytes come from a different host than the API
    (static.inaturalist.org / the open-data S3 bucket), so this does not consume
    the API rate-limit budget. Kept modest anyway.
    """
    stats = FetchStats()
    rows = m.pending(limit=limit)
    if not rows:
        return stats

    raw_dir = cfg.path("raw_dir")
    session = requests.Session()
    session.headers["User-Agent"] = cfg.user_agent()
    timeout = cfg.train_cfg["fetch"]["request_timeout_s"]

    def _one(row) -> tuple[str, str, dict[str, Any]]:
        slug = row["scientific_name"].lower().replace(" ", "_")
        dest = raw_dir / slug / f"{row['image_id'].replace(':', '_')}.jpg"
        if dest.is_file() and dest.stat().st_size > 0:
            return (
                row["image_id"],
                "downloaded",
                {
                    "local_path": str(dest.relative_to(cfg.root)),
                    "sha256": sha256_file(dest),
                },
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = session.get(row["source_url"], timeout=timeout, stream=True)
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    fh.write(chunk)
            if tmp.stat().st_size == 0:
                tmp.unlink(missing_ok=True)
                return row["image_id"], "failed", {"status_detail": "empty response"}
            tmp.replace(dest)
            return (
                row["image_id"],
                "downloaded",
                {
                    "local_path": str(dest.relative_to(cfg.root)),
                    "sha256": sha256_file(dest),
                },
            )
        except Exception as exc:  # noqa: BLE001 - one bad URL must not stop the run
            return row["image_id"], "failed", {"status_detail": str(exc)[:200]}

    logger.info("downloading %d images with %d workers", len(rows), workers)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, r): r for r in rows}
        for fut in as_completed(futures):
            image_id, status, updates = fut.result()
            m.mark(image_id, status=status, **updates)
            if status == "downloaded":
                stats.downloaded += 1
            else:
                stats.download_failed += 1
            done += 1
            if done % 100 == 0:
                m.commit()
                logger.info(
                    "  %d/%d downloaded (%d failed)", done, len(rows), stats.download_failed
                )
    m.commit()
    return stats


# -- entry point ---------------------------------------------------------------


def run(cfg: Config, tiers: list[str], per_species: int, workers: int = 4) -> None:
    taxa = get_resolved(cfg, tiers=tiers)
    manifest_path = cfg.path("manifest_db")

    with Manifest(manifest_path) as m:
        fetcher = INatFetcher(cfg, m)
        total = FetchStats()

        for name, taxon in taxa.items():
            if not taxon.resolved:
                logger.error("skipping %s: unresolved (%s)", name, taxon.warnings)
                continue
            spec = cfg.species_by_name[name]
            logger.info("%s (tier %s, scope %s)", name, spec.tier, spec.fetch_scope)

            # Annotated observations first -- they are the scarce resource.
            s1 = fetcher.fetch_species(taxon, spec.tier, spec.fetch_scope, per_species, "sexed")
            total.merge(s1)
            have = m.count(scientific_name=name)
            if have < per_species:
                s2 = fetcher.fetch_species(
                    taxon, spec.tier, spec.fetch_scope, per_species - have, "general"
                )
                total.merge(s2)

        logger.info(
            "listing done: %d observations seen, %d images kept, "
            "%d rejected on licence, %d without usable photo",
            total.seen,
            total.kept,
            total.rejected_license,
            total.rejected_no_photo,
        )

        dl = download_pending(cfg, m, workers=workers)
        logger.info("downloaded %d, failed %d", dl.downloaded, dl.download_failed)

        pq = cfg.path("manifest_parquet")
        n = m.export_parquet(pq)
        logger.info("exported %d rows to %s", n, pq)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    import argparse

    ap = argparse.ArgumentParser(description="Fetch images + annotations from iNaturalist.")
    ap.add_argument("--tiers", nargs="*", default=["A"])
    ap.add_argument("--per-species", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config()
    per = args.per_species or cfg.train_cfg["fetch"]["max_images_per_species"]
    run(cfg, tiers=args.tiers, per_species=per, workers=args.workers)


if __name__ == "__main__":
    main()
