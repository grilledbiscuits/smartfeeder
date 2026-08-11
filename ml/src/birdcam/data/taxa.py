"""GBIF + iNaturalist taxon resolution, cached to disk.

Taxon IDs are NEVER hardcoded. Every species in config/species.yaml is resolved
at runtime against the GBIF species-match API and the iNaturalist taxa API, and
the result is cached to config/taxon_cache.json.

Two checks make this more than a lookup:

1. The genus derived from the binomial is compared against the genus the GBIF
   backbone actually returns. A mismatch means the species has been moved, and
   is surfaced rather than silently accepted. (Known case: GBIF resolves
   *Melaenornis silens* to *Sigelus silens* -- see ASSUMPTIONS.md A4.)
2. The family is compared against config/taxonomy.yaml's genus_family map, since
   the rollup hierarchy depends on it being right.

Names that fail to resolve are logged and recorded with a null key. They are
never silently skipped: a species missing from the corpus and a species that
failed to resolve are different problems with different fixes.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import requests

from birdcam.config import Config, load_config

logger = logging.getLogger(__name__)

GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
INAT_TAXA_URL = "https://api.inaturalist.org/v1/taxa"
INAT_PLACES_URL = "https://api.inaturalist.org/v1/places"

# iNaturalist place_id for South Africa. Verified against GET /v1/places/6986 on
# 2026-07-31 -> {"name": "South Africa", "admin_level": 0}. Re-verified at
# runtime by resolve_place() rather than trusted.
ZA_PLACE_ID = 6986


@dataclass
class ResolvedTaxon:
    """One species resolved against both backbones."""

    scientific_name: str
    common_name: str
    tier: str
    fetch_scope: str

    gbif_usage_key: int | None = None
    gbif_scientific_name: str | None = None
    gbif_genus: str | None = None
    gbif_family: str | None = None
    gbif_match_type: str | None = None
    gbif_confidence: int | None = None

    inat_taxon_id: int | None = None
    inat_name: str | None = None
    inat_rank: str | None = None

    warnings: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        """Both backbones are needed: GBIF drives occurrence search, iNat drives
        the annotation join. Half a resolution is not usable."""
        return self.gbif_usage_key is not None and self.inat_taxon_id is not None


class TaxonResolver:
    def __init__(self, cfg: Config, session: requests.Session | None = None) -> None:
        self.cfg = cfg
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = cfg.user_agent()
        self.timeout = cfg.train_cfg["fetch"]["request_timeout_s"]
        # iNaturalist asks for roughly one request per second. Honoured here as
        # well as in the fetchers -- resolution is only ~40 calls, but there is
        # no reason to be rude about it.
        self._inat_min_interval = 1.0 / cfg.train_cfg["fetch"]["inaturalist"]["rate_limit_rps"]
        self._last_inat_call = 0.0

    # -- HTTP ------------------------------------------------------------------

    def _get(self, url: str, params: dict[str, Any], *, throttle: bool = False) -> dict[str, Any]:
        if throttle:
            elapsed = time.monotonic() - self._last_inat_call
            if elapsed < self._inat_min_interval:
                time.sleep(self._inat_min_interval - elapsed)
            self._last_inat_call = time.monotonic()

        max_retries = self.cfg.train_cfg["fetch"]["max_retries"]
        backoff = self.cfg.train_cfg["fetch"]["backoff_base_s"]
        for attempt in range(max_retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429:
                    wait = backoff * (2**attempt)
                    logger.warning("429 from %s; backing off %.1fs", url, wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == max_retries - 1:
                    raise
                wait = backoff * (2**attempt)
                logger.warning("%s failed (%s); retry in %.1fs", url, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"exhausted retries for {url}")

    # -- resolution ------------------------------------------------------------

    def resolve_gbif(self, taxon: ResolvedTaxon) -> None:
        data = self._get(
            GBIF_MATCH_URL,
            {"name": taxon.scientific_name, "kingdom": "Animalia", "class": "Aves"},
        )
        if not data.get("usageKey"):
            taxon.warnings.append(f"GBIF returned no usageKey (matchType={data.get('matchType')})")
            return

        taxon.gbif_usage_key = data["usageKey"]
        taxon.gbif_scientific_name = data.get("scientificName")
        taxon.gbif_genus = data.get("genus")
        taxon.gbif_family = data.get("family")
        taxon.gbif_match_type = data.get("matchType")
        taxon.gbif_confidence = data.get("confidence")

        if data.get("matchType") != "EXACT":
            taxon.warnings.append(
                f"GBIF matchType={data.get('matchType')} "
                f"(confidence={data.get('confidence')}) -> {data.get('scientificName')}"
            )

        # Genus drift: derived genus vs the backbone's. Surfaced, not hidden.
        derived_genus = taxon.scientific_name.split()[0]
        if taxon.gbif_genus and taxon.gbif_genus != derived_genus:
            taxon.warnings.append(
                f"genus mismatch: species.yaml implies {derived_genus!r}, GBIF says "
                f"{taxon.gbif_genus!r} (accepted name {taxon.gbif_scientific_name!r}). "
                "Rollup uses the taxonomy.yaml mapping; verify it is still correct."
            )

        # Family must agree with the rollup hierarchy, or predictions roll up
        # into the wrong parent.
        expected_family = self.cfg.genus_to_family.get(derived_genus)
        if expected_family and taxon.gbif_family and taxon.gbif_family != expected_family:
            taxon.warnings.append(
                f"family mismatch: taxonomy.yaml maps {derived_genus!r} to "
                f"{expected_family!r}, GBIF says {taxon.gbif_family!r}. "
                "ROLLUP WILL BE WRONG until this is reconciled."
            )

    def resolve_inat(self, taxon: ResolvedTaxon) -> None:
        data = self._get(
            INAT_TAXA_URL,
            {"q": taxon.scientific_name, "rank": "species", "is_active": "true"},
            throttle=True,
        )
        results = data.get("results") or []
        target = taxon.scientific_name.lower()

        # Exact name match only. iNat's fuzzy search will happily return a
        # congener, and quietly training on the wrong species is exactly the
        # failure this module exists to prevent.
        exact = [r for r in results if (r.get("name") or "").lower() == target]
        if not exact:
            names = [r.get("name") for r in results[:5]]
            taxon.warnings.append(f"iNat returned no exact name match (top results: {names})")
            return

        best = max(exact, key=lambda r: r.get("observations_count") or 0)
        taxon.inat_taxon_id = best["id"]
        taxon.inat_name = best.get("name")
        taxon.inat_rank = best.get("rank")

        if len(exact) > 1:
            taxon.warnings.append(
                f"iNat returned {len(exact)} exact matches; chose id={best['id']} "
                "by observation count"
            )

    def resolve_place(self, place_id: int = ZA_PLACE_ID) -> int:
        """Confirm the recorded ZA place_id still names South Africa."""
        data = self._get(f"{INAT_PLACES_URL}/{place_id}", {}, throttle=True)
        results = data.get("results") or []
        if not results:
            raise RuntimeError(f"iNat place_id {place_id} returned no result")
        name = results[0].get("name")
        if name != "South Africa":
            raise RuntimeError(
                f"iNat place_id {place_id} is {name!r}, expected 'South Africa'. "
                "Place IDs have changed; update taxa.ZA_PLACE_ID."
            )
        logger.info("verified iNat place_id %d = %s", place_id, name)
        return place_id

    def resolve_all(self, tiers: list[str] | None = None) -> list[ResolvedTaxon]:
        species = [s for s in self.cfg.species if tiers is None or s.tier in tiers]
        out: list[ResolvedTaxon] = []
        for i, s in enumerate(species, 1):
            t = ResolvedTaxon(
                scientific_name=s.scientific_name,
                common_name=s.common_name,
                tier=s.tier,
                fetch_scope=s.fetch_scope,
            )
            logger.info("[%d/%d] resolving %s", i, len(species), s.scientific_name)
            try:
                self.resolve_gbif(t)
            except Exception as exc:  # noqa: BLE001 - one failure must not stop the run
                t.warnings.append(f"GBIF resolution raised: {exc}")
            try:
                self.resolve_inat(t)
            except Exception as exc:  # noqa: BLE001
                t.warnings.append(f"iNat resolution raised: {exc}")
            out.append(t)
        return out


# -- cache ---------------------------------------------------------------------


def load_cache(cfg: Config) -> dict[str, ResolvedTaxon]:
    path = cfg.path("taxon_cache")
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: ResolvedTaxon(**v) for k, v in raw.get("taxa", {}).items()}


def save_cache(cfg: Config, taxa: list[ResolvedTaxon], place_id: int) -> None:
    path = cfg.path("taxon_cache")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": (
            "Generated by birdcam.data.taxa. Committed deliberately so runs stay "
            "reproducible across GBIF/iNat backbone revisions. Delete to force "
            "re-resolution."
        ),
        "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inat_place_id_za": place_id,
        "taxa": {t.scientific_name: asdict(t) for t in taxa},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", path)


def get_resolved(cfg: Config, tiers: list[str] | None = None) -> dict[str, ResolvedTaxon]:
    """Load from cache, resolving anything missing.

    Used by the fetchers so they never need to know how resolution works.
    """
    cache = load_cache(cfg)
    wanted = [s for s in cfg.species if tiers is None or s.tier in tiers]
    missing = [s for s in wanted if s.scientific_name not in cache]
    if missing:
        logger.info("%d species not in cache; resolving", len(missing))
        resolver = TaxonResolver(cfg)
        place_id = resolver.resolve_place()
        for t in resolver.resolve_all(tiers=tiers):
            cache[t.scientific_name] = t
        save_cache(cfg, list(cache.values()), place_id)
    return {
        s.scientific_name: cache[s.scientific_name] for s in wanted if s.scientific_name in cache
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    import argparse

    ap = argparse.ArgumentParser(description="Resolve GBIF + iNaturalist taxon IDs.")
    ap.add_argument("--tiers", nargs="*", default=None, help="e.g. --tiers A C")
    ap.add_argument("--force", action="store_true", help="ignore the existing cache")
    args = ap.parse_args()

    cfg = load_config()
    resolver = TaxonResolver(cfg)
    place_id = resolver.resolve_place()

    if args.force:
        taxa = resolver.resolve_all(tiers=args.tiers)
        save_cache(cfg, taxa, place_id)
    else:
        taxa = list(get_resolved(cfg, tiers=args.tiers).values())

    ok = [t for t in taxa if t.resolved]
    bad = [t for t in taxa if not t.resolved]
    warned = [t for t in taxa if t.warnings]

    print(f"\n{'species':<28} {'tier':<5} {'gbif':<10} {'inat':<9} status")
    print("-" * 72)
    for t in sorted(taxa, key=lambda x: (x.tier, x.scientific_name)):
        status = "ok" if t.resolved else "FAILED"
        if t.warnings:
            status += f" ({len(t.warnings)} warning{'s' if len(t.warnings) > 1 else ''})"
        print(
            f"{t.scientific_name:<28} {t.tier:<5} "
            f"{t.gbif_usage_key or '-':<10} {t.inat_taxon_id or '-':<9} {status}"
        )

    if warned:
        print("\nwarnings:")
        for t in warned:
            for w in t.warnings:
                print(f"  {t.scientific_name}: {w}")

    print(f"\nresolved {len(ok)}/{len(taxa)}")
    if bad:
        # Reported, never silently skipped.
        print("FAILED TO RESOLVE:")
        for t in bad:
            print(f"  {t.scientific_name}")


if __name__ == "__main__":
    main()
