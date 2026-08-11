"""Fetch the out-of-distribution evaluation set.

Reuses the iNaturalist fetcher, but writes rows with `tier='OOD'` so they never
enter the label space or the training splits. See config/ood.yaml for why these
are evaluation-only.
"""

from __future__ import annotations

import logging

import yaml

from birdcam.config import Config, load_config
from birdcam.data.fetch_inat import INatFetcher, download_pending
from birdcam.data.manifest import Manifest
from birdcam.data.taxa import ResolvedTaxon, TaxonResolver

logger = logging.getLogger(__name__)

OOD_TIER = "OOD"


def load_ood_config(cfg: Config) -> dict:
    path = cfg.root / "config" / "ood.yaml"
    if not path.is_file():
        raise RuntimeError(f"missing {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_ood(cfg: Config) -> list[ResolvedTaxon]:
    """Resolve OOD taxa. Not cached alongside target taxa -- different purpose."""
    ood = load_ood_config(cfg)
    resolver = TaxonResolver(cfg)
    out: list[ResolvedTaxon] = []
    for entry in ood["taxa"]:
        t = ResolvedTaxon(
            scientific_name=entry["scientific_name"],
            common_name=entry["common_name"],
            tier=OOD_TIER,
            fetch_scope=ood.get("fetch_scope", "za"),
        )
        logger.info("resolving OOD taxon %s", t.scientific_name)
        try:
            resolver.resolve_gbif(t)
        except Exception as exc:  # noqa: BLE001
            t.warnings.append(f"GBIF: {exc}")
        try:
            resolver.resolve_inat(t)
        except Exception as exc:  # noqa: BLE001
            t.warnings.append(f"iNat: {exc}")
        if not t.inat_taxon_id:
            # Reported, never silently skipped.
            logger.error("FAILED to resolve OOD taxon %s: %s", t.scientific_name, t.warnings)
        out.append(t)
    return out


def run(cfg: Config, workers: int = 5) -> None:
    ood = load_ood_config(cfg)
    per = ood.get("per_taxon", 150)
    scope = ood.get("fetch_scope", "za")

    with Manifest(cfg.path("manifest_db")) as m:
        fetcher = INatFetcher(cfg, m)
        for t in resolve_ood(cfg):
            if not t.inat_taxon_id:
                continue
            logger.info("%s (OOD)", t.scientific_name)
            # `general` variant only: sex annotations are irrelevant here, and
            # most of these taxa have none anyway.
            fetcher.fetch_species(t, OOD_TIER, scope, per, "general")
        dl = download_pending(cfg, m, workers=workers)
        logger.info("OOD downloaded %d, failed %d", dl.downloaded, dl.download_failed)

        counts = m.conn.execute(
            "SELECT scientific_name, COUNT(*) n FROM images "
            "WHERE tier=? AND status='downloaded' GROUP BY scientific_name ORDER BY n DESC",
            (OOD_TIER,),
        ).fetchall()
        print(f"\n{'OOD taxon':<30}{'images':>8}")
        print("-" * 38)
        for r in counts:
            print(f"{r['scientific_name']:<30}{r['n']:>8}")
        print(f"{'TOTAL':<30}{sum(r['n'] for r in counts):>8}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    import argparse

    ap = argparse.ArgumentParser(description="Fetch the OOD evaluation set.")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    run(load_config(), workers=args.workers)


if __name__ == "__main__":
    main()
