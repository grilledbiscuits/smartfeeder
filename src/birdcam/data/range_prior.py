"""Build a per-site range prior from real local observation density.

The capture application runs at one feeder in one garden. Most of the label
space is irrelevant there: *Cinnyris neergaardi* occurs in coastal KwaZulu-Natal
and will never visit Cape Town. Downweighting it costs nothing and removes a
whole class of confident mistakes.

Deliberately a SOFT prior with a floor, not a filter. Two reasons:

* Vagrants happen. A model that *cannot* emit a rare species will never let you
  discover one, and discovering one is among the more interesting things a
  feeder camera could do.
* Observation density measures where *people* record birds, not purely where
  birds are. Near a university suburb it is biased toward accessible, popular
  sites. Treating it as ground truth would encode that bias as fact.

Weights come from the iNaturalist species_counts endpoint over a radius around
the site, so they are measured rather than assumed.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
import yaml

from birdcam.config import Config, load_config

logger = logging.getLogger(__name__)

SPECIES_COUNTS_URL = "https://api.inaturalist.org/v1/observations/species_counts"


def fetch_local_counts(cfg: Config, lat: float, lon: float, radius_km: int) -> dict[str, int]:
    """Research-grade bird observation counts per species within the radius."""
    session = requests.Session()
    session.headers["User-Agent"] = cfg.user_agent()
    counts: dict[str, int] = {}
    page = 1
    while True:
        r = session.get(
            SPECIES_COUNTS_URL,
            params={
                "lat": lat,
                "lng": lon,
                "radius": radius_km,
                "quality_grade": "research",
                "iconic_taxa": "Aves",
                "per_page": 500,
                "page": page,
            },
            timeout=cfg.train_cfg["fetch"]["request_timeout_s"],
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        for row in results:
            name = (row.get("taxon") or {}).get("name")
            if name:
                counts[name] = row.get("count", 0)
        if len(counts) >= (data.get("total_results") or 0) or not results:
            break
        page += 1
        time.sleep(1.0)  # iNat asks for ~1 rps
    logger.info("%d bird species recorded within %dkm", len(counts), radius_km)
    return counts


def build_weights(
    cfg: Config, counts: dict[str, int], alpha: float, floor: float, neutral: list[str]
) -> dict[str, float]:
    """Map local counts onto a weight per taxon class."""
    local_max = max(counts.values()) if counts else 1
    weights: dict[str, float] = {}

    for s in cfg.species:
        n = counts.get(s.scientific_name, 0)
        w = max(floor, (n / local_max) ** alpha) if n > 0 else floor
        weights[s.slug] = round(float(w), 4)

    # Fallback classes inherit the strongest of their members: if any species in
    # a genus is plausible here, "that genus, species unknown" is plausible too.
    head = cfg.taxonomy_cfg["taxon_head"]
    by_genus: dict[str, list[float]] = {}
    for s in cfg.species:
        by_genus.setdefault(s.genus, []).append(weights[s.slug])
    for name, g in head["genus_fallback"].items():
        members = by_genus.get(g["genus"], [floor])
        weights[name] = round(float(max(members)), 4)
    by_family: dict[str, list[float]] = {}
    for s in cfg.species:
        fam = cfg.genus_to_family.get(s.genus)
        if fam:
            by_family.setdefault(fam, []).append(weights[s.slug])
    for name, fdef in head["family_fallback"].items():
        members = by_family.get(fdef["family"], [floor])
        weights[name] = round(float(max(members)), 4)
    for name in head["guild_fallback"]:
        weights[name] = 1.0
    for name in neutral:
        weights[name] = 1.0
    return weights


def load_site(cfg: Config, site: str) -> dict[str, Any]:
    path = cfg.root / "config" / "sites" / f"{site}.yaml"
    if not path.is_file():
        raise RuntimeError(f"no site config at {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run(cfg: Config, site: str) -> dict[str, float]:
    sc = load_site(cfg, site)
    s, p = sc["site"], sc["prior"]
    counts = fetch_local_counts(cfg, s["lat"], s["lon"], s["radius_km"])
    weights = build_weights(cfg, counts, p["alpha"], p["floor"], p.get("neutral_classes", []))

    path = cfg.root / "config" / "sites" / f"{site}.yaml"
    text = path.read_text(encoding="utf-8")
    lines = ["weights:"]
    for slug in sorted(weights):
        obs = ""
        for sp in cfg.species:
            if sp.slug == slug:
                obs = f"   # {counts.get(sp.scientific_name, 0)} obs within {s['radius_km']}km"
        lines.append(f"  {slug}: {weights[slug]}{obs}")
    marker = "weights: {}"
    text = text.replace(marker, "\n".join(lines)) if marker in text else text
    path.write_text(text, encoding="utf-8")

    # Report what the prior actually does, ranked.
    ranked = sorted(
        ((sp.scientific_name, sp.common_name, sp.tier, counts.get(sp.scientific_name, 0),
          weights[sp.slug]) for sp in cfg.species),
        key=lambda r: -r[3],
    )
    print(f"\n{sc['site']['name']} -- {len(counts)} bird species recorded within "
          f"{s['radius_km']}km\n")
    print(f"{'species':<28}{'common name':<32}{'tier':<5}{'local obs':>10}{'weight':>8}")
    print("-" * 84)
    for name, common, tier, n, w in ranked:
        print(f"{name:<28}{common[:31]:<32}{tier:<5}{n:>10}{w:>8.3f}")
    return weights


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    import argparse

    ap = argparse.ArgumentParser(description="Build a site range prior from iNaturalist.")
    ap.add_argument("--site", default="rondebosch")
    args = ap.parse_args()
    run(load_config(), args.site)


if __name__ == "__main__":
    main()
