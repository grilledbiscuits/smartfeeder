"""Common names for everything a person reads.

The label space is keyed on scientific names because that is the only thing
GBIF and iNaturalist will resolve a taxon by. That is correct for the data
layer and wrong for everything else: a confusion matrix reading
`cinnyris chalybeus` against `chalcomitra amethystina` is unreadable at a
glance, and the two are not remotely alike in the field.

So: slugs stay scientific, and every human-facing surface routes through
`display()`. Nothing here is ever a key, a filename, or a lookup -- if a
display name reaches a dict, that is a bug.

The fallback classes need care. `cinnyris_indet` does not mean "a Cinnyris";
it means "one of the double-collared sunbirds, and the model will not commit
to which". Rendering that as "Cinnyris" would read like an identification.
Hence the "unsure which" suffix -- the uncertainty is the message.
"""

from __future__ import annotations

from birdcam.config import Config, load_config

UNSURE = "unsure which"


def _display_cfg(cfg: Config) -> dict:
    d = cfg.taxonomy_cfg.get("display")
    if not d:
        raise RuntimeError("taxonomy.yaml has no `display:` section; see birdcam.names")
    return d


def build_display_map(cfg: Config) -> dict[str, str]:
    """Map every taxon class -- plus `unknown` -- to an English name.

    Fatal if a class has no name. A missing entry means a slug would leak into
    a report, and a scientific slug in a report is exactly what this module
    exists to prevent.
    """
    disp = _display_cfg(cfg)
    head = cfg.taxonomy_cfg["taxon_head"]
    out: dict[str, str] = {}

    for s in cfg.species:
        out[s.slug] = s.common_name

    for slug, node in head["genus_fallback"].items():
        group = disp["genus"].get(node["genus"])
        if group:
            out[slug] = f"{group} ({UNSURE})"

    for slug, node in head["family_fallback"].items():
        group = disp["family"].get(node["family"])
        if group:
            out[slug] = f"{group} ({UNSURE})"

    for slug in head["guild_fallback"]:
        group = disp["guild"].get(slug.removesuffix("_indet"))
        if group:
            out[slug] = f"{group} ({UNSURE})"

    out.update(disp["negative"])
    out["unknown"] = disp["unknown"]

    missing = [c for c in cfg.taxon_classes if c not in out]
    if missing:
        raise RuntimeError(
            f"no display name for {missing}; add it under `display:` in taxonomy.yaml"
        )
    return out


_CACHE: dict[int, dict[str, str]] = {}


def display_map(cfg: Config | None = None) -> dict[str, str]:
    cfg = cfg or load_config()
    key = id(cfg)
    if key not in _CACHE:
        _CACHE[key] = build_display_map(cfg)
    return _CACHE[key]


def display(label: str, cfg: Config | None = None) -> str:
    """English name for one class label.

    Accepts either a class slug (`cinnyris_chalybeus`) or a scientific name as
    written in the configs (`Cinnyris chalybeus`), because the field-ingest side
    carries the latter and callers should not have to know which they hold.

    Unknown labels are returned unchanged rather than raising: this is called
    from logging, and a formatting error must never take down a long run.
    """
    m = display_map(cfg)
    if label in m:
        return m[label]
    return m.get(label.lower().replace(" ", "_"), label)


def display_all(labels: list[str], cfg: Config | None = None) -> list[str]:
    m = display_map(cfg)
    return [m.get(x, x) for x in labels]
