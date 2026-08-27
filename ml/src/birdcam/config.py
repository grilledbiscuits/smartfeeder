"""Configuration loading and label-space construction.

Single source of truth for everything the rest of the package needs to know
about species, the label hierarchy, and hyperparameters. Nothing downstream
should ever open a YAML file directly, and no module should ever contain a
hardcoded species name, taxon ID or path.

The label space is *derived*, not written down twice: taxon head classes are
built from species.yaml plus the fallback nodes in taxonomy.yaml. That way
adding a species to species.yaml is the only edit required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when config files are internally inconsistent.

    Deliberately fatal. A silently-wrong label hierarchy produces a model that
    trains happily and means nothing.
    """


def repo_root() -> Path:
    """Locate the ML project root by walking up for `config/train.yaml`.

    Every path in the project is resolved against this, so getting it wrong
    does not raise -- it silently resolves `data/processed` somewhere empty and
    reports zero images.

    The marker used to be `pyproject.toml`. That stopped being safe when the
    repo gained a sibling `web/` application: pyproject.toml lives at the repo
    root, one level ABOVE this project, so the walk-up would return the repo
    root and every data path would land outside `ml/`. `config/train.yaml` is
    the file this package actually needs, which makes it the honest marker --
    and it still works unchanged from a Kaggle notebook, where the working
    directory is not the repo.
    """
    marker = Path("config") / "train.yaml"
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / marker).is_file():
            return parent
    # Kaggle fallback: config/ sits beside the notebook's working directory.
    cwd = Path.cwd()
    if (cwd / marker).is_file():
        return cwd
    raise ConfigError(
        f"Could not locate the ML project root: no {marker} in any parent of "
        f"{here}, and none in {cwd}. Run from inside the repo, or place "
        "config/ beside the notebook."
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} did not parse to a mapping.")
    return data


@dataclass(frozen=True)
class Species:
    """One target species, before taxon-ID resolution."""

    scientific_name: str
    common_name: str
    tier: str
    fetch_scope: str

    @property
    def genus(self) -> str:
        """Genus from the binomial.

        Verified against the genus GBIF returns during resolution (taxa.py);
        a mismatch is a hard error, which is what catches a species being moved
        between genera by a taxonomic revision.
        """
        return self.scientific_name.split()[0]

    @property
    def slug(self) -> str:
        """Class label, e.g. 'Cinnyris chalybeus' -> 'cinnyris_chalybeus'."""
        return self.scientific_name.lower().replace(" ", "_").replace("-", "_")


@dataclass
class Config:
    """Loaded, cross-validated configuration."""

    species_cfg: dict[str, Any]
    taxonomy_cfg: dict[str, Any]
    train_cfg: dict[str, Any]
    root: Path = field(default_factory=repo_root)

    # -- loading ---------------------------------------------------------------

    @classmethod
    def load(cls, root: Path | None = None) -> Config:
        root = root or repo_root()
        cfg_dir = root / "config"
        obj = cls(
            species_cfg=_load_yaml(cfg_dir / "species.yaml"),
            taxonomy_cfg=_load_yaml(cfg_dir / "taxonomy.yaml"),
            train_cfg=_load_yaml(cfg_dir / "train.yaml"),
            root=root,
        )
        obj.validate()
        return obj

    # -- species ---------------------------------------------------------------

    @cached_property
    def species(self) -> list[Species]:
        out: list[Species] = []
        seen: set[str] = set()
        for tier_name, tier in self.species_cfg["tiers"].items():
            scope = tier.get("fetch_scope", "za")
            for entry in tier["species"]:
                name = entry["scientific_name"].strip()
                if name in seen:
                    raise ConfigError(f"Species {name!r} listed more than once in species.yaml")
                seen.add(name)
                if len(name.split()) != 2:
                    raise ConfigError(
                        f"Expected a binomial (two words) but got {name!r}. "
                        "Subspecies and bare genera are not valid target species."
                    )
                out.append(
                    Species(
                        scientific_name=name,
                        common_name=entry["common_name"].strip(),
                        tier=tier_name,
                        fetch_scope=scope,
                    )
                )
        return out

    def species_by_tier(self, tier: str) -> list[Species]:
        return [s for s in self.species if s.tier == tier]

    @cached_property
    def species_by_name(self) -> dict[str, Species]:
        return {s.scientific_name: s for s in self.species}

    # -- taxon label space -----------------------------------------------------

    @cached_property
    def taxon_classes(self) -> list[str]:
        """Full Head 1 label space, in a stable order.

        Order matters: it fixes the meaning of every logit index, so it must be
        deterministic across runs and machines. Species come first (sorted),
        then fallbacks by ascending generality, then negatives.
        """
        head = self.taxonomy_cfg["taxon_head"]
        classes = sorted(s.slug for s in self.species)
        classes += sorted(head["genus_fallback"])
        classes += sorted(head["family_fallback"])
        classes += sorted(head["guild_fallback"])
        classes += list(head["negative"])
        dupes = {c for c in classes if classes.count(c) > 1}
        if dupes:
            raise ConfigError(f"Duplicate taxon class labels: {sorted(dupes)}")
        return classes

    @cached_property
    def taxon_class_index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.taxon_classes)}

    @cached_property
    def sex_classes(self) -> list[str]:
        return list(self.taxonomy_cfg["sex_plumage_head"]["classes"])

    @cached_property
    def sex_class_index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.sex_classes)}

    @cached_property
    def partial_label_groups(self) -> dict[str, list[str]]:
        """Masked-loss groups for Head 2.

        `male_unspecified` maps to [male_breeding, male_eclipse]: nothing we can
        fetch distinguishes the two, so an annotated male trains the summed mass
        of the group rather than either member. See taxonomy.yaml.
        """
        return dict(self.taxonomy_cfg["sex_plumage_head"].get("partial_label_groups", {}))

    def genus_of_class(self, label: str) -> str | None:
        """Genus parent of a species class label, for rollup."""
        for s in self.species:
            if s.slug == label:
                return s.genus
        return None

    @cached_property
    def genus_to_family(self) -> dict[str, str]:
        return dict(self.taxonomy_cfg["taxon_head"]["genus_family"])

    @cached_property
    def species_per_genus(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in self.species:
            out.setdefault(s.genus, []).append(s.scientific_name)
        return out

    @cached_property
    def family_to_guild(self) -> dict[str, str]:
        ff = self.taxonomy_cfg["taxon_head"]["family_fallback"]
        return {v["family"]: v["guild"] for v in ff.values()}

    # -- paths -----------------------------------------------------------------

    def path(self, key: str) -> Path:
        """Resolve a configured path relative to the repo root."""
        paths = self.train_cfg["paths"]
        if key not in paths:
            raise ConfigError(f"Unknown path key {key!r}. Known: {sorted(paths)}")
        return self.root / paths[key]

    def contact_email(self) -> str:
        """Contact email for the User-Agent, from the environment.

        Fetchers refuse to run without it. Sending thousands of unidentified
        requests to a free public API is how you get the project blocked.
        """
        var = self.train_cfg["fetch"]["contact_env_var"]
        value = os.environ.get(var, "").strip()
        if not value or "@" not in value:
            raise ConfigError(
                f"Environment variable {var} must be set to a contact email "
                "address before fetching. iNaturalist and GBIF both expect a "
                "descriptive User-Agent with a way to reach the operator.\n"
                f"  export {var}='you@example.com'"
            )
        return value

    def user_agent(self) -> str:
        from birdcam import __version__

        tpl = self.train_cfg["fetch"]["user_agent_template"]
        return tpl.format(version=__version__, contact=self.contact_email())

    # -- validation ------------------------------------------------------------

    def validate(self) -> None:
        """Cross-check the config files against each other.

        Every check here corresponds to a mistake that would otherwise surface
        as a quietly wrong model rather than an error.
        """
        problems: list[str] = []
        head = self.taxonomy_cfg["taxon_head"]

        # Every species genus must have a family mapping, or rollup has nowhere
        # to roll up to.
        genus_family = head["genus_family"]
        for missing in sorted({s.genus for s in self.species} - set(genus_family)):
            problems.append(
                f"Genus {missing!r} appears in species.yaml but has no entry in "
                "taxon_head.genus_family -- rollup would have no parent for it."
            )

        # Every mapped family needs a family fallback node.
        known_families = {v["family"] for v in head["family_fallback"].values()}
        for genus, fam in sorted(genus_family.items()):
            if fam not in known_families:
                problems.append(
                    f"genus_family maps {genus!r} to family {fam!r}, which has "
                    "no family_fallback node."
                )

        # Genus fallback classes must reference a known genus...
        for gname, g in sorted(head["genus_fallback"].items()):
            genus = g["genus"]
            if genus not in genus_family:
                problems.append(
                    f"genus_fallback {gname!r} references genus {genus!r}, which "
                    "is not in genus_family."
                )
                continue
            # ...and must not be a dead class. A genus holding one species gives
            # the model nothing to be uncertain between: its genus probability
            # would merely duplicate the species probability, and the class
            # could never accumulate training examples of its own.
            n = len(self.species_per_genus.get(genus, []))
            if n < 2:
                problems.append(
                    f"genus_fallback {gname!r} covers genus {genus!r}, which has "
                    f"only {n} species in the label space. Single-species genera "
                    "must roll straight up to family instead -- a genus fallback "
                    "here would be an untrainable dead class."
                )

        # Conversely, a multi-species genus without a fallback loses the ability
        # to express the single most important uncertainty in this project
        # (female Cinnyris chalybeus vs Cinnyris afer).
        declared = {g["genus"] for g in head["genus_fallback"].values()}
        for genus, members in sorted(self.species_per_genus.items()):
            if len(members) >= 2 and genus not in declared:
                problems.append(
                    f"Genus {genus!r} holds {len(members)} species in the label "
                    "space but has no genus_fallback class -- the model would be "
                    "unable to express 'this genus, species unknown'."
                )

        # Guild members must be declared families.
        for guild_name, guild in head["guild_fallback"].items():
            for fam in guild["members"]:
                if fam not in known_families:
                    problems.append(
                        f"guild_fallback {guild_name!r} lists family {fam!r}, "
                        "which has no family_fallback node."
                    )

        sex_head = self.taxonomy_cfg["sex_plumage_head"]
        sex_classes = set(sex_head["classes"])

        # Partial-label groups must reference real sex classes.
        for gname, members in sex_head.get("partial_label_groups", {}).items():
            unknown = set(members) - sex_classes
            if unknown:
                problems.append(
                    f"partial_label_group {gname!r} references unknown sex "
                    f"classes {sorted(unknown)}."
                )
            if len(members) < 2:
                problems.append(
                    f"partial_label_group {gname!r} needs at least two members "
                    "to be meaningful as a masked-loss group."
                )

        # Annotation mappings must land on a real class or a declared group.
        valid_targets = sex_classes | set(sex_head.get("partial_label_groups", {}))
        for source, mapping in sex_head["annotation_mapping"].items():
            for key, target in mapping.items():
                if target not in valid_targets:
                    problems.append(
                        f"annotation_mapping[{source}][{key}] -> {target!r} is "
                        "neither a sex class nor a partial-label group."
                    )

        if sex_head["unannotated_label"] not in sex_classes:
            problems.append(
                f"unannotated_label {sex_head['unannotated_label']!r} is not a sex class."
            )

        # Monomorphic species must actually be in the species list, or the entry
        # is a typo doing nothing.
        for name in sex_head.get("monomorphic_forced_na", []):
            if name not in self.species_by_name:
                problems.append(
                    f"monomorphic_forced_na lists {name!r}, which is not in species.yaml."
                )

        # Rollup thresholds must be ordered: a more general prediction should
        # require more confidence, not less.
        th = self.taxonomy_cfg["rollup"]["thresholds"]
        order = ["species", "genus", "family", "guild"]
        for a, b in zip(order, order[1:], strict=False):
            if th[a] > th[b]:
                problems.append(
                    f"rollup threshold for {a} ({th[a]}) exceeds {b} ({th[b]}). "
                    "Thresholds must increase with generality."
                )

        # Split fractions must sum to 1.
        split = self.train_cfg["preprocess"]["split"]
        total = split["train"] + split["val"] + split["test"]
        if abs(total - 1.0) > 1e-6:
            problems.append(f"Split fractions sum to {total}, expected 1.0.")

        if problems:
            raise ConfigError("Configuration is inconsistent:\n  - " + "\n  - ".join(problems))


def load_config(root: Path | None = None) -> Config:
    return Config.load(root)


def main() -> None:
    """Print a summary of the loaded configuration.

    `uv run python -m birdcam.config` -- the Phase 1 smoke test. Confirms the
    config files parse, cross-validate, and produce the label space the rest of
    the pipeline will be built against.
    """
    from birdcam.utils.runtime import describe_environment

    cfg = load_config()
    print(f"repo root       : {cfg.root}")
    print(
        f"species         : {len(cfg.species)} "
        f"(A={len(cfg.species_by_tier('A'))}, "
        f"B={len(cfg.species_by_tier('B'))}, "
        f"C={len(cfg.species_by_tier('C'))})"
    )
    print(f"taxon classes   : {len(cfg.taxon_classes)}")
    print(f"sex classes     : {len(cfg.sex_classes)} -> {', '.join(cfg.sex_classes)}")
    print(f"masked groups   : {cfg.partial_label_groups}")

    head = cfg.taxonomy_cfg["taxon_head"]
    n_species = len(cfg.species)
    n_genus = len(head["genus_fallback"])
    n_family = len(head["family_fallback"])
    n_guild = len(head["guild_fallback"])
    n_neg = len(head["negative"])
    print(
        f"\nlabel space     : {n_species} species + {n_genus} genus + "
        f"{n_family} family + {n_guild} guild + {n_neg} negative "
        f"= {len(cfg.taxon_classes)}"
    )

    print("\nmulti-species genera (genus fallback is meaningful):")
    for genus, members in sorted(cfg.species_per_genus.items()):
        if len(members) >= 2:
            print(f"  {genus:<14} {len(members):>2} species -> {cfg.genus_to_family[genus]}")

    print("\nrollup thresholds:")
    for level, th in cfg.taxonomy_cfg["rollup"]["thresholds"].items():
        print(f"  {level:<8} {th}")

    print("\nenvironment:")
    for k, v in describe_environment().items():
        print(f"  {k:<16} {v}")


if __name__ == "__main__":
    main()
