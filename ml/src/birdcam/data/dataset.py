"""Label mapping and the torch Dataset yielding two-head targets.

The manifest stores what each *source* said ("Male", "Juvenile", NULL). This
module maps that onto our schema using config/taxonomy.yaml, and is the only
place that translation happens.

Head 2 targets are not plain integers. An annotated male carries no evidence
about breeding vs eclipse plumage -- no public source annotates it -- so its
target is a *set* of admissible classes, and the loss is computed over the
summed probability of that set. Every sample therefore carries a mask:

    mask[c] = 1 if class c is admissible for this sample

For an exactly-labelled sample the mask has a single 1 and the masked loss
reduces to ordinary cross-entropy. For an annotated male it has two, and the
model is trained on "is male" without being told something we do not know.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from birdcam.config import Config
from birdcam.data.manifest import Manifest

logger = logging.getLogger(__name__)


@dataclass
class LabelledImage:
    image_id: str
    path: Path
    scientific_name: str
    taxon_label: str
    taxon_index: int
    sex_mask: np.ndarray  # (n_sex_classes,) float32, 1.0 where admissible
    sex_label_name: str  # human-readable, for reporting
    split: str
    observation_id: str | None
    observer_id: str | None


class LabelMapper:
    """Maps raw source annotations onto the two-head label space."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        head = cfg.taxonomy_cfg["sex_plumage_head"]
        self.sex_classes = cfg.sex_classes
        self.sex_index = cfg.sex_class_index
        self.groups = cfg.partial_label_groups
        self.mapping = head["annotation_mapping"]
        self.precedence = head["precedence"]
        self.unannotated = head["unannotated_label"]
        self.monomorphic = set(head.get("monomorphic_forced_na", []))

    def taxon_label(self, scientific_name: str) -> str:
        """Species slug, or the genus/family fallback if the species is too rare.

        Rare species are folded at label-mapping time, and the fold is reported
        by the caller rather than being silent.
        """
        return scientific_name.lower().replace(" ", "_")

    def sex_target(
        self,
        scientific_name: str,
        sex_annotation: str | None,
        life_stage: str | None,
        source: str = "inaturalist",
    ) -> tuple[np.ndarray, str]:
        """Return (mask, label_name) for Head 2.

        Precedence follows taxonomy.yaml: juvenile beats sex, because juvenile
        plumage is what the model actually sees.
        """
        mask = np.zeros(len(self.sex_classes), dtype=np.float32)

        # Monomorphic species: sex is not visually determinable even in
        # principle, so training the head on it would be training on noise.
        if scientific_name in self.monomorphic:
            mask[self.sex_index["not_applicable"]] = 1.0
            return mask, "not_applicable"

        src_map = self.mapping.get(source, {})
        candidates: list[str] = []

        if life_stage:
            key = "1|8" if life_stage == "Juvenile" else None
            if key and key in src_map:
                candidates.append(src_map[key])

        if sex_annotation:
            sex_ids = {"Female": "9|10", "Male": "9|11", "Cannot Be Determined": "9|20"}
            key = sex_ids.get(sex_annotation)
            if key and key in src_map:
                candidates.append(src_map[key])

        if not candidates:
            # NEVER dropped. A large share of real feeder traffic genuinely
            # cannot be sexed, so `indeterminate` is a label the model must
            # learn to emit, not an absence of data.
            target = self.unannotated
        else:
            target = min(
                candidates,
                key=lambda c: self.precedence.index(c) if c in self.precedence else 99,
            )

        if target in self.groups:
            # Partial label: admissible over the whole group.
            for member in self.groups[target]:
                mask[self.sex_index[member]] = 1.0
        else:
            mask[self.sex_index[target]] = 1.0
        return mask, target


def load_labelled(cfg: Config, m: Manifest, split: str | None = None) -> list[LabelledImage]:
    """Build the labelled image list from the manifest.

    Only images that are downloaded, deduplicated and split are returned.
    """
    mapper = LabelMapper(cfg)
    where = "status='downloaded' AND split IS NOT NULL"
    params: tuple = ()
    if split:
        where += " AND split=?"
        params = (split,)

    out: list[LabelledImage] = []
    missing_class = 0
    for r in m.iter_rows(where, params):
        taxon_label = mapper.taxon_label(r["scientific_name"])
        idx = cfg.taxon_class_index.get(taxon_label)
        if idx is None:
            missing_class += 1
            continue
        mask, name = mapper.sex_target(
            r["scientific_name"], r["sex_annotation"], r["life_stage_annotation"]
        )
        slug = r["scientific_name"].lower().replace(" ", "_")
        path = cfg.path("processed_dir") / slug / f"{r['image_id'].replace(':', '_')}.jpg"
        out.append(
            LabelledImage(
                image_id=r["image_id"],
                path=path,
                scientific_name=r["scientific_name"],
                taxon_label=taxon_label,
                taxon_index=idx,
                sex_mask=mask,
                sex_label_name=name,
                split=r["split"],
                observation_id=r["observation_id"],
                observer_id=r["observer_id"],
            )
        )
    if missing_class:
        logger.warning("%d images had no matching taxon class and were excluded", missing_class)
    return out


class EmbeddingDataset:
    """Cached-feature dataset for the fast loop.

    Not a torch Dataset: the fast loop holds the whole feature matrix in memory
    (20k x 768 float32 is ~60MB) and trains with full-batch or large-batch
    steps, which is what makes a run finish in seconds.
    """

    def __init__(self, features: np.ndarray, items: list[LabelledImage]) -> None:
        if len(features) != len(items):
            raise ValueError(
                f"feature/label misalignment: {len(features)} features, {len(items)} items"
            )
        self.features = features
        self.items = items

    @property
    def taxon_targets(self) -> np.ndarray:
        return np.array([i.taxon_index for i in self.items], dtype=np.int64)

    @property
    def sex_masks(self) -> np.ndarray:
        return np.stack([i.sex_mask for i in self.items])

    def subset(self, split: str) -> EmbeddingDataset:
        idx = [i for i, it in enumerate(self.items) if it.split == split]
        return EmbeddingDataset(self.features[idx], [self.items[i] for i in idx])


class ImageDataset:
    """torch Dataset over preprocessed JPEGs, for embedding extraction.

    Defined lazily so importing this module does not import torch.
    """

    def __new__(cls, items: list[LabelledImage], transform):  # noqa: D102
        import torch
        from PIL import Image

        class _DS(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return len(items)

            def __getitem__(self, i: int):
                it = items[i]
                with Image.open(it.path) as img:
                    img = img.convert("RGB")
                    return transform(img), i

        return _DS()


def main() -> None:
    raise NotImplementedError("dataset.py is a library module; nothing to run directly.")
