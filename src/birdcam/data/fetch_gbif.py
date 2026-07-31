"""GBIF occurrence-API fetcher (non-iNaturalist datasets only).

NOT YET IMPLEMENTED -- scheduled for Phase 2.

This file is a placeholder created during Phase 1 scaffolding so the package
layout is fixed and imports resolve. It deliberately raises rather than
returning empty or synthetic results: a fetcher that quietly returns nothing is
indistinguishable from a species with no data, and that distinction matters.
"""

from __future__ import annotations

PHASE = 2


def main() -> None:
    raise NotImplementedError(
        "birdcam.data.fetch_gbif is a Phase 2 deliverable and is not implemented yet."
    )


if __name__ == "__main__":
    main()
