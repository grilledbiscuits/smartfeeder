"""Runtime environment setup: threads, seeds, and platform detection.

Kept separate from config.py because it has side effects (it mutates global
torch state) and because train_full.py must be runnable *unmodified* in a
Kaggle notebook -- which means thread and device selection has to adapt to the
environment rather than being pinned in a config file.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def on_kaggle() -> bool:
    """True when running inside a Kaggle notebook.

    Kaggle sets KAGGLE_KERNEL_RUN_TYPE; /kaggle/input is the mounted dataset
    directory and is a reliable secondary signal.
    """
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").is_dir()


def physical_core_count() -> int:
    """Physical cores, not logical.

    Hyperthreads share execution units, and on a convolution workload the
    contention makes torch *slower* with threads set to the logical count. Falls
    back to a conservative estimate if the topology cannot be read.
    """
    try:
        # Counting distinct (physical_id, core_id) pairs is the portable way to
        # get physical cores on Linux without a third-party dependency.
        cores: set[tuple[str, str]] = set()
        phys = core = None
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip() and phys is not None and core is not None:
                    cores.add((phys, core))
                    phys = core = None
        if phys is not None and core is not None:
            cores.add((phys, core))
        if cores:
            return len(cores)
    except OSError:
        pass
    logical = os.cpu_count() or 2
    return max(1, logical // 2)


def setup_torch(cfg) -> None:
    """Configure torch threading and seeds.

    On Kaggle the GPU does the work and thread count is irrelevant, so the
    configured CPU thread count is ignored there.
    """
    import torch

    seed = cfg.train_cfg["compute"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    if on_kaggle():
        logger.info("Kaggle environment detected; leaving torch thread count at default.")
        return

    configured = cfg.train_cfg["compute"]["torch_num_threads"]
    actual = physical_core_count()
    if configured != actual:
        logger.warning(
            "config torch_num_threads=%d but this machine has %d physical cores. "
            "Using %d. Update config/train.yaml if this machine is the new target.",
            configured,
            actual,
            actual,
        )
    torch.set_num_threads(actual)
    logger.info("torch threads set to %d (physical cores)", actual)


def describe_environment() -> dict[str, object]:
    """Environment summary, for logging into reports.

    Metrics without the environment that produced them are hard to compare
    across a Kaggle run and a local run.
    """
    info: dict[str, object] = {
        "kaggle": on_kaggle(),
        "physical_cores": physical_core_count(),
        "logical_cores": os.cpu_count(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = None
    return info
