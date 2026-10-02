# Smart Feeder CAD

CadQuery designs for the v6 feeder cradle, Raspberry Pi 4B housing, and
Waveshare power management box. The three designs share one dovetail geometry.

| Folder | Contents |
|---|---|
| [`src/`](src/) | Parametric CadQuery models and render scripts |
| [`step/`](step/) | Editable STEP exports for the printable parts |
| [`stl/`](stl/) | STL exports for slicing |
| [`docs/`](docs/) | Part dimensions, assembly, and printing notes |

The source scripts are `concept.py` (cradle), `split.py` (printable cradle
halves), `rpi_housing.py`, and `pmu_housing.py`. They require CadQuery; the
render scripts also require Matplotlib, NumPy, and `numpy-stl`. Run the scripts
in that order from `src/` to regenerate the models. Each build prints its
geometry checks and should end with `ALL OK`. The scripts write exports to the
current working directory; copy the printable STEP/STL outputs into `step/`
and `stl/` after checking them.

The Pi housing fits the deployment board, a **Raspberry Pi 4B**. Confirm cable
and connector clearance with the physical board before printing.
