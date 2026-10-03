# Smart Feeder CAD

CadQuery designs for the v6 feeder cradle, Raspberry Pi 4B housing, PMU box,
camera housing, and ToF sensor box. The cradle, Pi, and PMU share one dovetail
geometry.

| Folder | Contents |
|---|---|
| [`src/`](src/) | Parametric CadQuery models and render scripts |
| [`step/`](step/) | Editable STEP exports for the printable parts |
| [`stl/`](stl/) | STL exports for slicing |
| [`docs/`](docs/) | Part dimensions, assembly, and printing notes |

The source scripts are `concept.py` (cradle), `split.py` (printable cradle
halves), `rpi_housing.py`, `pmu_housing.py`, and `sensor_housings.py`. They require CadQuery; the
render scripts also require Matplotlib, NumPy, and `numpy-stl`. Run the scripts
in that order from `src/` to regenerate the models. Each build prints its
geometry checks and should end with `ALL OK`. The scripts write exports to the
current working directory; copy the printable STEP/STL outputs into `step/`
and `stl/` after checking them.

Run `pmu_port_test_wall.py` after `pmu_housing.py` to make a cropped PMU
port-fit print. Its inner wall face is flat on the print bed; it includes the
seven port openings with a 4 mm border, without awnings.

For the PMU full print, slice `stl/pmu_housing_roof_down.stl` and
`stl/pmu_housing_lid_flat.stl`. The test coupon is
`stl/pmu_port_test_wall.stl`. These include the corrected inside-facing port
layout. See the [full-print preview](docs/pmu_full_print_preview.png).

For the camera and ToF boxes, slice `stl/camera_housing_lens_down.stl` and
`stl/tof_housing_face_down.stl`. The unsuffixed STEP/STL files retain their CAD
coordinates. The [camera render](docs/camera_housing_draft.png) shows its
rails on the cradle's 38 × 32 × 5 mm arm pad and a 30 × 28 mm camera removal
opening. The [ToF render](docs/tof_housing_draft.png) shows the 7 mm optical opening and
four-wire exit. The ToF box uses the approximate 12 × 17 mm PCB size supplied
by the user and has no mounting feature. Check board and connector clearance
against the physical parts before a full print.

The Pi housing fits the deployment board, a **Raspberry Pi 4B**. Confirm cable
and connector clearance with the physical board before printing.
