# Smart Feeder CAD

Each component folder contains its model files, STEP/STL exports, notes and previews.
Existing part names are preserved. Sources shared by several components live in `shared/`.

| Component folder | Contents / build entrypoint |
|---|---|
| [cradle/](cradle/) | Cradle halves and pins; `split.py`, shared geometry in `concept.py` |
| [rpi_housing/](rpi_housing/) | Raspberry Pi 4B body, lid and notes; `rpi_housing.py` |
| [pmu_housing/](pmu_housing/) | PMU body, lid and test coupon; `pmu_housing.py`, `pmu_port_test_wall.py` |
| [camera_housing/](camera_housing/) | Camera box and preview; built by `shared/sensor_housings.py` |
| [tof_housing/](tof_housing/) | ToF box and preview; built by `shared/sensor_housings.py` |
| [solar_panel_frame/](solar_panel_frame/) | Six flat frame sections and print layouts; `build.py` |

## Existing CadQuery models

Run the build entrypoints with CadQuery installed. The render scripts additionally
use Matplotlib, NumPy and `numpy-stl`. Export paths now resolve to the component
folders regardless of the current working directory; the shared models import
the cradle's geometry to retain their existing interfaces.

The PMU print orientations are `pmu_housing/pmu_housing_roof_down.stl` and
`pmu_housing/pmu_housing_lid_flat.stl`. Its coupon is
`pmu_housing/pmu_port_test_wall.stl`. For the camera and ToF, use
`camera_housing/camera_housing_lens_down.stl` and
`tof_housing/tof_housing_face_down.stl`. Their unsuffixed exports retain assembly
coordinates. These pre-existing geometries were moved without changing them.

## Solar panel frame

See [frame dimensions and assembly](solar_panel_frame/solar-panel-frame.md).
The frame fits a 340 × 232 × 17 mm panel, with 1 mm clearance per side,
a 13 mm inner support ledge, an open center and four corner suspension holes.
Its six pieces print flat and have plain faces for superglue assembly.

The frame uses the text-to-cad skill's pinned cadgen environment. From the repo root:

```bash
uvx --no-config --managed-python --python 3.13 --from cadgen==0.7.21 python cad/solar_panel_frame/build.py
uvx --no-config --managed-python --python 3.13 --from cadgen==0.7.21 python cad/solar_panel_frame/checks.py
```

The Ender-3 V3 SE build volume is 220 × 220 × 250 mm. Solar-frame print files
are placed flat at Z=0; supports are not required by their modeled geometry.
