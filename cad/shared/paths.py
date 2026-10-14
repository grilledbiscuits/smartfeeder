"""Component paths shared by the existing CadQuery builds and renderers."""
from pathlib import Path

CAD = Path(__file__).resolve().parents[1]


def model_path(name, extension):
    component = "cradle"
    for prefix, folder in (("rpi", "rpi_housing"), ("pmu", "pmu_housing"),
                           ("camera", "camera_housing"), ("tof", "tof_housing")):
        if name.startswith(prefix):
            component = folder
            break
    return CAD / component / f"{name}.{extension}"


def export_part(part, name, extension):
    import cadquery as cq
    cq.exporters.export(part, str(model_path(name, extension)))
