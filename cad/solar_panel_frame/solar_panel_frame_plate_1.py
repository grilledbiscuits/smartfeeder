"""All six frame pieces packed flat on one Ender-3 V3 SE build plate."""
from cadgen import build123d as bd, step, stl
from solar_panel_frame_left_lower import solar_panel_frame_left_lower
from solar_panel_frame_left_upper import solar_panel_frame_left_upper
from solar_panel_frame_right_lower import solar_panel_frame_right_lower
from solar_panel_frame_right_upper import solar_panel_frame_right_upper
from solar_panel_frame_center_lower import solar_panel_frame_center_lower
from solar_panel_frame_center_upper import solar_panel_frame_center_upper


@step(out="solar_panel_frame_plate_1.step")
@stl(out="solar_panel_frame_plate_1.stl", mesh_tolerance=0.0001)
def solar_panel_frame_plate_1():
    parts = []
    for model, angle, x, y in (
        (solar_panel_frame_left_lower, 180, 54.931848, 75.907707),
        (solar_panel_frame_left_upper, 180, 78.965582, 34.961489),
        (solar_panel_frame_right_lower, 270, 40.649999, 12.869710),
        (solar_panel_frame_right_upper, 90, 12.148672, 51.188674),
        (solar_panel_frame_center_lower, 180, 87.500307, 6.486115),
        (solar_panel_frame_center_upper, 270, 5.211000, 32.912879),
    ):
        part = model().rotate(bd.Axis.Z, angle)
        low = part.bounding_box().min
        part = bd.Pos(x - low.X, y - low.Y, -low.Z) * part
        parts.append(part)
    plate = bd.Compound(children=parts, label="six_parts_one_print")
    box = plate.bounding_box()
    assert len(plate.solids()) == 6
    assert box.min.X > 3 and box.min.Y > 3
    assert box.max.X < 217 and box.max.Y < 217
    return plate


if __name__ == "__main__":
    solar_panel_frame_plate_1()
