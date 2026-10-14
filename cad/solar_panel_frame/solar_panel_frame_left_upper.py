from cadgen import step, stl
from split_frame import print_section


@step(out="solar_panel_frame_left_upper.step")
@stl(out="solar_panel_frame_left_upper.stl", mesh_tolerance=0.0001)
def solar_panel_frame_left_upper():
    part = print_section("left", True)
    part.label = "solar_panel_frame_left_upper"
    return part


if __name__ == "__main__":
    solar_panel_frame_left_upper()
