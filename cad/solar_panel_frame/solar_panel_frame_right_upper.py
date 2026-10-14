from cadgen import step, stl
from split_frame import print_section


@step(out="solar_panel_frame_right_upper.step")
@stl(out="solar_panel_frame_right_upper.stl", mesh_tolerance=0.0001)
def solar_panel_frame_right_upper():
    part = print_section("right", True)
    part.label = "solar_panel_frame_right_upper"
    return part


if __name__ == "__main__":
    solar_panel_frame_right_upper()
