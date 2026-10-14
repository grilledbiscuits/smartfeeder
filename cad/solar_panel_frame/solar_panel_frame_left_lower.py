from cadgen import step, stl
from split_frame import print_section


@step(out="solar_panel_frame_left_lower.step")
@stl(out="solar_panel_frame_left_lower.stl", mesh_tolerance=0.0001)
def solar_panel_frame_left_lower():
    part = print_section("left", False)
    part.label = "solar_panel_frame_left_lower"
    return part


if __name__ == "__main__":
    solar_panel_frame_left_lower()
