from cadgen import step, stl
from split_frame import print_section


@step(out="solar_panel_frame_center_lower.step")
@stl(out="solar_panel_frame_center_lower.stl", mesh_tolerance=0.0001)
def solar_panel_frame_center_lower():
    part = print_section("center", False)
    part.label = "solar_panel_frame_center_lower"
    return part


if __name__ == "__main__":
    solar_panel_frame_center_lower()
