"""Build all six flat, support-free solar frame pieces."""
from solar_panel_frame_plate_1 import solar_panel_frame_plate_1
from solar_panel_frame_exploded import solar_panel_frame_exploded
from solar_panel_frame_left_lower import solar_panel_frame_left_lower
from solar_panel_frame_left_upper import solar_panel_frame_left_upper
from solar_panel_frame_center_lower import solar_panel_frame_center_lower
from solar_panel_frame_center_upper import solar_panel_frame_center_upper
from solar_panel_frame_right_lower import solar_panel_frame_right_lower
from solar_panel_frame_right_upper import solar_panel_frame_right_upper


if __name__ == "__main__":
    solar_panel_frame_left_lower()
    solar_panel_frame_left_upper()
    solar_panel_frame_center_lower()
    solar_panel_frame_center_upper()
    solar_panel_frame_right_lower()
    solar_panel_frame_right_upper()
    solar_panel_frame_plate_1()
    solar_panel_frame_exploded()
