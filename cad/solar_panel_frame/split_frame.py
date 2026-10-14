"""Three lengthwise sections, each divided across Y for flat Ender prints."""
from cadgen import build123d as bd
from solar_panel_frame import solar_panel_frame, HEIGHT, PANEL_L


def section_shape(frame, section, upper):
    cut = PANEL_L / 6
    x0, x1 = {"left": (-250, -cut), "center": (-cut, cut),
              "right": (cut, 250)}[section]
    mask = bd.Pos((x0 + x1) / 2, 150 if upper else -150,
                  HEIGHT / 2) * bd.Box(x1 - x0, 300, HEIGHT + 2)
    return frame & mask


def print_section(section, upper):
    part = section_shape(solar_panel_frame(), section, upper)
    low = part.bounding_box().min
    part = bd.Pos(-low.X, -low.Y, -low.Z) * part
    size = part.bounding_box().size
    assert len(part.solids()) == 1 and part.is_valid
    assert size.X < 220 and size.Y < 220 and size.Z < 250
    return part
