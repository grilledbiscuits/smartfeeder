"""Six-piece exploded reference assembly; not a print layout."""
from cadgen import build123d as bd, step, srgb
from solar_panel_frame import solar_panel_frame
from split_frame import section_shape


@step(out="solar_panel_frame_exploded.step")
def solar_panel_frame_exploded():
    frame = solar_panel_frame()
    parts = []
    colors = ("#5288B5", "#70A58C", "#C79751", "#946CA8", "#BA7060", "#6AADB8")
    for section, dx in (("left", -28), ("center", 0), ("right", 28)):
        for upper in (False, True):
            row = "upper" if upper else "lower"
            part = section_shape(frame, section, upper)
            part = bd.Pos(dx, 22 if upper else -22, 0) * part
            part.label = f"{section}_{row}"
            part.color = srgb(colors[len(parts)])
            parts.append(part)
    return bd.Compound(children=parts, label="six_piece_exploded_frame")


if __name__ == "__main__":
    solar_panel_frame_exploded()
