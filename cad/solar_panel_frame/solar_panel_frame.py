"""Solar-panel frame. Dimensions in mm; XY horizontal, Z upward."""
from cadgen import build123d as bd, step, srgb

PANEL_L, PANEL_W, PANEL_T = 340.0, 232.0, 17.0
CLEARANCE = 1.0
WALL = 6.0
LEDGE_W, LEDGE_T = 13.0, 5.0
WIRE_D = 5.0
LUG_R, LUG_T = 12.0, 12.0
INNER_L, INNER_W = PANEL_L + 2 * CLEARANCE, PANEL_W + 2 * CLEARANCE
OUTER_L, OUTER_W = INNER_L + 2 * WALL, INNER_W + 2 * WALL
HEIGHT = LEDGE_T + PANEL_T
HOLE_X, HOLE_Y = OUTER_L / 2 + 4.0, OUTER_W / 2 + 4.0


@step(out="solar_panel_frame.step")
def solar_panel_frame():
    base_align = (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)
    frame = bd.Box(OUTER_L, OUTER_W, HEIGHT, align=base_align)
    # Two nested cuts leave a peripheral ledge and an open underside.
    frame -= bd.Pos(0, 0, LEDGE_T) * bd.Box(
        INNER_L, INNER_W, HEIGHT, align=base_align)
    frame -= bd.Pos(0, 0, -1) * bd.Box(
        INNER_L - 2 * LEDGE_W, INNER_W - 2 * LEDGE_W,
        LEDGE_T + 2, align=base_align)
    for sx in (-1, 1):
        for sy in (-1, 1):
            pos = bd.Pos(sx * HOLE_X, sy * HOLE_Y, 0)
            frame += pos * bd.Cylinder(LUG_R, LUG_T, align=base_align)
            frame -= bd.Pos(sx * HOLE_X, sy * HOLE_Y, -1) * bd.Cylinder(
                WIRE_D / 2, LUG_T + 2, align=base_align)
            # Break the hole mouths so suspension wire avoids a sharp edge.
            frame -= pos * bd.Cone(WIRE_D / 2 + 0.8, WIRE_D / 2, 0.8,
                                    align=base_align)
            frame -= bd.Pos(sx * HOLE_X, sy * HOLE_Y, LUG_T - 0.8) * bd.Cone(
                WIRE_D / 2, WIRE_D / 2 + 0.8, 0.8, align=base_align)
    frame.label = "solar_panel_frame"
    frame.color = srgb("#D9DEE5")
    assert len(frame.solids()) == 1
    assert frame.is_valid
    return frame


if __name__ == "__main__":
    solar_panel_frame()
