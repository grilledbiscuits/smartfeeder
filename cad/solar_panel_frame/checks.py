"""Check the saved draft against its fit, support and suspension requirements."""
import sys
from pathlib import Path
from cadgen import read_step, build123d as bd
from cadgen.geometry import overlap_volume, closest_points

COMPONENT = Path(__file__).resolve().parent
sys.path.insert(0, str(COMPONENT))
from solar_panel_frame import (INNER_L, INNER_W, LEDGE_W, LEDGE_T, HEIGHT,
                              PANEL_L, PANEL_W, PANEL_T, HOLE_X, HOLE_Y,
                              WIRE_D, LUG_T)

frame = read_step(COMPONENT / "solar_panel_frame.step")
align = (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)
assert frame.is_valid and len(frame.solids()) == 1
def volume_in(shape):
    return sum(s.volume for s in (frame & shape).solids())

cavity = bd.Pos(0, 0, LEDGE_T) * bd.Box(INNER_L, INNER_W, HEIGHT, align=align)
assert volume_in(cavity) < 0.001
opening = bd.Pos(0, 0, -1) * bd.Box(
    INNER_L - 2 * LEDGE_W, INNER_W - 2 * LEDGE_W, HEIGHT + 2, align=align)
assert volume_in(opening) < 0.001
for dx, dy in ((0, 0), (-0.9, -0.9), (0.9, 0.9)):
    panel = bd.Pos(dx, dy, LEDGE_T) * bd.Box(PANEL_L, PANEL_W, PANEL_T, align=align)
    assert volume_in(panel) < 0.001
for sx in (-1, 1):
    for sy in (-1, 1):
        bore = bd.Pos(sx * HOLE_X, sy * HOLE_Y, -1) * bd.Cylinder(
            WIRE_D / 2 - 0.001, LUG_T + 2, align=align)
        assert volume_in(bore) < 0.001
for x, y in ((INNER_L / 2 - 12, 0), (-INNER_L / 2 + 12, 0),
             (0, INNER_W / 2 - 12), (0, -INNER_W / 2 + 12)):
    assert volume_in(bd.Pos(x, y, 1) * bd.Box(2, 2, 3, align=align)) > 11.9
print('PASS: 342x234 cavity, 1 mm panel clearance per side, 316x208 open center')
print('PASS: 13 mm ledge on all four sides, 5 mm thickness, four 5 mm through holes')
print('PASS: one valid solid, centered and offset panel fit without interference')
print('Overall envelope (mm):', frame.bounding_box().size)

from split_frame import section_shape
names = ("left_lower", "left_upper", "center_lower", "center_upper", "right_lower", "right_upper")
parts = []
for name in names:
    part = read_step(COMPONENT / f"solar_panel_frame_{name}.step")
    assert len(part.solids()) == 1 and part.is_valid
    box = part.bounding_box()
    assert box.size.X < 220 and box.size.Y < 220 and abs(box.size.Z - HEIGHT) < 0.001
    assert abs(box.min.Z) < 0.001
    section, row = name.split("_")
    expected = section_shape(frame, section, row == "upper")
    low = expected.bounding_box().min
    restored = bd.Pos(low.X, low.Y, low.Z) * part
    assert abs(restored.volume - expected.volume) < 0.01
    assert abs(overlap_volume(restored.solids()[0], expected.solids()[0]) - expected.volume) < 0.01
    parts.append(part)
    print(name, "flat envelope (mm):", box.size)
assert abs(sum(p.volume for p in parts) - frame.volume) < 0.01
print("PASS: all six flat pieces fit the Ender, restore their assembly positions and conserve frame volume")

plate = read_step(COMPONENT / "solar_panel_frame_plate_1.step")
solids = plate.solids()
assert len(solids) == 6 and plate.is_valid
box = plate.bounding_box()
assert box.min.X > 3 and box.min.Y > 3 and box.max.X < 217 and box.max.Y < 217
assert abs(sum(s.volume for s in solids) - frame.volume) < 0.01
gaps = []
for i, part in enumerate(solids):
    assert abs(part.bounding_box().min.Z) < 0.001
    for other in solids[:i]:
        gap = closest_points(part, other).distance
        assert gap >= 2.99, gap
        gaps.append(gap)
print("PASS: one plate, six flat pieces; minimum part gap (mm):", min(gaps))
print("Packed print envelope (mm):", box.size)
