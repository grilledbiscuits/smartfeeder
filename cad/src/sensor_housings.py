"""Camera Module 3 and VL53L1X enclosure drafts.

The camera housing shares the cradle coordinate frame. Its local x follows
the camera pad, local y points toward the feeder port, and local z rises from
the pad's top. The ToF box is independent and has no mounting features.
"""

import cadquery as cq
import concept as cradle


def box(x0, x1, y0, y1, z0, z1):
    return (cq.Workplane("XY")
            .box(x1 - x0, y1 - y0, z1 - z0, centered=(False, False, False))
            .translate((x0, y0, z0)))


def camera_local():
    """Open-bottom box with two C rails around the 38 × 32 × 5 mm arm pad."""
    part = box(-20.5, 20.5, -19.5, 19.5, 0.4, 35)
    part = part.cut(box(-18, 18, -17, 17, -1, 32.5))
    for side in (-1, 1):
        outer = (16.6, 19.5) if side > 0 else (-19.5, -16.6)
        lip = (14.5, 19.5) if side > 0 else (-19.5, -14.5)
        end = -2 if side > 0 else 18  # clear the arm handle at this shoulder
        part = part.union(box(-20, end, *outer, -5.3, 0.5))
        part = part.union(box(-20, end, *lip, -7.5, -5.3))
    # Clear the whole 25 × 24 mm camera PCB for removal through this face.
    return part.cut(box(-15, 15, 16, 21, 4, 32))


def camera_housing(local=None):
    """Place the box on the existing camera arm's rectangular pad."""
    part = camera_local() if local is None else local
    return (part.rotate((0, 0, 0), (1, 1, 1), 120)
            .rotate((0, 0, 0), (1, 0, 0), cradle.CAM_SLOT_ANGLE)
            .translate((cradle.CAM_X + cradle.CAM_T, *cradle.CAM_SLOT_POS)))


def tof_housing():
    """Open-back shell for an approximately 12 × 17 mm PCB and four leads."""
    part = box(-12, 12, -9.5, 9.5, 0, 14)
    part = part.cut(box(-10, 10, -7.5, 7.5, -1, 11.5))
    aperture = cq.Workplane(obj=cq.Solid.makeCylinder(
        3.5, 4, cq.Vector(0, 0, 11), cq.Vector(0, 0, 1)))
    return part.cut(aperture).cut(box(-7.5, 7.5, -10, -7, -1, 10))


def bed_ready(part):
    bb = part.val().BoundingBox()
    return part.translate((-bb.xmin, -bb.ymin, -bb.zmin))


def build():
    local = camera_local()
    camera = camera_housing(local)
    tof = tof_housing()
    arm = cradle.camera_arm()
    pad = (cq.Workplane("YZ").workplane(offset=cradle.CAM_X)
           .polyline(cradle.pad_corners()).close().extrude(cradle.CAM_T))
    assert all(len(part.val().Solids()) == 1 for part in (local, camera, tof))
    assert camera.intersect(pad).val().Volume() < 0.01
    assert local.intersect(box(-12.5, 12.5, 0, 22, 7, 31)).val().Volume() < 0.01
    u, _v = cradle._pad_axes()
    for back in (0, 10, 20, 30):
        shifted = camera.translate((0, -back * u[0], -back * u[1]))
        assert shifted.intersect(arm).val().Volume() < 0.01, back

    camera_print = bed_ready(local.rotate((0, 0, 0), (1, 0, 0), -90))
    tof_print = bed_ready(tof.rotate((0, 0, 0), (1, 0, 0), 180))
    for part in (camera_print, tof_print):
        assert len(part.val().Solids()) == 1
        assert abs(part.val().BoundingBox().zmin) < 0.01

    for name, part in (("camera_housing", camera), ("tof_housing", tof)):
        cq.exporters.export(part, name + ".step")
        cq.exporters.export(part, name + ".stl")
    for name, part in (("camera_housing_lens_down", camera_print),
                       ("tof_housing_face_down", tof_print)):
        cq.exporters.export(part, name + ".stl")
    print("Camera box: 41 × 39 × 35 mm above pad; removal opening 30 × 28 mm")
    print("ToF box: 24 × 19 × 14 mm; optical hole Ø7 mm; wire exit 15 × 10 mm")
    print("Arm and slide path clear; both print STLs on the bed; ALL OK")


if __name__ == "__main__":
    build()
