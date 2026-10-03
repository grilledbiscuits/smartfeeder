"""Printable PMU end-wall coupon for checking port alignment before the box."""

import cadquery as cq
import pmu_housing as pmu


def wall():
    # Leave a 4 mm border around the outermost ports for a faster fit print.
    cutters = pmu.port_cutters()
    bounds = [cutter.val().BoundingBox() for cutter in cutters.values()]
    border = 4.0
    x0 = min(box.xmin for box in bounds) - border
    x1 = max(box.xmax for box in bounds) + border
    z0 = min(box.zmin for box in bounds) - border
    z1 = max(box.zmax for box in bounds) + border
    part = pmu._box(x0, x1, pmu.Y_INT, pmu.Y_BOX, z0, z1)
    for cutter in cutters.values():
        part = part.cut(cutter)
    return part.intersect(pmu.housing())


def build():
    part = wall()
    assert len(part.val().Solids()) == 1
    assert pmu.R.vol(part.cut(pmu.housing())) < 0.5
    for name, cutter in pmu.port_cutters().items():
        assert pmu.R.vol(part.intersect(cutter)) < 0.5, name

    # Cycle model X/Y/Z to print Y/Z/X: flat inner wall face becomes Z=0.
    source_bb = part.val().BoundingBox()
    printable = (part.rotate((0, 0, 0), (1, 1, 1), 120)
                 .translate((-source_bb.zmin, -source_bb.xmin, -pmu.Y_INT)))
    bb = printable.val().BoundingBox()
    assert abs(bb.zmin) < 0.01
    assert abs(bb.zlen - pmu.WALL) < 0.01
    print("PMU port test wall: %.1f × %.1f × %.1f mm, %d solid" %
          (bb.xlen, bb.ylen, bb.zlen, len(printable.val().Solids())))
    cq.exporters.export(printable, "pmu_port_test_wall.step")
    cq.exporters.export(printable, "pmu_port_test_wall.stl")
    print("ALL OK")


if __name__ == "__main__":
    build()
