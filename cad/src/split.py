"""Option 4: split the v6 cradle in two, joined by sliding dovetails + cross pins.

Why this joint shape. All three members that cross the split - the two side
beams and the bottom spine - have to engage in ONE motion, and the only motion
available is along the bottle axis. So each tongue is a constant-section
dovetail PRISM pointing aft: the undercut is in the member's cross-section
plane, so the joint cannot open radially or sideways, and axial pull-out (the
one direction a dovetail like this leaves free) is taken by a cross pin.

Front part carries the tongues, rear part the sockets.
"""
import math
import cadquery as cq
import concept as c

SPLIT_Z = 158.0
TONGUE_L = 18.0
CLEAR = 0.20                 # per face, tongue -> socket
PIN_D = 4.0
BED = 180.0
SPINE_FACE_X_TIP = c.SPINE_FACE_X - c.RAIL_H   # -67: the rail's outer face

# (label, azimuth, r_in, r_out, width at r_in, width at r_out, pin axis)
# width shrinks with radius, so the tongue cannot be drawn radially outward.
JOINTS = [
    ("beam +Y", 90.0, 51.0, 56.0, 14.0, 11.5),
    ("beam -Y", 270.0, 51.0, 56.0, 14.0, 11.5),
    ("spine", 180.0, 48.0, 55.0, 18.0, 14.5),
]


def _prism(r_in, r_out, w_in, w_out, z0, z1, az, grow=0.0):
    pts = [(r_in - grow, (w_in + 2 * grow) / 2),
           (r_out + grow, (w_out + 2 * grow) / 2),
           (r_out + grow, -(w_out + 2 * grow) / 2),
           (r_in - grow, -(w_in + 2 * grow) / 2)]
    return (cq.Workplane("XY").workplane(offset=z0).polyline(pts).close()
            .extrude(z1 - z0).rotate((0, 0, 0), (0, 0, 1), az))


def host_section(lab):
    """(r_in, r_out), width of the member a joint sits in, at SPLIT_Z -
    interpolated from concept.py rather than copied, so it cannot go stale."""
    secs = c.BEAM_SECTIONS if "beam" in lab else c.SPINE_SECTIONS
    for a, b in zip(secs, secs[1:]):
        if a[0] <= SPLIT_Z <= b[0]:
            f = (SPLIT_Z - a[0]) / (b[0] - a[0])
            return ((a[2] + f * (b[2] - a[2]), a[3] + f * (b[3] - a[3])),
                    a[1] + f * (b[1] - a[1]))
    raise ValueError("SPLIT_Z outside %s" % lab)


def whole():
    w = (c.make_frame().union(c.rpi_rail())
         .union(c.camera_arm()).union(c.perch_lugs()))
    for cut in c.wire_hole_cutters():
        w = w.cut(cut)
    return w


def pin_cutter(lab, az):
    """Cross pin at mid-tongue, drilled RADIALLY.

    The axis matters. A tangential pin runs across the tongue's wide face but
    eats its thin dimension: the beam tongue is only 5mm deep radially, so a
    D4 hole would have left 0.5mm of material either side and all but severed
    it (1.5mm on the spine). Radial puts the hole through the thin dimension
    and eats the wide one instead, leaving 3.8mm on the beams and 5.2mm on the
    spine. It also means the pin goes in from the outside, where you can
    reach it.

    Drilled through the assembled solid, so tongue and socket line up by
    construction rather than by arithmetic.
    """
    zc = SPLIT_Z + TONGUE_L / 2.0
    a = math.radians(az)
    d = (math.cos(a), math.sin(a), 0.0)          # radial, outward
    # Start just outside the member and run only through it. A cutter swept
    # across the whole diameter would also drill the opposite beam and sweep
    # through the bottle's space on the way.
    (r_in, r_out), _w = host_section(lab)
    r0, L = r_out + 4.0, (r_out - r_in) + 8.0
    return c._cyl(PIN_D / 2, (r0 * d[0], r0 * d[1], zc),
                  (-d[0], -d[1], 0.0), L)


def main():
    w = whole()

    tongues, sockets = None, None
    for (_lab, az, ri, ro, wi, wo) in JOINTS:
        t = _prism(ri, ro, wi, wo, SPLIT_Z - 1.0, SPLIT_Z + TONGUE_L, az)
        s = _prism(ri, ro, wi, wo, SPLIT_Z - 1.0, SPLIT_Z + TONGUE_L + CLEAR,
                   az, grow=CLEAR)
        tongues = t if tongues is None else tongues.union(t)
        sockets = s if sockets is None else sockets.union(s)

    big = 500.0
    fwd_half = (cq.Workplane("XY").workplane(offset=SPLIT_Z - big)
                .box(big, big, big, centered=(True, True, False)))
    aft_half = (cq.Workplane("XY").workplane(offset=SPLIT_Z)
                .box(big, big, big, centered=(True, True, False)))

    front = w.intersect(fwd_half).union(w.intersect(tongues))
    rear = w.intersect(aft_half).cut(sockets)

    for (lab, az, _ri, _ro, _wi, _wo) in JOINTS:
        pc = pin_cutter(lab, az)
        front = front.cut(pc)
        rear = rear.cut(pc)

    # One pin length per joint: the beam holes are 13mm deep, the spine's runs
    # 23mm because it also passes through the rail. A single length would leave
    # a pin proud of the rail face, which is exactly where the RPi housing
    # slides.
    pins = {}
    for lab in ("beam +Y", "spine"):
        (r_in, r_out), _w = host_section(lab)
        depth = (r_out - r_in) if "beam" in lab else (abs(SPINE_FACE_X_TIP) - r_in)
        pins["cross_pin_%s" % ("beam" if "beam" in lab else "spine")] = (
            c._cyl(PIN_D / 2 - 0.15, (0, 0, 0), (0, 0, 1), depth - 0.4))

    ok = True
    for name, part in ([("cradle_front", front), ("cradle_rear", rear)]
                       + sorted(pins.items())):
        v = part.val()
        bb = v.BoundingBox()
        fits = max(bb.xlen, bb.ylen, bb.zlen) <= BED
        solids = len(v.Solids())
        ok = ok and fits and solids == 1
        cq.exporters.export(part, name + ".stl")
        cq.exporters.export(part, name + ".step")
        print("%-13s %6.0f x %6.0f x %6.0f mm   z %5.1f..%5.1f   %5.0f g   "
              "solids %d   %s"
              % (name, bb.xlen, bb.ylen, bb.zlen, bb.zmin, bb.zmax,
                 v.Volume() * 1.24e-3, solids,
                 "FITS %g" % BED if fits else "*** TOO BIG ***"))

    # --- joint checks ------------------------------------------------------
    k0, k1 = c.JOINT_KEEPOUT_Z
    assert k0 <= SPLIT_Z and SPLIT_Z + TONGUE_L <= k1, \
        "joint runs outside concept.py's declared keep-out %s" % (c.JOINT_KEEPOUT_Z,)
    for zw in c.WIRE_AFT_ZS + (c.WIRE_FWD_Z,):
        assert not (k0 <= zw <= k1), "wire hole z=%.0f is inside the joint keep-out" % zw
    print("\njoint: sliding dovetails at z=%.0f, %.0fmm engagement, %.2fmm/face"
          " clearance, %.0fmm cross pins" % (SPLIT_Z, TONGUE_L, CLEAR, PIN_D))
    for (lab, az, ri, ro, wi, wo) in JOINTS:
        taper = math.degrees(math.atan((wi - wo) / 2.0 / (ro - ri)))
        host, hw = host_section(lab)
        print("  %-8s taper %4.1f deg | socket walls: radial %.1f / %.1f mm,"
              "  side %.1f mm | pin leaves %.1f mm of tongue"
              % (lab, taper, ri - CLEAR - host[0], host[1] - ro - CLEAR,
                 (hw - wi) / 2 - CLEAR, wo / 2 - PIN_D / 2))
        assert wo / 2 - PIN_D / 2 >= 2.0, "%s: pin too close to the tongue edge" % lab
    rail = c.rpi_rail()
    rp = pin_cutter("spine", 180.0)
    railhit = abs(rp.intersect(rail).val().Volume())
    print("  spine pin through the rail: %.0f mm^3 %s"
          % (railhit, "- pin must finish flush/recessed in the rail face"
             if railhit > 1 else "- clear of the rail"))
    for zw in c.WIRE_AFT_ZS + (c.WIRE_FWD_Z,):
        gap = abs(zw - (SPLIT_Z + TONGUE_L / 2.0)) - PIN_D / 2 - c.WIRE_HOLE_D / 2
        flag = "OK" if (zw < SPLIT_Z - 2 or zw > SPLIT_Z + TONGUE_L + 4) else "*** IN THE JOINT ***"
        print("  wire hole z=%3.0f: %5.1fmm from the pin, %s" % (zw, gap, flag))
    inter = abs(front.intersect(rear).val().Volume())
    print("  front ^ rear interference: %.1f mm^3  %s"
          % (inter, "OK" if inter < 1.0 else "*** PARTS CLASH ***"))
    print("ALL OK" if ok and inter < 1.0 else "*** CHECK ***")


if __name__ == "__main__":
    main()
