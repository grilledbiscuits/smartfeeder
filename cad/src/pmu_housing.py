"""
PMU housing for the v6 feeder cradle.

Sits on the dovetail rail immediately AFT of the Pi housing. Same model frame
as concept.py / rpi_housing.py, so it assembles with no transform.

The unit is treated as ONE block, 110 x 70 x 25mm with the 18650 included,
because that is how it is built and how it comes out. It is carried on corner
pads on the LID, so undoing four screws drops board, battery and all out of
the box in one piece.

Retention on the rail differs from the Pi housing. There the groove is blind
at the front and seats on the rail's front face. That is impossible here - the
rail runs straight through this box's position, so a blind wall would foul it.
Instead the groove is open at BOTH ends and the box seats, under gravity,
against the Pi housing's aft face, which is itself stopped by the rail's front
end. A grub screw stops it creeping back up the slope.
"""
import cadquery as cq
import concept as c
import rpi_housing as R

# ------------------------------------------------ rail interface (shared) --
X_TOP = R.X_TOP                        # -58.25, roof top just under the spine
X_GROOVE_FLOOR = R.X_GROOVE_FLOOR      # -67.25
KEEL_HW = R.KEEL_HW
WALL, ROOF_T = R.WALL, R.ROOF_T
DRIP_W, DRIP_H, VENT_W = R.DRIP_W, R.DRIP_H, R.VENT_W
X_KEEL_BOT = X_GROOVE_FLOOR - R.GROOVE_FLOOR_T             # -70.0
X_ROOF_UNDER = X_TOP - ROOF_T                              # -61.25

# ------------------------------------------------- the PMU as one unit --
# Measured overall dimensions supplied by the user; battery included.
UNIT_L, UNIT_W = 110.0, 70.0        # along y, along z
UNIT_H = 25.0
UNIT_UNDER = 0.0                    # whole unit included in UNIT_H
UNIT_CLR = 1.0
# Corner pads lift it 4mm off the lid for small protrusions and cable dressing.
PAD_H = 4.0
PAD_XY, LIP_H, LIP_T = 10.0, 10.0, 2.5
UNIT_UNDER_INSET = 12.0             # unused while UNIT_UNDER is 0

# ------------------------------------------------------------- the shell ---
GAP_TO_PI = 0.5                     # closes under gravity; the roofs butt
Z_ROOF_FRONT = R.Z_ROOF_AFT + GAP_TO_PI                    # 148.0
OVH_SIDE, OVH_AFT = 8.0, 8.0                               # no front overhang
Z_BOX_FRONT = Z_ROOF_FRONT
Z_INT_FRONT = Z_BOX_FRONT + WALL
Z_INT_AFT = Z_INT_FRONT + UNIT_W + 2 * (UNIT_CLR + LIP_T + 1.0)
Z_BOX_AFT = Z_INT_AFT + WALL
Z_ROOF_AFT = Z_BOX_AFT + OVH_AFT
Y_INT = UNIT_L / 2 + UNIT_CLR + LIP_T + 1.0   # room for the corner lips
Y_BOX = Y_INT + WALL
Y_ROOF = Y_BOX + OVH_SIDE

X_LID_IN = X_KEEL_BOT - (PAD_H + UNIT_H + 2.0)
LID_T = 3.0
X_LID_OUT = X_LID_IN - LID_T
X_UNIT_BOT = X_LID_IN + PAD_H
X_UNIT_TOP = X_UNIT_BOT + UNIT_H
UNIT_Z0 = Z_INT_FRONT + UNIT_CLR + LIP_T + 1.0
UNIT_Z1 = UNIT_Z0 + UNIT_W

# --------------------------------------------------------------- IO ports --
# The photo establishes order, but not measured port centers or diameters.
# These openings are proportional placements to check against the real PMU.
# (name, shape, position along 70mm side viewed from inside, distance below top,
#  width, height). The outside view has the opposite left-to-right order.
PORTS = (
    ("switch", "rect", 0.24, 14.5, 9, 7),
    ("usb_a", "rect", 0.45, 14.5, 16, 10),
    ("usb_c", "rect", 0.68, 16.5, 12, 8),
    ("solar_in", "round", 0.87 - 1.0 / UNIT_W, 14, 8, 8),
    ("led_warning", "round", 0.60, 9, 3, 3),
    ("led_charge", "round", 0.67, 9, 3, 3),
    ("led_done", "round", 0.74, 9, 3, 3),
)

# -------------------------------------------------------------- lid fixings -
BOSS_D, BOSS_H, BOSS_PILOT = 7.0, 7.0, 2.5
BOSS_Y = Y_BOX + 3.0
BOSS_ZS = (Z_BOX_FRONT + 11.0, Z_BOX_AFT - 11.0)
LID_Y = BOSS_Y + BOSS_D / 2
SPIGOT_CLR, SPIGOT_T, SPIGOT_H = 0.3, 2.0, 3.0

# ------------------------------------------------------------ rail locking --
GRUB_D, GRUB_X = 2.5, (X_TOP + X_GROOVE_FLOOR) / 2
GRUB_Z = Z_BOX_AFT + OVH_AFT / 2
GROOVE_RELIEF = 0.30              # extra PMU flank/floor clearance beyond Pi

_box = R._box


def unit_keepout():
    """The complete PMU envelope, including any underside projection."""
    body = _box(X_UNIT_BOT, X_UNIT_TOP, -UNIT_L / 2, UNIT_L / 2, UNIT_Z0, UNIT_Z1)
    if UNIT_UNDER <= 0:
        return body
    i = UNIT_UNDER_INSET
    return body.union(_box(X_UNIT_BOT - UNIT_UNDER, X_UNIT_BOT,
                           -UNIT_L / 2 + i, UNIT_L / 2 - i,
                           UNIT_Z0 + i, UNIT_Z1 - i))


def groove_cutter():
    """Open at BOTH ends - the rail passes straight through. Lead-in flare at
    the aft end, the one it goes on over."""
    z0, z1 = Z_ROOF_FRONT - 1.0, Z_ROOF_AFT + 1.0
    main = (cq.Workplane("XY").workplane(offset=z0)
            .polyline(R.groove_profile(GROOVE_RELIEF)).close().extrude(z1 - z0))
    w0 = (cq.Workplane("XY").workplane(offset=Z_ROOF_AFT - 4.0)
          .polyline(R.groove_profile(GROOVE_RELIEF)).close().wires().val())
    w1 = (cq.Workplane("XY").workplane(offset=Z_ROOF_AFT + 0.01)
          .polyline(R.groove_profile(GROOVE_RELIEF + 0.8)).close().wires().val())
    return main.union(cq.Workplane(obj=cq.Solid.makeLoft([w0, w1], True)))


def housing():
    roof = _box(X_ROOF_UNDER, X_TOP, -Y_ROOF, Y_ROOF, Z_ROOF_FRONT, Z_ROOF_AFT)
    keel = _box(X_KEEL_BOT, X_ROOF_UNDER, -KEEL_HW, KEEL_HW,
                Z_ROOF_FRONT, Z_ROOF_AFT)
    shell = (_box(X_LID_IN, X_ROOF_UNDER, -Y_BOX, Y_BOX, Z_BOX_FRONT, Z_BOX_AFT)
             .cut(_box(X_LID_IN - 1, X_ROOF_UNDER, -Y_INT, Y_INT,
                       Z_INT_FRONT, Z_INT_AFT)))
    drip = (_box(X_ROOF_UNDER - DRIP_H, X_ROOF_UNDER, -Y_ROOF, Y_ROOF,
                 Z_ROOF_FRONT, Z_ROOF_AFT)
            .cut(_box(X_ROOF_UNDER - DRIP_H - 1, X_ROOF_UNDER + 1,
                      -(Y_ROOF - DRIP_W), Y_ROOF - DRIP_W,
                      Z_ROOF_FRONT - 1, Z_ROOF_AFT - DRIP_W)))
    h = roof.union(shell).union(keel).union(drip)

    for sy in (1, -1):
        for z in BOSS_ZS:
            h = h.union(c._cyl(BOSS_D / 2, (X_LID_IN, sy * BOSS_Y, z),
                               (1, 0, 0), BOSS_H))
            yo, ye, xg = Y_BOX - 0.5, BOSS_Y + BOSS_D / 2, X_LID_IN + BOSS_H
            h = h.union(cq.Workplane("XY").workplane(offset=z - BOSS_D / 2)
                        .polyline([(xg, sy * yo), (xg, sy * ye),
                                   (xg + (ye - yo), sy * yo)]).close()
                        .extrude(BOSS_D))

    h = h.cut(groove_cutter())
    for sy in (1, -1):
        for z in BOSS_ZS:
            h = h.cut(c._cyl(BOSS_PILOT / 2, (X_LID_IN - 0.1, sy * BOSS_Y, z),
                             (1, 0, 0), BOSS_H - 1.0))
    for cutter in port_cutters().values():
        h = h.cut(cutter)
    for sy in (1, -1):
        for xv in (X_ROOF_UNDER - 2.4, X_ROOF_UNDER - 5.9):
            h = h.cut(_box(xv - VENT_W / 2, xv + VENT_W / 2,
                           sy * Y_INT - 1.5 if sy > 0 else -Y_BOX - 0.5,
                           Y_BOX + 0.5 if sy > 0 else -Y_INT + 1.5,
                           Z_INT_AFT - 26.0, Z_INT_AFT - 8.0))
    h = h.cut(c._cyl(GRUB_D / 2, (GRUB_X, KEEL_HW + 1.0, GRUB_Z), (0, -1, 0),
                     KEEL_HW + 1.0 - R.groove_hw(GRUB_X) + 1.0))
    return h.union(port_awnings())


def port_cutters():
    """Open the +Y wall and any gusset behind a port."""
    cuts = {}
    for name, shape, fraction, down, width, height in PORTS:
        z, x = UNIT_Z0 + fraction * UNIT_W, X_UNIT_TOP - down
        if shape == "rect":
            cuts[name] = _box(x - height / 2, x + height / 2,
                              Y_INT - 1, Y_BOX + 8, z - width / 2, z + width / 2)
        else:
            cuts[name] = c._cyl(width / 2, (x, Y_INT - 1, z),
                                (0, 1, 0), Y_BOX + 8 - (Y_INT - 1))
    return cuts


def port_awnings():
    """Straight staple hoods; round openings use an upper half-ring."""
    y0 = Y_BOX - 0.5
    parts = []
    for name, shape, fraction, down, width, height in PORTS:
        z, x = UNIT_Z0 + fraction * UNIT_W, X_UNIT_TOP - down
        projection = 2.5 if name.startswith("led_") else 4.0
        thickness = 0.8 if name.startswith("led_") else 1.0
        y1 = Y_BOX + projection
        if shape == "rect":
            outer = _box(x - height / 2, x + height / 2 + thickness,
                         y0, y1, z - width / 2 - thickness, z + width / 2 + thickness)
            inner = _box(x - height / 2, x + height / 2,
                         y0 - 1, y1 + 1, z - width / 2, z + width / 2)
            part = outer.cut(inner)
        else:
            outer = c._cyl(width / 2 + thickness, (x, y0, z),
                           (0, 1, 0), projection + 0.5)
            inner = c._cyl(width / 2, (x, y0 - 1, z),
                           (0, 1, 0), projection + 2.5)
            lower = _box(X_LID_OUT - 10, x, y0 - 1, y1 + 1, z - 10, z + 10)
            part = outer.cut(inner).cut(lower)
        parts.append(part)
    awnings = parts[0]
    for part in parts[1:]:
        awnings = awnings.union(part)
    for cutter in port_cutters().values():
        awnings = awnings.cut(cutter)
    return awnings


def lid():
    plate = _box(X_LID_OUT, X_LID_IN, -LID_Y, LID_Y, Z_BOX_FRONT, Z_BOX_AFT)
    yo = Y_INT - SPIGOT_CLR
    z0o, z1o = Z_INT_FRONT + SPIGOT_CLR, Z_INT_AFT - SPIGOT_CLR
    spig = (_box(X_LID_IN - 0.01, X_LID_IN + SPIGOT_H, -yo, yo, z0o, z1o)
            .cut(_box(X_LID_IN - 1, X_LID_IN + SPIGOT_H + 1,
                      -(yo - SPIGOT_T), yo - SPIGOT_T,
                      z0o + SPIGOT_T, z1o - SPIGOT_T)))
    L = plate.union(spig)

    # four corner pads carry the unit clear of the lid; each has
    # an L of lip above it to trap the unit in plan. No screw standoffs: the
    # mounting-hole pattern of the assembled unit is not known.
    for sy in (1, -1):
        for sz in (1, -1):
            # inner faces of this corner's lips, offset off the unit by UNIT_CLR
            ey = sy * (UNIT_L / 2 + UNIT_CLR)
            ez = UNIT_Z1 + UNIT_CLR if sz > 0 else UNIT_Z0 - UNIT_CLR
            iny, inz = -sy, -sz            # directions back under the unit
            py = sorted((ey, ey + iny * PAD_XY))
            pz = sorted((ez, ez + inz * PAD_XY))
            L = L.union(_box(X_LID_IN, X_LID_IN + PAD_H, py[0], py[1], pz[0], pz[1]))
            # lip on the y face, running the pad's length plus the corner
            ly = sorted((ey, ey - iny * LIP_T))
            lz = sorted((ez - inz * LIP_T, ez + inz * PAD_XY))
            L = L.union(_box(X_LID_IN, X_LID_IN + PAD_H + LIP_H,
                             ly[0], ly[1], lz[0], lz[1]))
            # lip on the z face
            mz = sorted((ez, ez - inz * LIP_T))
            my = sorted((ey, ey + iny * PAD_XY))
            L = L.union(_box(X_LID_IN, X_LID_IN + PAD_H + LIP_H,
                             my[0], my[1], mz[0], mz[1]))

    for sy in (1, -1):
        for z in BOSS_ZS:
            L = L.cut(c._cyl(1.7, (X_LID_OUT - 1, sy * BOSS_Y, z), (1, 0, 0),
                             LID_T + 2))
    for z in (Z_INT_FRONT + 4.0, Z_INT_FRONT + 8.0, Z_INT_FRONT + 12.0):
        L = L.cut(_box(X_LID_OUT - 1, X_LID_IN + 1, -18.0, 18.0,
                       z - VENT_W / 2, z + VENT_W / 2))
    return L


def build(export=True):
    H, L, U = housing(), lid(), unit_keepout()
    PI_H, PI_L = R.housing(), R.lid()
    import split as sp
    cradle = sp.whole()
    v = R.vol
    ok = True

    def rep(m, good):
        nonlocal ok
        ok = ok and good
        print("  %-61s %s" % (m, "OK" if good else "*** CHECK ***"))

    print("PMU housing  (unit taken as %.0f x %.0f x %.0f mm)"
          % (UNIT_L, UNIT_W, UNIT_H))
    for nm, part in (("housing", H), ("lid", L)):
        x0, x1, y0, y1, z0, z1 = c.tight_bb(part)
        print("  %-8s %5.1f x %5.1f x %5.1f mm   %4.0f g PLA   solids %d"
              % (nm, x1 - x0, y1 - y0, z1 - z0, part.val().Volume() * 1.24e-3,
                 len(part.val().Solids())))
        rep("%s one solid, fits 180" % nm, len(part.val().Solids()) == 1
            and max(x1 - x0, y1 - y0, z1 - z0) <= 180)

    print("\n on the rail")
    rep("housing ^ cradle  %5.1f mm^3" % v(H.intersect(cradle)),
        v(H.intersect(cradle)) < 0.5)
    worst = max(v(H.translate((0, 0, dz)).intersect(cradle))
                for dz in (5., 15., 30., 45., 60., 80., 100.))
    rep("sliding on from behind, 7 stations, worst %5.1f mm^3" % worst, worst < 0.5)
    eng = min(Z_ROOF_AFT, c.RAIL_Z1) - Z_ROOF_FRONT
    rep("groove engages the rail over %.0f mm, open both ends" % eng, eng > 60)
    floor_left = X_GROOVE_FLOOR - GROOVE_RELIEF - X_KEEL_BOT
    rep("PMU-only groove relief %.2f mm; floor left %.2f mm"
        % (GROOVE_RELIEF, floor_left), floor_left >= 2.0)
    rep("housing ^ Pi housing %5.1f mm^3 (butts it, %.1f mm gap)"
        % (v(H.intersect(PI_H)), GAP_TO_PI), v(H.intersect(PI_H)) < 0.5)
    rep("housing ^ Pi lid     %5.1f mm^3" % v(H.intersect(PI_L)),
        v(H.intersect(PI_L)) < 0.5)
    pin_z = sp.SPLIT_Z + sp.TONGUE_L / 2
    rep("slides over the spine cross-pin at z=%.0f -> that pin MUST be flush"
        % pin_z, Z_ROOF_FRONT < pin_z < Z_ROOF_AFT)

    print("\n inside")
    rep("unit keep-out ^ housing %5.1f mm^3" % v(U.intersect(H)), v(U.intersect(H)) < 0.5)
    rep("unit keep-out ^ lid     %5.1f mm^3" % v(U.intersect(L)), v(U.intersect(L)) < 0.5)
    rep("housing ^ lid           %5.1f mm^3" % v(H.intersect(L)), v(H.intersect(L)) < 0.5)
    rep("unit clears the keel by %.1f mm" % (X_KEEL_BOT - X_UNIT_TOP),
        X_KEEL_BOT - X_UNIT_TOP >= 1.0)
    rep("corner lips stand %.0f mm up the unit's %.0f mm side" % (LIP_H, UNIT_H),
        LIP_H >= 8)

    print("\n cable entries")
    for name, cutter in port_cutters().items():
        overlap = v(H.intersect(cutter))
        rep("%s passage unobstructed: %.1f mm^3 overlap" % (name, overlap),
            overlap < 0.5)

    if export:
        for nm, part in (("pmu_housing", H), ("pmu_housing_lid", L)):
            cq.exporters.export(part, nm + ".stl")
            cq.exporters.export(part, nm + ".step")
        cq.exporters.export(U, "pmu_keepout.stl")
    print("\nALL OK" if ok else "\n*** CHECK ***")
    return H, L


if __name__ == "__main__":
    build()
