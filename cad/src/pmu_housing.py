"""
Waveshare Power Management HAT (B) housing for the v6 feeder cradle.

Sits on the dovetail rail immediately AFT of the Pi housing. Same model frame
as concept.py / rpi_housing.py, so it assembles with no transform.

The unit is treated as ONE block, 115 x 75mm in plan with the 18650 included,
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

# --------------------------------------------- the Waveshare unit, as one --
# 115 x 75 in plan, battery included. Height is the one number I am guessing:
# the 18650 sets it at about 26mm above the board plane, with the unused
# 40-pin socket hanging ~9mm below.
UNIT_L, UNIT_W = 115.0, 75.0        # along y, along z
UNIT_H = 40.0                       # overall, measured by Jack
UNIT_UNDER = 0.0                    # 40 is the whole unit, nothing below it
UNIT_CLR = 1.0
# Corner pads still lift it 4mm off the lid: clearance for anything small
# protruding from the underside, for cable dressing, and so condensation on
# the lid cannot wick into the board. Raise PAD_H if the 40-pin socket or
# anything else sticks out more than 4mm below the unit's bottom face.
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

# ------------------------------------------------------------ cable entries -
# USB-A out to the Pi goes through the FRONT wall: the Pi housing is directly
# in front of it, so the run is short and doubly sheltered.
USBA_Y, USBA_X = -25.0, X_UNIT_BOT + UNIT_H / 2   # centred on the unit
USBA_W, USBA_H, USBA_R = 18.0, 9.0, 3.5          # passes a USB-A overmould
# PV input through the LID, which faces straight down when hung - the most
# sheltered face on the box, and it gives the drip loop for free.
PV_Y, PV_D = 40.0, 12.0
PV_Z = Z_INT_AFT - 9.0

# -------------------------------------------------------------- lid fixings -
BOSS_D, BOSS_H, BOSS_PILOT = 7.0, 7.0, 2.5
BOSS_Y = Y_BOX + 3.0
BOSS_ZS = (Z_BOX_FRONT + 11.0, Z_BOX_AFT - 11.0)
LID_Y = BOSS_Y + BOSS_D / 2
SPIGOT_CLR, SPIGOT_T, SPIGOT_H = 0.3, 2.0, 3.0

# ------------------------------------------------------------ rail locking --
GRUB_D, GRUB_X = 2.5, (X_TOP + X_GROOVE_FLOOR) / 2
GRUB_Z = Z_BOX_AFT + OVH_AFT / 2

_box = R._box


def unit_keepout():
    """The whole Waveshare assembly, plus the socket hanging below it."""
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
            .polyline(R.groove_profile()).close().extrude(z1 - z0))
    w0 = (cq.Workplane("XY").workplane(offset=Z_ROOF_AFT - 4.0)
          .polyline(R.groove_profile()).close().wires().val())
    w1 = (cq.Workplane("XY").workplane(offset=Z_ROOF_AFT + 0.01)
          .polyline(R.groove_profile(0.8)).close().wires().val())
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
    h = h.cut(usba_cutter())
    for sy in (1, -1):
        for xv in (X_ROOF_UNDER - 2.4, X_ROOF_UNDER - 5.9):
            h = h.cut(_box(xv - VENT_W / 2, xv + VENT_W / 2,
                           sy * Y_INT - 1.5 if sy > 0 else -Y_BOX - 0.5,
                           Y_BOX + 0.5 if sy > 0 else -Y_INT + 1.5,
                           Z_INT_AFT - 26.0, Z_INT_AFT - 8.0))
    h = h.cut(c._cyl(GRUB_D / 2, (GRUB_X, KEEL_HW + 1.0, GRUB_Z), (0, -1, 0),
                     KEEL_HW + 1.0 - R.groove_hw(GRUB_X) + 1.0))
    return h


def usba_cutter():
    return (cq.Workplane("XY").workplane(offset=Z_BOX_FRONT - 1.0)
            .center(USBA_X, USBA_Y).sketch().rect(USBA_H, USBA_W)
            .vertices().fillet(USBA_R).finalize().extrude(WALL + 2.0))


def pv_cutter():
    return c._cyl(PV_D / 2, (X_LID_OUT - 1, PV_Y, PV_Z), (1, 0, 0), LID_T + 2)


def lid():
    plate = _box(X_LID_OUT, X_LID_IN, -LID_Y, LID_Y, Z_BOX_FRONT, Z_BOX_AFT)
    yo = Y_INT - SPIGOT_CLR
    z0o, z1o = Z_INT_FRONT + SPIGOT_CLR, Z_INT_AFT - SPIGOT_CLR
    spig = (_box(X_LID_IN - 0.01, X_LID_IN + SPIGOT_H, -yo, yo, z0o, z1o)
            .cut(_box(X_LID_IN - 1, X_LID_IN + SPIGOT_H + 1,
                      -(yo - SPIGOT_T), yo - SPIGOT_T,
                      z0o + SPIGOT_T, z1o - SPIGOT_T)))
    L = plate.union(spig)

    # four corner pads carry the unit clear of its underside socket; each has
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
    L = L.cut(pv_cutter())
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

    print("Waveshare PMU housing  (unit taken as %.0f x %.0f x %.0f mm)"
          % (UNIT_L, UNIT_W, UNIT_H))
    for nm, part in (("housing", H), ("lid", L)):
        x0, x1, y0, y1, z0, z1 = c.tight_bb(part)
        print("  %-8s %5.1f x %5.1f x %5.1f mm   %4.0f g PETG   solids %d"
              % (nm, x1 - x0, y1 - y0, z1 - z0, part.val().Volume() * 1.27e-3,
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
    wall = _box(X_LID_IN, X_ROOF_UNDER, -Y_BOX, Y_BOX, Z_BOX_FRONT, Z_INT_FRONT)
    u = usba_cutter()
    rep("USB-A out, front wall: %.0f mm^2 open, %.1f mm^3 left in it"
        % (v(u.intersect(wall)) / WALL, v(H.intersect(u.intersect(wall)))),
        v(H.intersect(u.intersect(wall))) < 0.5)
    rep("PV in, through the lid: %.1f mm^3 left in it" % v(L.intersect(pv_cutter())),
        v(L.intersect(pv_cutter())) < 0.5)
    rep("USB-A slot is level with the unit (x %.1f, unit %.1f..%.1f)"
        % (USBA_X, X_UNIT_BOT, X_UNIT_TOP), X_UNIT_BOT < USBA_X < X_UNIT_TOP)

    if export:
        for nm, part in (("pmu_housing", H), ("pmu_housing_lid", L)):
            cq.exporters.export(part, nm + ".stl")
            cq.exporters.export(part, nm + ".step")
        cq.exporters.export(U, "pmu_keepout.stl")
    print("\nALL OK" if ok else "\n*** CHECK ***")
    return H, L


if __name__ == "__main__":
    build()
