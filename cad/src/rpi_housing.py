"""
Raspberry Pi 4B housing for the v6 feeder cradle.

Hangs from the cradle's dovetail rail, on the FRONT half (forward of the split
at z=158). Same model frame as concept.py, so it drops straight onto the
cradle with no transform:
    Z = bottle axis, +Z aft.   +X = "up" side when hung.   -X = down.
The whole rig hangs 30 deg nose-down, so in the world:
    front face (-Z) looks forward and 30 deg DOWN  -> sheltered, undercut
    aft face   (+Z) looks aft and 30 deg UP        -> takes rain
    lid        (-X) looks down                      -> sheltered
The USB-C entry is in the front wall; the camera ribbon and GPIO cable exit
through the microSD side wall. The lid is underneath.

Parts: housing body, lid. The Pi mounts upside-down on standoffs from the
ceiling, component side facing the lid, so taking the lid off exposes the
camera connector, GPIO header and USB-C without disturbing anything.
"""
import math
import cadquery as cq
import concept as c

# ----------------------------------------------------------- rail interface --
# Everything here is derived from the cradle's rail so the two cannot drift.
CLR = 0.25                                   # sliding clearance, every face
RAIL_ROOT_X = c.SPINE_FACE_X + 1.0           # -57 (embedded 1mm in the spine)
RAIL_TIP_X = c.SPINE_FACE_X - c.RAIL_H       # -67
RAIL_ROOT_HW, RAIL_TIP_HW = c.RAIL_ROOT_W / 2, c.RAIL_TIP_W / 2      # 8, 13
X_TOP = c.SPINE_FACE_X - CLR                 # -58.25  roof top, under the spine
X_GROOVE_FLOOR = RAIL_TIP_X - CLR            # -67.25
GROOVE_FLOOR_T = 2.75                        # material under the groove

# ------------------------------------------------------------ shell layout --
# The spine perch lug hangs to x=-65.2 over z 45..60.5, which is exactly the
# height band the roof and keel occupy - so the roof's front edge sits aft of
# it. Everything BELOW the roof (the box) is clear of the lug regardless.
Z_ROOF_FRONT = 65.5
OVH_SIDE, OVH_FRONT, OVH_AFT = 8.0, 6.0, 8.0
WALL = 2.5
ROOF_T = 3.0
KEEL_HW = 17.0                               # keel carries the dovetail groove
DRIP_W, DRIP_H = 1.5, 2.0                    # lip under every roof edge

# ------------------------------------------------------------- the Pi 4B ----
PI_L, PI_W, PI_T = 85.0, 56.0, 1.4
PI_HOLES = [(3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5)]
PORT_GAP = 3.0        # board front edge to wall: the audio jack sticks out 2.5
AFT_GAP = 4.0
STACK = 24.0          # below the component face: Dupont on GPIO + a wire bend
STANDOFF_D, PILOT_D = 6.0, 2.2               # M2.5 self-tapping

# derived stations
Z_BOX_FRONT = Z_ROOF_FRONT + OVH_FRONT                     # 71.5
Z_INT_FRONT = Z_BOX_FRONT + WALL                           # 74.0
Z_BOARD_FRONT = Z_INT_FRONT + PORT_GAP                     # 77.0
Z_INT_AFT = Z_BOARD_FRONT + PI_W + AFT_GAP                 # 137.0
Z_BOX_AFT = Z_INT_AFT + WALL                               # 139.5
Z_ROOF_AFT = Z_BOX_AFT + OVH_AFT                           # 147.5
Y_INT = PI_L / 2 + 3.5                                     # 46.0
Y_BOX = Y_INT + WALL                                       # 48.5
Y_ROOF = Y_BOX + OVH_SIDE                                  # 56.5
X_ROOF_UNDER = X_TOP - ROOF_T                              # -61.25
X_KEEL_BOT = X_GROOVE_FLOOR - GROOVE_FLOOR_T               # -70.0
X_BOARD_BACK = X_KEEL_BOT - 2.0                            # -72.0
X_COMP = X_BOARD_BACK - PI_T                               # -73.4
X_LID_IN = X_COMP - STACK - 1.0                            # -98.4
LID_T = 3.0
X_LID_OUT = X_LID_IN - LID_T                               # -101.4


def board_to_model(bx, by, bz):
    """Pi board frame (bx along the 85mm edge, by along the 56mm edge from the
    PORT edge, bz up from the component face) -> model frame. Board is upside
    down: component side faces -X, port edge faces forward (-Z)."""
    return (X_COMP - bz, PI_L / 2 - bx, Z_BOARD_FRONT + by)


def _box(x0, x1, y0, y1, z0, z1):
    return (cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False)
            .translate((x0, y0, z0)))


def board_box(bx0, bx1, by0, by1, bz0, bz1):
    xa, ya, za = board_to_model(bx0, by0, bz0)
    xb, yb, zb = board_to_model(bx1, by1, bz1)
    return _box(min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb),
                min(za, zb), max(za, zb))


# Keep-out model of a Pi 4B, board coords. Generous where I'm unsure.
PI_PARTS = {
    "board":   (0, 85, 0, 56, -1.4, 0),
    "usb_c":   (7.7, 14.7, -1.2, 6.3, 0, 3.3),
    "hdmi0":   (22.5, 29.5, -1.2, 6.3, 0, 3.2),
    "hdmi1":   (36.0, 43.0, -1.2, 6.3, 0, 3.2),
    "audio":   (50.5, 56.5, -2.5, 12.0, 0, 6.0),
    "csi":     (43.0, 47.0, 1.5, 23.5, 0, 5.5),
    "soc_hs":  (22.0, 37.0, 21.0, 36.0, 0, 10.0),
    "usb2":    (69.0, 87.5, 2.5, 15.5, 0, 16.0),
    "usb3":    (69.0, 87.5, 20.5, 33.5, 0, 16.0),
    "eth":     (65.0, 87.5, 38.0, 54.0, 0, 13.5),
    "gpio":    (7.1, 57.9, 49.96, 55.04, 0, 8.5),
    "dupont":  (5.6, 59.4, 48.5, 56.5, 8.5, STACK),
    "sd":      (-2.5, 13.0, 22.0, 34.0, -2.8, -1.4),
    "under":   (4.0, 81.0, 4.0, 52.0, -2.4, -1.4),     # underside passives
}
USB_C_BX, CSI_BX = 11.2, 45.0


PAD_D = 6.4          # component-free pad round each mounting hole, both sides


def pi_keepout():
    out = None
    for nm, dims in PI_PARTS.items():
        b = board_box(*dims)
        if nm in ("under", "sd"):
            for (hx, hy) in PI_HOLES:
                x, y, z = board_to_model(hx, hy, 0)
                b = b.cut(c._cyl(PAD_D / 2, (x + 5, y, z), (-1, 0, 0), 20))
        out = b if out is None else out.union(b)
    return out


# ------------------------------------------------------------ cable entries --
# USB-C is in the front wall; camera ribbon and GPIO exit the microSD side.
USBC_Y = board_to_model(USB_C_BX, 0, 0)[1]            # +31.3
USBC_X = X_COMP - 1.63                                # receptacle centre
USBC_W, USBC_H, USBC_R = 16.0, 10.0, 3.0              # passes a 14x8 overmould
USBC_AWNING_T, USBC_AWNING_OUT = 1.5, 5.0             # straight staple, flush inner edge
CSI_Y = board_to_model(CSI_BX, 0, 0)[1]               # -2.5
CSI_Z = board_to_model(CSI_BX, 12.5, 0)[2]            # connector midpoint
CSI_X = X_COMP - 6.0                                  # level with cable exit
CSI_SLOT_L, CSI_SLOT_W = 22.0, 3.5                    # 16mm FFC + margin
CSI_AWNING_R, CSI_AWNING_T = 4.5, 1.5                 # quarter-circle inner radius, roof thickness
CSI_AWNING_SIDE = 1.2                                # walls outside each slot end
GPIO_X = CSI_X
GPIO_D = 8.0                                         # 4-6 Dupont jumpers
GPIO_Z = CSI_Z + CSI_SLOT_L / 2 + CSI_AWNING_SIDE + GPIO_D / 2 + 4.0
GPIO_AWNING_T, GPIO_AWNING_OUT = 1.5, 5.0          # half-ring, flush inner edge


# --------------------------------------------------------------- vents ------
VENT_W = 2.0                                          # insect-limiting
EXH_XS = (X_ROOF_UNDER - 2.4, X_ROOF_UNDER - 5.9)    # high on the side walls
EXH_Z = (Z_INT_AFT - 29.0, Z_INT_AFT - 11.0)
INTAKE_ZS = (80.0, 84.0, 88.0, 92.0)                  # lid, front = low end
INTAKE_HALF = 20.0

# --------------------------------------------------------- lid fixings ------
BOSS_D, BOSS_H, BOSS_PILOT = 7.0, 7.0, 2.5            # M3 self-tapping
BOSS_Y = Y_BOX + 3.0
BOSS_ZS = (Z_BOX_FRONT + 10.0, Z_BOX_AFT - 10.0)
LID_Y = BOSS_Y + BOSS_D / 2                           # lid covers the bosses
SPIGOT_CLR, SPIGOT_T, SPIGOT_H = 0.3, 2.0, 3.0

# --------------------------------------------------------- rail locking -----
GRUB_D = 2.5                                          # tap M3
GRUB_Z = Z_BOX_AFT + OVH_AFT / 2                      # in the keel, OUTSIDE the box
GRUB_X = (X_TOP + X_GROOVE_FLOOR) / 2


def groove_hw(x):
    """Half-width of the groove at height x: the rail flank, offset CLR along
    its normal."""
    fy = RAIL_TIP_HW - RAIL_ROOT_HW                 # 5
    fx = RAIL_ROOT_X - RAIL_TIP_X                   # 10
    L = math.hypot(fy, fx)
    ny, nx = fx / L, fy / L                         # outward normal (y, x)
    y0, x0 = RAIL_TIP_HW + CLR * ny, RAIL_TIP_X + CLR * nx
    return y0 + (x - x0) * (-fy / fx)


def groove_profile(grow=0.0, x_top=X_TOP + 4.0):
    xf = X_GROOVE_FLOOR - grow
    return [(xf, -(groove_hw(xf) + grow)), (xf, groove_hw(xf) + grow),
            (x_top, groove_hw(x_top) + grow), (x_top, -(groove_hw(x_top) + grow))]


def groove_cutter():
    main = (cq.Workplane("XY").workplane(offset=Z_ROOF_FRONT - 1.0)
            .polyline(groove_profile()).close()
            .extrude(Z_ROOF_AFT - Z_ROOF_FRONT + 2.0))
    # lead-in at the open (aft) end: flare 0.8mm over the last 4mm
    w0 = (cq.Workplane("XY").workplane(offset=Z_ROOF_AFT - 4.0)
          .polyline(groove_profile()).close().wires().val())
    w1 = (cq.Workplane("XY").workplane(offset=Z_ROOF_AFT + 0.01)
          .polyline(groove_profile(0.8)).close().wires().val())
    lead = cq.Workplane(obj=cq.Solid.makeLoft([w0, w1], True))
    return main.union(lead)


def _yz_hole_cut(shape, cutter_2d_fn):
    return shape.cut(cutter_2d_fn)


def front_wall_cutters():
    z0 = Z_BOX_FRONT - 1.0
    d = WALL + 2.0
    usbc = (cq.Workplane("XY").workplane(offset=z0).center(USBC_X, USBC_Y)
            .sketch().rect(USBC_H, USBC_W).vertices().fillet(USBC_R)
            .finalize().extrude(d))
    return {"usb_c": usbc}


def usb_c_awning():
    """Straight rounded staple whose inner outline matches the USB-C hole."""
    z0 = Z_BOX_FRONT - USBC_AWNING_OUT
    depth = USBC_AWNING_OUT + 0.5             # embed in front wall
    outer = (cq.Workplane("XY").workplane(offset=z0).center(USBC_X, USBC_Y)
             .sketch().rect(USBC_H + 2 * USBC_AWNING_T,
                            USBC_W + 2 * USBC_AWNING_T)
             .vertices().fillet(USBC_R + USBC_AWNING_T)
             .finalize().extrude(depth))
    inner = (cq.Workplane("XY").workplane(offset=z0 - 1).center(USBC_X, USBC_Y)
             .sketch().rect(USBC_H, USBC_W).vertices().fillet(USBC_R)
             .finalize().extrude(depth + 2))
    open_bottom = _box(X_LID_OUT - 10, USBC_X - USBC_H / 2,
                       USBC_Y - 20, USBC_Y + 20, z0 - 1, z0 + depth + 1)
    return outer.cut(inner).cut(open_bottom)


def ribbon_cutter():
    return _box(CSI_X - CSI_SLOT_W / 2, CSI_X + CSI_SLOT_W / 2,
                Y_INT - 1.0, Y_BOX + 1.0,
                CSI_Z - CSI_SLOT_L / 2, CSI_Z + CSI_SLOT_L / 2)


def gpio_cutter():
    return cq.Workplane(obj=cq.Solid.makeCylinder(
        GPIO_D / 2, WALL + 2, cq.Vector(GPIO_X, Y_INT - 1, GPIO_Z),
        cq.Vector(0, 1, 0)))


def gpio_awning():
    """Straight upper half-ring around the GPIO cable exit."""
    y = Y_BOX - 0.5                            # embed in side wall
    outer = c._cyl(GPIO_D / 2 + GPIO_AWNING_T,
                   (GPIO_X, y, GPIO_Z), (0, 1, 0), GPIO_AWNING_OUT + 0.5)
    inner = c._cyl(GPIO_D / 2,
                   (GPIO_X, y - 1, GPIO_Z), (0, 1, 0), GPIO_AWNING_OUT + 2.5)
    lower_open = _box(X_LID_OUT - 10, GPIO_X, y - 1,
                      Y_BOX + GPIO_AWNING_OUT + 1, GPIO_Z - 10, GPIO_Z + 10)
    return outer.cut(inner).cut(lower_open)


def camera_awning():
    """Quarter-circle roof with two end walls around the side ribbon slot."""
    ri, ro = CSI_AWNING_R, CSI_AWNING_R + CSI_AWNING_T
    x = CSI_X - CSI_SLOT_W / 2 + 1.0             # outer lip below slot centre
    y = Y_BOX - 0.5                             # embed in wall to fuse
    z = CSI_Z - CSI_SLOT_L / 2 - CSI_AWNING_SIDE
    length = CSI_SLOT_L + 2 * CSI_AWNING_SIDE
    mid = math.sqrt(0.5)
    roof = (cq.Workplane("XY").workplane(offset=z)
            .moveTo(x + ri, y)
            .threePointArc((x + ri * mid, y + ri * mid), (x, y + ri))
            .lineTo(x, y + ro)
            .threePointArc((x + ro * mid, y + ro * mid), (x + ro, y))
            .close().extrude(length))
    wall = (cq.Workplane("XY").workplane(offset=z)
            .moveTo(CSI_X - CSI_SLOT_W / 2 - 1.2, y)
            .lineTo(CSI_X - CSI_SLOT_W / 2 - 1.2, y + ro)
            .lineTo(x, y + ro).lineTo(x, y + ri)
            .threePointArc((x + ri * mid, y + ri * mid), (x + ri, y))
            .close().extrude(CSI_AWNING_SIDE))
    return roof.union(wall).union(wall.translate((0, 0, length - CSI_AWNING_SIDE)))


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
                      Z_ROOF_FRONT + DRIP_W, Z_ROOF_AFT - DRIP_W)))
    h = roof.union(shell).union(keel).union(drip)

    # standoffs
    for (bx, by) in PI_HOLES:
        _x, y, z = board_to_model(bx, by, 0)
        post = c._cyl(STANDOFF_D / 2, (X_ROOF_UNDER + 0.5, y, z), (-1, 0, 0),
                      X_ROOF_UNDER + 0.5 - X_BOARD_BACK)
        h = h.union(post)

    # external lid bosses, each with a 45 deg gusset so it prints roof-down
    for sy in (1, -1):
        for z in BOSS_ZS:
            boss = c._cyl(BOSS_D / 2, (X_LID_IN, sy * BOSS_Y, z), (1, 0, 0), BOSS_H)
            yo = Y_BOX - 0.5
            ye = BOSS_Y + BOSS_D / 2
            xg = X_LID_IN + BOSS_H
            tri = (cq.Workplane("XY").workplane(offset=z - BOSS_D / 2)
                   .polyline([(xg, sy * yo), (xg, sy * ye),
                              (xg + (ye - yo), sy * yo)]).close()
                   .extrude(BOSS_D))
            h = h.union(boss).union(tri)

    # --- cuts ---
    h = h.cut(groove_cutter())
    for (bx, by) in PI_HOLES:
        _x, y, z = board_to_model(bx, by, 0)
        h = h.cut(c._cyl(PILOT_D / 2, (X_BOARD_BACK - 0.1, y, z), (1, 0, 0), 8.0))
    for sy in (1, -1):
        for z in BOSS_ZS:
            h = h.cut(c._cyl(BOSS_PILOT / 2, (X_LID_IN - 0.1, sy * BOSS_Y, z),
                             (1, 0, 0), BOSS_H - 1.0))
    for cut in front_wall_cutters().values():
        h = h.cut(cut)
    h = h.union(usb_c_awning())
    h = h.cut(ribbon_cutter())
    h = h.union(camera_awning())
    h = h.cut(gpio_cutter())
    h = h.union(gpio_awning())
    for sy in (1, -1):
        for xv in EXH_XS:
            h = h.cut(_box(xv - VENT_W / 2, xv + VENT_W / 2,
                           sy * Y_INT - 1.5 if sy > 0 else -Y_BOX - 0.5,
                           Y_BOX + 0.5 if sy > 0 else -Y_INT + 1.5,
                           EXH_Z[0], EXH_Z[1]))
    h = h.cut(c._cyl(GRUB_D / 2, (GRUB_X, KEEL_HW + 1.0, GRUB_Z), (0, -1, 0),
                     KEEL_HW + 1.0 - groove_hw(GRUB_X) + 1.0))
    return h


def lid():
    plate = _box(X_LID_OUT, X_LID_IN, -LID_Y, LID_Y, Z_BOX_FRONT, Z_BOX_AFT)
    yo, z0o, z1o = Y_INT - SPIGOT_CLR, Z_INT_FRONT + SPIGOT_CLR, Z_INT_AFT - SPIGOT_CLR
    spig = (_box(X_LID_IN - 0.01, X_LID_IN + SPIGOT_H, -yo, yo, z0o, z1o)
            .cut(_box(X_LID_IN - 1, X_LID_IN + SPIGOT_H + 1,
                      -(yo - SPIGOT_T), yo - SPIGOT_T,
                      z0o + SPIGOT_T, z1o - SPIGOT_T)))
    L = plate.union(spig)
    for sy in (1, -1):
        for z in BOSS_ZS:
            L = L.cut(c._cyl(1.7, (X_LID_OUT - 1, sy * BOSS_Y, z), (1, 0, 0),
                             LID_T + 2))
    for z in INTAKE_ZS:
        L = L.cut(_box(X_LID_OUT - 1, X_LID_IN + 1, -INTAKE_HALF, INTAKE_HALF,
                       z - VENT_W / 2, z + VENT_W / 2))
    return L


def vol(a):
    try:
        return abs(a.val().Volume())
    except Exception:
        return 0.0


def build(export=True):
    H, L, P = housing(), lid(), pi_keepout()
    rail = c.rpi_rail()
    import split as sp
    front = sp.whole().intersect(
        cq.Workplane("XY").workplane(offset=sp.SPLIT_Z - 500)
        .box(500, 500, 500, centered=(True, True, False)))

    ok = True

    def rep(msg, good):
        nonlocal ok
        ok = ok and good
        print("  %-58s %s" % (msg, "OK" if good else "*** CHECK ***"))

    print("RPi 4B housing")
    for nm, part in (("housing", H), ("lid", L)):
        x0, x1, y0, y1, z0, z1 = c.tight_bb(part)
        print("  %-8s %5.1f x %5.1f x %5.1f mm   %4.0f g PLA   solids %d"
              % (nm, x1 - x0, y1 - y0, z1 - z0, part.val().Volume() * 1.24e-3,
                 len(part.val().Solids())))
        rep("%s is one solid, fits 180" % nm,
            len(part.val().Solids()) == 1 and max(x1 - x0, y1 - y0, z1 - z0) <= 180)

    print("\n mating to the cradle")
    rep("housing ^ cradle front half (seated)   %6.1f mm^3" % vol(H.intersect(front)),
        vol(H.intersect(front)) < 0.5)
    # installation: comes on from ahead and slides aft; no axial stop in CAD
    worst = 0.0
    for dz in (5.0, 15.0, 25.0, 40.0, 55.0, 70.0, 85.0, 100.0, 120.0, 150.0):
        worst = max(worst, vol(H.translate((0, 0, -dz)).intersect(front)))
    rep("sliding on from ahead, 10 stations, worst %5.1f mm^3" % worst, worst < 0.5)
    front_groove = _box(X_GROOVE_FLOOR, X_TOP, -RAIL_TIP_HW, RAIL_TIP_HW,
                         Z_ROOF_FRONT, Z_ROOF_FRONT + 1.0).intersect(groove_cutter())
    rep("groove open through front end", vol(H.intersect(front_groove)) < 0.5)
    engage = min(Z_ROOF_AFT, sp.SPLIT_Z) - c.RAIL_Z0
    rep("rail engagement %.0f mm (open-ended groove)" % engage,
        engage >= 50)
    rep("housing clear of the split joint: aft end %.1f vs split %.0f"
        % (Z_ROOF_AFT, sp.SPLIT_Z), Z_ROOF_AFT < sp.SPLIT_Z - 5)
    gap_mouth = groove_hw(c.SPINE_FACE_X - CLR) - (RAIL_ROOT_HW + (RAIL_ROOT_X - (c.SPINE_FACE_X - CLR)) * 0.5)
    rep("groove clearance at the mouth %.2f mm/side" % gap_mouth, 0.15 < gap_mouth < 0.4)

    print("\n the Pi inside")
    rep("Pi keep-out ^ housing  %6.1f mm^3" % vol(P.intersect(H)), vol(P.intersect(H)) < 0.5)
    rep("Pi keep-out ^ lid      %6.1f mm^3" % vol(P.intersect(L)), vol(P.intersect(L)) < 0.5)
    rep("housing ^ lid          %6.1f mm^3" % vol(H.intersect(L)), vol(H.intersect(L)) < 0.5)
    rep("board clears the keel by %.1f mm" % (X_KEEL_BOT - X_BOARD_BACK),
        X_KEEL_BOT - X_BOARD_BACK >= 1.5)

    print("\n cable entries")
    cuts = front_wall_cutters()
    wall = _box(X_LID_IN, X_ROOF_UNDER, -Y_BOX, Y_BOX, Z_BOX_FRONT, Z_INT_FRONT)
    for nm, cut in cuts.items():
        # measure the PART, not the cutter: whatever housing material is left
        # inside the hole's footprint through the wall must be nil
        left = vol(H.intersect(cut.intersect(wall)))
        area = vol(cut.intersect(wall)) / WALL
        rep("%-7s clear through the wall (%.0f mm^2 open, %.1f mm^3 left in it)"
            % (nm, area, left), left < 0.5 and area > 10)
    z0 = Z_BOX_FRONT - USBC_AWNING_OUT
    plug = _box(USBC_X - 4, USBC_X + 4, USBC_Y - 7, USBC_Y + 7,
                z0 - 1, Z_INT_FRONT + 1)
    rep("14 x 8 mm rigid USB-C plug clears straight awning",
        vol(H.intersect(plug)) < 0.5)
    side_wall = _box(CSI_X - CSI_SLOT_W / 2, CSI_X + CSI_SLOT_W / 2,
                     Y_INT, Y_BOX, CSI_Z - CSI_SLOT_L / 2, CSI_Z + CSI_SLOT_L / 2)
    rep("ribbon clear through microSD side wall",
        vol(H.intersect(side_wall)) < 0.5)
    rep("ribbon passage clear under camera awning",
        vol(H.intersect(ribbon_cutter())) < 0.5)
    rep("GPIO clear through microSD side wall and half-ring awning",
        vol(H.intersect(gpio_cutter())) < 0.5)
    # USB-C: plug shell must line up with the receptacle
    rx, ry, _ = board_to_model(USB_C_BX, 0, 1.63)
    rep("USB-C cutout centred on the receptacle (dy %.2f, dx %.2f)"
        % (USBC_Y - ry, USBC_X - rx), abs(USBC_Y - ry) < 0.1 and abs(USBC_X - rx) < 0.1)
    # ribbon length estimate: connector -> side slot -> camera slot on the pad
    cx, cy, cz = board_to_model(CSI_BX, 12.5, 5.5)
    sx, sy_, sz = CSI_X, Y_BOX, CSI_Z
    rep("side slot aligned with CSI cable exit (dx %.1f, dz %.1f mm)"
        % (sx - cx, sz - cz), abs(sx - cx) < 1.0 and abs(sz - cz) < 0.1)
    px, py_, pz = c.CAM_X + c.CAM_T / 2, c.CAM_SLOT_POS[0], c.CAM_SLOT_POS[1]
    d1 = math.dist((cx, cy, cz), (sx, sy_, sz))
    ribbon = d1 + math.dist((sx, sy_, sz), (px, py_, pz)) + 8.0
    rep("camera ribbon route ~%.0f mm (+ slack) vs 200mm cable" % ribbon, ribbon < 170)

    print("\n weather")
    rep("roof overhang: sides %.0f, front %.0f, aft %.0f mm, drip lip all round"
        % (OVH_SIDE, OVH_FRONT, OVH_AFT), True)
    rep("roof side edge (y=%.1f) clears the lid bosses (y=%.1f)"
        % (Y_ROOF, BOSS_Y + BOSS_D / 2), Y_ROOF > BOSS_Y + BOSS_D / 2)

    if export:
        for nm, part in (("rpi_housing", H), ("rpi_housing_lid", L)):
            cq.exporters.export(part, nm + ".stl")
            cq.exporters.export(part, nm + ".step")
        cq.exporters.export(P, "rpi4b_keepout.stl")
    print("\nALL OK" if ok else "\n*** CHECK ***")
    return H, L, P


if __name__ == "__main__":
    build()
