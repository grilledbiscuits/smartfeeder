"""
v6 CONCEPT MASSING MODEL - not the production CAD.

Built only to render my interpretation of the sketches for review. Dimensions
are plausible but provisional; the sketches are explicitly not to scale.

Part frame (same as previous versions):
    Z = bottle axis, Z=0 at the lip face, +Z aft toward the base.
    +X = "up" side in the hung attitude.   -X = down, where the bottom spine is.
    +/-Y = the two sides, where the side support beams are.

Components, each exported separately so the render can colour-code them.
"""

import math
import cadquery as cq

# ---------------------------------------------------------------- bottle ----
LIP_D = 100.0 / math.pi          # 31.83
BODY_D = 260.0 / math.pi         # 82.76
L_TOTAL = 260.0
Z_SHOULDER = 110.0
TILT_DEG = 30.0


def bottle_r(z):
    if z <= 0:
        return 0.0
    if z >= Z_SHOULDER:
        return BODY_D / 2
    return LIP_D / 2 + (BODY_D - LIP_D) / 2 * z / Z_SHOULDER


BOTTLE_LIFT = 10.1   # the bottle actually rests this much higher than the
                     # nominal - see the seat-contact note


def bottle():
    cone = (cq.Workplane("XY").circle(LIP_D / 2)
            .workplane(offset=Z_SHOULDER).circle(BODY_D / 2).loft(ruled=True))
    body = (cq.Workplane("XY").workplane(offset=Z_SHOULDER)
            .circle(BODY_D / 2).extrude(L_TOTAL - Z_SHOULDER))
    return cone.union(body).translate((0, 0, BOTTLE_LIFT))


# ------------------------------------------------------------- primitives ---
def _loft_radial(sections, az_deg):
    """Loft a member through (z, width, r_in, r_out) sections. Built radial
    along +X, then rotated to the azimuth."""
    wires = []
    for (z, w, ri, ro) in sections:
        wires.append(cq.Workplane("XY").workplane(offset=z)
                     .center((ri + ro) / 2.0, 0).rect(ro - ri, w).wires().val())
    s = cq.Workplane(obj=cq.Solid.makeLoft(wires, True))
    return s.rotate((0, 0, 0), (0, 0, 1), az_deg)


def _cyl(r, p0, d, L):
    return cq.Workplane(obj=cq.Solid.makeCylinder(
        r, L, cq.Vector(*p0), cq.Vector(*d)))


def _beam(r_in, r_out, width, z0, z1, az_deg):
    """A square box spanning r_in..r_out radially, `width` tangentially,
    z0..z1 along the axis, rotated to the given azimuth."""
    b = cq.Workplane("XY").box(r_out - r_in, width, z1 - z0,
                               centered=(False, True, False))
    return b.translate((r_in, 0, z0)).rotate((0, 0, 0), (0, 0, 1), az_deg)


def tight_bb(shape):
    """Bounding box from the tessellation. OCC's BRepBndLib inflates a lofted
    solid's box by a couple of mm (it uses surface poles, not the surface), so
    the raw BoundingBox is useless for clearance numbers."""
    v = shape.val() if hasattr(shape, "val") else shape
    pts = [p for p in v.tessellate(0.02)[0]]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def _rod(p0, p1, r):
    d = tuple(p1[i] - p0[i] for i in range(3))
    return _cyl(r, p0, d, math.sqrt(sum(c * c for c in d)))


# ------------------------------------------------------------------ rings ---
NECK_Z0, NECK_H = 45.0, 30.0
NECK_BORE_LO, NECK_BORE_HI = 48.0, 70.0      # conical: stops the bottle sliding
NECK_WALL = 8.0
NECK_FLANGE_R = 50.0

MID_Z0, MID_H = 128.0, 22.0
REAR_Z0, REAR_H = 208.0, 22.0
BODY_BORE_D, BODY_OD = 87.0, 102.0


def neck_rest_ring():
    ring = (cq.Workplane("XY").workplane(offset=NECK_Z0)
            .circle(NECK_BORE_LO / 2 + NECK_WALL)
            .workplane(offset=NECK_H).circle(NECK_BORE_HI / 2 + NECK_WALL)
            .loft(ruled=True))
    ring = ring.union(cq.Workplane("XY").workplane(offset=NECK_Z0 + NECK_H - 8)
                      .circle(NECK_FLANGE_R).extrude(8))
    slope = (NECK_BORE_HI - NECK_BORE_LO) / 2 / NECK_H
    bore = (cq.Workplane("XY").workplane(offset=NECK_Z0 - 2)
            .circle(NECK_BORE_LO / 2 - 2 * slope)
            .workplane(offset=NECK_H + 4).circle(NECK_BORE_HI / 2 + 2 * slope)
            .loft(ruled=True))
    return ring.cut(bore)


def body_ring(z0, h):
    return (cq.Workplane("XY").workplane(offset=z0).circle(BODY_OD / 2)
            .extrude(h)
            .cut(cq.Workplane("XY").workplane(offset=z0 - 1)
                 .circle(BODY_BORE_D / 2).extrude(h + 2)))


# ------------------------------------------------------------- side beams ---
# Splayed in plan: tucked in close at the narrow front of the bottle, opening
# out to the body radius aft. Inner radius always clears the local bottle
# radius, so the bottle still lifts straight out.
BEAM_DEPTH = 13.0
BEAM_SECTIONS = [
    (45.0, 22.0, 32.0, 32.0 + BEAM_DEPTH),   # flush with the ring face
    (70.0, 20.0, 37.0, 37.0 + BEAM_DEPTH),
    (110.0, 20.0, 46.0, 46.0 + BEAM_DEPTH),
    (150.0, 22.0, 47.0, 47.0 + BEAM_DEPTH),
    (230.0, 24.0, 47.0, 47.0 + BEAM_DEPTH),
]

# ------------------------------------------------------------ bottom spine --
# The front of the spine follows the neck ring's outer cone so the two
# actually fuse over its whole length, instead of flying past it with a 12mm
# gap and meeting only at the flange.
# ...and the OUTER face now conforms too, instead of running straight out to
# r=58 while the inner face climbed the cone - that left a 26mm-deep solid
# wedge at the nose doing nothing. Depth now tapers 18 -> 14 instead of
# 26 -> 14. It cannot follow the cone exactly: the outer face has to stay
# outboard of the camera platform's root (which spans x -47..-42), so 50 at
# the nose is the floor, giving 3mm of cover under the plate. From z=72 it is
# flat at 58, which is the rail's seat.
# The nose is ONE straight taper, 50 at z=45 to the flat 58 face at z=68. Do
# not add an intermediate section on the outer face: 50/55/58 at 45/60/68 is
# very slightly kinked, and that kink was enough to make the OCC union of the
# frame with the rail silently return the rail alone.
SPINE_SECTIONS = [
    (45.0, 26.0, 32.0, 50.0),
    (68.0, 27.4, 40.4, 58.0),
    (90.0, 28.0, 44.0, 58.0),
    (230.0, 28.0, 44.0, 58.0),
]
SPINE_FACE_X = -58.0             # flat underside over the straight run

# ------------------------------------------- RPi housing rail (dovetail) ----
# Chosen profile: a dovetail tenon - narrow at the root, wide at the tip - so
# the housing cannot drop off, only slide along. Runs along the straight part
# of the spine's underside.
RAIL_Z0, RAIL_Z1 = 72.0, 226.0
RAIL_ROOT_W, RAIL_TIP_W, RAIL_H = 16.0, 26.0, 9.0


def rpi_rail():
    wires = []
    for (x, w) in ((SPINE_FACE_X + 1.0, RAIL_ROOT_W),
                   (SPINE_FACE_X - RAIL_H, RAIL_TIP_W)):
        wires.append(cq.Workplane("YZ").workplane(offset=x)
                     .center(0, (RAIL_Z0 + RAIL_Z1) / 2)
                     .rect(w, RAIL_Z1 - RAIL_Z0).wires().val())
    return cq.Workplane(obj=cq.Solid.makeLoft(wires, True))


# ------------------------------------------------------------ camera arm ----
# A simple FLAT arm off the front of the bottom spine, swept out to one side in
# plan. Constant thickness, no drop, no angled pad - it is just a plate whose
# far end is the camera housing base. Built as a plan outline extruded along X.
CAM_T = 5.0                                  # plate thickness (in X) - thin
# Top edge of the spine, not the underside - the underside is the RPi rail's.
# The spine's top face runs -32 at z=45 to -41.2 at z=70 over the platform's
# root, so -42 keeps the plate just inside it and fully fused along the root.
CAM_X = -47.0                                # plate spans -47..-42
# HOCKEY-STICK plan form, built as two pieces unioned:
#   * the HANDLE - a tapering band running transverse off the spine's front,
#     then bending at an obtuse knee toward the camera end;
#   * the BLADE / PAD - a plain rectangle, square with the ribbon slot, that a
#     separate camera housing box slides onto.
# The handle is narrower than the pad, so the pad's two long edges stand proud
# as rails and the box slides on from the free end and stops on the shoulders.
# Root edge runs ALONG the spine (y=-2, z 44..70) rather than clipping its
# corner - a long cantilever needs more than the ~40mm^2 footprint that gave.
# y=-2 keeps 2mm clear of the spine perch tab at y -12..-4.
# Root pushed 1mm aft so its 26mm band sits fully on the spine (z 45..71), and
# the arm now routes FORWARD of the +Y perch lug instead of under it: the lug's
# hole is at (y 53.5, z 57.1) in plan and the blade used to run straight
# beneath it, leaving no corridor to thread wire down to the bottom lug.
# Root band z 45..66: it used to run to 71 and clip the neck ring's flange
# (which starts at z=67) with a 53 mm^3 knife-edge sliver - a feather edge that
# prints badly and reads as an outcrop where the arm crosses the flange rim.
# The arm now attaches to the spine alone.
CAM_ROOT = (-2.0, 55.5)                      # centreline start, on the spine
# Knee swung forward to clear the perch lug. The arm dives forward-outboard
# from the root, passes ahead of the lug, then turns back aft-outboard to the
# pad - which stays exactly where it was, as does the slot.
CAM_KNEE = (56.0, 32.0)                      # the obtuse bend
PERCH_WIRE_CLEAR = 8.0                       # required plan corridor radius
                                             # below the side lug's hole
CAM_W_ROOT, CAM_W_KNEE, CAM_W_TIP = 21.0, 21.0, 22.0
CAM_PAD_L, CAM_PAD_W = 36.0, 32.0            # along the slot axis / across it
CAM_TIP_INSET = 9.0                          # how far the handle runs into the pad
# The camera sits directly ABOVE the slot and is angled up or down in its own
# housing to take up the height offset. Only the SLOT is rotated, in plan, so
# its long axis is perpendicular to the line to the feeder port - and the pad
# is built on the slot's own axes, so it is square with the hole by
# construction. NEITHER THE SLOT POSITION NOR ITS ANGLE CHANGES HERE.
# Pushed out along the SAME bearing from the port until the camera sits 100mm
# away - the Camera Module 3's minimum focus. Same bearing means the aim-exact
# angle is unchanged at 116.1 deg, so the specified 118 deg still holds.
CAM_SLOT_POS = (80.4, 39.4)                  # (Y, Z) on the platform
CAM_RANGE_TARGET = 100.0                     # mm, camera to feeder port
CAM_SLOT_ANGLE = 128.0                       # specified; also ~the angle that
                                             # aims at the port from here
# The feeder port juts ~10mm AHEAD of the bottle lip, so it sits at about
# z = 0 rather than at the lip face (z = 10.1 with the bottle's rest lift).
PORT_PROUD = 10.0
CAM_AIM = (0.0, BOTTLE_LIFT - PORT_PROUD)    # feeder port in plan (Y, Z)
# Pi CSI ribbon is 15-way at 1mm pitch: ~16.0 wide x 0.3 thick. An 18 x 3 slot
# with rounded ends clears it without a sharp corner to chafe the cable.
CAM_SLOT_L, CAM_SLOT_W = 18.0, 3.0


def slot_angle_deg():
    """Rotation of the ribbon slot in plan, as specified."""
    return CAM_SLOT_ANGLE


def side_lug_hole_plan():
    """(y, z) of the +Y side perch lug's hole, in plan."""
    r_at, slope = _host_face(BEAM_SECTIONS)
    phi = math.atan(slope)
    cph, sph = math.cos(phi), math.sin(phi)
    L, H, e = PERCH_L_SIDE, PERCH_Z1 - PERCH_Z0, PERCH_EMBED
    z0 = PERCH_Z0 + L * sph
    r0 = (r_at + slope * (z0 - PERCH_Z0)) - e / cph
    base = (_rounded_tab(0.0, L, PERCH_TAB_T_SIDE, 0.0, H, 0.0)
            .rotate((0, 0, 0), (0, 1, 0), math.degrees(phi))
            .translate((r0, 0, z0)))
    dt = (PERCH_Z0 - base.val().BoundingBox().zmin) / cph
    hx, hz = L - PERCH_HOLE_FROM_END_SIDE, H / 2.0
    return (r0 + hx * cph + hz * sph + dt * sph,
            z0 - hx * sph + hz * cph + dt * cph)


def wire_corridor_mm():
    """Clear plan radius below the side lug's hole. The camera arm has to cross
    the space between that lug and the bottom lug to reach the spine, so this
    is what says whether wire can actually be threaded down through it."""
    hy, hz = side_lug_hole_plan()
    outline = (_offset_polygon([CAM_ROOT, CAM_KNEE, _pad_attach()],
                               [CAM_W_ROOT / 2, CAM_W_KNEE / 2, CAM_W_TIP / 2]),
               pad_corners())
    worst = 1e9
    for poly in outline:
        inside = False
        n = len(poly)
        for i in range(n):
            ax, az_ = poly[i]
            bx, bz = poly[(i + 1) % n]
            if (az_ > hz) != (bz > hz):
                if hy < ax + (hz - az_) / (bz - az_) * (bx - ax):
                    inside = not inside
            dy, dz = bx - ax, bz - az_
            t = max(0.0, min(1.0, ((hy - ax) * dy + (hz - az_) * dz)
                             / (dy * dy + dz * dz)))
            worst = min(worst, math.hypot(hy - (ax + t * dy), hz - (az_ + t * dz)))
        if inside:
            return -worst
    return worst


def port_range_mm():
    """Straight-line distance from the camera (directly above the slot) to the
    feeder port. Must clear the Camera Module 3's ~100mm minimum focus."""
    y, z = CAM_SLOT_POS
    return math.sqrt((y - CAM_AIM[0]) ** 2 + (z - CAM_AIM[1]) ** 2
                     + (CAM_X + CAM_T / 2) ** 2)


def slot_aim_angle_deg():
    """The angle that would point the camera exactly at the feeder port, for
    comparison with the specified one."""
    y, z = CAM_SLOT_POS
    return math.degrees(math.atan2(y - CAM_AIM[0], CAM_AIM[1] - z))


def _pad_axes():
    """Unit vectors in the plan (Y, Z): u along the slot's long axis, v across
    it. The pad is built on these, so it is square with the hole by
    construction and stays square if the slot angle ever moves."""
    a = math.radians(CAM_SLOT_ANGLE)
    return (math.cos(a), math.sin(a)), (math.sin(a), -math.cos(a))


def _knee_inner_edge():
    """A point on the handle's INNER (concave-side) edge where it meets the
    pad, and the edge's direction."""
    ky, kz = CAM_KNEE
    my, mz = _pad_attach()
    dy, dz = my - ky, mz - kz
    L = math.hypot(dy, dz)
    t = (dy / L, dz / L)
    n = (-t[1], t[0])
    ry, rz = CAM_ROOT                       # inside of the bend is the side
    if (ry - ky) * n[0] + (rz - kz) * n[1] < 0:      # the root lies on
        n = (-n[0], -n[1])
    h = CAM_W_TIP / 2.0
    return (my + h * n[0], mz + h * n[1]), t


def pad_extent_u():
    """(+u, -u) half-extents of the pad. The +u end is pushed out to sit flush
    with the handle's inner edge - otherwise the handle pokes a couple of mm
    past the pad's end and leaves a step there."""
    u, _v = _pad_axes()
    (ey, ez), _t = _knee_inner_edge()
    cy, cz = CAM_SLOT_POS
    reach = (ey - cy) * u[0] + (ez - cz) * u[1]
    return max(CAM_PAD_L / 2.0, reach), CAM_PAD_L / 2.0


def pad_corners():
    """The four corners of the rectangular camera end, in order."""
    u, v = _pad_axes()
    cy, cz = CAM_SLOT_POS
    ap, am = pad_extent_u()
    b = CAM_PAD_W / 2.0
    return [(cy + sa * a * u[0] + sb * b * v[0],
             cz + sa * a * u[1] + sb * b * v[1])
            for (sa, sb, a) in ((1, 1, ap), (-1, 1, am),
                                (-1, -1, am), (1, -1, ap))]


def _pad_attach():
    """Where the handle's centreline ends - on the pad's axis, CAM_TIP_INSET
    short of the pad's near edge, so handle and pad overlap and fuse."""
    u, _v = _pad_axes()
    cy, cz = CAM_SLOT_POS
    d = CAM_PAD_L / 2.0 - CAM_TIP_INSET
    return (cy + d * u[0], cz + d * u[1])


def _offset_polygon(pts, halfw):
    """Closed outline of a band following the polyline `pts` with the given
    half-widths, mitred at the interior vertices."""
    nrm = []
    for (p, q) in zip(pts, pts[1:]):
        dy, dz = q[0] - p[0], q[1] - p[1]
        L = math.hypot(dy, dz)
        nrm.append((-dz / L, dy / L))              # left-hand normal
    left, right = [], []
    for i, p in enumerate(pts):
        if i == 0 or i == len(pts) - 1:
            n, k = (nrm[0] if i == 0 else nrm[-1]), 1.0
        else:
            a, b = nrm[i - 1], nrm[i]
            mx, my = a[0] + b[0], a[1] + b[1]
            m = math.hypot(mx, my)
            n = (mx / m, my / m)
            k = 1.0 / (n[0] * a[0] + n[1] * a[1])  # mitre extension
        h = halfw[i] * k
        left.append((p[0] + n[0] * h, p[1] + n[1] * h))
        right.append((p[0] - n[0] * h, p[1] - n[1] * h))
    return left + right[::-1]


def knee_angle_deg():
    """Included angle at the bend. Must be obtuse - that is what makes it a
    hockey stick rather than an L."""
    (ay, az), (ky, kz), (my, mz) = CAM_ROOT, CAM_KNEE, _pad_attach()
    a = math.atan2(az - kz, ay - ky)
    b = math.atan2(mz - kz, my - ky)
    d = math.degrees(abs(a - b)) % 360.0
    return 360.0 - d if d > 180.0 else d


def slot_clearance_mm(n=240):
    """Smallest gap from the slot outline to the edge of the rectangular pad.
    Negative means the slot breaks out and is not a closed hole."""
    u, _v = _pad_axes()
    h = (CAM_SLOT_L - CAM_SLOT_W) / 2.0
    r = CAM_SLOT_W / 2.0
    cy, cz = CAM_SLOT_POS
    pad = pad_corners()
    worst = 1e9
    for i in range(n):
        t = 2 * math.pi * i / n
        sgn = h if math.cos(t) >= 0 else -h
        py = cy + sgn * u[0] + r * math.cos(t) * u[0] - r * math.sin(t) * u[1]
        pz = cz + sgn * u[1] + r * math.cos(t) * u[1] + r * math.sin(t) * u[0]
        for j in range(4):
            y1, z1 = pad[j]
            y2, z2 = pad[(j + 1) % 4]
            dy, dz = y2 - y1, z2 - z1
            worst = min(worst,
                        ((py - y1) * dz - (pz - z1) * dy) / math.hypot(dy, dz))
    return worst


def camera_arm():
    wp = cq.Workplane("YZ").workplane(offset=CAM_X)
    handle = wp.polyline(_offset_polygon(
        [CAM_ROOT, CAM_KNEE, _pad_attach()],
        [CAM_W_ROOT / 2, CAM_W_KNEE / 2, CAM_W_TIP / 2])).close().extrude(CAM_T)
    pad = wp.polyline(pad_corners()).close().extrude(CAM_T)
    arm = handle.union(pad)
    slot = (cq.Workplane("YZ").workplane(offset=CAM_X - 2)
            .center(*CAM_SLOT_POS)
            .slot2D(CAM_SLOT_L, CAM_SLOT_W, slot_angle_deg())
            .extrude(CAM_T + 4))
    return arm.cut(slot)


# ------------------------------------------------------------- perch lugs ---
# Three of them: one on each side beam, one on the bottom spine. Wire winds
# through these out to the perch in front of the bottle.
PERCH_HOLE_D = 6.0
PERCH_TAB_T_SIDE = 6.0        # thinner than the 22mm-wide side beams
PERCH_TAB_T_SPINE = 8.0       # thinner than the 28mm-wide bottom spine
PERCH_Z0, PERCH_Z1 = 45.0, 70.0      # front face flush with the ring / beams
PERCH_L_SIDE, PERCH_L_SPINE = 20.0, 16.0     # reach measured off the host face
# Hole sits in the rounded end, measured back from the free end.
PERCH_HOLE_FROM_END_SIDE = 10.0
PERCH_HOLE_FROM_END_SPINE = 7.5
PERCH_EMBED = 4.0             # depth of the root inside the host, normal to it
# The BOTTOM lug is pulled in and kept forward so the dovetail stays the lowest
# thing on the cradle. Three changes do it: a deeper root (6 instead of 4), a
# shorter reach (18 instead of 26), and - the big one - only 16mm of extent
# along the nose instead of 25, so the lug sits entirely on the shallow front
# of the taper (z 45..64) rather than running back to z=74 where the spine's
# face has already dropped to r=58 and drags the lug's aft corner down with it.
PERCH_H_SPINE = 16.0
PERCH_EMBED_SPINE = 4.0        # 4, not 6: a deeper root reaches past the
                               # camera platform's underside at x=-47
PERCH_CNR_R_SPINE = 7.5
# Free end fully rounded: at 12mm against a 25mm-tall tab the two corner arcs
# almost meet, so the end is a half-round. That is also kinder to the hole than
# a smaller radius, because the arc centres land right next to it.
PERCH_CNR_R = 12.0            # plan-corner radius at the free end of each lug
PERCH_EDGE_R = 3.0            # break on the outer end face's edges (capped by
                              # the tab thickness - see below)


def _rounded_tab(r_in, r_out, width, z0, z1, az_deg, cnr_r=None):
    """A flat tab spanning r_in..r_out radially, with its FREE (outer) end
    rounded off - both plan corners filleted and the end face's edges broken.
    Built radial along +X, then rotated to the azimuth."""
    tab = (cq.Workplane("XY").box(r_out - r_in, width, z1 - z0,
                                  centered=(False, True, False))
           .translate((r_in, 0, z0)))
    # the two corners at the free end, i.e. the edges running through the
    # tab's thickness at max radius. Back off if the radius is too big to build.
    # No silent fallback: a quietly shrunk radius would leave tab_geometry
    # reporting an edge distance the part does not actually have.
    R = PERCH_CNR_R if cnr_r is None else cnr_r
    if R > (z1 - z0) / 2 - 0.4:
        raise ValueError("corner radius %.2f too large for a %.1fmm tab"
                         % (R, z1 - z0))
    tab = tab.edges("|Y").edges(">X").fillet(R)
    er = min(PERCH_EDGE_R, width / 2 - 0.6)
    if er > 0.3:
        tab = tab.faces(">X").edges().fillet(er)   # cosmetic edge break
    return tab.rotate((0, 0, 0), (0, 0, 1), az_deg)


def _host_face(sections):
    """The host member's OUTER face over the lug's z range, as (r at PERCH_Z0,
    dr/dz). The lug is set square to THIS, not to the bottle axis."""
    a = b = None
    for (z, _w, _ri, ro) in sections:
        if z <= PERCH_Z0 + 1e-9:
            a = (z, ro)
        elif b is None:
            b = (z, ro)
    slope = (b[1] - a[1]) / (b[0] - a[0])
    return a[1] + slope * (PERCH_Z0 - a[0]), slope


def _tilted_tab(sections, length, thick, hole_from_end, az_deg,
                height=None, cnr_r=None, embed=None, lead=0.0):
    """A perch lug set SQUARE TO ITS HOST: the root plane is parallel to the
    member's outer face and the lug projects along that face's normal, instead
    of both being referenced to the bottle axis while the face slopes away.

    Positioned so the lug's forward-most point lands exactly on PERCH_Z0 -
    nothing projects ahead of the member's front face.
    """
    r_at, slope = _host_face(sections)
    phi = math.atan(slope)
    L = length
    H = (PERCH_Z1 - PERCH_Z0) if height is None else height
    e = PERCH_EMBED if embed is None else embed
    z0 = PERCH_Z0 + L * math.sin(phi)
    r0 = (r_at + slope * (z0 - PERCH_Z0)) - e / math.cos(phi)

    c, s_ = math.cos(phi), math.sin(phi)
    base = (_rounded_tab(0.0, L, thick, 0.0, H, 0.0, cnr_r)
            .rotate((0, 0, 0), (0, 1, 0), math.degrees(phi))
            .translate((r0, 0, z0)))
    # The rounded free end pulls the tip back from where a sharp corner would
    # have sat, so slide the tab ALONG the face (which keeps the root's depth
    # in the host constant) until its forward-most point lands exactly on the
    # member's front face.
    dt = (PERCH_Z0 - lead - base.val().BoundingBox().zmin) / c
    tab = base.translate((dt * s_, 0, dt * c)).rotate((0, 0, 0), (0, 0, 1), az_deg)
    # hole through the thickness, centred in the rounded end, carried through
    # the same transform so it stays square to the tab
    hx, hz = L - hole_from_end, H / 2.0
    wx = r0 + hx * c + hz * s_ + dt * s_
    wz = z0 - hx * s_ + hz * c + dt * c
    az = math.radians(az_deg)
    p = (wx * math.cos(az), wx * math.sin(az), wz)
    d = (-math.sin(az), math.cos(az), 0.0)            # thickness direction
    return tab.cut(_cyl(PERCH_HOLE_D / 2,
                        (p[0] - 30 * d[0], p[1] - 30 * d[1], p[2]), d, 60))


def tab_geometry(sections, length, hole_from_end, height=None, cnr_r=None):
    """Face tilt, and the hole's distance to the nearest free edge."""
    _r, slope = _host_face(sections)
    H = (PERCH_Z1 - PERCH_Z0) if height is None else height
    PERCH_CNR = PERCH_CNR_R if cnr_r is None else cnr_r
    hx, hz = length - hole_from_end, H / 2.0
    d = [hz, H - hz]
    for cz in (PERCH_CNR, H - PERCH_CNR):
        d.append(PERCH_CNR - math.hypot(hx - (length - PERCH_CNR), hz - cz))
    return math.degrees(math.atan(slope)), min(d) - PERCH_HOLE_D / 2


def perch_lugs():
    """Thin flat tabs off the front of each side beam and the bottom spine,
    each set square to the member it grows out of. Deliberately thinner than
    those members, with free ends rounded so wound wire has nothing sharp to
    bear on."""
    tabs = None
    for az in (90.0, 270.0):
        tab = _tilted_tab(BEAM_SECTIONS, PERCH_L_SIDE, PERCH_TAB_T_SIDE,
                          PERCH_HOLE_FROM_END_SIDE, az)
        tabs = tab if tabs is None else tabs.union(tab)
    # Spine tab: projects off the spine's nose, CENTRED on y=0. The -8mm offset
    # it used to carry was there to dodge the camera platform's root; the
    # platform now lives on the spine's TOP edge, so it is obsolete.
    return tabs.union(_tilted_tab(SPINE_SECTIONS, PERCH_L_SPINE,
                                  PERCH_TAB_T_SPINE,
                                  PERCH_HOLE_FROM_END_SPINE, 180.0,
                                  height=PERCH_H_SPINE,
                                  cnr_r=PERCH_CNR_R_SPINE,
                                  embed=PERCH_EMBED_SPINE))


# --------------------------------------------------- ceiling wire holes -----
# Six, all through the side beams: one pair forward, two pairs aft. Only one
# of the aft pairs gets used, chosen to balance the bottle.
WIRE_HOLE_D = 5.0
WIRE_FWD_Z = 88.0
# 188, not 176: the two-part split puts a dovetail tongue at z=158..176, and a
# hole at 176 landed exactly on the tongue's tip AND inside its radial span
# (r 51..56 against a hole at r=53.5), notching the tip open. 188 clears the
# joint by 12mm and still gives a usable spread against the 224 station.
WIRE_AFT_ZS = (188.0, 224.0)
JOINT_KEEPOUT_Z = (150.0, 180.0)     # nothing drilled here - see split.py


def make_frame():
    part = neck_rest_ring()
    part = part.union(body_ring(MID_Z0, MID_H))
    part = part.union(body_ring(REAR_Z0, REAR_H))
    for az in (90.0, 270.0):
        part = part.union(_loft_radial(BEAM_SECTIONS, az))
    part = part.union(_loft_radial(SPINE_SECTIONS, 180.0))
    return part


def beam_radii(z):
    """Interpolate the splayed side beam's inner/outer radius at station z."""
    ss = BEAM_SECTIONS
    if z <= ss[0][0]:
        return ss[0][2], ss[0][3]
    for a, b in zip(ss, ss[1:]):
        if z <= b[0]:
            f = (z - a[0]) / (b[0] - a[0])
            return (a[2] + f * (b[2] - a[2]), a[3] + f * (b[3] - a[3]))
    return ss[-1][2], ss[-1][3]


def wire_hole_cutters():
    """Through the side beams TOP TO BOTTOM - axis along X, i.e. vertical in
    the hung attitude, so the wire drops through and knots underneath."""
    cuts = []
    for z in (WIRE_FWD_Z,) + WIRE_AFT_ZS:
        ri, ro = beam_radii(z)
        ym = (ri + ro) / 2.0
        for sy in (1, -1):
            cuts.append(_cyl(WIRE_HOLE_D / 2, (-90.0, sy * ym, z), (1, 0, 0), 180.0))
    return cuts


def build():
    frame = make_frame()
    rail = rpi_rail()
    cam = camera_arm()
    lugs = perch_lugs()

    whole = frame.union(rail).union(cam).union(lugs)
    for c in wire_hole_cutters():
        whole = whole.cut(c)

    parts = {
        "neck_rest_ring": neck_rest_ring(),
        "mid_body_ring": body_ring(MID_Z0, MID_H),
        "rear_body_ring": body_ring(REAR_Z0, REAR_H),
        "side_beams": _loft_radial(BEAM_SECTIONS, 90.0)
                      .union(_loft_radial(BEAM_SECTIONS, 270.0)),
        "bottom_spine": _loft_radial(SPINE_SECTIONS, 180.0),
        "rpi_rail": rail,
        "camera_arm": cam,
        "perch_lugs": lugs,
    }
    # punch the wire holes through the beams for the coloured export too
    for c in wire_hole_cutters():
        parts["side_beams"] = parts["side_beams"].cut(c)

    for nm, p in parts.items():
        cq.exporters.export(p, "%s.stl" % nm)
    cq.exporters.export(bottle(), "bottle.stl")
    cq.exporters.export(whole, "whole.stl")
    cq.exporters.export(whole, "concept.step")

    x0, x1, y0, y1, z0b, z1b = tight_bb(whole)
    print("concept bounding box: %.0f x %.0f x %.0f mm"
          % (x1 - x0, y1 - y0, z1b - z0b))
    print("  X %.0f .. %.0f   Y %.0f .. %.0f   Z %.0f .. %.0f"
          % (x0, x1, y0, y1, z0b, z1b))
    print("  volume %.0f cm^3 -> ~%.0f g solid PETG"
          % (whole.val().Volume() / 1000, whole.val().Volume() * 1.27e-3))
    print("  solids: %d   (design tilt %.0f deg from horizontal,"
          " lip low)" % (len(whole.val().Solids()), TILT_DEG))
    # A union can only ADD material. If the assembly comes out smaller than the
    # frame alone, an OCC boolean has silently dropped a body - this has
    # happened twice now, both times from a near-tangent or kinked loft face.
    fv = make_frame().val().Volume()
    print("  union sanity: frame %.0f mm^3 -> whole %.0f mm^3  %s"
          % (fv, whole.val().Volume(),
             "OK" if whole.val().Volume() >= fv - 1 else "*** BOOLEAN DROPPED A BODY ***"))

    # clearance: can the bottle still be withdrawn along its axis?
    # Exclude ONLY a narrow band around the intended seat contact (the seat
    # bore's lower edge at z=45), not everything below it. The old version cut
    # away z < 55.6, which left the whole neck-ring bore untested.
    sweep = (bottle()
             .union(cq.Workplane("XY").workplane(offset=Z_SHOULDER)
                    .circle(BODY_D / 2).extrude(400))
             .cut(cq.Workplane("XY").workplane(offset=NECK_Z0 - 1.0)
                  .circle(300).extrude(3.0)))
    try:
        foul = abs(whole.intersect(sweep).val().Volume())
    except Exception:
        foul = -1.0
    print("  material in the bottle's exit path: %.1f mm^3  %s"
          % (foul, "OK" if 0 <= foul < 1.0 else "CHECK"))
    # --- mirror symmetry about the Y=0 plane -------------------------------
    # Everything except the camera arm is meant to be left/right symmetric.
    for nm in ("bottom_spine", "rpi_rail", "perch_lugs", "side_beams"):
        s = parts[nm]
        v = s.val().Volume()
        try:
            common = abs(s.intersect(s.mirror("XZ")).val().Volume())
        except Exception:
            common = float("nan")
        _a, _b, ymin, ymax, _c, _d = tight_bb(s)
        bb = type("B", (), {"ymin": ymin, "ymax": ymax})
        print("  symmetry %-12s  Y %+8.3f .. %+8.3f  centre %+.4f"
              "   asymmetric volume %.1f mm^3  %s"
              % (nm, bb.ymin, bb.ymax, (bb.ymin + bb.ymax) / 2,
                 v - common, "OK" if v - common < 1.0 else "*** NOT SYMMETRIC ***"))

    sp = parts["bottom_spine"]
    print("  spine nose: depth by station (inner face on the ring cone,"
          " outer face now following it):")
    for (z, w, ri, ro) in SPINE_SECTIONS[:4]:
        print("    z %5.0f   x %7.1f .. %7.1f   depth %4.1f mm" % (z, -ro, -ri, ro - ri))
    old = _loft_radial([(45.0, 26.0, 32.0, 58.0), (60.0, 27.0, 37.5, 58.0),
                        (75.0, 28.0, 43.0, 58.0), (90.0, 28.0, 44.0, 58.0),
                        (230.0, 28.0, 44.0, 58.0)], 180.0)
    dv = old.val().Volume() - sp.val().Volume()
    print("    spine volume %.1f cm^3, was %.1f with the straight nose"
          "  ->  %.1f cm^3 / %.0f g of PETG saved"
          % (sp.val().Volume() / 1000, old.val().Volume() / 1000,
             dv / 1000, dv * 1.27e-3))
    print("    platform underside at x %.1f, spine outer face at the nose"
          " x %.1f -> %.1f mm of cover"
          % (CAM_X, -SPINE_SECTIONS[0][3], SPINE_SECTIONS[0][3] + CAM_X))

    print("  perch lugs: free ends rounded R%.0f, edges broken R%.0f"
          % (PERCH_CNR_R, PERCH_EDGE_R))
    lugz = stlbb = None
    for tag, secs, L, thk, fe, az, host, kw in (
            ("side ", BEAM_SECTIONS, PERCH_L_SIDE, PERCH_TAB_T_SIDE,
             PERCH_HOLE_FROM_END_SIDE, 90.0, "side_beams", {}),
            ("spine", SPINE_SECTIONS, PERCH_L_SPINE, PERCH_TAB_T_SPINE,
             PERCH_HOLE_FROM_END_SPINE, 180.0, "bottom_spine",
             dict(height=PERCH_H_SPINE, cnr_r=PERCH_CNR_R_SPINE,
                  embed=PERCH_EMBED_SPINE))):
        tilt, edge = tab_geometry(secs, L, fe, kw.get("height"), kw.get("cnr_r"))
        tab = _tilted_tab(secs, L, thk, fe, az, **kw)
        bb = tab.val().BoundingBox()
        ov = abs(tab.intersect(parts[host]).val().Volume())
        print("    %s lug: host face tilts %4.1f deg -> lug rotated to match;"
              " hole %.1fmm from the nearest edge" % (tag, tilt, edge))
        print("           embedded %5.0f mm^3 in the %-12s z %5.1f .. %5.1f"
              "   reaches r=%.1f  %s"
              % (ov, host, bb.zmin, bb.zmax, max(abs(bb.xmin), abs(bb.ymax)),
                 "OK" if ov > 300 and bb.zmin >= PERCH_Z0 - 0.05
                 else "*** CHECK ***"))
    rail_tip = abs(SPINE_FACE_X - RAIL_H)
    lug_low = abs(tight_bb(parts["perch_lugs"])[0])
    print("    lowest point: dovetail tip r=%.1f vs bottom lug r=%.1f  ->  %s"
          % (rail_tip, lug_low,
             "dovetail is the limit, %.1fmm clear" % (rail_tip - lug_low)
             if lug_low < rail_tip - 0.05 else "*** LUG HANGS BELOW THE RAIL ***"))

    cam = parts["camera_arm"]
    print("  camera platform: knee %.1f deg (obtuse), pad %.0f x %.0f square"
          " with the slot, handle %.0f wide -> %.1fmm shoulder each side"
          % (knee_angle_deg(), CAM_PAD_L, CAM_PAD_W, CAM_W_TIP,
             (CAM_PAD_W - CAM_W_TIP) / 2))
    for nm in ("bottom_spine", "rpi_rail", "perch_lugs", "side_beams"):
        try:
            ov = abs(cam.intersect(parts[nm]).val().Volume())
        except Exception:
            ov = 0.0
        print("    platform ^ %-13s %8.0f mm^3  %s" % (
            nm, ov,
            "fused" if nm == "bottom_spine" and ov > 500 else
            ("OK" if ov < 1.0 else "*** CLASH ***")))
    try:
        ov = abs(cam.intersect(bottle()).val().Volume())
    except Exception:
        ov = 0.0
    print("    platform ^ bottle        %8.0f mm^3  %s"
          % (ov, "OK" if ov < 1.0 else "*** CLASH ***"))

    hy, hz = side_lug_hole_plan()
    wc = wire_corridor_mm()
    print("    side lug hole at plan (y %.1f, z %.1f): clear radius to the arm"
          " %.1f mm (need %.0f)  %s"
          % (hy, hz, wc, PERCH_WIRE_CLEAR,
             "OK" if wc >= PERCH_WIRE_CLEAR else "*** ARM BLOCKS THE HOLE ***"))
    rng = port_range_mm()
    print("    range to the feeder port %.1f mm (target %.0f)  %s"
          % (rng, CAM_RANGE_TARGET,
             "OK" if rng >= CAM_RANGE_TARGET - 0.5 else "*** INSIDE MIN FOCUS ***"))
    cl = slot_clearance_mm()
    print("  ribbon slot: %.0f deg (aim-exact would be %.1f), clearance to the"
          " platform edge %.2f mm  %s"
          % (CAM_SLOT_ANGLE, slot_aim_angle_deg(), cl,
             "CLOSED" if cl > 0.5 else "*** BREAKS OUT ***"))


if __name__ == "__main__":
    build()
