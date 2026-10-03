# Feeder bottle cradle — v6, two-part split

Four printed parts: two cradle halves and two pin types (2 × beam, 1 × spine).
No fasteners, no glue required.

| Part | Size (mm) | Solid mass | Qty |
|---|---|---|---|
| `cradle_front` | 118 × 168 × 161 | 309 g | 1 |
| `cradle_rear` | 118 × 120 × 72 | 147 g | 1 |
| `cross_pin_beam` | Ø3.7 × 12.6 | — | 2 |
| `cross_pin_spine` | Ø3.7 × 22.6 | — | 1 |

Both halves fit a 180mm bed. The front is the tight one — 168mm across the
camera platform, 12mm to spare.

STL for slicing, STEP for editing (imports to FreeCAD as a single
`Part::Feature`, no parametric tree). `concept.py` builds the whole cradle,
`split.py` cuts it in two — both re-derive every number and re-run the checks.

## The joint

Sliding dovetails at z=158, one on each side beam and one on the spine, 18mm
of engagement, 0.20mm per face of clearance. The front half carries the
tongues, the rear the sockets.

The undercut is in each member's **cross-section plane**, not along the slide.
All three tongues have to engage in one motion and the only motion available is
along the bottle axis, so the dovetail keys the joint against opening radially
or sideways — which is how the rear ring's load wants to break it — and axial
pull-out, the one direction left free, is taken by the cross pins.

The pins are **radial**, entering from outside. This matters: a tangential pin
would run across each tongue's wide face but eat its thin one, leaving 0.5mm of
material on the beams. Radial leaves 3.8mm on the beams and 5.2mm on the spine.

```
beam +Y / -Y   taper 14.0 deg   socket walls 3.8 / 3.8 radial, 3.9 side
spine          taper 14.0 deg   socket walls 3.8 / 2.8 radial, 4.8 side
```

The spine's 2.8mm outer wall is backed by the full 9mm of the RPi rail
underneath it — it is not a free wall.

## Assembly

1. Slide the rear half onto the front along the bottle axis until the dovetail
   shoulders meet. It only goes one way.
2. Push the two short pins in radially through the side beams, and the long one
   up through the rail into the spine joint.
3. **Trim the spine pin flush with the rail's outer face.** It passes through
   the rail, and that is the surface the RPi housing slides along.

Pins are modelled 0.3mm under the hole for a slip fit. Print them, or use 4mm
rod. For outdoors, a drop of glue in each hole is worth it.

## Printing

- **White PLA.** White keeps solar heat gain down.
- Print both halves **axis-vertical, split face down on the bed**. That puts
  the dovetail flanks on as vertical walls rather than layer-line steps, and
  the flanks are what carry the joint.
- 3–4 perimeters, 30%+ infill. Load path is neck ring → spine and beams →
  ceiling wire holes.
- Supports under the beam flares, the seat cone's outer face, and the camera
  platform.

## Verified in the model

Both scripts re-run these on every build; nothing below is copied by hand.

- Each part is a single connected solid; both meshes are watertight
  (0 open edges, 35 874 and 7 164 triangles).
- Bottle withdraws straight out: **0.0 mm³** in its exit path, checked against
  the true swept volume with only a 3mm band excluded at the seat contact.
- Assembled halves interfere by **0.0 mm³**.
- Spine, rail, perch lugs and side beams are mirror-symmetric about Y=0 to
  0.0 mm³.
- Union sanity: the assembly is never smaller than the frame it started from
  (this catches OCC silently dropping a body — it has happened twice).
- Dovetail rail is the lowest point, 1.8mm below the bottom perch lug.
- Camera sits 99.9mm from the feeder port; ribbon slot is a closed hole with
  9.00mm to the nearest platform edge.
- Side perch lug's hole has a 10.5mm clear plan corridor for wire.
- Both parts inside 180mm.

## Known open items — read before printing a final part

1. **Seat cone radius not applied.** The seat is steeper (20.1°) than the
   bottle's taper (13.0°), so contact is a *line* on the seat bore's lower
   edge at z=45, not a band. Pressure is low (~0.07 N/mm) so this is a
   denting/creep concern over seasons, not a strength one, but a ~R4 on that
   edge is the fix and it is not in these files.
2. **Ribbon slot is 12° off aim.** Aim-exact is 116.0°, the slot is at 128° as
   specified. The image will be rolled by that much. Fine if intentional.
3. **Rail crosses the split as a plain butt.** If the RPi housing has to slide
   over the seam it needs a lead-in chamfer both sides, and joint alignment
   becomes a functional tolerance. Depends where the housing sits — undecided.
4. **Spine perch lug has 4.0mm edge distance** around its hole, against 6.9mm
   on the side lugs. The cost of tucking it above the rail.
5. Rear ring position, T-slot vs dovetail for the rail, and which side the
   camera platform sweeps to are all still open.

## Check against the real bottle first

1. **Where it seats.** Drop the bottle through a 60mm hole in cardboard and
   measure up from the lip. The model says it rests 10.1mm higher than nominal
   — contact at z=45. If yours differs, adjust `NECK_BORE_LO`/`NECK_BORE_HI`.
2. **Body diameter**, measured directly rather than from circumference. Over
   84mm and you need to open `BODY_BORE_D`.
3. **Port protrusion.** The camera standoff assumes the port juts 10mm ahead of
   the lip (`PORT_PROUD`). That number sets the 100mm range.
