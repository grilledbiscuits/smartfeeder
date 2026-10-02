# Raspberry Pi 4B housing — for the v6 cradle

Two printed parts: the housing body and a bottom lid. It hangs from the
cradle's dovetail rail on the **front half**, forward of the split.

| Part | Size (mm) | Solid mass |
|---|---|---|
| `rpi_housing` | 113 × 82 × 40 | 85 g |
| `rpi_housing_lid` | 110 × 68 × 6 | 29 g |

STL for slicing, STEP for editing. `rpi_housing.py` builds both and re-runs
every check below; it imports `concept.py`, so the dovetail groove is derived
from the cradle's rail and cannot drift out of step with it.

## Why the holes are where they are

The rig hangs 30° nose-down, which gives each face of the housing a different
exposure:

| Face | Looks | Rain |
|---|---|---|
| Front | forward and **30° down** | sheltered — and undercut, so roof drips fall clear of it |
| Aft | aft and 30° up | takes rain directly — **no holes** |
| Lid | down | sheltered |
| Sides | horizontal | driven rain only, under an 8mm roof overhang |

The USB-C entry is in the sheltered front wall. The camera ribbon and GPIO
cable exit the microSD side wall under the roof overhang:

- **USB-C** — 16 × 10mm rounded slot, centred on the Pi's USB-C receptacle.
  The plug's overmould passes through it and plugs straight in; seal the gap
  round the overmould with a split grommet. Passes overmoulds up to ~14 × 8mm.
  A straight 5mm staple awning has the same 16 × 10mm rounded inner outline,
  with 1.5mm roof and side walls. It leaves the rigid plug path open.
- **Camera ribbon** — 22 × 3.5mm slot in the side wall on the microSD edge of
  the Pi, centred on the modeled CSI connector's cable exit. The ribbon can
  leave the box sideways without turning toward the front wall. Estimated
  route to the camera pad is about 128mm with slack, within the 200mm cable.
  A staple-shaped awning hugs the slot: two 1.2mm side walls and a 1.5mm roof
  with a 4.5mm inner radius. Its quarter-circle profile guides the exiting
  ribbon downward beneath the case's existing perimeter brim.
- **GPIO cable** — Ø8mm hole beside the camera slot on the microSD side wall.
  Takes 4–6 Dupont jumpers one at a time. A straight 5mm half-ring awning
  wraps the upper half of the hole; its inner edge is flush with the opening.

## The Pi inside

Mounted **upside down** on four standoffs from the ceiling (the official
58 × 49mm pattern, M2.5 self-tapping into Ø2.2 pilots), component side facing
the lid. Take the lid off and the camera connector, GPIO header and USB-C are
right there; nothing has to be unplugged to open it.

Interior is sized for standard Dupont jumpers on the GPIO header plus a wire
bend (24mm below the board). If you use right-angle jumpers the housing could
lose ~10mm of height — say so and I'll re-derive it.

## Mounting on the cradle

The groove is **open at both ends**. This CAD has no positive axial stop;
provide the external stopping mechanism before hanging the feeder.

That means it goes on **from the front**:

1. Hold it ahead of the cradle's nose, groove up, and slide it **aft** onto the
   rail.
2. The spine's perch lug fits inside the groove and passes through its open
   front end. The groove also drains at that end.
3. Position the housing with your external stop.
4. Tighten an **M3 grub screw** (tap the Ø2.5 hole in the keel, aft end, +Y
   side) against the rail flank. It's outside the box, so you can reach it with
   everything assembled. The grub screw restrains the housing on the rail.

You can do this with the cradle fully assembled — the housing never crosses
the split joint.

76mm of rail engagement, 0.25mm clearance per face.

## Lid

Four M3 × 8 self-tapping screws from below into bosses on the outside of the
side walls (outside, so they don't compete with the Pi for interior space).
A spigot inside the walls locates it. Screw heads face down — sheltered.

## Cooling

The 4B needs airflow in a box in the sun. Cool air comes in through four 2mm
slots in the lid at the front (the low end); warm air leaves through 2mm slots
high on both side walls at the aft end (the high corner), under the roof
overhang. 2mm keeps most insects out. **Print it in a light colour.**

## Printing

- **PETG or ASA.**
- **Housing: roof down on the bed.** The roof top is flat by design. The
  standoffs point up, the lid bosses
  have 45° gussets, and the dovetail's flanks overhang at only 27° from
  vertical. The groove's floor is a 27mm bridge; the USB-C awning projects
  5mm from the front wall. Check bridging and support the awning if needed.
- Printed this way the groove's layer lines run along the slide direction, so
  it slides smoothly.
- **Lid: flat, spigot up.**

## Verified in the model

- Both parts single solids, meshes watertight (0 open edges).
- **0.0 mm³** interference with the cradle, seated.
- **0.0 mm³** at ten stations along the installation path from ahead of the
  cradle — including the spine lug passing through the groove.
- Groove is open through the front end; the housing slides onto the rail
  without hitting the cradle or perch lug.
- Housing's aft end at z=147.5, 10mm clear of the split joint.
- **0.0 mm³** between a keep-out model of the Pi 4B and either part; board
  clears the keel by 2mm.
- Each cable entry is clear through its wall (0.0 mm³ left inside it). The
  USB-C slot is centred on the receptacle to 0.00mm; the side ribbon slot is
  within 0.5mm of the modeled CSI exit height. All three awnings leave their
  cable passages clear, including a 14 × 8mm rigid USB-C plug.
- Roof's side edge clears the lid bosses, so drips don't land on them.

## Check before printing

1. **The CSI connector's orientation.** The side exit follows the modeled
   connector, whose exact orientation remains unverified. Check your Pi and
   ribbon direction before printing.
2. **Your USB-C lead's overmould.** Measure it. Over 14 × 8mm and you need to
   open `USBC_W` / `USBC_H`.
3. **The Pi keep-out model** is built from the published 4B dimensions with
   generous margins, not from a scan of your board. Clearances of 1–2mm
   (keel, audio jack to the wall) are the ones to eyeball.
4. **Balance.** The housing and Pi add ~160g below the axis. Because the rig
   hangs 30° nose-down, "below" works out as *aft* in the world: the pair sit
   at world X ≈ 127mm, against wire lines at 76 (front) and 163 / 194 (the two
   aft stations). So they pull the centre of mass aft, but stay well inside
   the wires. If the hang looks marginal, the rearmost station (z=224) gives
   the most aft margin.
