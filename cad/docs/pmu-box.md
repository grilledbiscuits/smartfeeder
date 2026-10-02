# Waveshare Power Management HAT (B) box — for the v6 cradle

Two printed parts. Sits on the dovetail rail immediately behind the Pi
housing, so the rail now carries a train of three: cradle → Pi → power.

| Part | Size (mm) | Solid mass |
|---|---|---|
| `pmu_housing` | 145 × 97 × 58 | 140 g |
| `pmu_housing_lid` | 142 × 89 × 17 | 55 g |

Built around the unit as **one 115 × 75 × 40mm block, battery included** —
that's how it's assembled and how it comes out. Same lid design as the Pi housing,
since that worked.

## Getting at it

Undo four M3 screws from below and the lid comes away with **the board and the
18650 still on it**. Nothing to unplug except the two cables, and the PV lead
comes out with the lid anyway.

The unit sits on four corner pads 4mm tall, with an L of lip at each corner
trapping it in plan — 10mm up its 40mm side. The 4mm gap underneath takes any
small protrusions, lets you dress cables under it, and keeps the board off a
lid that may carry condensation. No screw standoffs — I don't know the
mounting-hole pattern of the assembled unit. Tell me where the holes are and
I'll add them.

Put a strip of closed-cell foam between the top of the unit and the box
ceiling before closing it; there's 2mm of gap and that stops it rattling.

## Cables

- **USB-A out to the Pi** — 18 × 9mm slot in the front wall. The Pi housing is
  directly in front of it, so the run is short and the wall is doubly
  sheltered: it faces 30° down and is undercut, and the Pi box shields it.
- **PV panel in** — Ø12mm hole through the lid. The lid faces straight down
  when hung, which is the most sheltered face on the box, and it gives the
  cable a drip loop for free. Grommet it.

No battery hole (cell is internal) and no PWR button hole, as you asked.

## On the rail — this one seats differently

Both housings have grooves open at both ends. The PMU box seats, under
gravity, against the **Pi housing's aft face**. The Pi housing needs the
user's external axial stop to hold both boxes on the rail. The PMU has 78mm
of rail engagement.

Fitting it:

1. Easiest before joining the two cradle halves: slide it onto the rear half's
   rail from the rear half's front face and push it aft out of the way.
2. Join the cradle halves, then fit the Pi housing.
3. Let the power box slide forward until it butts the Pi housing.
4. Tighten the M3 grub screw in the keel (aft end, +Y side, outside the box).

You can also fit it with the cradle fully assembled by sliding it on from
behind the feeder — it just has to travel over the split seam at z=158.

**One thing this makes non-optional:** the spine cross-pin of the cradle's
split joint comes through the rail at z=167, and this box slides right over
that spot. That pin **must** finish flush or slightly recessed in the rail's
face, or the box will jam on it. It was a "should" before; it's a "must" now.

## Cooling

Same scheme as the Pi housing — 2mm intake slots in the lid at the front (the
low end when hung) and 2mm exhaust slots high on the side walls at the aft
(the high corner), under the roof overhang. A lithium cell in a sealed box in
Johannesburg sun is worth venting. **Print it in a light colour.**

## Printing

Roof down on the bed, no supports, same as the Pi housing. Lid flat, pads and
lips up. PETG or ASA.

## Verified in the model

- Both parts single solids, meshes watertight (0 open edges).
- **0.0 mm³** against the cradle, seated; **0.0 mm³** at seven stations along
  the slide-on path from behind.
- **0.0 mm³** against the Pi housing and its lid — they butt with a 0.5mm gap
  that closes under gravity.
- **0.0 mm³** between the unit keep-out and either part; 2mm over the unit to
  the keel.
- Both cable entries clear through their walls.
- Both parts inside 180mm.

## Assumptions to check

1. **Nothing protrudes more than 4mm** below the unit's bottom face. That's
   the gap the corner pads leave. If the 40-pin socket hangs lower, raise
   `PAD_H` — the box gets deeper by the same amount and nothing else moves.
2. **USB-A overmould** up to about 16 × 7mm passes the front slot. It's
   centred on the unit's height; if the socket sits high or low on the real
   unit, move `USBA_X`.
3. **Where the connectors sit** on the unit. Both holes are placed for cable
   routing, not aligned to specific sockets, so the leads need a little slack
   inside. That's deliberate given I don't have the board layout.
4. **Balance.** This adds roughly 250g further aft again. Check the hang and
   use the rearmost ceiling-wire station if it looks nose-up.

Board dimensions (56.5 × 65mm, Ø3.0 holes, USB input, PH2.0 battery
connector, 40-pin header) from the Waveshare wiki.

Sources: [Power Management HAT (B) — Waveshare Wiki](https://www.waveshare.com/wiki/Power_Management_HAT_(B))
