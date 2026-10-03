# PMU box for the v6 cradle

Two printed parts sit on the dovetail rail immediately behind the Pi housing.
The modeled PMU envelope is **110 × 70 × 25 mm**, battery included.

| Part | Size (mm) | Estimated PLA mass |
|---|---|---|
| `pmu_housing` | 43 × 140 × 92 | 109 g |
| `pmu_housing_lid` | 17 × 137 × 84 | 50 g |

## Access and fit

Four M3 lid screws release the lid and its four 4 mm corner pads. The pads
hold the unit 4 mm above the lid; 10 mm lips locate its corners. The unit has
2 mm clearance above it. The mounting-hole pattern is unknown, so the PMU is
not screwed to the lid. Check the physical assembly for protrusions below its
measured 25 mm envelope before printing.

## IO end wall

All openings are on the +Y wall, along the PMU's 70 mm short end. Viewed from
inside the housing, the left-to-right order is power switch, USB-A, USB-C,
and SOLAR-IN; the outside view reverses that order. Three LED apertures sit above
the USB-C and solar ports. The BOOT and rightmost LED openings were removed
after a test print. The SOLAR-IN opening is Ø8 mm, with its center 1 mm left
and 2 mm up from the first test print. The round opening and LEDs have upper
half-ring awnings; the rectangular ports have straight staple awnings. The
main awnings project 4 mm and the LED awnings project 2.5 mm.

The photo establishes port order, but it has no scale or measured hole centers.
Port positions and opening sizes are estimates from the photo. Compare a print
or dimensioned drawing with the actual PMU before using this as a final fit.
The awning-free port test wall is cropped to a 4 mm border around the seven
remaining openings: 59.6 × 22.0 × 2.5 mm, flat on the print bed.

## Rail and printing

The groove is open at both ends. The PMU box slides on from behind and rests
against the Pi housing with a 0.5 mm modeled gap; the user's external stop
retains the Pi housing. The PMU groove engages 78 mm of rail. Tighten the M3
grub screw in the keel to stop the PMU box sliding backward. The cradle's
cross-pin at z=167 mm must finish flush or recessed so the PMU can slide over
it.
The PMU groove has 0.30 mm extra clearance at each flank and the floor beyond
the Pi housing's groove. The remaining floor thickness is 2.45 mm.

Print the housing roof down and the lid flat with pads up, in white PLA.
The lid has intake slots; the shell has high side vents under the roof brim.
The bed-oriented exports are `stl/pmu_housing_roof_down.stl` and
`stl/pmu_housing_lid_flat.stl`; [preview](pmu_full_print_preview.png).

## Model checks

`pmu_housing.py` reports `ALL OK`: each part is one solid and fits within a
180 mm build volume; the unit, lid, Pi housing, and cradle have zero modeled
interference; the seven checked slide-on stations are clear; and all seven
port passages have zero obstruction.
