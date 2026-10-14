# Solar panel support frame — six flat pieces, one print

Dimensions are millimeters. XY is horizontal and +Z is upward.

| Feature | Dimension |
|---|---|
| Panel | 340 × 232 × 17 |
| Inner fit | 342 × 234; 1 clearance per side |
| Outer rectangular rim | 354 × 246 |
| Inner support ledge | 13 wide × 5 thick |
| Open center | 316 × 208; no floor |
| Side walls | 6 thick; 17 above the ledge |
| Corner tabs | Ø24 × 12 thick |
| Wire holes | Four Ø5 through holes; 0.8 bevel at each mouth |
| Full envelope with tabs | 386 × 278 × 22 |

The ledge is 5 mm wider than the initial draft. The panel drops in from above
and rests on all four sides. Its top is flush with the rim. Corner wire holes
lie outside the panel fit and do not obstruct placement or removal.

## Print

Use **`solar_panel_frame_plate_1.stl`** for one print of all six pieces.
It is already packed for the Ender-3 V3 SE's 220 × 220 mm bed, with nominal
3 mm gaps and more than 3 mm clearance to the bed edges. All parts have their
flat underside at Z=0 and are 22 mm tall. Enable neither supports nor a brim;
a brim would occupy the deliberately small gaps. Keep the packed arrangement
when importing; use ordinary layer-by-layer printing, not sequential objects.

The six individual STEP/STL pairs are retained for inspecting or replacing a
single part. Four corner pieces measure approximately 136.3 × 139 × 22 mm;
the two center rails measure 113.3 × 19 × 22 mm. The long dimension is cut at
X = ±56.667 mm, forming three lengthwise sections. Each end is also cut at
Y=0, producing four corners plus the two separate center rails.

## Glue assembly

Arrange the lower row as left_lower, center_lower, right_lower, and the upper
row as left_upper, center_upper, right_upper. Dry-fit around the panel to check
alignment. Glue the six plain seam faces with the user's chosen superglue;
keep the panel removable and the support ledge flush across every seam.
There are no printed joining pins or dovetails. The exploded STEP and Desktop
render are assembly references, not layouts for printing.

The full-frame STEP is the continuous design reference; it is not a printable
one-piece model on the Ender. Panel weight, glue-joint strength, wire strength
and outdoor material performance have not been physically tested.

## Regenerate

From the repository root:

```bash
uvx --no-config --managed-python --python 3.13 --from cadgen==0.7.21 python cad/solar_panel_frame/build.py
uvx --no-config --managed-python --python 3.13 --from cadgen==0.7.21 python cad/solar_panel_frame/checks.py
```

Saved-STEP checks verify the panel clearance, 13 mm ledge, open center,
corner holes, six solid pieces, conservation of geometry across all cuts,
restored assembly positions and the packed plate's exact clearances.
