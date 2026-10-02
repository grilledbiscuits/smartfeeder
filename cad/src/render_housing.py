"""Renders of the RPi 4B housing, alone and hung on the cradle's front half."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh as stlmesh
from render_concept import hex2rgb, shade, tilt


def scene(png, title, items, elev, azim, hung=0.0, focus=None, zoom=1.0,
          legend=True, section_z=None):
    tris, cols, handles = [], [], []
    for (nm, col, lab, alpha, offset) in items:
        t = stlmesh.Mesh.from_file(nm + ".stl").vectors.copy()
        t = t + np.array(offset, dtype=float)
        if section_z is not None:                      # drop everything aft of the cut
            t = t[t[..., 2].max(axis=1) <= section_z + 0.01]
        t = tilt(t, hung)
        tris.append(t)
        cols.append(shade(t, hex2rgb(col), alpha=alpha))
        if lab:
            handles.append(Patch(facecolor=col, alpha=max(alpha, 0.5), label=lab))
    T = np.concatenate(tris); C = np.concatenate(cols)
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(T, facecolors=C, linewidths=0)
    pc.set_sort_zpos(None)
    ax.add_collection3d(pc)
    pts = T.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ctr = (lo + hi) / 2 if focus is None else np.array(focus, dtype=float)
    span = (hi - lo).max() / 2 * 0.62 / zoom
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(ctr[2] - span, ctr[2] + span)
    ax.set_box_aspect((1, 1, 1)); ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    ax.set_title(title, fontsize=15, pad=4)
    if legend and handles:
        ax.legend(handles=handles, loc="upper left", fontsize=11, frameon=False,
                  bbox_to_anchor=(-0.02, 0.98))
    fig.tight_layout(); fig.savefig(png, dpi=150); plt.close(fig)
    print("wrote", png)


H = ("rpi_housing", "#4a7fa5", "Housing body", 1.0, (0, 0, 0))
LID = ("rpi_housing_lid", "#d1603d", "Lid", 1.0, (0, 0, 0))
PI = ("rpi4b_keepout", "#3d9a4a", "Pi 4B (keep-out model)", 1.0, (0, 0, 0))
CR = ("cradle_front", "#9aa7b0", "Cradle, front half", 1.0, (0, 0, 0))
BOT = ("bottle", "#e8d5b0", "Bottle", 0.18, (0, 0, 0))

if __name__ == "__main__":
    # front view, straight on: USB-C and ToF entries
    # front wall only (cut behind its inner face) so the holes read as holes -
    # through a whole box you just see the aft wall's inside, same colour
    scene("housing_front.png", "RPi housing - FRONT WALL (USB-C and ToF)",
          [H, LID], elev=0, azim=180, section_z=74.2,
          focus=(70.0, 0.0, -80.0), zoom=1.2)
    scene("housing_csi_side.png", "RPi housing - microSD side (CSI ribbon exit)",
          [H, LID], elev=16, azim=75)
    # from below-front with the lid dropped and the Pi visible inside
    scene("housing_exploded.png", "RPi housing - EXPLODED (lid dropped 35mm)",
          [H, ("rpi_housing_lid", "#d1603d", "Lid", 1.0, (-35, 0, 0)), PI],
          elev=-28, azim=-40)
    # from above-aft: roof, keel groove, drip lips
    scene("housing_roof.png", "RPi housing - ROOF and dovetail groove (from above, aft)",
          [H, LID], elev=38, azim=140)
    # looking straight down the rail, section at mid-housing
    scene("housing_section.png", "Section at z=105 - rail seated in the groove",
          [H, LID, PI, ("cradle_front", "#9aa7b0", "Cradle", 1.0, (0, 0, 0))],
          elev=0, azim=0, section_z=105.0, focus=(100.0, 0.0, -75.0), zoom=1.45)
    # hung on the cradle, as installed (30 deg nose-down, world frame)
    scene("housing_on_cradle.png", "Housing on the cradle - AS HUNG (30 deg, lip low)",
          [CR, H, LID, BOT], elev=4, azim=-90, hung=30.0)
    scene("housing_on_cradle_iso.png", "Housing on the cradle - ISOMETRIC (part frame)",
          [CR, H, LID], elev=-22, azim=-128)
    pmu_views()


PMU = ("pmu_housing", "#7a9e5c", "PMU housing", 1.0, (0, 0, 0))
PMUL = ("pmu_housing_lid", "#c9a227", "PMU lid", 1.0, (0, 0, 0))
PMUK = ("pmu_keepout", "#9b5fa8", "Waveshare unit + 18650", 1.0, (0, 0, 0))


def pmu_views():
    scene("pmu_exploded.png", "PMU housing - EXPLODED (lid + unit dropped 45mm)",
          [PMU, ("pmu_housing_lid", "#c9a227", "PMU lid", 1.0, (-45, 0, 0)),
           ("pmu_keepout", "#9b5fa8", "Waveshare unit + 18650", 1.0, (-45, 0, 0))],
          elev=-26, azim=-42)
    scene("pmu_front.png", "PMU housing - FRONT WALL (USB-A out to the Pi)",
          [PMU, PMUL], elev=0, azim=180, section_z=150.4,
          focus=(150.0, 0.0, -85.0), zoom=1.2)
    scene("pmu_roof.png", "PMU housing - ROOF and groove (from above, aft)",
          [PMU, PMUL], elev=38, azim=140)
    scene("train_hung.png",
          "Cradle, Pi housing and PMU box - AS HUNG (30 deg, lip low)",
          [CR, H, LID, PMU, PMUL, BOT], elev=4, azim=-90, hung=30.0)
    scene("train_iso.png", "The whole train - ISOMETRIC",
          [CR, H, LID, PMU, PMUL], elev=-20, azim=-126)
