"""Renders of the two-part split: assembled, exploded, and a joint close-up."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh as stlmesh
from render_concept import hex2rgb, shade, tilt

PARTS = [("cradle_front", "#4a7fa5", "Front part  (tongues)"),
         ("cradle_rear", "#d1603d", "Rear part  (sockets)")]


def scene(png, title, elev, azim, explode=0.0, focus=None, zoom=1.0,
          legend=True):
    tris, cols = [], []
    for i, (nm, col, _l) in enumerate(PARTS):
        t = stlmesh.Mesh.from_file(nm + ".stl").vectors.copy()
        if i == 1:
            t[..., 2] += explode
        t = tilt(t, 0.0)
        tris.append(t)
        cols.append(shade(t, hex2rgb(col)))
    T = np.concatenate(tris)
    C = np.concatenate(cols)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(T, facecolors=C, linewidths=0)
    pc.set_sort_zpos(None)
    ax.add_collection3d(pc)
    pts = T.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ctr = (lo + hi) / 2
    span = (hi - lo).max() / 2 * 0.62 / zoom
    if focus is not None:
        ctr = np.array(focus, dtype=float)
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(ctr[2] - span, ctr[2] + span)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=15, pad=4)
    if legend:
        ax.legend(handles=[Patch(facecolor=c, label=l) for (_n, c, l) in PARTS],
                  loc="upper left", fontsize=11, frameon=False,
                  bbox_to_anchor=(-0.02, 0.98))
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print("wrote", png)


if __name__ == "__main__":
    scene("split_assembled.png", "Split cradle - ASSEMBLED (front / lip at left)",
          elev=18, azim=-128)
    scene("split_exploded.png", "Split cradle - EXPLODED 70mm along the axis",
          elev=18, azim=-128, explode=70.0)
    scene("split_joint.png",
          "Dovetail tongues - exploded 60mm, from aft and above",
          elev=26, azim=-38, explode=60.0, focus=(195.0, 0.0, -5.0), zoom=1.7,
          legend=False)
    scene("split_joint_end.png",
          "Dovetail tongues end-on - the undercut that keys the joint",
          elev=0, azim=0, explode=60.0, focus=(200.0, 0.0, -10.0), zoom=1.5,
          legend=False)
