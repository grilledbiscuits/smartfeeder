"""Shaded, colour-coded renders of the v6 concept for design review."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh as stlmesh

PARTS = [
    ("neck_rest_ring", "#d1603d", "Neck rest ring (conical)"),
    ("mid_body_ring", "#4a7fa5", "Mid body ring"),
    ("rear_body_ring", "#6699bb", "Rear body ring"),
    ("side_beams", "#8fa8b8", "Side support beams"),
    ("bottom_spine", "#5c6b73", "Bottom spine"),
    ("rpi_rail", "#c9a227", "RPi housing rail (dovetail)"),
    ("camera_arm", "#7a9e5c", "Camera arm + housing base"),
    ("perch_lugs", "#9b5fa8", "Perch mounting lugs (x3)"),
]


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def load(name):
    return stlmesh.Mesh.from_file(name + ".stl").vectors.copy()


def shade(tris, rgb, alpha=1.0, light=(0.45, -0.75, 0.55)):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln < 1e-12, 1.0, ln)
    l = np.array(light, dtype=float)
    l /= np.linalg.norm(l)
    d = np.abs(n @ l)
    f = 0.42 + 0.58 * d
    c = np.clip(rgb[None, :] * f[:, None], 0, 1)
    return np.concatenate([c, np.full((len(c), 1), alpha)], axis=1)


def tilt(v, deg):
    """Part frame -> hung frame (X horizontal along the tilt, Z up)."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    out = np.empty_like(v)
    out[..., 0] = c * v[..., 2] - s * v[..., 0]
    out[..., 1] = v[..., 1]
    out[..., 2] = s * v[..., 2] + c * v[..., 0]
    return out


def scene(png, title, elev, azim, with_bottle=True, hung=None, legend=True,
          focus=None, zoom=1.0):
    tris, cols = [], []
    for (nm, col, _lab) in PARTS:
        t = load(nm)
        if hung is not None:
            t = tilt(t, hung)
        tris.append(t)
        cols.append(shade(t, hex2rgb(col)))
    if with_bottle:
        b = load("bottle")
        if hung is not None:
            b = tilt(b, hung)
        tris.append(b)
        cols.append(shade(b, hex2rgb("#e8d5b0"), alpha=0.22))

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
    # Orthographic, so an elevation view reads true - in perspective anything
    # well off the view centre (the rail, down at x=-62) skews and looks
    # lopsided even when it is dead symmetric.
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=15, pad=4)
    if legend:
        handles = [Patch(facecolor=c, label=l) for (_n, c, l) in PARTS]
        handles.append(Patch(facecolor="#e8d5b0", alpha=0.5, label="Bottle (reference)"))
        ax.legend(handles=handles, loc="upper left", fontsize=10,
                  frameon=False, bbox_to_anchor=(-0.02, 0.98))
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print("wrote", png)


if __name__ == "__main__":
    # Side view, drawn in the part frame to match the sketch (bottle axis
    # horizontal). Lip/front at the left.
    # Part frame, bottle axis horizontal, lip at the left - matching the sketch.
    scene("v6_side.png", "v6 concept - SIDE VIEW (front / lip at left)",
          elev=0, azim=-90, hung=0.0)
    scene("v6_iso.png", "v6 concept - ISOMETRIC (camera side, from above)",
          elev=24, azim=-133, hung=0.0, with_bottle=False)
    scene("v6_iso_far.png", "v6 concept - ISOMETRIC (opposite side, from above)",
          elev=26, azim=-46, hung=0.0, with_bottle=False)
    # Detail: the rounded perch lug on the +Y side beam.
    scene("v6_lug_detail.png",
          "v6 concept - PERCH LUG DETAIL (top view, orthographic)",
          elev=89, azim=-90, hung=0.0, with_bottle=False, legend=False,
          focus=(57.0, 45.0, 0.0), zoom=5.0)
    scene("v6_top.png", "v6 concept - TOP VIEW (front / lip at left)",
          elev=89, azim=-90, hung=0.0)
    # Detail: the hockey-stick platform and its rectangular camera end.
    scene("v6_platform_detail.png",
          "v6 concept - CAMERA PLATFORM (top view, orthographic)",
          elev=89, azim=-90, hung=0.0, with_bottle=False, legend=False,
          # plot coords are (z, y, x) - centre on the arm's midspan
          focus=(48.0, 52.0, -44.5), zoom=1.55)
    scene("v6_front.png", "v6 concept - FRONT VIEW (looking aft along the axis)",
          elev=0, azim=0, hung=0.0)
    # Detail: the dovetail and the spine perch lug, straight down the axis.
    scene("v6_rail_detail.png",
          "v6 concept - DOVETAIL DETAIL (front view, orthographic)",
          elev=0, azim=0, hung=0.0, with_bottle=False, legend=False,
          # plot coords with hung=0 are (z, y, x): mid-rail, centreline, rail
          focus=(150.0, 0.0, -58.0), zoom=4.0)
