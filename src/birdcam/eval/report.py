"""Self-contained HTML metrics report, plus a terminal summary.

    uv run python -m birdcam.eval.report

Everything is embedded -- charts as base64 PNGs, error crops inline -- so the
file can be opened from anywhere or emailed without breaking.

The report leads with a plain-language summary naming the three things most
worth fixing next, because a wall of per-class tables does not answer "what do I
do on Monday".
"""

from __future__ import annotations

import base64
import datetime as _dt
import io
import json
import logging
from dataclasses import dataclass

import numpy as np

from birdcam.config import Config, load_config
from birdcam.eval.metrics import (
    error_flow,
    expected_calibration_error,
    per_class_metrics,
    sex_breakdown,
    wilson_interval,
)

logger = logging.getLogger(__name__)


@dataclass
class Evaluated:
    """Everything the renderer needs, computed once."""

    cfg: Config
    labels: list[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    probs: np.ndarray
    species: np.ndarray
    sex: np.ndarray
    image_paths: list
    backbone: str
    temperature: float
    n_train: int


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img_to_b64(path, max_px: int = 120) -> str | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_px, max_px))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001 - a missing crop must not kill the report
        return None


def evaluate(cfg: Config, feature_file: str, epochs: int = 200) -> Evaluated:
    import torch
    import torch.nn as nn

    from birdcam.data.dataset import load_labelled
    from birdcam.data.manifest import open_manifest

    X = np.load(cfg.path("embeddings_dir") / "sweep" / feature_file)
    with open_manifest(cfg.path("manifest_db")) as m:
        items = load_labelled(cfg, m)
    if len(X) != len(items):
        raise RuntimeError(
            f"feature/label misalignment: {len(X)} features vs {len(items)} items. "
            "Re-extract features after changing the corpus."
        )

    split = np.array([i.split for i in items])
    y = np.array([i.taxon_index for i in items])
    tr, va, te = split == "train", split == "val", split == "test"

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr, Xva, Xte = ((X[m_] - mu) / sd for m_ in (tr, va, te))

    head = nn.Linear(X.shape[1], len(cfg.taxon_classes))
    opt = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(y[tr])
    for _ in range(epochs):
        opt.zero_grad()
        ce(head(xt), yt).backward()
        opt.step()
    head.eval()

    with torch.no_grad():
        lg_va = head(torch.tensor(Xva, dtype=torch.float32))
        lg_te = head(torch.tensor(Xte, dtype=torch.float32))

    # Temperature fitted on VAL, never test.
    T = torch.ones(1, requires_grad=True)
    o = torch.optim.LBFGS([T], lr=0.1, max_iter=60)
    yva = torch.tensor(y[va])

    def closure():
        o.zero_grad()
        loss = ce(lg_va / T.clamp(min=1e-2), yva)
        loss.backward()
        return loss

    o.step(closure)
    temp = float(T.detach().clamp(min=1e-2))

    probs = torch.softmax(lg_te / temp, dim=1).numpy()
    return Evaluated(
        cfg=cfg,
        labels=cfg.taxon_classes,
        y_true=y[te],
        y_pred=probs.argmax(1),
        probs=probs,
        species=np.array([i.scientific_name for i in items])[te],
        sex=np.array([i.sex_label_name for i in items])[te],
        image_paths=[i.path for i in items if i.split == "test"],
        backbone=feature_file.rsplit("_", 1)[0],
        temperature=temp,
        n_train=int(tr.sum()),
    )


# --- charts -------------------------------------------------------------------


def chart_confusion(ev: Evaluated, sex_filter: str | None = None) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = np.ones(len(ev.y_true), bool) if sex_filter is None else (ev.sex == sex_filter)
    if sel.sum() == 0:
        return ""
    present = sorted(set(ev.y_true[sel].tolist()) | set(ev.y_pred[sel].tolist()))
    names = [ev.labels[i].replace("_", " ") for i in present]
    pos = {c: i for i, c in enumerate(present)}
    m = np.zeros((len(present), len(present)))
    for t, p in zip(ev.y_true[sel], ev.y_pred[sel], strict=True):
        m[pos[t], pos[p]] += 1
    row = m.sum(1, keepdims=True)
    norm = m / np.maximum(row, 1)

    fig, ax = plt.subplots(figsize=(max(6, len(present) * 0.42), max(5, len(present) * 0.38)))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    title = "confusion (row-normalised)"
    if sex_filter:
        title += f" -- {sex_filter} only, n={int(sel.sum())}"
    ax.set_title(title, fontsize=9)
    if len(present) <= 26:
        for i in range(len(present)):
            for j in range(len(present)):
                if m[i, j]:
                    ax.text(
                        j,
                        i,
                        int(m[i, j]),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black",
                    )
    return _fig_to_b64(fig)


def chart_reliability(ev: Evaluated) -> tuple[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ece, bins = expected_calibration_error(ev.probs, ev.y_true)
    xs = [(b["lo"] + b["hi"]) / 2 for b in bins if b["n"]]
    accs = [b["acc"] for b in bins if b["n"]]
    confs = [b["conf"] for b in bins if b["n"]]
    ns = [b["n"] for b in bins if b["n"]]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(5.2, 5.4), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(confs, accs, "o-", color="#2a6f97", label="model")
    ax.fill_between(confs, accs, confs, alpha=0.2, color="#d62828")
    ax.set_ylabel("accuracy")
    ax.set_title(f"reliability -- ECE {ece:.3f} (after temperature scaling)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax2.bar(xs, ns, width=1 / len(bins) * 0.9, color="#8d99ae")
    ax2.set_xlabel("confidence")
    ax2.set_ylabel("n")
    return _fig_to_b64(fig), ece


def chart_pr_tier_a(ev: Evaluated) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    thr_default = ev.cfg.taxonomy_cfg["rollup"]["thresholds"]["species"]
    for s in ev.cfg.species_by_tier("A"):
        idx = ev.cfg.taxon_class_index.get(s.slug)
        if idx is None:
            continue
        pos = ev.y_true == idx
        if pos.sum() == 0:
            continue
        score = ev.probs[:, idx]
        order = np.argsort(score)[::-1]
        tp = np.cumsum(pos[order])
        prec = tp / np.arange(1, len(order) + 1)
        rec = tp / pos.sum()
        (line,) = ax.plot(rec, prec, lw=1.4, label=s.common_name)
        # Mark the operating threshold.
        k = int((score[order] >= thr_default).sum())
        if 0 < k <= len(rec):
            ax.plot(rec[k - 1], prec[k - 1], "o", color=line.get_color(), ms=5)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Tier A precision-recall (dot = threshold {thr_default})", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


# --- rollup top-k -------------------------------------------------------------


def rollup_accuracy(ev: Evaluated) -> dict[str, float]:
    """Accuracy when correctness is judged at species / genus / family level."""
    cfg = ev.cfg
    genus_of = {s.slug: s.genus for s in cfg.species}
    fam_of = {g: f for g, f in cfg.genus_to_family.items()}

    def to_genus(i):
        return genus_of.get(cfg.taxon_classes[i])

    def to_family(i):
        g = to_genus(i)
        return fam_of.get(g) if g else None

    out = {"species": float((ev.y_pred == ev.y_true).mean())}
    for name, fn in (("genus", to_genus), ("family", to_family)):
        t = [fn(i) for i in ev.y_true]
        p = [fn(i) for i in ev.y_pred]
        ok = [a is not None and a == b for a, b in zip(t, p, strict=True)]
        out[name] = float(np.mean(ok))
    # Top-3 at species level.
    top3 = np.argsort(ev.probs, axis=1)[:, -3:]
    out["species_top3"] = float(np.mean([y in row for y, row in zip(ev.y_true, top3, strict=True)]))
    return out


def tier_cross_contamination(ev: Evaluated) -> dict:
    cfg = ev.cfg
    tier_a = {s.slug for s in cfg.species_by_tier("A")}
    tier_of = {s.scientific_name: s.tier for s in cfg.species}
    is_a_true = np.array([tier_of.get(n) == "A" for n in ev.species])
    is_a_pred = np.array([cfg.taxon_classes[p] in tier_a for p in ev.y_pred])

    fp, n_c = int((~is_a_true & is_a_pred).sum()), int((~is_a_true).sum())
    fn, n_a = int((is_a_true & ~is_a_pred).sum()), int((is_a_true).sum())
    worst = []
    for n in sorted(set(ev.species[~is_a_true].tolist())):
        sel = ev.species == n
        if sel.sum():
            worst.append((n, float(is_a_pred[sel].mean()), int(sel.sum())))
    worst.sort(key=lambda t: -t[1])
    return {
        "c_called_nectarivore": (fp, n_c, fp / max(n_c, 1), wilson_interval(fp, n_c)),
        "a_called_non_target": (fn, n_a, fn / max(n_a, 1), wilson_interval(fn, n_a)),
        "worst": worst[:6],
    }


def high_confidence_errors(ev: Evaluated, k: int = 12) -> list[dict]:
    """Most confident mistakes -- usually mislabelled source images."""
    wrong = np.where(ev.y_pred != ev.y_true)[0]
    if len(wrong) == 0:
        return []
    conf = ev.probs[wrong, ev.y_pred[wrong]]
    order = wrong[np.argsort(conf)[::-1]][:k]
    out = []
    for i in order:
        out.append(
            {
                "true": ev.labels[ev.y_true[i]],
                "pred": ev.labels[ev.y_pred[i]],
                "conf": float(ev.probs[i, ev.y_pred[i]]),
                "sex": str(ev.sex[i]),
                "b64": _img_to_b64(ev.image_paths[i]) if i < len(ev.image_paths) else None,
            }
        )
    return out


def recommendations(ev: Evaluated, rows) -> list[tuple[str, str, str]]:
    """Rank the worst classes and say what to DO about each."""
    cfg = ev.cfg
    out = []
    for r in sorted([x for x in rows if x.support > 0], key=lambda x: x.f1)[:8]:
        flows = error_flow(ev.y_true, ev.y_pred, ev.labels, cfg.taxon_class_index[r.label])
        top = flows[0] if flows else None
        if not r.verdict_reliable:
            action = f"MORE DATA -- only {r.support} test images, no verdict possible"
        elif top and top[1] >= 0.4 * (r.support - r.tp):
            action = (
                f"MERGE with {top[0].replace('_', ' ')} -- "
                f"{top[1]} of {r.support - r.tp} errors go there"
            )
        elif r.recall < 0.5:
            action = "MORE DATA -- errors scatter, no single confusable sibling"
        else:
            action = "acceptable; revisit after fine-tuning"
        detail = ", ".join(f"{a.replace('_', ' ')}={b}" for a, b in flows) or "no errors"
        out.append((r.label, action, detail))
    return out


# --- rendering ----------------------------------------------------------------

_CSS = """
body{font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
 margin:0;background:#f6f7f9;color:#1d2129}
.wrap{max-width:1100px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:25px;margin:0 0 4px} h2{font-size:18px;margin:34px 0 10px;
 padding-bottom:6px;border-bottom:2px solid #e3e6ea}
h3{font-size:14px;margin:20px 0 8px;color:#42506b}
.sub{color:#6b7684;font-size:13px;margin-bottom:20px}
.card{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:16px 18px;margin:14px 0}
.tldr{background:#fff8e6;border:1px solid #f0d68a}
.tldr ol{margin:8px 0 0;padding-left:20px} .tldr li{margin:7px 0}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:6px 9px;text-align:left;border-bottom:1px solid #eceff3}
th{background:#f2f4f7;font-weight:600;position:sticky;top:0}
td.n{text-align:right;font-variant-numeric:tabular-nums}
tr:hover{background:#fafbfc}
.bad{color:#b3261e;font-weight:600} .ok{color:#1b7f3b} .warn{color:#a56a00}
.pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;
 background:#eef1f5;color:#42506b}
img.chart{max-width:100%;height:auto;display:block;margin:6px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}
.err{border:1px solid #e3e6ea;border-radius:6px;padding:6px;background:#fff;font-size:11px}
.err img{width:100%;border-radius:4px;display:block;margin-bottom:4px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:13px}
.kv div:nth-child(odd){color:#6b7684}
code{background:#eef1f5;padding:1px 5px;border-radius:3px;font-size:12px}
.note{font-size:12.5px;color:#6b7684;margin-top:8px}
"""


def render_html(ev: Evaluated, ece: float, charts: dict, open_set: dict | None) -> str:
    cfg = ev.cfg
    rows = per_class_metrics(ev.y_true, ev.y_pred, ev.labels)
    present = [r for r in rows if r.support > 0]
    tier_a_slugs = {s.scientific_name: s.slug for s in cfg.species_by_tier("A")}
    roll = rollup_accuracy(ev)
    cross = tier_cross_contamination(ev)
    recs = recommendations(ev, present)
    overall = float((ev.y_pred == ev.y_true).mean())
    lo, hi = wilson_interval(int((ev.y_pred == ev.y_true).sum()), len(ev.y_true))

    def f(x, d=3):
        return f"{x:.{d}f}"

    # --- the three things most worth fixing -----------------------------------
    tldr = []
    worst_a = sorted(
        [r for r in present if r.label in tier_a_slugs.values()], key=lambda r: r.recall
    )
    if worst_a:
        w = worst_a[0]
        flows = error_flow(ev.y_true, ev.y_pred, ev.labels, cfg.taxon_class_index[w.label])
        where = flows[0][0].replace("_", " ") if flows else "several classes"
        tldr.append(
            f"<b>{w.label.replace('_', ' ')} is the weakest Tier A class</b> "
            f"(recall {f(w.recall)}, n={w.support}). Most errors go to <i>{where}</i>. "
            f"{recs[0][1] if recs else ''}"
        )
    if cross["worst"]:
        n, rate, cnt = cross["worst"][0]
        tldr.append(
            f"<b>{n} is mistaken for a nectarivore {rate * 100:.0f}% of the time</b> "
            f"(n={cnt}) -- the largest false-positive source. More data for this "
            "species, or accept it will trigger captures."
        )
    fem = [r for r in sex_breakdown(ev.y_true, ev.y_pred, ev.species, ev.sex)]
    gaps = []
    for b in fem:
        m_, f_ = b.recall("male_unspecified"), b.recall("female")
        if m_ is not None and f_ is not None:
            gaps.append((m_ - f_, b.species, m_, f_))
    gaps.sort(reverse=True)
    if gaps:
        g, sp, m_, f_ = gaps[0]
        tldr.append(
            f"<b>Females remain the hard case</b>: {sp} scores {f(m_)} on males "
            f"but {f(f_)} on females (gap {f(g)}). Sex-annotated females are the "
            "scarce resource -- hand-labelling them is the highest-value effort."
        )

    h = [f"<style>{_CSS}</style><div class='wrap'>"]
    h.append("<h1>birdcam metrics report</h1>")
    h.append(
        f"<div class='sub'>{_dt.datetime.now():%Y-%m-%d %H:%M} &middot; "
        f"backbone <code>{ev.backbone}</code> &middot; frozen features + linear head "
        f"&middot; {ev.n_train} train / {len(ev.y_true)} test images</div>"
    )

    h.append("<div class='card tldr'><b>Three things most worth fixing next</b><ol>")
    for t in tldr:
        h.append(f"<li>{t}</li>")
    h.append("</ol></div>")

    # --- headline -------------------------------------------------------------
    h.append("<h2>Headline</h2><div class='card'><div class='kv'>")
    h.append(
        f"<div>taxon accuracy</div><div><b>{f(overall)}</b> [95% CI {f(lo)}&ndash;{f(hi)}]</div>"
    )
    h.append(f"<div>species top-3</div><div>{f(roll['species_top3'])}</div>")
    h.append(f"<div>genus-level</div><div>{f(roll['genus'])}</div>")
    h.append(f"<div>family-level</div><div>{f(roll['family'])}</div>")
    h.append(
        f"<div>calibration (ECE)</div><div>{f(ece)} after temperature "
        f"T={f(ev.temperature, 2)}</div>"
    )
    h.append(
        "</div><div class='note'>Rollup levels show what the fallback hierarchy buys: "
        "if genus-level accuracy is far above species-level, emitting a genus label when "
        "uncertain is worth doing.</div></div>"
    )

    # --- open set -------------------------------------------------------------
    if open_set:
        h.append("<h2>Open-set failsafe</h2><div class='card'>")
        h.append(
            f"<div class='note'>{open_set['n_ood']} real out-of-distribution photographs "
            "(squirrels, mice, cats, rats, baboons, bees, butterflies, agamas, skinks). "
            "No detector ever saw one during fitting.</div>"
        )
        h.append(
            "<table><tr><th>scorer</th><th class='n'>AUROC</th>"
            "<th class='n'>caught @1% FPR</th><th class='n'>@5%</th><th class='n'>@10%</th></tr>"
        )
        for r in sorted(open_set["scorers"], key=lambda x: -x["auroc"]):
            t = r["tpr_at_fpr"]
            cls = "ok" if r["auroc"] > 0.95 else ("warn" if r["auroc"] > 0.85 else "bad")
            h.append(
                f"<tr><td>{r['scorer']}</td><td class='n {cls}'>{r['auroc']:.3f}</td>"
                f"<td class='n'>{t['0.01']:.3f}</td><td class='n'><b>{t['0.05']:.3f}</b></td>"
                f"<td class='n'>{t['0.10']:.3f}</td></tr>"
            )
        h.append("</table>")
        best = max(open_set["scorers"], key=lambda x: x["auroc"])
        h.append(
            f"<h3>Per-group recall at 5% false-alarm rate ({best['scorer']})</h3><table>"
            "<tr><th>group</th><th class='n'>n</th><th class='n'>caught</th></tr>"
        )
        for g, v in sorted(best["per_group"].items()):
            h.append(
                f"<tr><td>{g}</td><td class='n'>{v['n']}</td>"
                f"<td class='n'>{v['caught_at_5pct_fpr']:.3f}</td></tr>"
            )
        h.append("</table></div>")

    # --- Tier A vs C ----------------------------------------------------------
    h.append("<h2>Tier A vs Tier C</h2><div class='card'>")
    fp, n_c, rate_c, ci_c = cross["c_called_nectarivore"]
    fn, n_a, rate_a, ci_a = cross["a_called_non_target"]
    h.append("<div class='kv'>")
    h.append(
        f"<div>non-target called nectarivore</div><div>{fp}/{n_c} = "
        f"<b>{f(rate_c)}</b> [{f(ci_c[0])}&ndash;{f(ci_c[1])}]</div>"
    )
    h.append(
        f"<div>nectarivore called non-target</div><div>{fn}/{n_a} = "
        f"<b>{f(rate_a)}</b> [{f(ci_a[0])}&ndash;{f(ci_a[1])}]</div>"
    )
    h.append(
        "</div><h3>Worst offenders</h3><table>"
        "<tr><th>Tier C species</th><th class='n'>n</th>"
        "<th class='n'>called nectarivore</th></tr>"
    )
    for n, r, c in cross["worst"]:
        cls = "bad" if r > 0.25 else ("warn" if r > 0.1 else "")
        h.append(
            f"<tr><td><i>{n}</i></td><td class='n'>{c}</td><td class='n {cls}'>{r:.3f}</td></tr>"
        )
    h.append("</table></div>")

    # --- per class ------------------------------------------------------------
    h.append(
        "<h2>Per-class metrics</h2><div class='card'><table>"
        "<tr><th>class</th><th>tier</th><th class='n'>n</th><th class='n'>prec</th>"
        "<th class='n'>recall</th><th class='n'>F1</th><th class='n'>95% CI</th></tr>"
    )
    tier_of = {s.slug: s.tier for s in cfg.species}
    for r in sorted(present, key=lambda x: x.f1):
        cls = "bad" if r.f1 < 0.5 else ("warn" if r.f1 < 0.7 else "ok")
        flag = "" if r.verdict_reliable else " <span class='pill'>low n</span>"
        h.append(
            f"<tr><td>{r.label.replace('_', ' ')}{flag}</td>"
            f"<td>{tier_of.get(r.label, '-')}</td><td class='n'>{r.support}</td>"
            f"<td class='n'>{r.precision:.3f}</td><td class='n'>{r.recall:.3f}</td>"
            f"<td class='n {cls}'>{r.f1:.3f}</td>"
            f"<td class='n'>{r.recall_ci[0]:.2f}&ndash;{r.recall_ci[1]:.2f}</td></tr>"
        )
    h.append("</table></div>")

    # --- sex breakdown --------------------------------------------------------
    h.append(
        "<h2>Accuracy by sex &mdash; Tier A</h2><div class='card'>"
        "<div class='note'>An aggregate confusion matrix hides this. Abundant, "
        "easily-identified males swamp the scarce females.</div><table>"
        "<tr><th>species</th><th class='n'>male</th><th class='n'>female</th>"
        "<th class='n'>gap</th><th class='n'>female 95% CI</th></tr>"
    )
    for b in sex_breakdown(ev.y_true, ev.y_pred, ev.species, ev.sex):
        if b.species not in tier_a_slugs:
            continue
        m_, f_ = b.recall("male_unspecified"), b.recall("female")
        if m_ is None or f_ is None:
            continue
        mn = b.by_sex["male_unspecified"][1]
        fn_ = b.by_sex["female"][1]
        ci = b.ci("female")
        gap = m_ - f_
        cls = "bad" if gap > 0.25 else ("warn" if gap > 0.12 else "")
        h.append(
            f"<tr><td><i>{b.species}</i></td>"
            f"<td class='n'>{m_:.3f} <span class='pill'>n={mn}</span></td>"
            f"<td class='n'>{f_:.3f} <span class='pill'>n={fn_}</span></td>"
            f"<td class='n {cls}'>{gap:+.3f}</td>"
            f"<td class='n'>{ci[0]:.2f}&ndash;{ci[1]:.2f}</td></tr>"
        )
    h.append("</table></div>")

    # --- charts ---------------------------------------------------------------
    h.append("<h2>Confusion matrices</h2>")
    for key, title in (
        ("confusion", "All test images"),
        ("confusion_female", "Females only"),
        ("confusion_male", "Males only"),
    ):
        if charts.get(key):
            h.append(
                f"<div class='card'><h3>{title}</h3>"
                f"<img class='chart' src='data:image/png;base64,{charts[key]}'></div>"
            )

    h.append(
        "<h2>Calibration</h2><div class='card'>"
        f"<img class='chart' src='data:image/png;base64,{charts['reliability']}'>"
        "<div class='note'>Confidence gates the video-capture decision downstream, so a "
        "miscalibrated model is worse than a slightly less accurate one. Red area is the "
        "gap between confidence and accuracy.</div></div>"
    )

    h.append(
        "<h2>Precision-recall, Tier A</h2><div class='card'>"
        f"<img class='chart' src='data:image/png;base64,{charts['pr']}'>"
        "<div class='note'>Dots mark the configured operating threshold.</div></div>"
    )

    # --- recommendations ------------------------------------------------------
    h.append(
        "<h2>Worst classes &mdash; and what to do</h2><div class='card'><table>"
        "<tr><th>class</th><th>recommendation</th><th>errors go to</th></tr>"
    )
    for label, action, detail in recs:
        cls = "bad" if action.startswith(("MORE", "MERGE")) else ""
        h.append(
            f"<tr><td>{label.replace('_', ' ')}</td>"
            f"<td class='{cls}'>{action}</td><td>{detail}</td></tr>"
        )
    h.append("</table></div>")

    # --- error gallery --------------------------------------------------------
    errs = high_confidence_errors(ev)
    if errs:
        h.append(
            "<h2>Highest-confidence errors</h2><div class='card'>"
            "<div class='note'>These are worth opening individually: a confidently wrong "
            "prediction is very often a mislabelled source image, and finding them is worth "
            "real accuracy.</div><div class='grid'>"
        )
        for e in errs:
            img = f"<img src='data:image/jpeg;base64,{e['b64']}'>" if e["b64"] else ""
            h.append(
                f"<div class='err'>{img}<b>{e['conf']:.2f}</b> "
                f"<span class='pill'>{e['sex']}</span><br>"
                f"pred <span class='bad'>{e['pred'].replace('_', ' ')}</span><br>"
                f"true {e['true'].replace('_', ' ')}</div>"
            )
        h.append("</div></div>")

    h.append("</div>")
    return "\n".join(h)


def print_summary(ev: Evaluated, ece: float, open_set: dict | None) -> None:
    rows = [r for r in per_class_metrics(ev.y_true, ev.y_pred, ev.labels) if r.support > 0]
    overall = float((ev.y_pred == ev.y_true).mean())
    lo, hi = wilson_interval(int((ev.y_pred == ev.y_true).sum()), len(ev.y_true))
    roll = rollup_accuracy(ev)
    print(f"\nbackbone {ev.backbone}   {ev.n_train} train / {len(ev.y_true)} test")
    print(f"taxon accuracy  {overall:.3f} [95% CI {lo:.3f}-{hi:.3f}]")
    print(f"species top-3   {roll['species_top3']:.3f}")
    print(f"genus level     {roll['genus']:.3f}      family level {roll['family']:.3f}")
    print(f"calibration     ECE {ece:.3f} (T={ev.temperature:.2f})")
    if open_set:
        best = max(open_set["scorers"], key=lambda x: x["auroc"])
        print(
            f"open-set        {best['scorer']} AUROC {best['auroc']:.3f}, "
            f"catches {best['tpr_at_fpr']['0.05']:.1%} of intruders at 5% false alarm"
        )

    print(f"\n{'worst 8 classes':<30}{'n':>5}{'prec':>7}{'rec':>7}{'f1':>7}")
    print("-" * 56)
    for r in sorted(rows, key=lambda x: x.f1)[:8]:
        flag = "" if r.verdict_reliable else "  (low n)"
        print(
            f"{r.label.replace('_', ' '):<30}{r.support:>5}{r.precision:>7.3f}"
            f"{r.recall:>7.3f}{r.f1:>7.3f}{flag}"
        )

    print("\nTier A by sex:")
    tier_a = {s.scientific_name for s in ev.cfg.species_by_tier("A")}
    for b in sex_breakdown(ev.y_true, ev.y_pred, ev.species, ev.sex):
        if b.species not in tier_a:
            continue
        m_, f_ = b.recall("male_unspecified"), b.recall("female")
        if m_ is None or f_ is None:
            continue
        print(f"  {b.species:<26} male {m_:.3f}   female {f_:.3f}   gap {m_ - f_:+.3f}")


def build(cfg: Config, feature_file: str, epochs: int = 200):
    ev = evaluate(cfg, feature_file, epochs=epochs)
    rel, ece = chart_reliability(ev)
    charts = {
        "confusion": chart_confusion(ev),
        "confusion_female": chart_confusion(ev, "female"),
        "confusion_male": chart_confusion(ev, "male_unspecified"),
        "reliability": rel,
        "pr": chart_pr_tier_a(ev),
    }
    os_path = cfg.path("reports_dir") / "open_set.json"
    open_set = json.loads(os_path.read_text(encoding="utf-8")) if os_path.is_file() else None

    out = cfg.path("reports_dir") / "metrics_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(ev, ece, charts, open_set), encoding="utf-8")
    return ev, ece, open_set, out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    import argparse

    ap = argparse.ArgumentParser(description="Build the HTML metrics report.")
    ap.add_argument("--features", default="tf_efficientnetv2_b0.in1k_18146.npy")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--quiet", action="store_true", help="write the file, skip the summary")
    args = ap.parse_args()

    cfg = load_config()
    ev, ece, open_set, out = build(cfg, args.features, args.epochs)
    if not args.quiet:
        print_summary(ev, ece, open_set)
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
