"""Shared figure builders for the FPL sweeps, with 95% confidence intervals.

pareto_figure: (speed, survival) small-multiple with x error bars (CI on mean speed) and
    y error bars (Wilson CI on survival), one panel per commanded speed.
robustness_figure: productive-speed and survival vs traction-loss, with shaded CI bands.

Each config's stats dict must carry the precomputed CI fields:
  Pareto:     vx, vx_ci, fall, surv, surv_lo, surv_hi
  Robustness: prod, prod_ci, surv, surv_lo, surv_hi
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIN_C, FPL_C = "#4c72b0", "#dd8452"
FPL_RED = "#c44e52"


def pareto_figure(per_tv, fpl_label, suptitle, out_path, lin_prefix="lin",
                  xlabel="achieved forward speed  (m/s)", gain_word="speed",
                  panel_title="commanded speed = {tv} m/s"):
    tvs = list(per_tv.keys())
    n = len(tvs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.3), sharey=True)
    if n == 1:
        axes = [axes]
    for ax_i, (ax, tv) in enumerate(zip(axes, tvs)):
        is_last = ax_i == len(tvs) - 1
        items = [(l, s) for l, s in per_tv[tv].items() if l.startswith(lin_prefix)]
        items.sort(key=lambda kv: kv[1]["vx"])
        xs = [s["vx"] for _, s in items]
        ys = [s["surv"] for _, s in items]
        xerr = [s["vx_ci"] for _, s in items]
        yerr = [[s["surv"] - s["surv_lo"] for _, s in items],
                [s["surv_hi"] - s["surv"] for _, s in items]]
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="o-", color=LIN_C, lw=2, ms=7,
                    mec="white", mew=1, ecolor=LIN_C, elinewidth=1, capsize=2,
                    label="linear-weight family", zorder=3)
        for l, s in items:
            ax.annotate(l.replace(lin_prefix + " wv=", "w="), (s["vx"], s["surv"]),
                        fontsize=6.5, color=LIN_C, xytext=(0, -12),
                        textcoords="offset points", ha="center")
        fs = per_tv[tv][fpl_label]
        fx, fy = fs["vx"], fs["surv"]
        ax.errorbar(fx, fy, xerr=fs["vx_ci"],
                    yerr=[[fy - fs["surv_lo"]], [fs["surv_hi"] - fy]],
                    fmt="*", color=FPL_C, ms=22, mec="k", mew=1.2, ecolor="k",
                    elinewidth=1.2, capsize=2, label="FPL (one fixed spec)", zorder=5)
        # advantage arrow: fastest linear at least as safe as FPL (fall <= FPL's fall)
        safe = [(s["vx"], s["surv"]) for l, s in items if s["fall"] <= fs["fall"] + 1e-9]
        if safe:
            bx, by = max(safe)
            ax.annotate("", xy=(fx, fy), xytext=(bx, by),
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.6))
            # Only annotate the % on the hardest-speed panel: there FPL's own fall rate is
            # highest so the most linear weights qualify as "equally safe", giving the most
            # conservative (defensible) gap. At easy speeds the only equally-safe linear is
            # the crawling one, which inflates the % — the arrow already shows dominance.
            if fx > bx and is_last:
                ax.text((fx + bx) / 2, min(fy, by) - 0.06,
                        f"+{100*(fx-bx)/max(bx,1e-6):.0f}% {gain_word}\nat equal safety",
                        fontsize=8, ha="center", va="top", fontweight="bold")
        ax.set_title(panel_title.format(tv=tv), fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("survival rate  (1 − fall)")
    axes[0].legend(fontsize=8, loc="lower left", framealpha=0.95)
    fig.suptitle(suptitle.replace("\\n", "\n"), fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure (95% CI) -> {out_path}")


def robustness_figure(per_fr, fpl_label, suptitle, out_path,
                      prod_title="Productive speed  (speed × survival)",
                      prod_ylabel="m/s × survival"):
    frs = list(per_fr.keys())
    x = [1.0 - fr for fr in frs]
    labels = list(per_fr[frs[0]].keys())
    lin_labels = [l for l in labels if l != fpl_label]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(lin_labels)))
    fig, (axP, axS) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    def ser(l, k):
        return np.array([per_fr[fr][l][k] for fr in frs])

    for i, l in enumerate(lin_labels):
        prod, pci = ser(l, "prod"), ser(l, "prod_ci")
        axP.plot(x, prod, "-o", color=cmap[i], lw=1.5, ms=4, label=l.replace("lin ", ""))
        axP.fill_between(x, prod - pci, prod + pci, color=cmap[i], alpha=0.13, lw=0)
        surv, lo, hi = ser(l, "surv"), ser(l, "surv_lo"), ser(l, "surv_hi")
        axS.plot(x, surv, "-o", color=cmap[i], lw=1.5, ms=4, label=l.replace("lin ", ""))
        axS.fill_between(x, lo, hi, color=cmap[i], alpha=0.13, lw=0)
    prod, pci = ser(fpl_label, "prod"), ser(fpl_label, "prod_ci")
    axP.plot(x, prod, "-*", color=FPL_RED, lw=3, ms=13, zorder=6, label="FPL (one fixed spec)")
    axP.fill_between(x, prod - pci, prod + pci, color=FPL_RED, alpha=0.2, lw=0)
    surv, lo, hi = ser(fpl_label, "surv"), ser(fpl_label, "surv_lo"), ser(fpl_label, "surv_hi")
    axS.plot(x, surv, "-*", color=FPL_RED, lw=3, ms=13, zorder=6, label="FPL (one fixed spec)")
    axS.fill_between(x, lo, hi, color=FPL_RED, alpha=0.2, lw=0)
    axP.set(title=prod_title, ylabel=prod_ylabel,
            xlabel="traction loss  (1 − friction;  0 = nominal)")
    axS.set(title="Survival", ylabel="survival rate (1 − fall)",
            xlabel="traction loss  (1 − friction;  0 = nominal)")
    for ax in (axP, axS):
        ax.grid(alpha=0.3)
    axP.legend(fontsize=8, title="cost", loc="lower left")
    fig.suptitle(suptitle.replace("\\n", "\n"), y=1.06, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure (95% CI band) -> {out_path}")
