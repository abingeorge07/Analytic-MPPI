"""Plot com_z(t) and com_x(t) for a hopper config to eyeball hop quality.

Good hopping = clean PERIODIC com_z (regular jumps, low noise) with steadily advancing
com_x and uprightness held the whole 15 s. Usage:
  .venv/bin/python verification/_hopper_plot.py '<param JSON>' [out.png]
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hopper_search import evaluate, DT  # noqa: E402


def main():
    p = json.loads(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "runs" / "hopper_fix" / "hopper_com.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    p.setdefault("steps", 750)
    p.setdefault("seeds", [0, 1, 2, 3])
    p["want_traj"] = True
    agg = evaluate(p, want_traj=True)
    trajs = agg.pop("trajs")

    n = len(trajs)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, tr in enumerate(trajs):
        t = np.arange(len(tr["comz"])) * DT
        c = cmap[i % len(cmap)]
        s = tr["survive_steps"]
        axes[0].plot(t, tr["comz"], color=c, lw=1.0, label=f"seed {tr['seed']} (survive {s*DT:.1f}s)")
        axes[1].plot(t, tr["comx"], color=c, lw=1.0)
        tu = np.arange(len(tr["up"])) * DT          # up is (T,), com is (T+1,)
        axes[2].plot(tu, tr["up"], color=c, lw=1.0)
        if s < len(t):
            axes[0].axvline(s * DT, color=c, ls=":", lw=0.8, alpha=0.6)
    # NOTE: com_z (whole-body COM) sits ~0.6 m BELOW torso height (the hanging leg drags the
    # COM down: torso 1.19 -> com_z 0.54 at standing). The FPL *height* atom is defined on
    # TORSO height (floor 0.92), NOT com_z, so we do NOT shade a com_z "danger" band here — it
    # would sit right inside the normal hop range and misleadingly imply FPL is failing. See
    # hopper_WINNER_fpl_signal.png for the correct torso-height-vs-FPL-fulfillment view.
    axes[0].set_ylabel("com_z (m)"); axes[0].set_title(
        f"Hopper COM over 15 s   |   periodicity={agg['periodicity']:.2f} "
        f"period={agg['period_s']:.2f}s  survive_min={agg['survive_min']}/750  minup={agg['minup']:.2f}")
    axes[0].legend(fontsize=7, loc="upper right", ncol=2)
    axes[1].set_ylabel("com_x (m)"); axes[1].set_title("forward progress")
    axes[2].axhline(0.6, color="k", ls="--", lw=0.7, alpha=0.5)
    axes[2].set_ylabel("upright (cos tilt)"); axes[2].set_title("uprightness (0.6 = topple line)")
    axes[2].set_xlabel("time (s)"); axes[2].set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    print(json.dumps({k: v for k, v in agg.items() if k != "per_seed_survive"}, indent=2))


if __name__ == "__main__":
    main()
