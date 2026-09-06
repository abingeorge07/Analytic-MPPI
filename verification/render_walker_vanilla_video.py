"""Render a video + provenance bundle for configs/exp/walker_vanilla_mppi.py.

Mirrors verification/render_hopper_vanilla_video.py's layout (video.mp4, config.json,
metrics.json, model/ + INFO.json) so this config's artifact sits under
runs/capability/walker/, driven straight from the config.py file (not
capability_probe's ENVS/CLI sweep machinery) since config.resolve() is the source of
truth for this run. Camera is "floating" (walker's own trackcom camera, matching
capability_probe.py's CAMERAS["walker"]), not hopper's "track".

Run:  a-mppi/bin/python verification/render_walker_vanilla_video.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("DISPLAY", ":1")
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analytic_mppi.config import load_config_file, resolve
from analytic_mppi.eval import run_episode, render_video, make_task
from _ci import mean_ci, wilson_ci

CONFIG_PATH = "configs/exp/walker_vanilla_mppi.py"
TAG = "mppi_normal_K256_10s_weighttuned"
OUT_DIR = REPO / "runs" / "capability" / "walker" / TAG
N_SEEDS = 30
FALL_UPRIGHT = 0.6


def _git_commit() -> str:
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True, timeout=10)
    return out.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = load_config_file(CONFIG_PATH)
    r = resolve(cfg)
    build = dict(r.build)
    cost_mode = build.pop("cost_mode")
    task = make_task(r.task_name, **r.task_kwargs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "model").mkdir(exist_ok=True)

    # --- metrics: fresh N_SEEDS run straight from the resolved config -------------
    rows = []
    for seed in range(N_SEEDS):
        res = run_episode(r.task_name, r.controller, steps=r.steps, seed=seed,
                          cost_mode=cost_mode, init_fn=r.init_fn,
                          task_kwargs=r.task_kwargs, **build)
        sd = res["sd"]
        vx = float(sd[..., task._vel_adr].mean())
        zax_min = float(sd[..., task._zax_adr + 2].min())
        h_mean = float(sd[..., task._pos_adr + 2].mean())
        rows.append(dict(vx=vx, survived=float(zax_min >= FALL_UPRIGHT),
                         zax_min=zax_min, h_mean=h_mean))
    vx_m, vx_h = mean_ci([row["vx"] for row in rows])
    surv_k = sum(1 for row in rows if row["survived"])
    p, lo, hi = wilson_ci(surv_k, N_SEEDS)
    metrics = dict(
        config_path=CONFIG_PATH, n_seeds=N_SEEDS,
        vx=vx_m, vx_ci=vx_h, target_velocity=r.task_kwargs.get("target_velocity"),
        frac_of_target=vx_m / r.task_kwargs["target_velocity"],
        survival=p, survival_lb=lo, survival_hi=hi, per_seed=rows,
    )
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"metrics: vx={vx_m:+.3f}±{vx_h:.3f} ({metrics['frac_of_target']:.0%} of target) "
          f"survival={p:.2f} (lb={lo:.2f})")

    # --- config.json: full resolved config + provenance ----------------------------
    (OUT_DIR / "config.json").write_text(json.dumps(dict(
        config_source=CONFIG_PATH, resolved_config=cfg.to_dict(),
        controller=r.controller, task_name=r.task_name, task_kwargs=r.task_kwargs,
        steps=r.steps, build_kwargs=r.build, git_commit=_git_commit(),
    ), indent=2, default=str))

    # --- model provenance (mirrors capability_probe.py's save_bundle) --------------
    import mujoco
    src = Path(task.model_path)
    (OUT_DIR / "model" / src.name).write_bytes(src.read_bytes())
    (OUT_DIR / "model" / "INFO.json").write_text(json.dumps(dict(
        source_path=str(src), sha256=_sha256(src), mujoco_version=mujoco.__version__,
        git_commit=_git_commit(),
        note="Entry-point MJCF only; meshes are pinned by git_commit, not vendored.",
    ), indent=2))

    # --- video (seed=0, walker's tracking camera) -----------------------------------
    render_video(r.task_name, r.controller, steps=r.steps, out_path=OUT_DIR / "video.mp4",
                seed=0, cost_mode=cost_mode, init_fn=r.init_fn, camera="floating",
                task_kwargs=r.task_kwargs, **build)
    print(f"bundle written to {OUT_DIR.relative_to(REPO)}")
    for f in sorted(OUT_DIR.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUT_DIR)}  ({f.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
