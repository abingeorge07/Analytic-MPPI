"""One-time fetch of the reference walking clip for the humanoid_mocap env.

The reference motion is retargeted motion-capture data for the Unitree G1 from
the LocoMuJoCo dataset:

    https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main

We *vendor* the downloaded ``.npz`` next to this file so the env has no runtime
network dependency (and ``huggingface_hub`` is not needed at runtime). Re-run
this script only to regenerate / change the vendored clip:

    .venv/bin/python -m analytic_mppi.envs.humanoid_mocap._fetch_reference

Requires the optional ``mocap`` extra (``huggingface_hub``).
"""
from __future__ import annotations

import shutil
from pathlib import Path

# Default reference clip (matches the original hydrax HumanoidMocap default).
REPO_ID = "robfiras/loco-mujoco-datasets"
REFERENCE_FILENAME = "Lafan1/mocap/UnitreeG1/walk1_subject1.npz"

HERE = Path(__file__).resolve().parent
# Vendored file name: the basename of the reference file.
VENDORED_NPZ = HERE / Path(REFERENCE_FILENAME).name


def fetch(reference_filename: str = REFERENCE_FILENAME,
          dest: Path = VENDORED_NPZ) -> Path:
    """Download ``reference_filename`` from the LocoMuJoCo dataset and copy it to
    ``dest`` (default: alongside this module). Returns the destination path."""
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=REPO_ID,
        filename=reference_filename,
        repo_type="dataset",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)
    return dest


if __name__ == "__main__":
    out = fetch()
    print(f"Vendored reference clip -> {out}")
