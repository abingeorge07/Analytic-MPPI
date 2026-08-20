"""Crash-safe, resumable trial store for the FPL-MPPI experiment campaign.

Every experiment (portability matrix, Pareto sweeps, zero-tuning, robustness, the
FPL-native-sampler race) writes ONE line per (config, seed, difficulty) trial to an
append-only JSONL file, flushed+fsynced immediately. If the process dies (OOM, a
network hiccup, a MuJoCo segfault on one config) nothing already computed is lost:
re-running skips every trial whose key is already present and continues.

Layout (one JSON object per line):
    {"key": "<canonical>", "fields": {...}, "metrics": {...}, "ts": 1699.9}

`fields` identifies the trial (experiment / env / sampler / cost / label / seed /
difficulty / ...); `metrics` is whatever scalar summary that trial produced. The key
is the canonical JSON of `fields` (sorted keys) so it is stable across runs.

Usage
-----
    ckpt = Checkpoint("verification/checkpoints/portability.jsonl")
    fields = dict(exp="portability", env="hopper", sampler="mppi", cost="fpl", seed=3, tv=2.0)
    if not ckpt.has(fields):
        m = run_one_trial(...)          # expensive
        ckpt.record(fields, m)          # durable immediately
    ...
    rows = ckpt.rows()                  # list[dict] of every trial for aggregation
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _canonical(fields: Dict[str, Any]) -> str:
    """Stable string key: JSON with sorted keys and normalized numbers.

    Floats are rounded to 6 significant places so that e.g. a target speed of 2.0
    read back from JSON matches a freshly passed 2.0 exactly (no 1.9999999 keys)."""
    norm: Dict[str, Any] = {}
    for k, v in fields.items():
        if isinstance(v, float):
            norm[k] = round(v, 6)
        else:
            norm[k] = v
    return json.dumps(norm, sort_keys=True, separators=(",", ":"))


class Checkpoint:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A partially-written final line from a hard crash: ignore it; the
                    # trial will simply be recomputed. Everything before it is intact.
                    continue
                self._records[rec["key"]] = rec

    # ---- membership / lookup ----

    def has(self, fields: Dict[str, Any]) -> bool:
        return _canonical(fields) in self._records

    def get(self, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rec = self._records.get(_canonical(fields))
        return None if rec is None else rec.get("metrics")

    # ---- write ----

    def record(self, fields: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Append (or overwrite) one trial's result durably."""
        key = _canonical(fields)
        rec = {"key": key, "fields": fields, "metrics": metrics, "ts": round(time.time(), 2)}
        self._records[key] = rec
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ---- read-back for aggregation ----

    def rows(self, **filt: Any) -> List[Dict[str, Any]]:
        """Flat list of {**fields, **metrics} for every trial matching `filt`
        (exact match on the given field keys). Metrics keys win on collision."""
        out = []
        for rec in self._records.values():
            fields = rec["fields"]
            if all(fields.get(k) == v for k, v in filt.items()):
                out.append({**fields, **rec["metrics"]})
        return out

    def field_values(self, key: str, **filt: Any) -> List[Any]:
        """Sorted unique values a field takes across the (optionally filtered) store."""
        vals = {r[key] for r in self.rows(**filt) if key in r}
        return sorted(vals, key=lambda x: (isinstance(x, str), x))

    def __len__(self) -> int:
        return len(self._records)

    def summary(self) -> str:
        exps = {}
        for rec in self._records.values():
            exps.setdefault(rec["fields"].get("exp", "?"), 0)
            exps[rec["fields"].get("exp", "?")] += 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(exps.items()))
        return f"{len(self)} trials in {self.path.name}  [{parts}]"


def aggregate(rows: Iterable[Dict[str, Any]], group_keys: List[str],
              metric_keys: List[str]) -> Dict[tuple, Dict[str, Any]]:
    """Group flat rows by `group_keys`, collecting each metric into a list.

    Returns {group_tuple: {"n": count, metric: [values...], **group_fields}}. Callers
    apply their own CI (mean_ci / wilson_ci) — this only bins the per-seed values."""
    import collections
    groups: Dict[tuple, Dict[str, Any]] = {}
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g = tuple(r.get(k) for k in group_keys)
        for m in metric_keys:
            if m in r and r[m] is not None:
                buckets[g][m].append(r[m])
    for g, mdict in buckets.items():
        entry = {k: v for k, v in zip(group_keys, g)}
        n = max((len(v) for v in mdict.values()), default=0)
        entry["n"] = n
        for m in metric_keys:
            entry[m] = mdict.get(m, [])
        groups[g] = entry
    return groups
