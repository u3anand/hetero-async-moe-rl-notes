"""JSONL schema + writers/readers for the B2 telemetry streams.

Two append-only JSONL streams plus a replay-side timeline, joined on
``(layer_id, expert_id, window_id)`` and one wall clock:

1. ``expert.jsonl`` — per (layer, expert, time-window) token load, emitted by the vLLM
   MoE-gate hook (``expert_load_logger`` in the fork). The contract is `EXPERT_FIELDS`.
2. ``segments.jsonl`` — the replay's segment timeline (one row per segment block with its
   wall-clock span), emitted by ``replay.py``. Used to label expert windows by domain
   (Q6) when the capture ran a multi-segment drifting trace. The contract is
   `SEGMENT_FIELDS`.
3. ``requests.jsonl`` — per-request latency from ``replay.py`` (for the measured p50/p95/p99
   sanity numbers). The contract is `REQUEST_FIELDS`.
4. ``system.jsonl`` — ~1 Hz per-GPU sidecar, only populated in the live B7 validation.

The field tuples below are the *contract*: the fork hook + replay emit these keys and the
sim/charts read them. Keep them in sync with the vault `Initial Plan 2.md` → Instrumentation.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable

try:
    import orjson as _json

    def _dumps(o: dict) -> bytes:
        return _json.dumps(o) + b"\n"

    def _loads(b: bytes) -> dict:
        return _json.loads(b)
except ModuleNotFoundError:  # stdlib fallback so the module imports without deps
    import json as _stdjson

    def _dumps(o: dict) -> bytes:
        return (_stdjson.dumps(o, separators=(",", ":")) + "\n").encode()

    def _loads(b: bytes) -> dict:
        return _stdjson.loads(b)


# ---- schema (the contract) -------------------------------------------------

# expert.jsonl — emitted by the vLLM fork hook (one row per layer/expert/phase/window).
# ``segment_label`` is present only for single-segment captures (VLLM_B2_SEGMENT); for a
# drifting replay it is filled offline by ``assign_segments``. ``expert_class`` (consistent
# /temporal, GEM taxonomy) and the sim-side ``tier_assigned``/``is_replica`` are derived
# offline, not emitted by the hook.
EXPERT_FIELDS = (
    "window_id", "t_start", "t_end", "layer_id", "expert_id", "token_load",
    "n_experts", "top_k", "phase",          # emitted by the hook
    "segment_label",                        # hook (single-segment) or assign_segments()
    "expert_class", "tier_assigned", "is_replica",  # derived offline (sim/taxonomy)
)

# segments.jsonl — the replay's domain timeline (Q6 labels).
SEGMENT_FIELDS = ("segment_label", "t_start", "t_end", "n_requests")

# requests.jsonl — per-request replay outcome (measured latency sanity).
REQUEST_FIELDS = (
    "req_id", "segment_label", "t_start", "t_end", "latency_s",
    "prompt_tokens", "completion_tokens", "ok", "status",
)

# system.jsonl — ~1 Hz sidecar, live B7 only.
SYSTEM_FIELDS = (
    "gpu_index", "tier", "sm_active", "dram_active",
    "cross_tier_alltoall_bytes", "per_tier_straggler_wait_s", "decode_queue_depth",
)


def monotonic() -> float:
    """Single monotonic clock for durations (perf_counter)."""
    return time.perf_counter()


# ---- writer ----------------------------------------------------------------


class JsonlWriter:
    """Append-only JSONL writer. ``O_APPEND`` + single ``write()`` per line = atomic
    append, so multiple producers may share one file."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    def write(self, **fields: Any) -> None:
        fields.setdefault("t_wall", time.time())
        os.write(self._fd, _dumps(fields))

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None  # type: ignore[assignment]

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---- readers / joiners -----------------------------------------------------


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    """Read a JSONL file (or a directory of ``*.jsonl``, e.g. one per pid) into a list."""
    p = Path(path)
    files: Iterable[Path]
    if p.is_dir():
        files = sorted(p.glob("*.jsonl"))
    else:
        files = [p]
    rows: list[dict] = []
    for f in files:
        with open(f, "rb") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(_loads(line))
    return rows


def assign_segments(
    expert_rows: list[dict], segment_rows: list[dict], clock: str = "t_start"
) -> list[dict]:
    """Label each expert window with its domain by wall-clock overlap with the replay's
    segment timeline. Mutates+returns ``expert_rows`` with ``segment_label`` set.

    A window is assigned the segment whose [t_start, t_end] span contains the window's
    midpoint; ties / gaps fall back to the nearest segment. Rows that already carry a
    ``segment_label`` (single-segment capture) are left untouched.
    """
    if not segment_rows:
        return expert_rows
    segs = sorted(segment_rows, key=lambda s: s["t_start"])
    for row in expert_rows:
        if row.get("segment_label"):
            continue
        mid = (row["t_start"] + row["t_end"]) / 2.0
        label = None
        for s in segs:
            if s["t_start"] <= mid <= s["t_end"]:
                label = s["segment_label"]
                break
        if label is None:  # gap/edge: nearest segment by center distance
            label = min(
                segs,
                key=lambda s: abs(mid - (s["t_start"] + s["t_end"]) / 2.0),
            )["segment_label"]
        row["segment_label"] = label
    return expert_rows
