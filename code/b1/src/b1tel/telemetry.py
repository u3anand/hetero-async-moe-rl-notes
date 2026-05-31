"""JSONL telemetry writers + schema for the three B1 streams.

Design:
- One ``TelemetryLogger`` per run, pointed at ``runs/<config>/<ts>/telemetry/``.
- Three streams, each append-only JSONL: ``episode.jsonl``, ``step.jsonl``, ``system.jsonl``.
- Every record carries ``t_wall`` (wall clock, for cross-stream joins) written by the logger.
- GPU phase durations are passed in by the caller, measured with CUDA events upstream.
- Writers are process-safe via O_APPEND + line-buffered orjson; multiple producers
  (trainer, orchestrator, sampler, udocker env) may write the same stream concurrently.

The field lists below are the *contract*: the prime-rl / vLLM / mini-swe-agent fork
edits emit these keys; ``charts.py`` reads them. Keep them in sync.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

try:
    import orjson as _json

    def _dumps(o: dict) -> bytes:
        return _json.dumps(o) + b"\n"
except ModuleNotFoundError:  # fallback so the module imports without deps
    import json as _stdjson

    def _dumps(o: dict) -> bytes:
        return (_stdjson.dumps(o, separators=(",", ":")) + "\n").encode()


# ---- schema (the contract) -------------------------------------------------

# Q1, Q3 (C_infer/C_tool), Q5, Q6 version tags
EPISODE_FIELDS = (
    "episode_id", "replica_id", "tier", "policy_version_start",
    "t_start", "t_end", "num_turns", "num_tokens", "reward",
    "staleness_at_consume",
    "tool_calls",  # list[ {category, t_start, t_end, dur_s, cores, in_udocker} ]
)
TOOL_CALL_FIELDS = ("category", "t_start", "t_end", "dur_s", "cores", "in_udocker")
TOOL_CATEGORIES = ("file_op", "git", "mid_test", "build", "reward_eval", "inference")

# Q3 (C_train), Q4 (broadcast split), Q7
STEP_FIELDS = (
    "step_id", "t_train",
    "broadcast",  # {router_bytes, router_s, expert_bytes, expert_s}
    "ep_a2a_bytes", "ep_a2a_s",
    "tokens", "useful_tokens",
    "batch_staleness",  # list[int]
    "mfu",
)
BROADCAST_FIELDS = ("router_bytes", "router_s", "expert_bytes", "expert_s")

# Q2, Q7
SYSTEM_FIELDS = (
    "gpu_index", "sm_active", "dram_active",       # per-GPU; sampler emits one record/GPU/tick
    "sandbox_queue_depth", "n_active_sandboxes", "cpu_util",
)


def monotonic() -> float:
    """Single monotonic clock all producers should use for durations."""
    return time.perf_counter()


class TelemetryLogger:
    """Append-only JSONL logger for one run's three streams.

    >>> log = TelemetryLogger("runs/homo-fast/2026-06-01T00:00:00")
    >>> log.episode(episode_id="e1", replica_id=0, tier="fast", reward=1.0, tool_calls=[])
    >>> log.step(step_id=42, t_train=3.1, broadcast={"router_bytes": 0, ...})
    >>> log.system(gpu_index=0, sm_active=0.91, dram_active=0.7)
    """

    def __init__(self, run_dir: str | os.PathLike):
        self.dir = Path(run_dir) / "telemetry"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fds: dict[str, int] = {}

    def _fd(self, stream: str) -> int:
        fd = self._fds.get(stream)
        if fd is None:
            path = self.dir / f"{stream}.jsonl"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            self._fds[stream] = fd
        return fd

    def _write(self, stream: str, record: dict[str, Any]) -> None:
        record.setdefault("t_wall", time.time())
        os.write(self._fd(stream), _dumps(record))  # single write() = atomic append per line

    def episode(self, **fields: Any) -> None:
        self._write("episode", fields)

    def step(self, **fields: Any) -> None:
        self._write("step", fields)

    def system(self, **fields: Any) -> None:
        self._write("system", fields)

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def classify_command(cmd: str) -> str:
    """Best-effort category for a bash command (Q1). Tune on the sandbox-spike data."""
    c = cmd.strip().lower()
    first = c.split()[0] if c.split() else ""
    if any(k in c for k in ("pytest", "python -m pytest", "tox", "nosetests", "unittest")):
        return "mid_test"
    if any(k in c for k in ("pip install", "make", "setup.py", "cmake", "gcc", "build")):
        return "build"
    if first == "git":
        return "git"
    if first in ("ls", "cat", "grep", "sed", "awk", "find", "head", "tail", "echo", "cd", "mkdir", "rm", "cp", "mv"):
        return "file_op"
    return "file_op"
