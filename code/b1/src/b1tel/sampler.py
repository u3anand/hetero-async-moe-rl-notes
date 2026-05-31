"""~1 Hz system sidecar → system.jsonl (Q2, Q7).

Polls per-GPU SM utilization and a HBM-bandwidth proxy plus host CPU. Runs as a
background process for the lifetime of a run; SIGTERM/SIGINT stops it cleanly.

GPU stats come from ``nvidia-smi --query-gpu`` (no DCGM dependency). ``utilization.memory``
is the % of time the memory interface was active — a coarse HBM-bandwidth proxy; swap to
DCGM ``DRAM_ACTIVE`` if/when the DCGM exporter is available on the node for a true number.

Sandbox queue depth / n_active_sandboxes are NOT visible here — the prime-rl orchestrator
writes those into the same ``system.jsonl`` via ``TelemetryLogger.system(...)``.

Usage:
    python -m b1tel.sampler --run-dir runs/homo-fast/<ts> --interval 1.0
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import time

from b1tel.telemetry import TelemetryLogger

_QUERY = "index,utilization.gpu,utilization.memory,memory.used"
_running = True


def _stop(*_: object) -> None:
    global _running
    _running = False


def _poll_gpus() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        try:
            idx, sm, mem, used = (p.strip() for p in line.split(","))
            rows.append({
                "gpu_index": int(idx),
                "sm_active": float(sm) / 100.0,
                "dram_active": float(mem) / 100.0,   # proxy; see module docstring
                "mem_used_mib": float(used),
            })
        except ValueError:
            continue
    return rows


def _cpu_util() -> float:
    # 1-minute load average / core count as a cheap, dependency-free CPU proxy.
    try:
        with open("/proc/loadavg") as f:
            load1 = float(f.read().split()[0])
        import os
        return load1 / max(1, os.cpu_count() or 1)
    except Exception:
        return -1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    with TelemetryLogger(args.run_dir) as log:
        while _running:
            t0 = time.perf_counter()
            cpu = _cpu_util()
            for g in _poll_gpus():
                log.system(cpu_util=cpu, **g)
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, args.interval - dt))


if __name__ == "__main__":
    main()
