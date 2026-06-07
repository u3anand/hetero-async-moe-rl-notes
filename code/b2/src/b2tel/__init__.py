"""b2tel — telemetry + offline analysis for the B2 / TierShift benchmark.

Serve a fixed MoE (OLMoE) on the heterogeneous watgpu308 node, replay a mixed drifting
request trace, capture per-(layer, expert, window) token load, micro-bench the tier gap
and migration cost, then simulate static-vs-dynamic expert placement and score each on
the objective: **p99 latency / SLO-attainment + $/token**.

Raw JSONL is the single source of truth; charts are derived offline. The capture-side
writer lives in the vLLM fork (``expert_load_logger``); this package owns the schema,
the readers/joiners, the trace builder, the replay client, the micro-benches, and the
offline simulator + charts.
"""

from b2tel.telemetry import (
    EXPERT_FIELDS,
    SYSTEM_FIELDS,
    SEGMENT_FIELDS,
    REQUEST_FIELDS,
    JsonlWriter,
    read_jsonl,
    assign_segments,
    monotonic,
)

__all__ = [
    "EXPERT_FIELDS",
    "SYSTEM_FIELDS",
    "SEGMENT_FIELDS",
    "REQUEST_FIELDS",
    "JsonlWriter",
    "read_jsonl",
    "assign_segments",
    "monotonic",
]
