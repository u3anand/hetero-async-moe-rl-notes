"""b1tel — telemetry for the OLMoE async MoE-RL bring-up (B1).

Three append-only JSONL streams (episode / step / system) joined on
episode_id / step_id / replica_id and one monotonic clock. Raw JSONL is the
single source of truth; charts are derived offline.
"""

from b1tel.telemetry import TelemetryLogger, EPISODE_FIELDS, STEP_FIELDS, SYSTEM_FIELDS

__all__ = ["TelemetryLogger", "EPISODE_FIELDS", "STEP_FIELDS", "SYSTEM_FIELDS"]
