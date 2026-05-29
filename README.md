# hetero-async-moe-rl-notes

Research notes and paper-direction working docs for a project on **throughput-optimal asynchronous MoE RL on heterogeneous GPU clusters for agentic workloads**.

This is an Obsidian vault. Open the directory in Obsidian to traverse the link graph, or read the files directly as markdown.

## Canonical doc

Start with **[Paper Direction - v2](Paper%20Direction%20-%20v2.md)** — the consolidated plan. Covers:

1. Workload (Qwen3-30B-A3B + mini-swe-agent + GRPO + R3 on prime-rl)
2. Benchmark stage (6 characterization questions + sweep dimensions)
3. RL problem walkthrough
4. Scheduling problem and solution (static MILP + dynamic runtime controllers, 3 mechanisms + 1 conditional)
5. Resource planning (AWS-first, cheap GPU pair recommendation, budget estimate)

Older revisions ([Paper Direction.md](Paper%20Direction.md) and [Plan Logic Check.md](Plan%20Logic%20Check.md)) are kept as history but superseded by v2.

## Working thesis

> Throughput-optimal asynchronous MoE RL on heterogeneous GPU clusters for agentic workloads, with a workload-characterization benchmark as the precondition and a hetero-aware scheduler with composable mechanisms as the system contribution.

## Layout

- `Paper Direction - v2.md` — canonical plan
- `Papers/` — per-paper notes + linked PDFs
- `Deep Dives/` — longer-form topic notes
- Top-level concept notes (`MoE Architecture.md`, `MoE vs Dense Workload.md`, `Training.md`, `Inference.md`, `Transformer.md`)

## Status

Pre-experiments. Plan written, scoping not yet begun.

Target venue: MLSys or EuroSys '27.
