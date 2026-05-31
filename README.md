# hetero-async-moe-rl-notes

Research notes and paper-direction working docs for a project on **throughput-optimal asynchronous MoE RL on heterogeneous GPU clusters for agentic workloads**.

This is an Obsidian vault. Open the directory in Obsidian to traverse the link graph, or read the files directly as markdown.

## Canonical doc

Start with **[Research Plan](Research%20Plan.md)** — the consolidated plan:

1. Motivation (why MoE × async × multi-replica × hetero × agentic is an open cell)
2. B1 — characterize first (six questions / six charts) and why
3. Main idea going forward (hetero-aware MILP scheduler + 3 composable mechanisms + 1 conditional)
4. How we'll use WatGPU (the initial phase)

Then **[Initial Plan](Initial%20Plan.md)** is the step-by-step plan for the initial phase, and **[WatGPU](WatGPU.md)** documents the cluster.

## Working thesis

> Throughput-optimal asynchronous MoE RL on heterogeneous GPU clusters for agentic workloads, with a workload-characterization benchmark as the precondition and a hetero-aware scheduler with composable mechanisms as the system contribution.

## Layout

- `Research Plan.md` — canonical plan
- `Initial Plan.md` — initial-phase execution plan
- `WatGPU.md` — cluster capability + constraints
- `Papers/` — per-paper notes + linked PDFs
- `Deep Dives/` — longer-form topic notes
- Top-level concept notes (`MoE Architecture.md`, `MoE vs Dense Workload.md`, `Training.md`, `Inference.md`, `Transformer.md`)

