# Repo guide for agents

Obsidian vault + research code for **throughput-optimal asynchronous MoE RL on
heterogeneous GPU clusters for agentic workloads**. Open in Obsidian to traverse the
link graph, or read the markdown directly.

## Canonical docs (read these first, in order)

1. [Research Plan](Research%20Plan.md) — the consolidated plan: motivation, the B1
   characterization benchmark (six questions / six charts), the system idea
   (hetero-aware MILP scheduler + composable mechanisms), and the WatGPU initial phase.
2. [Initial Plan](Initial%20Plan.md) — step-by-step plan for the initial phase, incl. the
   **Instrumentation** schema (the telemetry contract the forks emit).
3. [WatGPU](WatGPU.md) — the cluster: capabilities, constraints, login. **Local-only**
   (gitignored — do not commit or paste its contents into shared files).
4. [README](README.md) — short orientation + repo layout.

Code lives under `code/b1/` (the B1 benchmark: `b1tel` telemetry lib, configs, slurm,
sandbox helpers) with the three instrumented forks as submodules under `code/forks/`.
`code/b1/README.md`, `code/b1/RUNBOOK.md`, and `code/b1/TRACK_A.md` describe the run plan
and the fork-instrumentation work.

## Commit & GitHub guidelines

- **Author commits as the repo owner, never as Claude.** This repo's local git config is
  set to `u3anand <u3anand@uwaterloo.ca>`; commit with that identity.
- **Do NOT add agent trailers.** No `Co-Authored-By: Claude ...` line, no "Generated with
  Claude Code" in commit messages or PR bodies. Commits and PRs should read as authored by
  u3anand alone.
- Commit/push **only when asked.** If on `main`, branch first.
- **Message style** (match existing history): concise imperative subject, optionally
  `<area>: <subject>` (e.g. `check-node.sh: add cluster table`); a body with `-` bullets
  when the change spans multiple things. Reference the canonical docs by name when a change
  implements part of the plan (e.g. "implements Initial Plan → Instrumentation Q1").
- **Submodules / forks:** the three forks under `code/forks/` are pinned by submodule
  pointer + `code/b1/pins.txt`. After editing a fork, commit+push *in the fork* first, then
  bump the pointer and `pins.txt` in a separate notes-repo commit (see `code/b1/TRACK_A.md`).
- **Never commit:** `WatGPU.md` (gitignored), run/build artifacts, anything under the
  gitignore'd paths (`code/b1/runs|models|data`, `*.jsonl`, `.venv`, `__pycache__`).
