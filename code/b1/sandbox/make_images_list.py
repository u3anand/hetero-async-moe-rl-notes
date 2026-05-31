"""Generate a SWE-bench udocker image list for the sandbox spike (B2) / reward path.

Samples N instances spanning distinct repos (for tail coverage) and writes one image ref
per line. Default formats the epoch-research ghcr naming used for SWE-bench:
    ghcr.io/epoch-research/swe-bench.eval.x86_64.<instance_id>

NOTE: epoch-research images cover SWE-bench (Verified/full). Our training dataset is
SWE-rebench, a *different* instance set — for the B2 *spike* (validating udocker mechanics +
timing) any SWE-bench images are fine; for the real reward run, confirm the image source/registry
that matches SWE-rebench instances (the SWE-rebench project hosts its own images). Verify a ref
resolves before a full prewarm:  udocker pull <ref>

Usage:
    # from a known instance-id list (one id like 'astropy__astropy-12907' per line):
    python make_images_list.py --ids verified_ids.txt --n 15 --out images.txt
    # or sample from the downloaded SWE-rebench dataset parquet:
    python make_images_list.py --dataset /u3/u3anand/b1/data/swe-rebench --n 15 --out images.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

REGISTRY = "ghcr.io/epoch-research/swe-bench.eval.x86_64"


def ref(instance_id: str) -> str:
    return f"{REGISTRY}.{instance_id}"


def from_ids(path: str) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]


def from_dataset(path: str) -> list[tuple[str, str]]:
    """Return (instance_id, repo) pairs from a SWE-rebench parquet dir."""
    import polars as pl
    df = pl.read_parquet(str(Path(path) / "**/*.parquet"))
    cols = df.columns
    id_col = next((c for c in ("instance_id", "issue_id", "id") if c in cols), None)
    repo_col = next((c for c in ("repo", "repository") if c in cols), None)
    if id_col is None:
        raise SystemExit(f"no instance-id column in {cols}")
    rows = df.select([id_col, repo_col] if repo_col else [id_col]).unique().rows()
    return [(r[0], r[1] if repo_col else "") for r in rows]


def sample_spanning_repos(pairs: list[tuple[str, str]], n: int) -> list[str]:
    """Pick n ids, at most one per repo first (deterministic: sorted, no RNG)."""
    seen, out = set(), []
    for iid, repo in sorted(pairs):
        key = repo or iid
        if key in seen:
            continue
        seen.add(key); out.append(iid)
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids"); ap.add_argument("--dataset")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", default="images.txt")
    a = ap.parse_args()
    if a.ids:
        ids = from_ids(a.ids)[: a.n]
    elif a.dataset:
        ids = sample_spanning_repos(from_dataset(a.dataset), a.n)
    else:
        raise SystemExit("pass --ids <file> or --dataset <dir>")
    Path(a.out).write_text("\n".join(ref(i) for i in ids) + "\n")
    print(f"wrote {len(ids)} image refs → {a.out}  (verify one with: udocker pull <ref>)")


if __name__ == "__main__":
    main()
