#!/usr/bin/env python3
"""Assemble the **mixed drifting request trace** for B2 (Initial Plan 2 → step 2).

Pull prompts from public datasets in three domains and concatenate them into time-ordered
**segments** so the active domain shifts over wall-clock — this is the workload that is
supposed to make the hot MoE-expert set *drift* (the whole TierShift premise). Each emitted
request carries a ``segment_label`` (the Q6 domain tag).

Domains → sources:
  chat : ShareGPT  + LMSYS-Chat-1M   (--chat sharegpt,lmsys)
  code : HumanEval                   (--code humaneval)
  math : GSM8K                       (--math gsm8k)

Output is one JSONL request per line::

    {"req_id", "segment_label", "prompt", "max_tokens", "block_idx"}

ordered as the ``--segments`` list dictates; the special label ``mixed`` interleaves all
loaded domains round-robin. Downstream ``replay.py`` fires these at a vLLM endpoint and
records the per-segment wall-clock timeline.

Network/datasets are best-effort: any source that fails to load (gated, offline, missing
``datasets``) falls back to a deterministic **synthetic** generator for that domain, and the
substitution is logged. ``--synthetic`` forces the offline path for the whole trace (useful
for laptop pipeline tests). The trace is reproducible given ``--seed``.

Usage:
  python -m b2tel.make_trace --chat sharegpt,lmsys --code humaneval --math gsm8k \
      --segments chat,code,math,mixed --per-segment 200 --out traces/mixed.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

DOMAINS = ("chat", "code", "math")

# Output-length buckets for --length-mix (Research Plan 2: vary output length so the workload
# spans decode-light → decode-heavy and prefill/decode are both well represented). Inclusive
# (min, max) completion-token ranges; one is chosen per request and a value sampled within it.
LENGTH_BUCKETS = {
    "short": (16, 64),
    "medium": (128, 512),
    "long": (1024, 2048),
}

# Best-effort HF dataset coordinates per source name. (id, config, split, kind).
# NOTE: use NAMESPACED repo ids — datasets>=3 dropped script-based loading, so the bare
# "gsm8k"/"openai_humaneval" script datasets fail; the parquet-backed "openai/..." repos work.
# lmsys is gated → needs HF_TOKEN (huggingface-cli login) or it falls back to synthetic.
SOURCES = {
    "sharegpt": ("Aeala/ShareGPT_Vicuna_unfiltered", None, "train", "sharegpt"),
    "lmsys": ("lmsys/lmsys-chat-1m", None, "train", "lmsys"),
    "humaneval": ("openai/openai_humaneval", None, "test", "humaneval"),
    "gsm8k": ("openai/gsm8k", "main", "test", "gsm8k"),
}
SOURCE_DOMAIN = {
    "sharegpt": "chat", "lmsys": "chat", "humaneval": "code", "gsm8k": "math",
}


def _clean(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    return text[:max_chars].strip()


def _extract_prompts(kind: str, ds, n: int, max_chars: int) -> list[str]:
    """Pull up to ``n`` first-user-turn prompts from a loaded HF dataset."""
    out: list[str] = []
    for row in ds:
        if len(out) >= n:
            break
        p = ""
        if kind == "sharegpt":
            convs = row.get("conversations") or []
            for turn in convs:
                if turn.get("from") in ("human", "user"):
                    p = turn.get("value", "")
                    break
        elif kind == "lmsys":
            convo = row.get("conversation") or []
            for turn in convo:
                if turn.get("role") == "user":
                    p = turn.get("content", "")
                    break
        elif kind == "humaneval":
            p = "Complete this Python function:\n" + (row.get("prompt") or "")
        elif kind == "gsm8k":
            p = row.get("question") or ""
        p = _clean(p, max_chars)
        if len(p) >= 8:
            out.append(p)
    return out


def _load_source(name: str, n: int, max_chars: int) -> list[str]:
    """Load ``n`` prompts for a source, or [] on any failure (caller falls back)."""
    try:
        from datasets import load_dataset
    except Exception as e:  # noqa: BLE001
        print(f"[make_trace] datasets unavailable ({e}); synthetic for {name}", file=sys.stderr)
        return []
    ds_id, config, split, kind = SOURCES[name]
    try:
        ds = load_dataset(ds_id, config, split=split, streaming=True)
        prompts = _extract_prompts(kind, ds, n, max_chars)
        if prompts:
            print(f"[make_trace] {name}: loaded {len(prompts)} prompts from {ds_id}", file=sys.stderr)
        return prompts
    except Exception as e:  # noqa: BLE001
        print(f"[make_trace] {name} load failed ({e}); synthetic", file=sys.stderr)
        return []


# ---- synthetic fallbacks (deterministic, offline) --------------------------

_CHAT_TOPICS = [
    "the history of jazz", "how vaccines work", "planning a trip to Japan",
    "the rules of cricket", "why the sky is blue", "tips for growing tomatoes",
    "the plot of Hamlet", "how to brew good coffee", "the causes of WW1",
    "advice for a first marathon", "how compilers work", "the life of Ada Lovelace",
]
_CODE_TASKS = [
    "reverse a linked list", "check if a string is a palindrome",
    "compute the nth Fibonacci number", "merge two sorted arrays",
    "implement binary search", "find the longest common subsequence",
    "parse a CSV file", "debounce a function", "flatten a nested list",
    "implement an LRU cache", "validate a binary search tree", "rotate a matrix",
]


def _synth(domain: str, n: int, rng: random.Random) -> list[str]:
    out = []
    for i in range(n):
        if domain == "chat":
            t = rng.choice(_CHAT_TOPICS)
            out.append(f"Can you explain {t} in a few paragraphs?")
        elif domain == "code":
            t = rng.choice(_CODE_TASKS)
            out.append(f"Write a Python function to {t}. Include a docstring and example.")
        else:  # math
            a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 20)
            out.append(
                f"A shop sold {a} items on Monday and {b} on Tuesday, then returned {c}. "
                f"How many net items were sold? Show your reasoning step by step."
            )
    return out


def build_pools(args, rng: random.Random) -> dict[str, list[str]]:
    """Return {domain: [prompts]} large enough to fill every block of that domain."""
    # how many blocks reference each domain (mixed pulls from all)
    seg_list = [s.strip() for s in args.segments.split(",") if s.strip()]
    need = {d: 0 for d in DOMAINS}
    for seg in seg_list:
        if seg in DOMAINS:
            need[seg] += args.per_segment
        elif seg == "mixed":
            for d in DOMAINS:
                need[d] += args.per_segment  # mixed draws ~per_segment/len across all
    src_by_domain: dict[str, list[str]] = {d: [] for d in DOMAINS}
    chosen = {
        "chat": [s for s in args.chat.split(",") if s.strip()],
        "code": [s for s in args.code.split(",") if s.strip()],
        "math": [s for s in args.math.split(",") if s.strip()],
    }
    for domain in DOMAINS:
        want = max(need[domain], 1)
        pool: list[str] = []
        if not args.synthetic:
            srcs = [s for s in chosen[domain] if s in SOURCES]
            # request the full `want` from EACH source so a domain stays diverse even when
            # some sources fail (e.g. lmsys gated) — over-fetch is fine, take() uses `want`.
            for src in srcs:
                pool += _load_source(src, want + 16, args.max_chars)
        # prefer real prompts: only fall back to synthetic if NOTHING real loaded.
        # if a real source is small (e.g. HumanEval=164), take() cycles it to fill `want`.
        if not pool:
            pool = _synth(domain, want, rng)
            print(f"[make_trace] {domain}: 0 real -> 100% SYNTHETIC", file=sys.stderr)
        elif len(pool) < want:
            print(f"[make_trace] {domain}: {len(pool)} real prompts cycled to fill {want} "
                  f"(no synthetic)", file=sys.stderr)
        else:
            print(f"[make_trace] {domain}: {len(pool)} real prompts", file=sys.stderr)
        rng.shuffle(pool)
        src_by_domain[domain] = pool
    return src_by_domain


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the B2 mixed drifting request trace.")
    ap.add_argument("--chat", default="sharegpt,lmsys", help="chat sources (comma list)")
    ap.add_argument("--code", default="humaneval", help="code sources (comma list)")
    ap.add_argument("--math", default="gsm8k", help="math sources (comma list)")
    ap.add_argument("--segments", default="chat,code,math,mixed",
                    help="ordered segment blocks; 'mixed' interleaves all domains")
    ap.add_argument("--per-segment", type=int, default=200, help="requests per block")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="fixed completion cap per request (used when --length-mix is unset)")
    ap.add_argument("--length-mix", default="",
                    help="comma list of output-length buckets to sample per request "
                         f"(any of {','.join(LENGTH_BUCKETS)}); empty = fixed --max-tokens")
    ap.add_argument("--max-chars", type=int, default=8000,
                    help="prompt truncation cap; large so natural prompt lengths (hence prefill "
                         "sizes) vary across domains")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true", help="force offline synthetic prompts")
    ap.add_argument("--out", required=True, help="output trace .jsonl")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    pools = build_pools(args, rng)
    cursors = {d: 0 for d in DOMAINS}

    # output-length sampler (independent rng so it's reproducible regardless of pool building)
    len_rng = random.Random(args.seed + 1)
    mix = [b.strip() for b in args.length_mix.split(",") if b.strip()]
    bad = [b for b in mix if b not in LENGTH_BUCKETS]
    if bad:
        print(f"[make_trace] ignoring unknown length buckets {bad}", file=sys.stderr)
    mix = [b for b in mix if b in LENGTH_BUCKETS]

    def pick_length() -> tuple[int, str | None]:
        if not mix:
            return args.max_tokens, None
        bucket = len_rng.choice(mix)
        lo, hi = LENGTH_BUCKETS[bucket]
        return len_rng.randint(lo, hi), bucket

    def take(domain: str) -> str:
        pool = pools[domain]
        p = pool[cursors[domain] % len(pool)]
        cursors[domain] += 1
        return p

    seg_list = [s.strip() for s in args.segments.split(",") if s.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    req_id = 0
    with open(out_path, "w") as fh:
        for block_idx, seg in enumerate(seg_list):
            for i in range(args.per_segment):
                if seg == "mixed":
                    domain = DOMAINS[i % len(DOMAINS)]
                elif seg in DOMAINS:
                    domain = seg
                else:
                    print(f"[make_trace] unknown segment '{seg}', skipping", file=sys.stderr)
                    break
                max_tokens, len_bucket = pick_length()
                rec = {
                    "req_id": req_id,
                    "segment_label": seg,
                    "domain": domain,
                    "prompt": take(domain),
                    "max_tokens": max_tokens,
                    "block_idx": block_idx,
                }
                if len_bucket is not None:
                    rec["len_bucket"] = len_bucket
                fh.write(json.dumps(rec) + "\n")
                req_id += 1
                counts[seg] = counts.get(seg, 0) + 1

    print(f"[make_trace] wrote {req_id} requests to {out_path}")
    print(f"[make_trace] per-segment counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
