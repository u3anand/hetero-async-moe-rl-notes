#!/usr/bin/env bash
# Pull + R2-setup a set of SWE-bench udocker images so episodes don't pay cold-start.
# Images: ghcr.io/epoch-research/swe-bench.eval.x86_64.<repo>__<id>
#   prewarm.sh images.txt        # one image ref per line
set -euo pipefail
LIST=${1:?images.txt}
export UDOCKER_DIR=${UDOCKER_DIR:-/u3/u3anand/b1/sandbox/.udocker}
UD=${UDOCKER:-$HOME/.local/bin/udocker}

while read -r image; do
  [ -z "$image" ] && continue
  name=$(echo "$image" | tr '/:.' '___')
  echo "== $image -> $name =="
  "$UD" pull "$image"
  "$UD" create --name="$name" "$image" || true
  "$UD" setup --execmode=R2 "$name"     # runc/proot; fall back to P2 if R2 fails broadly
done < "$LIST"
