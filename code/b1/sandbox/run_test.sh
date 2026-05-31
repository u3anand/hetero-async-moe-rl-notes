#!/usr/bin/env bash
# Run a command inside a prewarmed udocker SWE-bench container (the reward path).
# Called by mini-swe-agent's UdockerEnvironment. Prints stdout+stderr; exit code = command's.
#   run_test.sh <container_name> <cwd> <command...>
set -uo pipefail
CONTAINER=${1:?container}; CWD=${2:?cwd}; shift 2
export UDOCKER_DIR=${UDOCKER_DIR:-/u3/u3anand/b1/sandbox/.udocker}
UD=${UDOCKER:-$HOME/.local/bin/udocker}
exec "$UD" run --rm -w "$CWD" "$CONTAINER" bash -lc "$*"
