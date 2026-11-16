#!/usr/bin/env bash
# Inject a node failure: SIGKILL the containers, which also wipes the checkpoints in their memory.
# Usage: ./cluster/kill_nodes.sh node0 node1
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 node0 [node1 ...]" >&2; exit 1; }
cd "$(dirname "$0")"
date -u +"kill at %Y-%m-%dT%H:%M:%S.%3NZ: $*"
docker compose kill "$@"
