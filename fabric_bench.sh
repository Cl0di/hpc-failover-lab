#!/usr/bin/env bash
# Raw TCP throughput between two node containers with iperf3.
# This is the ceiling for checkpoint replication on the emulated fabric. It is
# plain TCP over a Docker bridge, not RDMA: use it to sanity-check the lab
# network, not to predict InfiniBand numbers.
set -euo pipefail
SRC=${1:-node0}; DST=${2:-node1}
cd "$(dirname "$0")"
docker compose exec -d "$DST" iperf3 -s -1
sleep 1
docker compose exec "$SRC" iperf3 -c "$DST" -t 5 -P 4
