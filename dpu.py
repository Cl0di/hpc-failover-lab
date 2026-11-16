"""Mock DPU agent.

Stands in for the out-of-band health monitoring a BlueField-class DPU can
provide: it sits beside one host, heartbeats it over the fabric, and tells
the coordinator the moment the host stops answering. It runs in its own
process/container so it survives the death of the host it watches.
"""
from __future__ import annotations

import argparse
import logging
import time

from prometheus_client import Gauge, Histogram, start_http_server

from . import failover_pb2 as pb
from . import rpc

log = logging.getLogger("dpu")
M_HB = Histogram("dpu_heartbeat_seconds", "Heartbeat round-trip to the host agent",
                 buckets=(.0005, .001, .0025, .005, .01, .025, .05, .1, .25))
M_ALIVE = Gauge("dpu_host_alive", "1 while the watched host answers heartbeats")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--node", required=True, help="name of the watched host")
    ap.add_argument("--watch", required=True, help="host agent address")
    ap.add_argument("--coordinator", required=True)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--timeout", type=float, default=0.25, help="per-heartbeat deadline")
    ap.add_argument("--misses", type=int, default=3)
    ap.add_argument("--metrics-port", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s dpu-{a.node} %(message)s")
    if a.metrics_port:
        start_http_server(a.metrics_port)
    host, coord = rpc.agent_stub(a.watch), rpc.coordinator_stub(a.coordinator)
    misses, alive, seen = 0, True, False
    while True:
        t0 = time.perf_counter()
        try:
            host.Ping(pb.Empty(), timeout=a.timeout)
            M_HB.observe(time.perf_counter() - t0)
            misses, seen = 0, True
            if not alive:
                alive = True
                coord.ReportNode(pb.NodeEvent(node=a.node, reporter=f"dpu-{a.node}", alive=True,
                                              observed_at=time.time()), timeout=1)
        except Exception:
            misses += 1
            if seen and alive and misses >= a.misses:
                alive = False
                log.warning("host unresponsive after %d missed heartbeats", misses)
                try:
                    coord.ReportNode(pb.NodeEvent(node=a.node, reporter=f"dpu-{a.node}", alive=False,
                                                  observed_at=time.time()), timeout=1)
                except Exception as e:
                    log.error("could not reach coordinator: %s", e)
                    alive = True  # retry the report on the next miss
        M_ALIVE.set(1 if alive else 0)
        time.sleep(max(0.0, a.interval - (time.perf_counter() - t0)))


if __name__ == "__main__":
    main()
