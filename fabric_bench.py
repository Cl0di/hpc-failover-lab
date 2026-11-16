"""Measure checkpoint push throughput through the gRPC data plane.

Starts two agents on localhost (or targets --addr), pushes a payload of
--mb megabytes --repeat times and reports MB/s for the local write and for
the replica arriving on the peer.

    python -m experiments.fabric_bench --mb 64
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from concurrent import futures

import grpc

from failover_lab import GRPC_OPTS, rpc
from failover_lab import failover_pb2 as pb
from failover_lab import failover_pb2_grpc as pbg
from failover_lab.agent import Agent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mb", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=5)
    a = ap.parse_args()

    servers, addrs = [], []
    for _ in range(2):
        s = grpc.server(futures.ThreadPoolExecutor(max_workers=8), options=GRPC_OPTS)
        addrs.append(f"127.0.0.1:{s.add_insecure_port('127.0.0.1:0')}")
        servers.append(s)
    for i, s in enumerate(servers):
        agent = Agent(f"bench{i}", [addrs[1]] if i == 0 else [], (3 * a.mb + 16) << 20, "/tmp")
        agent.listen_addr = addrs[i]
        pbg.add_NodeAgentServicer_to_server(agent, s)
        s.start()
    src, dst = rpc.agent_stub(addrs[0]), rpc.agent_stub(addrs[1])

    local, replica = [], []
    for v in range(1, a.repeat + 1):
        data = os.urandom(a.mb << 20)
        meta = rpc.make_meta("bench", v, v, data, "bench0")
        t0 = time.perf_counter()
        rpc.put(src, meta, data)
        t1 = time.perf_counter()
        while True:
            r = dst.Latest(pb.LatestRequest(job_id="bench"))
            if r.found and r.meta.version == v:
                break
            time.sleep(0.001)
        t2 = time.perf_counter()
        local.append(a.mb / (t1 - t0))
        replica.append(a.mb / (t2 - t0))
    print(f"payload {a.mb} MB x {a.repeat}")
    print(f"local write (hash + gRPC + admit): {statistics.median(local):8.0f} MB/s median")
    print(f"write + replica on peer:           {statistics.median(replica):8.0f} MB/s median")
    for s in servers:
        s.stop(0)


if __name__ == "__main__":
    main()
