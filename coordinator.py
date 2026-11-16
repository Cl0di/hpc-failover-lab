"""Failover coordinator: launches the job, learns about node loss, relaunches on the survivors.

Two detection paths are modelled:
  dpu   node loss is reported by the mock DPU agents (sub-second heartbeats)
  poll  the coordinator notices a node only after it has been silent for
        --poll-timeout seconds, like a scheduler's node heartbeat timeout
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent import futures

import grpc
from prometheus_client import Counter, Gauge, start_http_server

from . import GRPC_OPTS
from . import failover_pb2 as pb
from . import failover_pb2_grpc as pbg
from . import rpc

log = logging.getLogger("coordinator")
M_FAILOVERS = Counter("failovers_total", "Completed failovers")
M_ALIVE = Gauge("nodes_alive", "Nodes the coordinator believes are alive")
M_EPOCH = Gauge("job_epoch", "Current job incarnation")
M_STEP = Gauge("job_step", "Highest training step seen in the current incarnation")
M_DETECT = Gauge("failover_last_detect_to_running_seconds", "Detection to all ranks relaunched, last failover")
M_LOST = Gauge("failover_last_lost_steps", "Steps that had to be replayed after the last failover")


class Coordinator(pbg.CoordinatorServicer):
    def __init__(self, a):
        self.a = a
        self.nodes = dict(kv.split("=", 1) for kv in a.nodes.split(","))
        self.stubs = {n: rpc.agent_stub(addr) for n, addr in self.nodes.items()}
        self.alive = list(self.nodes)
        self.epoch = 0
        self.step = 0
        self.done = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._events = open(a.events, "a", buffering=1)
        self._pool = futures.ThreadPoolExecutor(max_workers=max(8, 2 * len(self.nodes)))
        self._last_ok = {n: time.time() for n in self.nodes}

    def event(self, name, **kv):
        self._events.write(json.dumps({"t": time.time(), "event": name, **kv}) + "\n")

    # ---- failure intake -----------------------------------------------------
    def ReportNode(self, request, context):
        if self.a.detect == "dpu" and not request.alive:
            self.node_failed(request.node, request.reporter)
        return pb.Empty()

    def node_failed(self, node, source):
        with self._lock:
            if node not in self.alive:
                return
            self.alive.remove(node)
        log.warning("node %s declared failed by %s", node, source)
        self.event("node_failed", node=node, source=source)
        M_ALIVE.set(len(self.alive))
        self._wake.set()

    # ---- job control --------------------------------------------------------
    def launch(self):
        """Start one incarnation of the job on every node currently alive."""
        a = self.a
        with self._lock:
            self.epoch += 1
            epoch, members = self.epoch, list(self.alive)
        if not members:
            raise RuntimeError("no nodes left")
        restore_addr, restore_version, restore_step = "", 0, 0
        if a.mode == "inmem":
            best = None
            for n, reply in self._fanout(members, lambda s: s.Latest(pb.LatestRequest(job_id=a.job_id), timeout=1)):
                if reply is not None and reply.found and (best is None or reply.meta.version > best[1].version):
                    best = (n, reply.meta)
            if best:
                restore_addr, restore_version, restore_step = self.nodes[best[0]], best[1].version, best[1].step
        master = self.nodes[members[0]].rsplit(":", 1)[0]
        self.event("launch_begin", epoch=epoch, members=members, restore_from=restore_addr,
                   restore_step=restore_step)

        def start(rank):
            return self.stubs[members[rank]].StartRank(pb.StartRankRequest(
                job_id=a.job_id, epoch_id=epoch, rank=rank, world_size=len(members), master_addr=master,
                master_port=a.base_port + epoch, mode=a.mode, ckpt_interval_s=a.ckpt_interval,
                restore_addr=restore_addr, restore_version=restore_version, legacy_dir=a.legacy_dir,
                total_steps=a.total_steps, model_width=a.model_width), timeout=5)

        failed = []
        for rank, fut in [(r, self._pool.submit(start, r)) for r in range(len(members))]:
            try:
                fut.result()
            except Exception:
                failed.append(members[rank])
        for n in failed:  # a node died between detection and relaunch: go round again
            self.node_failed(n, "launch")
        self.event("launch_done", epoch=epoch, world=len(members), failed=failed)
        M_EPOCH.set(epoch)
        return epoch

    def _fanout(self, names, call):
        futs = {n: self._pool.submit(call, self.stubs[n]) for n in names}
        for n, f in futs.items():
            try:
                yield n, f.result()
            except Exception:
                yield n, None

    def failover_loop(self):
        while not self.done.is_set():
            if not self._wake.wait(0.2):
                continue
            t_detect = time.time()
            time.sleep(self.a.settle)  # let reports about simultaneous failures arrive together
            self._wake.clear()
            step_before = self.step
            self.event("failover_begin", alive=list(self.alive), step=step_before)
            # Stop the survivors, and fence the nodes declared dead (best effort) so a
            # node that was only slow cannot keep running a stale rank.
            alive = set(self.alive)
            futs = [self._pool.submit(self.stubs[n].StopRank, pb.Empty(), timeout=10 if n in alive else 0.5)
                    for n in self.nodes]
            futures.wait(futs)
            self.step = 0
            self.launch()
            M_FAILOVERS.inc()
            M_DETECT.set(time.time() - t_detect)

    def monitor_loop(self):
        a = self.a
        while not self.done.is_set():
            t0 = time.time()
            with self._lock:  # snapshot: a failover may bump the epoch while we are polling
                epoch, members = self.epoch, list(self.alive)
            step, finished = 0, 0
            for n, st in self._fanout(members, lambda s: s.Ping(pb.Empty(), timeout=0.3)):
                if st is None:
                    if a.detect == "poll" and time.time() - self._last_ok[n] > a.poll_timeout:
                        self.node_failed(n, "poll-timeout")
                    continue
                self._last_ok[n] = time.time()
                if st.epoch_id == epoch:
                    step = max(step, st.step)
                    if not st.rank_running and st.rank_exit_code == 0:
                        finished += 1
                    elif not st.rank_running and st.rank_exit_code not in (0, -999) and a.failover_on_rank_exit:
                        self._wake.set()
            if epoch == self.epoch and step > self.step:
                self.step = step
                M_STEP.set(step)
            self.event("progress", epoch=epoch, step=step)
            if members and finished == len(members) and step >= a.total_steps:
                self.event("job_done", step=step)
                self.done.set()
            time.sleep(max(0.0, a.sample_interval - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", required=True, help="name=host:port,name=host:port,...")
    ap.add_argument("--listen", default="0.0.0.0:50050")
    ap.add_argument("--job-id", default="job-1")
    ap.add_argument("--mode", choices=["inmem", "legacy"], default="inmem")
    ap.add_argument("--detect", choices=["dpu", "poll"], default="dpu")
    ap.add_argument("--ckpt-interval", type=float, default=15.0)
    ap.add_argument("--poll-timeout", type=float, default=5.0)
    ap.add_argument("--sample-interval", type=float, default=0.1)
    ap.add_argument("--settle", type=float, default=0.2)
    ap.add_argument("--legacy-dir", default="")
    ap.add_argument("--total-steps", type=int, default=10**9)
    ap.add_argument("--model-width", type=int, default=512)
    ap.add_argument("--base-port", type=int, default=29500)
    ap.add_argument("--events", default="experiments/run_logs/events.jsonl")
    ap.add_argument("--metrics-port", type=int, default=0)
    ap.add_argument("--failover-on-rank-exit", action="store_true",
                    help="also fail over when a rank process dies while its node stays up")
    ap.add_argument("--wait-nodes", type=float, default=60.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s coordinator %(message)s")

    c = Coordinator(a)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16), options=GRPC_OPTS)
    pbg.add_CoordinatorServicer_to_server(c, server)
    server.add_insecure_port(a.listen)
    server.start()
    if a.metrics_port:
        start_http_server(a.metrics_port)

    deadline = time.time() + a.wait_nodes
    while time.time() < deadline:
        if all(st is not None for _, st in c._fanout(list(c.nodes), lambda s: s.Ping(pb.Empty(), timeout=0.5))):
            break
        time.sleep(0.2)
    else:
        raise SystemExit("not all node agents came up")
    c._last_ok = {n: time.time() for n in c.nodes}
    M_ALIVE.set(len(c.alive))
    c.event("job_start", mode=a.mode, detect=a.detect, nodes=list(c.nodes), ckpt_interval=a.ckpt_interval)
    c.launch()
    threading.Thread(target=c.failover_loop, daemon=True).start()
    c.monitor_loop()
    server.stop(0)


if __name__ == "__main__":
    main()
