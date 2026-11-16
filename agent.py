"""Compute-node agent: emulated GPU, in-memory checkpoint store, peer replication, rank launcher."""
from __future__ import annotations

import argparse
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent import futures

import grpc
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from . import GRPC_OPTS
from . import failover_pb2 as pb
from . import failover_pb2_grpc as pbg
from . import rpc
from .store import CheckpointStore, GpuEmulator, Meta

log = logging.getLogger("agent")

M_PUT = Histogram("ckpt_put_seconds", "Time to receive and admit a checkpoint", ["kind"],
                  buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5))
M_REPL = Histogram("ckpt_replicate_seconds", "Time to push a checkpoint to one peer",
                   buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5))
M_REPL_FAIL = Counter("ckpt_replicate_failures_total", "Failed replication pushes")
M_BYTES = Counter("ckpt_bytes_total", "Checkpoint bytes admitted", ["kind"])
M_VERSION = Gauge("ckpt_latest_version", "Newest checkpoint version held on this node")
M_GPU_USED = Gauge("gpu_emulated_memory_used_bytes", "Emulated device memory in use")
M_GPU_TOTAL = Gauge("gpu_emulated_memory_total_bytes", "Emulated device memory capacity")
M_STEP = Gauge("train_step", "Last training step reported by the local rank")
M_RANK_UP = Gauge("train_rank_running", "1 while a training rank runs on this node")


class Agent(pbg.NodeAgentServicer):
    def __init__(self, node: str, peers: list[str], gpu_mem_bytes: int, log_dir: str):
        self.node = node
        self.gpu = GpuEmulator(gpu_mem_bytes)
        self.store = CheckpointStore(self.gpu)
        self.peers = [(p, rpc.agent_stub(p)) for p in peers]
        self.log_dir = log_dir
        self.proc: subprocess.Popen | None = None
        self.rank, self.epoch_id, self.step = -1, 0, 0
        self._lock = threading.Lock()
        self._repl_q: queue.Queue = queue.Queue(maxsize=4)
        M_GPU_TOTAL.set(gpu_mem_bytes)
        threading.Thread(target=self._replicate_loop, daemon=True).start()

    # ---- data plane ---------------------------------------------------------
    def Put(self, request_iterator, context):
        t0 = time.perf_counter()
        meta, parts = None, []
        for c in request_iterator:
            if c.HasField("meta"):
                meta = c.meta
            parts.append(c.data)
        if meta is None:
            return pb.PutReply(ok=False, error="stream carried no metadata")
        data = b"".join(parts)
        try:
            self.store.put(Meta(meta.job_id, meta.version, meta.step, meta.size_bytes, meta.sha256,
                                meta.origin_node), data)
        except Exception as e:  # integrity or out-of-memory: refuse, keep what we have
            log.warning("rejecting checkpoint v%d: %s", meta.version, e)
            return pb.PutReply(ok=False, error=str(e))
        kind = "replica" if meta.replica else "local"
        M_PUT.labels(kind).observe(time.perf_counter() - t0)
        M_BYTES.labels(kind).inc(len(data))
        M_VERSION.set(meta.version)
        M_GPU_USED.set(self.gpu.used)
        if not meta.replica:
            try:  # replication is asynchronous; training is never blocked on a slow peer
                self._repl_q.put_nowait((meta, data))
            except queue.Full:
                M_REPL_FAIL.inc()
        return pb.PutReply(ok=True)

    def _replicate_loop(self):
        while True:
            meta, data = self._repl_q.get()
            rmeta = pb.CheckpointMeta()
            rmeta.CopyFrom(meta)
            rmeta.replica = True
            for addr, stub in self.peers:
                t0 = time.perf_counter()
                try:
                    rpc.put(stub, rmeta, data, timeout=10)
                    M_REPL.observe(time.perf_counter() - t0)
                except Exception as e:
                    M_REPL_FAIL.inc()
                    log.warning("replication of v%d to %s failed: %s", meta.version, addr,
                                getattr(e, "code", lambda: e)())

    def Get(self, request, context):
        found = self.store.get(request.job_id, request.version)
        if found is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "no such checkpoint")
        meta, data = found
        pmeta = pb.CheckpointMeta(job_id=meta.job_id, version=meta.version, step=meta.step,
                                  size_bytes=meta.size_bytes, sha256=meta.sha256, origin_node=meta.origin_node)
        yield from rpc.chunks(pmeta, data)

    def Latest(self, request, context):
        meta = self.store.latest(request.job_id)
        if meta is None:
            return pb.LatestReply(found=False)
        return pb.LatestReply(found=True, meta=pb.CheckpointMeta(
            job_id=meta.job_id, version=meta.version, step=meta.step, size_bytes=meta.size_bytes,
            sha256=meta.sha256, origin_node=meta.origin_node))

    # ---- control plane ------------------------------------------------------
    def Ping(self, request, context):
        with self._lock:
            running = self.proc is not None and self.proc.poll() is None
            code = -999 if self.proc is None else (0 if running else self.proc.returncode)
            M_RANK_UP.set(1 if running else 0)
            return pb.NodeStatus(node=self.node, rank_running=running, rank=self.rank, step=self.step,
                                 epoch_id=self.epoch_id, gpu_mem_used=self.gpu.used,
                                 gpu_mem_total=self.gpu.total, rank_exit_code=code)

    def StartRank(self, r, context):
        with self._lock:
            self._stop_locked()
            cmd = [sys.executable, "-m", "failover_lab.trainer",
                   "--job-id", r.job_id, "--epoch-id", str(r.epoch_id), "--rank", str(r.rank),
                   "--world-size", str(r.world_size), "--master-addr", r.master_addr,
                   "--master-port", str(r.master_port), "--mode", r.mode,
                   "--ckpt-interval", str(r.ckpt_interval_s), "--agent", self.listen_addr,
                   "--node", self.node, "--total-steps", str(r.total_steps),
                   "--model-width", str(r.model_width or 512)]
            if r.restore_addr:
                cmd += ["--restore-addr", r.restore_addr, "--restore-version", str(r.restore_version)]
            if r.legacy_dir:
                cmd += ["--legacy-dir", r.legacy_dir]
            os.makedirs(self.log_dir, exist_ok=True)
            out = open(os.path.join(self.log_dir, f"{self.node}-rank.log"), "ab")
            self.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
            self.rank, self.epoch_id, self.step = r.rank, r.epoch_id, 0
            log.info("started rank %d/%d epoch %d", r.rank, r.world_size, r.epoch_id)
        return pb.PutReply(ok=True)

    def _stop_locked(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)

    def StopRank(self, request, context):
        with self._lock:
            self._stop_locked()
        return pb.PutReply(ok=True)

    def ReportProgress(self, request, context):
        if request.epoch_id == self.epoch_id:
            self.step = request.step
            M_STEP.set(request.step)
        return pb.Empty()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--node", required=True)
    ap.add_argument("--listen", default="0.0.0.0:50051")
    ap.add_argument("--advertise", default="", help="address the local rank uses to reach this agent")
    ap.add_argument("--peers", default="", help="comma-separated agents that receive replicas")
    ap.add_argument("--gpu-mem-mb", type=int, default=512)
    ap.add_argument("--metrics-port", type=int, default=0)
    ap.add_argument("--log-dir", default="experiments/run_logs")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s {a.node} %(message)s")
    agent = Agent(a.node, [p for p in a.peers.split(",") if p], a.gpu_mem_mb << 20, a.log_dir)
    port = a.listen.rsplit(":", 1)[1]
    agent.listen_addr = a.advertise or f"127.0.0.1:{port}"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16), options=GRPC_OPTS)
    pbg.add_NodeAgentServicer_to_server(agent, server)
    server.add_insecure_port(a.listen)
    server.start()
    if a.metrics_port:
        start_http_server(a.metrics_port)
    log.info("agent up on %s, replicas -> %s", a.listen, a.peers or "(none)")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
