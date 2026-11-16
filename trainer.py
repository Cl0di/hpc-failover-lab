"""One rank of a PyTorch DistributedDataParallel job with a synthetic workload.

The model and data are deliberately trivial: the lab is about what happens
around the training loop (checkpoint, replicate, detect, restore), not about
the loss curve. Every rank derives its batch from (step, rank), so a job
restored at step S replays exactly the batches it would have seen.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from . import failover_pb2 as pb
from . import rpc

FEATURES, BATCH = 128, 64


def emit(**kv):
    print(json.dumps({"t": time.time(), **kv}), flush=True)


def build_model(width: int) -> nn.Module:
    torch.manual_seed(0)  # identical initial weights on every rank
    return nn.Sequential(nn.Linear(FEATURES, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                         nn.Linear(width, 1))


def batch(step: int, rank: int):
    g = torch.Generator().manual_seed(step * 1009 + rank)
    x = torch.randn(BATCH, FEATURES, generator=g)
    y = torch.sin(x[:, :8].sum(dim=1, keepdim=True))
    return x, y


def serialize(model, opt, step: int) -> bytes:
    buf = io.BytesIO()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, buf)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--epoch-id", type=int, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--master-addr", required=True)
    ap.add_argument("--master-port", type=int, required=True)
    ap.add_argument("--mode", choices=["inmem", "legacy"], required=True)
    ap.add_argument("--ckpt-interval", type=float, required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--node", required=True)
    ap.add_argument("--total-steps", type=int, default=10**9)
    ap.add_argument("--model-width", type=int, default=512)
    ap.add_argument("--restore-addr", default="")
    ap.add_argument("--restore-version", type=int, default=0)
    ap.add_argument("--legacy-dir", default="")
    a = ap.parse_args()

    torch.set_num_threads(1)
    agent = rpc.agent_stub(a.agent)
    t_boot = time.time()

    model = build_model(a.model_width)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    step = 0

    # ---- restore (every rank loads the same state; DDP does not broadcast weights) ----
    t0 = time.time()
    state = None
    if a.mode == "inmem" and a.restore_addr:
        meta, data = rpc.fetch(rpc.agent_stub(a.restore_addr), a.job_id, a.restore_version)
        state = torch.load(io.BytesIO(data), weights_only=True)
        source = f"memory:{a.restore_addr}"
    elif a.mode == "legacy" and a.legacy_dir:
        path = os.path.join(a.legacy_dir, f"{a.job_id}.pt")
        if os.path.exists(path):
            state = torch.load(path, weights_only=True)
            source = f"disk:{path}"
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = int(state["step"])
        emit(event="restored", rank=a.rank, step=step, source=source, seconds=time.time() - t0)

    dist.init_process_group("gloo", init_method=f"tcp://{a.master_addr}:{a.master_port}",
                            rank=a.rank, world_size=a.world_size, timeout=timedelta(seconds=120))
    ddp = DDP(model)
    loss_fn = nn.MSELoss()
    emit(event="rank_ready", rank=a.rank, world=a.world_size, step=step, boot_seconds=time.time() - t_boot)

    last_ckpt = last_report = time.time()
    while step < a.total_steps:
        x, y = batch(step, a.rank)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(ddp(x), y)
        loss.backward()  # gradients are averaged across ranks here
        opt.step()
        step += 1

        now = time.time()
        if now - last_report >= 0.1:
            last_report = now
            try:
                agent.ReportProgress(pb.Progress(epoch_id=a.epoch_id, step=step), timeout=1)
            except Exception:
                pass
        if a.rank == 0 and now - last_ckpt >= a.ckpt_interval:
            t0 = time.time()
            data = serialize(model, opt, step)
            if a.mode == "inmem":
                rpc.put(agent, rpc.make_meta(a.job_id, step, step, data, a.node), data)
            else:
                os.makedirs(a.legacy_dir, exist_ok=True)
                tmp = os.path.join(a.legacy_dir, f".{a.job_id}.pt.tmp")
                with open(tmp, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, os.path.join(a.legacy_dir, f"{a.job_id}.pt"))
            last_ckpt = time.time()
            emit(event="checkpoint", mode=a.mode, step=step, bytes=len(data), seconds=last_ckpt - t0,
                 loss=float(loss))
    agent.ReportProgress(pb.Progress(epoch_id=a.epoch_id, step=step), timeout=1)
    emit(event="done", rank=a.rank, step=step)
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
