"""Fault-injection experiment: kill nodes mid-training and measure how long recovery takes.

Runs the whole lab as local processes (no Docker needed): N node agents, N mock
DPU agents and one coordinator. A node "fails" by SIGKILL of its process group
(agent + training rank), which also destroys the checkpoints held in its memory.

Two configurations are compared:
  legacy  checkpoint to a shared directory every hour, node loss noticed by
          scheduler-style heartbeat timeout, full restart from disk
  inmem   checkpoint to node memory every 15 minutes with replicas on two
          peers, node loss reported by the DPU agents, restart from a replica

Wall-clock intervals are compressed by --time-scale (default 60: one lab second
stands for one cluster minute). Results are reported in lab seconds.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time

PY = sys.executable


def spawn(args, log_path, new_group=False):
    return subprocess.Popen([PY, "-m", *args], stdout=open(log_path, "ab"), stderr=subprocess.STDOUT,
                            start_new_session=new_group)


def read_events(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass  # partially written last line
    return out


def rank_events(run_dir):
    out = []
    for path in glob.glob(os.path.join(run_dir, "*-rank.log")):
        with open(path, errors="replace") as f:
            for line in f:
                if line.startswith("{"):
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    return sorted(out, key=lambda e: e["t"])


def run_trial(mode, trial, a, u):
    run_dir = os.path.abspath(os.path.join(a.out, "run_logs", f"{mode}-trial{trial}"))
    os.makedirs(run_dir, exist_ok=True)
    for f in glob.glob(os.path.join(run_dir, "*")):
        if os.path.isfile(f):
            os.remove(f)
    port0 = 50100 + 100 * trial + (0 if mode == "inmem" else 50)
    coord_addr = f"127.0.0.1:{port0 - 1}"
    names = [f"node{i}" for i in range(a.nodes)]
    addrs = [f"127.0.0.1:{port0 + i}" for i in range(a.nodes)]
    legacy_ckpt, inmem_ckpt, poll_timeout = 3600 / a.time_scale, 900 / a.time_scale, 300 / a.time_scale
    events_path = os.path.join(run_dir, "events.jsonl")

    agents, others = [], []
    try:
        for i, (n, addr) in enumerate(zip(names, addrs)):
            peers = ",".join(addrs[(i + k) % a.nodes] for k in range(1, a.replicas + 1))
            agents.append(spawn(["failover_lab.agent", "--node", n, "--listen", addr, "--advertise", addr,
                                 "--peers", peers, "--gpu-mem-mb", str(a.gpu_mem_mb), "--log-dir", run_dir],
                                os.path.join(run_dir, f"{n}-agent.log"), new_group=True))
        for n, addr in zip(names, addrs):
            others.append(spawn(["failover_lab.dpu", "--node", n, "--watch", addr, "--coordinator", coord_addr],
                                os.path.join(run_dir, f"dpu-{n}.log")))
        coord = spawn(["failover_lab.coordinator", "--nodes", ",".join(f"{n}={ad}" for n, ad in zip(names, addrs)),
                       "--listen", coord_addr, "--mode", mode, "--detect", "dpu" if mode == "inmem" else "poll",
                       "--ckpt-interval", str(inmem_ckpt if mode == "inmem" else legacy_ckpt),
                       "--poll-timeout", str(poll_timeout), "--legacy-dir", os.path.join(run_dir, "shared_fs"),
                       "--model-width", str(a.model_width), "--base-port", str(29500 + 20 * trial + (0 if mode == "inmem" else 10)),
                       "--events", events_path], os.path.join(run_dir, "coordinator.log"))
        others.append(coord)

        def wait_for(pred, timeout, what):
            deadline = time.time() + timeout
            while time.time() < deadline:
                if coord.poll() is not None:
                    raise RuntimeError(f"coordinator exited while waiting for {what}")
                hit = pred(read_events(events_path))
                if hit is not None:
                    return hit
                time.sleep(0.1)
            raise TimeoutError(f"timed out waiting for {what}")

        t_train = wait_for(lambda ev: next((e["t"] for e in ev if e["event"] == "progress" and e["step"] > 0), None),
                           180, "training to start")
        t_kill_planned = t_train + u * legacy_ckpt
        time.sleep(max(0.0, t_kill_planned - time.time()))

        victims = list(range(a.kill))  # node0 runs rank 0 (checkpoint origin); node1 holds one of its replicas
        t_kill = time.time()
        for i in victims:
            os.killpg(agents[i].pid, signal.SIGKILL)
        ev = read_events(events_path)
        step_at_kill = max((e["step"] for e in ev if e["event"] == "progress" and e["t"] <= t_kill), default=0)

        def regained(ev):
            return next((e["t"] for e in ev if e["event"] == "progress" and e["t"] > t_kill
                         and e["epoch"] > 1 and e["step"] >= step_at_kill), None)

        t_regain = wait_for(regained, 4 * legacy_ckpt + 240, "progress to be regained")
        time.sleep(1.0)
    finally:
        for p in others:
            p.terminate()
        for p in agents:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in others + agents:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()

    ev = read_events(events_path)
    rk = rank_events(run_dir)
    failed = [e for e in ev if e["event"] == "node_failed"]
    t_detect = min(e["t"] for e in failed)
    final_epoch = max(e["epoch"] for e in ev if e["event"] == "progress")
    restored = [e for e in rk if e["event"] == "restored" and e["t"] > t_kill]
    restore_step = max((e["step"] for e in restored), default=0)
    if mode == "inmem" and not restored:
        raise RuntimeError(f"{mode} trial {trial}: job did not restore from a replica, see {run_dir}")
    t_resume = next(e["t"] for e in ev if e["event"] == "progress" and e["epoch"] == final_epoch
                    and e["step"] > restore_step)
    ckpts = [e for e in rk if e["event"] == "checkpoint"]
    if not [e for e in ckpts if e["t"] < t_kill]:
        raise RuntimeError(f"{mode} trial {trial}: no checkpoint was taken before the kill, see {run_dir}")
    pre = [e for e in ev if e["event"] == "progress" and e["epoch"] == 1 and e["step"] > 0 and e["t"] <= t_kill]
    rate_before = (pre[-1]["step"] - pre[0]["step"]) / max(pre[-1]["t"] - pre[0]["t"], 1e-9)
    return {
        "mode": mode, "trial": trial, "nodes": a.nodes, "killed": [names[i] for i in victims],
        "false_positive_failures": sorted({e["node"] for e in failed} - {names[i] for i in victims}),
        "kill_after_train_start_s": t_kill - t_train,
        "step_at_kill": step_at_kill, "restore_step": restore_step, "lost_steps": step_at_kill - restore_step,
        "restore_source": restored[0]["source"].split(":")[0] if restored else "none",
        "detect_s": t_detect - t_kill,
        "relaunch_s": t_resume - t_detect,
        "replay_s": max(0.0, t_regain - t_resume),
        "recovery_s": t_regain - t_kill,
        "survivor_world_size": a.nodes - len({e["node"] for e in failed}),
        "steps_per_s_before": rate_before,
        "checkpoints_taken": len(ckpts),
        "checkpoint_bytes": ckpts[-1]["bytes"] if ckpts else 0,
        "checkpoint_write_s_mean": statistics.mean(e["seconds"] for e in ckpts) if ckpts else None,
    }


def fmt(xs, nd=1):
    if not xs:
        return "n/a"
    return f"{statistics.mean(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})"


def write_report(results, a, path):
    rows = [("Checkpoint interval (lab s)", lambda m: f"{(900 if m == 'inmem' else 3600) / a.time_scale:g}"),
            ("Failure detection (s)", lambda m: fmt(col(m, "detect_s"), 2)),
            ("Relaunch + restore (s)", lambda m: fmt(col(m, "relaunch_s"))),
            ("Replay of lost work (s)", lambda m: fmt(col(m, "replay_s"))),
            ("**Total recovery (s)**", lambda m: "**" + fmt(col(m, "recovery_s")) + "**"),
            ("Steps lost", lambda m: fmt(col(m, "lost_steps"), 0)),
            ("Checkpoint write (ms)", lambda m: fmt([1000 * x for x in col(m, "checkpoint_write_s_mean")], 0)),
            ("Restored from", lambda m: ", ".join(sorted({r["restore_source"] for r in results if r["mode"] == m})))]

    def col(m, k):
        return [r[k] for r in results if r["mode"] == m and r[k] is not None]

    modes = [m for m in ("legacy", "inmem") if any(r["mode"] == m for r in results)]
    title = {"legacy": "Baseline (disk, scheduler timeout)", "inmem": "Proposed (memory replicas, DPU detection)"}
    r0 = results[0]
    lines = [f"# Failover experiment: {r0['nodes']} nodes, {len(r0['killed'])} killed simultaneously", "",
             f"Trials per configuration: {a.trials}. Time scale 1:{a.time_scale:g}. "
             f"Checkpoint size {r0['checkpoint_bytes'] / 1e6:.1f} MB. Values are mean (min–max) in lab seconds.", "",
             "| Metric | " + " | ".join(title[m] for m in modes) + " |", "|---|" + "---|" * len(modes)]
    for name, f in rows:
        lines.append(f"| {name} | " + " | ".join(f(m) for m in modes) + " |")
    fp = [r for r in results if r["false_positive_failures"]]
    lines += ["", f"False-positive node failures: {len(fp)} of {len(results)} runs."]
    if len(modes) == 2:
        sp = statistics.mean(col("legacy", "recovery_s")) / statistics.mean(col("inmem", "recovery_s"))
        lines += ["", f"Mean recovery is {sp:.1f}x faster with the proposed system in this setup."]
    lines += ["", "Recovery = time from the kill until the job is back at the step it had reached when the nodes died.",
              "Raw per-trial numbers are in `results.json`.", ""]
    open(path, "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--kill", type=int, default=2)
    ap.add_argument("--replicas", type=int, default=2)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["legacy", "inmem"], choices=["legacy", "inmem"])
    ap.add_argument("--time-scale", type=float, default=60.0)
    ap.add_argument("--model-width", type=int, default=512)
    ap.add_argument("--gpu-mem-mb", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="experiments/results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    offsets = [rng.uniform(1.15, 1.85) for _ in range(a.trials)]  # kill time, in legacy checkpoint intervals
    results = []
    for trial, u in enumerate(offsets):
        for mode in a.modes:  # same kill offset for both modes: paired comparison
            r = run_trial(mode, trial, a, u)
            results.append(r)
            print(json.dumps(r), flush=True)
            json.dump({"args": vars(a), "results": results}, open(os.path.join(a.out, "results.json"), "w"), indent=2)
            write_report(results, a, os.path.join(a.out, "results.md"))
    print(open(os.path.join(a.out, "results.md")).read())


if __name__ == "__main__":
    main()
