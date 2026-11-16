# Designing a Fault-Tolerant AI Training Architecture for Multi-GPU HPC Clusters

## Author: Claudia (This project was entirely designed, developed, and documented by me as a simulation of Distinguished Engineer-level work.)

[![ci](https://github.com/Cl0di/hpc-failover-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Cl0di/hpc-failover-lab/actions/workflows/ci.yml)

### ✨ Overview

This project explores how to keep a large distributed training job running when compute nodes fail. It is a lab simulation: the cluster is built from containers and processes, with emulated GPU memory, a gRPC data plane standing in for the fabric, and mock DPU agents. Inside that lab everything is real: PyTorch DDP ranks train, checkpoints are replicated between nodes, nodes are killed, and recovery is measured.

## 🔎 Problem Statement

Large-scale AI training (e.g., LLMs) requires thousands of GPUs over several weeks. At that scale node failures are routine. Every failure costs the time to notice it, the time to restart, and all the work done since the last checkpoint. Checkpoints written to a shared file system are expensive, so they are taken rarely, and the lost work is large.

## ⚖️ Objective

- Cut recovery time after node loss in multi-GPU training jobs
- Make checkpoints cheap enough to take often, by keeping them **in memory** and **replicating them to peer nodes** instead of writing to shared storage
- Detect node loss in well under a second with an **independent DPU-style monitor**
- Fit the orchestration people already use: **Slurm** and **PyTorch DDP**

## 🛠️ System Design Summary

1. **Infrastructure:** 8 emulated compute nodes, each running a node agent with a GPU memory emulator. Runs as plain local processes or as a Docker Compose cluster.
2. **Checkpointing:** rank 0 serialises model and optimizer state and hands it to the agent on its own node. The agent verifies it, stores it in memory and replicates it to two peers over gRPC streams. The newest two versions are kept.
3. **Failure detection:** one mock DPU agent per node heartbeats its host every 100 ms and reports loss to the coordinator after three misses.
4. **Failover:** the coordinator stops and fences the old job incarnation, finds the newest checkpoint that survived, and relaunches on the surviving nodes. Every rank restores from the replica and training resumes at the checkpointed step.
5. **Orchestration:** a Slurm batch wrapper runs the coordinator as the job and requeues it on failure.
6. **Monitoring:** every component exports Prometheus metrics; a Grafana dashboard is provisioned with the cluster.

Diagram and walkthrough: [docs/architecture.md](docs/architecture.md). Trade-offs and limits: [docs/design.md](docs/design.md).

## 🔢 Key Metrics (Lab Simulation)

Measured with `make experiment`: 8 nodes, 2 killed at the same moment (the node running rank 0 and one of its replica holders), 5 trials per configuration with paired kill times. Intervals are compressed 1:60, so 60 lab seconds stand for one hour.

| Metric | Baseline (disk checkpoint, scheduler timeout) | Proposed (memory replicas, DPU detection) |
| --- | --- | --- |
| Checkpoint interval | 60 s (1 h scaled) | 15 s (15 min scaled) |
| Failure detection | 4.96 s | 0.25 s |
| Relaunch + restore | 6.6 s | 6.7 s |
| Replay of lost work | 15.7 s | 3.4 s |
| **Total recovery (2-node failure)** | **27.2 s** (20.3–39.0) | **10.3 s** (7.2–15.2) |
| Training steps lost | 491 | 111 |
| Restored from | shared directory | peer memory |

Recovery is the time from the kill until the job is back at the step it had reached. Means over 5 trials, range in brackets. Full table and raw data: [experiments/results](experiments/results/results.md).

What the numbers say:

- Recovery was **2.6x faster** on average. The gain comes from faster detection and from losing less work. The restart itself costs the same in both.
- The restart (about 6.6 s) is now the largest part of recovery. Lowering it needs survivors to keep running while lost ranks rejoin, which DDP does not support.
- Writing a checkpoint was not faster in memory here (24 ms vs 15 ms for 4 MB), because the lab's "shared file system" is a fast local disk. The baseline's storage cost is understated.
- No healthy node was wrongly declared dead in 10 runs.

These are results from a 2-core sandbox with CPU ranks. They show how the design behaves, not what a GPU cluster would achieve.

## 📊 Technical Stack

- **Compute**: node agents with an emulated GPU memory pool, mock DPU agents, Docker Compose
- **Fabric**: gRPC streams over TCP (stand-in for RDMA), iperf3 for raw link throughput
- **Scheduler**: Slurm (single-node config, checkpoint-aware job wrapper)
- **AI Framework**: PyTorch DistributedDataParallel (gloo backend) with a synthetic workload
- **Tools**: Prometheus, Grafana, pytest, GitHub Actions

## 🚀 Run it

Requires Python 3.10+ on Linux (macOS should work but is untested; on Windows use WSL). Docker is only needed for the container cluster.

```bash
pip install -r requirements.txt
make test         # unit and replication tests
make smoke        # one failover with 4 local nodes, about 30 seconds
make experiment   # the full comparison above, about 25 minutes
```

Container cluster with dashboards:

```bash
make cluster-up                       # 8 nodes, 8 DPUs, coordinator, Prometheus, Grafana
open http://localhost:3000            # dashboard "HPC failover lab"
./cluster/kill_nodes.sh node0 node1   # inject a 2-node failure and watch the recovery
./cluster/fabric_bench.sh             # iperf3 between two nodes
make cluster-down
```

## 🗂️ Repository layout

| Path | Contents |
| --- | --- |
| `failover_lab/` | node agent, checkpoint store, trainer, mock DPU, coordinator |
| `proto/` | gRPC service definitions |
| `experiments/` | fault-injection harness, fabric benchmark, measured results |
| `cluster/` | Dockerfile, Compose file, fault-injection and iperf3 scripts |
| `slurm/` | `slurm.conf` and the job wrapper |
| `monitoring/` | Prometheus config and Grafana dashboard |
| `tests/` | store and replication tests |
| `docs/` | architecture and design notes |

## 🧪 What is tested and what is not

| Part | Status |
| --- | --- |
| Checkpoint store, gRPC replication | Unit and integration tests in CI |
| Detection, failover, restore, baseline comparison | Measured with the local-process harness; smoke test in CI |
| Docker Compose cluster, Prometheus, Grafana | Config validated; not yet run end to end |
| Slurm wrapper | Written; not yet run end to end |

## ⚡ Challenges

- There is no way to emulate GPUDirect without real GPU memory buffers, so the "GPU memory" here is host RAM with a capacity limit and the transfer path is ordinary TCP
- Swapping checkpoints inside a fixed memory budget without ever being left without a valid one
- Coordinating restore across ranks: DDP does not broadcast weights, so every rank must load the same checkpoint before the process group forms
- Telling a dead node from a slow one quickly without false alarms
- Balancing checkpoint frequency against memory use and training throughput

## 🧭 Next steps

- Run the Compose cluster and the Slurm wrapper end to end and add their results
- Periodic drain of memory checkpoints to durable storage
- Emulate shared file system latency in the baseline
- Re-admit recovered nodes; explore keeping survivors running during failover
