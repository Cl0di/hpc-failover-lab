# Design notes

## What the lab is and is not

This is a functional model of a failover design, built from containers and
processes. It exercises the control flow (checkpoint, replicate, detect, fence,
restore, resume) with real PyTorch DDP ranks, real gRPC transfers and real
process death. It does not have GPUs, RDMA, a DPU or a parallel file system,
so it cannot say anything about absolute performance on that hardware.

| Real system | Stand-in here | What that means for the results |
|---|---|---|
| GPU device memory, GPUDirect | Host RAM with a capacity limit (`GpuEmulator`) | Memory budget and eviction logic are exercised; copy speed is not representative. |
| InfiniBand / RDMA | gRPC streams over TCP (loopback or Docker bridge) | Replication correctness is tested; throughput is far below RDMA and includes serialisation and hashing on the CPU. |
| BlueField-3 DPU | A separate process that heartbeats the host | Shows the value of an independent, fast failure detector. No offload, no isolation guarantees. |
| NFS / parallel file system | A local directory with `fsync` | **Much faster than a loaded shared file system**, so the baseline's checkpoint write cost is understated. |
| Weeks-long job on thousands of GPUs | Minutes-long job on 8 CPU ranks | Intervals are compressed by a time scale (default 1:60). |

## Where the recovery time goes

Recovery = detection + relaunch and restore + replay of the work done since the
last checkpoint. The design attacks the first and the third term:

- **Detection** moves from a scheduler heartbeat timeout to an independent
  sub-second heartbeat.
- **Replay** shrinks because checkpoints are cheap enough to take four times as
  often. A checkpoint that never touches the file system costs the training
  loop one serialisation and one local stream.

The middle term (stopping the survivors, starting processes, importing the
framework, re-forming the process group) is the same in both configurations
and is the floor for this design. Keeping the survivors alive and re-admitting
only the lost ranks would lower it, but DDP has no support for changing the
world size of a running job, so that needs an elastic runtime.

## Decisions and trade-offs

**Synchronous local write, asynchronous replication.** Training blocks only
until the checkpoint is in the local agent. If the origin node dies before the
replicas land, that one checkpoint is lost and recovery uses the previous
version. Blocking until one replica acknowledges would close that window at
the cost of a longer pause per checkpoint.

**Two replicas on ring neighbours.** Three copies survive any two failures.
Ring placement is simple but correlated: neighbours in the ring should not
share a chassis or a power feed. A real deployment would place replicas across
failure domains.

**Keep two versions.** The store admits the new checkpoint before evicting the
old one, so a transfer that fails half way never leaves a node empty. The cost
is up to two checkpoints of memory per node, three when a swap is in flight
and memory allows.

**Memory cost.** In-memory checkpoints are not free: with two replicas the
cluster holds up to 3 copies x 2 versions of the job state. This competes with
the training job for device memory. For a model that fills the GPU, the
checkpoints would have to live in host RAM or on node-local NVMe instead.

**Periodic drain to durable storage is still needed.** Memory replicas do not
survive a full power loss or a failure wider than the replication factor. The
intended design keeps an infrequent disk checkpoint as the last line of
defence; the lab does not implement the drain yet.

**Fencing is best effort.** A node that was only slow gets a stop request, but
a partitioned node could keep a stale rank running. It cannot corrupt the new
incarnation (different process group and epoch id), but it wastes the node.
Real fencing would go through the DPU or the BMC.

**The coordinator is a single point of failure.** If it dies the ranks keep
training and checkpointing, but nobody handles the next failure. The Slurm
wrapper restarts it and the job resumes from memory; a production version
would make it replicated or let the scheduler own this logic.

## Known limitations

- Rank crashes with the node still up are only handled with
  `--failover-on-rank-exit`.
- Failed nodes are not re-admitted when they come back.
- No authentication or encryption on the gRPC channels.
- The Slurm scripts and the Docker Compose cluster have been validated for
  syntax but the published measurements come from the local-process harness.
