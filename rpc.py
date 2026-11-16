"""Small helpers shared by every gRPC client in the lab."""
from __future__ import annotations

import hashlib

import grpc

from . import CHUNK_BYTES, GRPC_OPTS
from . import failover_pb2 as pb
from . import failover_pb2_grpc as pbg


def agent_stub(addr: str) -> pbg.NodeAgentStub:
    return pbg.NodeAgentStub(grpc.insecure_channel(addr, options=GRPC_OPTS))


def coordinator_stub(addr: str) -> pbg.CoordinatorStub:
    return pbg.CoordinatorStub(grpc.insecure_channel(addr, options=GRPC_OPTS))


def chunks(meta: pb.CheckpointMeta, data: bytes):
    first = True
    for off in range(0, max(len(data), 1), CHUNK_BYTES):
        yield pb.Chunk(meta=meta if first else None, data=data[off:off + CHUNK_BYTES])
        first = False


def make_meta(job_id: str, version: int, step: int, data: bytes, origin: str, replica=False) -> pb.CheckpointMeta:
    return pb.CheckpointMeta(job_id=job_id, version=version, step=step, size_bytes=len(data),
                             sha256=hashlib.sha256(data).hexdigest(), origin_node=origin, replica=replica)


def put(stub, meta: pb.CheckpointMeta, data: bytes, timeout: float = 30.0) -> None:
    reply = stub.Put(chunks(meta, data), timeout=timeout)
    if not reply.ok:
        raise RuntimeError(f"Put rejected: {reply.error}")


def fetch(stub, job_id: str, version: int = 0, timeout: float = 30.0) -> tuple[pb.CheckpointMeta, bytes]:
    meta, parts = None, []
    for c in stub.Get(pb.GetRequest(job_id=job_id, version=version), timeout=timeout):
        if c.HasField("meta"):
            meta = c.meta
        parts.append(c.data)
    data = b"".join(parts)
    if meta is None:
        raise LookupError(f"no checkpoint for job {job_id} v{version}")
    if hashlib.sha256(data).hexdigest() != meta.sha256:
        raise ValueError("checkpoint corrupted in transit")
    return meta, data
