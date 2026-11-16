"""Three real agents over gRPC on localhost: write to one, read the replicas from the others."""
import os
import time
from concurrent import futures

import grpc
import pytest

from failover_lab import GRPC_OPTS, rpc
from failover_lab import failover_pb2 as pb
from failover_lab import failover_pb2_grpc as pbg
from failover_lab.agent import Agent


@pytest.fixture
def cluster(tmp_path):
    servers, addrs = [], []
    for _ in range(3):
        s = grpc.server(futures.ThreadPoolExecutor(max_workers=8), options=GRPC_OPTS)
        addrs.append(f"127.0.0.1:{s.add_insecure_port('127.0.0.1:0')}")
        servers.append(s)
    for i, s in enumerate(servers):
        agent = Agent(f"node{i}", [addrs[(i + 1) % 3], addrs[(i + 2) % 3]], 64 << 20, str(tmp_path))
        agent.listen_addr = addrs[i]
        pbg.add_NodeAgentServicer_to_server(agent, s)
        s.start()
    yield servers, addrs
    for s in servers:
        s.stop(0)


def wait_latest(stub, version, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = stub.Latest(pb.LatestRequest(job_id="j"))
        if r.found and r.meta.version == version:
            return r.meta
        time.sleep(0.02)
    raise AssertionError("replica never arrived")


def test_checkpoint_survives_loss_of_origin_and_one_replica(cluster):
    servers, addrs = cluster
    data = os.urandom(3 * (1 << 20) + 123)  # spans several chunks
    rpc.put(rpc.agent_stub(addrs[0]), rpc.make_meta("j", 7, 7, data, "node0"), data)
    for addr in addrs[1:]:
        wait_latest(rpc.agent_stub(addr), 7)
    servers[0].stop(0)  # origin dies
    servers[1].stop(0)  # one replica dies
    meta, got = rpc.fetch(rpc.agent_stub(addrs[2]), "j")
    assert meta.version == 7 and meta.origin_node == "node0" and got == data


def test_corrupt_put_is_refused_over_the_wire(cluster):
    _, addrs = cluster
    data = b"x" * 1000
    meta = rpc.make_meta("j", 1, 1, data, "node0")
    meta.sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="sha256"):
        rpc.put(rpc.agent_stub(addrs[0]), meta, data)
    assert not rpc.agent_stub(addrs[0]).Latest(pb.LatestRequest(job_id="j")).found


def test_replicas_do_not_fan_out_again(cluster):
    _, addrs = cluster
    data = b"y" * 10
    rpc.put(rpc.agent_stub(addrs[0]), rpc.make_meta("j", 1, 1, data, "node0", replica=True), data)
    time.sleep(0.3)
    assert not rpc.agent_stub(addrs[1]).Latest(pb.LatestRequest(job_id="j")).found
