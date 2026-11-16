PY ?= python3

.PHONY: proto test smoke experiment cluster-up cluster-down clean

proto:            ## generate gRPC stubs
	$(PY) -m grpc_tools.protoc -Iproto --python_out=failover_lab --grpc_python_out=failover_lab proto/failover.proto
	sed -i.bak 's/^import failover_pb2/from . import failover_pb2/' failover_lab/failover_pb2_grpc.py && rm -f failover_lab/failover_pb2_grpc.py.bak

test: proto       ## unit + integration tests (no Docker needed)
	$(PY) -m pytest -q tests

smoke: proto      ## one short failover run with 4 local nodes
	$(PY) -m experiments.run_failover_experiment --nodes 4 --kill 1 --trials 1 --modes inmem --time-scale 600 --out experiments/run_logs/smoke

experiment: proto ## full baseline-vs-proposed comparison, 8 local nodes, 2-node failure
	$(PY) -m experiments.run_failover_experiment --nodes 8 --kill 2 --trials 3 --modes legacy inmem --out experiments/results

cluster-up:       ## 8 node containers + 8 DPU containers + coordinator + Prometheus + Grafana
	docker compose -f cluster/docker-compose.yml up --build -d

cluster-down:
	docker compose -f cluster/docker-compose.yml down -v

clean:
	rm -rf experiments/run_logs failover_lab/failover_pb2*.py
