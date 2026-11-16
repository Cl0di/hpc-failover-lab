"""hpc-failover-lab: in-memory checkpointing and fast failover for emulated multi-GPU training."""
CHUNK_BYTES = 1 << 20
GRPC_OPTS = [("grpc.max_send_message_length", 8 << 20), ("grpc.max_receive_message_length", 8 << 20)]
