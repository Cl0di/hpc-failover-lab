"""Emulated GPU memory pool and the in-memory checkpoint store that lives in it."""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass


class GpuOutOfMemory(MemoryError):
    pass


class GpuEmulator:
    """Accounting-only stand-in for a GPU's device memory.

    There is no real device behind this. It enforces a capacity so the
    checkpoint store has to live inside a fixed budget, the way pinned
    device buffers would.
    """

    def __init__(self, total_bytes: int):
        self.total = int(total_bytes)
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        return self._used

    def allocate(self, nbytes: int) -> None:
        with self._lock:
            if self._used + nbytes > self.total:
                raise GpuOutOfMemory(f"need {nbytes} B, {self.total - self._used} B free")
            self._used += nbytes

    def free(self, nbytes: int) -> None:
        with self._lock:
            self._used = max(0, self._used - nbytes)


@dataclass(frozen=True)
class Meta:
    job_id: str
    version: int
    step: int
    size_bytes: int
    sha256: str
    origin_node: str = ""


class IntegrityError(ValueError):
    pass


class CheckpointStore:
    """Keeps the newest `keep_last` checkpoints per job in (emulated) device memory.

    A new checkpoint is verified before it is admitted, and an older one is
    only evicted once the new one is safely stored, so a failed or corrupt
    transfer can never leave a node without a valid checkpoint.
    """

    def __init__(self, gpu: GpuEmulator, keep_last: int = 2):
        self.gpu = gpu
        self.keep_last = keep_last
        self._jobs: dict[str, OrderedDict[int, tuple[Meta, bytes]]] = {}
        self._lock = threading.Lock()

    def put(self, meta: Meta, data: bytes) -> None:
        if len(data) != meta.size_bytes:
            raise IntegrityError(f"size mismatch: got {len(data)}, expected {meta.size_bytes}")
        if hashlib.sha256(data).hexdigest() != meta.sha256:
            raise IntegrityError("sha256 mismatch")
        with self._lock:
            versions = self._jobs.setdefault(meta.job_id, OrderedDict())
            if meta.version in versions:
                return  # idempotent re-send
            # Make room, but never drop the newest valid checkpoint to do so.
            while True:
                try:
                    self.gpu.allocate(len(data))
                    break
                except GpuOutOfMemory:
                    if len(versions) <= 1:
                        raise
                    self._evict_oldest(versions)
            versions[meta.version] = (meta, data)
            for v in sorted(versions):
                versions.move_to_end(v)
            while len(versions) > self.keep_last:
                self._evict_oldest(versions)

    def _evict_oldest(self, versions) -> None:
        _, (_, old) = versions.popitem(last=False)
        self.gpu.free(len(old))

    def latest(self, job_id: str) -> Meta | None:
        with self._lock:
            versions = self._jobs.get(job_id)
            if not versions:
                return None
            return versions[next(reversed(versions))][0]

    def get(self, job_id: str, version: int = 0) -> tuple[Meta, bytes] | None:
        with self._lock:
            versions = self._jobs.get(job_id)
            if not versions:
                return None
            if version == 0:
                version = next(reversed(versions))
            return versions.get(version)
