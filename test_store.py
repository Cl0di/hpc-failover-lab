import hashlib

import pytest

from failover_lab.store import CheckpointStore, GpuEmulator, GpuOutOfMemory, IntegrityError, Meta


def meta(version, data, job="j"):
    return Meta(job, version, version, len(data), hashlib.sha256(data).hexdigest())


def test_keeps_only_newest_versions_and_frees_memory():
    gpu = GpuEmulator(1000)
    store = CheckpointStore(gpu, keep_last=2)
    for v in (1, 2, 3):
        store.put(meta(v, bytes([v]) * 100), bytes([v]) * 100)
    assert store.latest("j").version == 3
    assert store.get("j", 1) is None
    assert store.get("j", 2)[1] == bytes([2]) * 100
    assert gpu.used == 200


def test_corrupt_checkpoint_is_rejected_and_old_one_survives():
    store = CheckpointStore(GpuEmulator(1000))
    good = b"a" * 50
    store.put(meta(1, good), good)
    bad = Meta("j", 2, 2, 50, hashlib.sha256(b"something else").hexdigest())
    with pytest.raises(IntegrityError):
        store.put(bad, b"b" * 50)
    assert store.latest("j").version == 1


def test_evicts_older_version_to_make_room():
    gpu = GpuEmulator(250)
    store = CheckpointStore(gpu, keep_last=2)
    for v in (1, 2, 3):
        store.put(meta(v, bytes([v]) * 100), bytes([v]) * 100)  # the third only fits after evicting v1
    assert [store.get("j", v) is not None for v in (1, 2, 3)] == [False, True, True]


def test_never_drops_last_valid_checkpoint_when_new_one_cannot_fit():
    store = CheckpointStore(GpuEmulator(150))
    store.put(meta(1, b"a" * 100), b"a" * 100)
    with pytest.raises(GpuOutOfMemory):
        store.put(meta(2, b"b" * 120), b"b" * 120)
    assert store.latest("j").version == 1


def test_out_of_order_arrival_keeps_highest_version_as_latest():
    store = CheckpointStore(GpuEmulator(1000), keep_last=2)
    for v in (5, 3):
        store.put(meta(v, bytes([v]) * 10), bytes([v]) * 10)
    assert store.latest("j").version == 5
