from types import SimpleNamespace

import pytest

from src.config import ConfigError
from src.utils.device import enable_amp_if_available, get_device, move_batch_to_device


class FakeDevice:
    def __init__(self, name):
        self.type = name

    def __str__(self):
        return self.type


class FakeTensor:
    def __init__(self, value):
        self.value = value
        self.device = None

    def to(self, device):
        moved = FakeTensor(self.value)
        moved.device = device
        return moved


def fake_torch(cuda_available=False, mps_available=False):
    return SimpleNamespace(
        device=FakeDevice,
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)),
    )


def test_get_device_auto_priority_and_forced_unavailable():
    assert get_device({"mode": "auto"}, fake_torch(cuda_available=True, mps_available=True)).type == "cuda"
    assert get_device({"mode": "auto"}, fake_torch(cuda_available=False, mps_available=True)).type == "mps"
    assert get_device({"mode": "auto"}, fake_torch(cuda_available=False, mps_available=False)).type == "cpu"

    with pytest.raises(ConfigError) as cuda_error:
        get_device({"mode": "cuda"}, fake_torch(cuda_available=False))
    assert cuda_error.value.code == "ERR_DEVICE_UNAVAILABLE"

    with pytest.raises(ConfigError) as mps_error:
        get_device({"mode": "mps"}, fake_torch(mps_available=False))
    assert mps_error.value.code == "ERR_DEVICE_UNAVAILABLE"


def test_amp_only_enabled_on_cuda():
    assert enable_amp_if_available({"amp": True}, FakeDevice("cuda")) is True
    assert enable_amp_if_available({"amp": False}, FakeDevice("cuda")) is False
    assert enable_amp_if_available({"amp": True}, FakeDevice("cpu")) is False
    assert enable_amp_if_available({"amp": True}, FakeDevice("mps")) is False


def test_move_batch_to_device_recurses_nested_batch():
    device = FakeDevice("cpu")
    batch = {
        "features": FakeTensor("x"),
        "nested": [FakeTensor("y"), (FakeTensor("z"), "keep")],
    }

    moved = move_batch_to_device(batch, device)

    assert moved["features"].device is device
    assert moved["nested"][0].device is device
    assert moved["nested"][1][0].device is device
    assert moved["nested"][1][1] == "keep"
