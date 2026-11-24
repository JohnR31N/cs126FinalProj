import gzip
import struct

import numpy as np
import pytest

from external_data import read_idx, to_gray28


def test_rgb_planes_use_the_same_luminance_scale_across_channels():
    images = np.zeros((4, 3, 32, 32), dtype=np.uint8)
    for channel in range(3):
        images[channel, channel] = 255
    images[3] = 255
    expected = np.broadcast_to(np.float32([0.299, 0.587, 0.114, 1])[:, None], (4, 784))
    np.testing.assert_allclose(to_gray28(images), expected, atol=2e-7)


def test_usps_resizing_preserves_the_horizontal_direction_of_a_ramp():
    ramp = np.linspace(0, 1, 16, dtype=np.float32)
    images = np.broadcast_to(ramp, (1, 1, 16, 16)).copy()
    resized = to_gray28(images).reshape(28, 28)
    np.testing.assert_allclose(resized, np.broadcast_to(resized[0], (28, 28)), atol=1e-7)
    assert resized[0, 0] == 0 and resized[-1, -1] == 1 and np.all(np.diff(resized[14]) > 0)


def test_idx_checks_declared_dimensions_and_count(tmp_path):
    data = (np.arange(784) % 256).astype(np.uint8)
    file = tmp_path / "images.gz"
    file.write_bytes(gzip.compress(struct.pack(">IIII", 2051, 1, 28, 28) + data.tobytes()))
    np.testing.assert_array_equal(read_idx(file, 2051, 1), data[None])
    with pytest.raises(ValueError):
        read_idx(file, 2051, 2)
    file.write_bytes(gzip.compress(struct.pack(">IIII", 2051, 1, 14, 56) + data.tobytes()))
    with pytest.raises(ValueError):
        read_idx(file, 2051, 1)
