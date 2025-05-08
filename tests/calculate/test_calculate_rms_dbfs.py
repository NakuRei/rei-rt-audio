from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from rei_rt_audio.calculate import calculate_rms_dbfs


@pytest.mark.parametrize(
    "signal, expected",
    [
        (np.array([1.0, -1.0, 1.0, -1.0]), 0.0),  # positive and negative values
        (np.array([0.0, 0.0, 0.0]), -200.0),  # all zeros
        (np.array([2.0, 2.0, 2.0]), 20 * np.log10(2.0)),  # all positive values
        (np.array([-3.0, -3.0]), 20 * np.log10(3.0)),  # all negative values
        (np.array([0.5, -0.5]), 20 * np.log10(0.5)),
        (np.array([2.0, -2.0]), 20 * np.log10(2.0)),
    ],
)
def test_calculate_rms_dbfs_basic(signal: npt.NDArray[Any], expected: float):
    result = calculate_rms_dbfs(signal)
    np.testing.assert_allclose(result, expected, rtol=1e-9)


@pytest.mark.parametrize(
    "signal, full_scale, expected",
    [
        (np.array([1.0, -1.0]), 2.0, 20 * np.log10(0.5)),
        (np.array([2.0, -2.0]), 2.0, 0.0),
    ],
)
def test_calculate_rms_dbfs_full_scale(
    signal: npt.NDArray[Any],
    full_scale: float,
    expected: float,
):
    result = calculate_rms_dbfs(signal, full_scale=full_scale)
    np.testing.assert_allclose(result, expected, rtol=1e-9)


@pytest.mark.parametrize(
    "signal, expected",
    [
        (
            np.array([[1.0, -2.0], [1.0, -2.0]]),
            20 * np.log10(np.array([1.0, 2.0])),
        ),  # positive and negative values
        (
            np.zeros((3, 2)),
            np.array([-200.0, -200.0]),
        ),  # all zeros
        (
            np.array([[3.0, 4.0], [3.0, 4.0]]),
            20 * np.log10(np.array([3.0, 4.0])),
        ),  # all positive values
        (
            np.array([[-5.0, -6.0], [-5.0, -6.0]]),
            20 * np.log10(np.array([5.0, 6.0])),
        ),  # all negative values
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            20 * np.log10(np.sqrt(np.array([5.0, 10.0]))),
        ),  # mixed values
        (
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            20 * np.log10(np.sqrt(np.array([35 / 3, 56 / 3]))),
        ),  # more rows than columns
    ],
)
def test_calculate_rms_dbfs_2d(
    signal: npt.NDArray[Any],
    expected: npt.NDArray[Any],
):
    result = calculate_rms_dbfs(signal)
    np.testing.assert_allclose(result, expected, rtol=1e-9)


@pytest.mark.parametrize(
    "signal",
    [
        np.array([]),
        np.empty((0, 2)),
    ],
)
def test_calculate_rms_dbfs_empty(signal: npt.NDArray[Any]):
    with pytest.raises(ValueError, match="Input signal is empty."):
        calculate_rms_dbfs(signal)


@pytest.mark.parametrize(
    "signal",
    [
        np.zeros((2, 2, 2)),  # 3D
        np.array(1.0),  # scalar
    ],
)
def test_calculate_rms_dbfs_invalid_shape(signal: npt.NDArray[Any]):
    with pytest.raises(ValueError, match="Input signal must be 1D or 2D."):
        calculate_rms_dbfs(signal)


@pytest.mark.parametrize(
    "invalid_input",
    ["a", None, [], {}, object()],
)
def test_calculate_rms_dbfs_invalid_type(invalid_input: Any):
    with pytest.raises(AttributeError, match="object has no attribute 'size'"):
        calculate_rms_dbfs(invalid_input)
