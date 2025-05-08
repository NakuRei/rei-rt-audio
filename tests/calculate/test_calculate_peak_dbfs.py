from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from rei_rt_audio.calculate import calculate_peak_dbfs


@pytest.mark.parametrize(
    "signal, expected",
    [
        (np.array([1.0, -1.0, 1.0, -1.0]), 0.0),  # positive and negative values
        (np.array([0.0, 0.0, 0.0]), -200.0),  # all zeros
        (
            np.array([2.0, 2.0, 2.0]),
            20 * np.log10(2.0),
        ),  # all positive values
        (np.array([-3.0, -3.0]), 20 * np.log10(3.0)),  # all negative values
        (np.array([0.5, -0.5]), 20 * np.log10(0.5)),  # peak=0.5 → -6.0206dBFS
    ],
)
def test_calculate_peak_dbfs_basic(signal: npt.NDArray[Any], expected: float):
    result = calculate_peak_dbfs(signal)
    np.testing.assert_allclose(result, expected, rtol=1e-9)


@pytest.mark.parametrize(
    "signal, full_scale, expected",
    [
        (
            np.array([1.0, -1.0]),
            2.0,
            20 * np.log10(0.5),
        ),  # peak=1.0 / 2.0 → -6.0206dBFS
        (np.array([2.0, -2.0]), 2.0, 0.0),  # peak=2.0 / 2.0 → 0dBFS
    ],
)
def test_calculate_peak_dbfs_full_scale(
    signal: npt.NDArray[Any],
    full_scale: float,
    expected: float,
):
    result = calculate_peak_dbfs(signal, full_scale=full_scale)
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
            20 * np.log10(np.array([3.0, 4.0])),
        ),  # mixed values
        (
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            20 * np.log10(np.array([5.0, 6.0])),
        ),  # more rows than columns
    ],
)
def test_calculate_peak_dbfs_2d(
    signal: npt.NDArray[Any],
    expected: npt.NDArray[Any],
):
    result = calculate_peak_dbfs(signal)
    np.testing.assert_allclose(result, expected, rtol=1e-9)


@pytest.mark.parametrize(
    "signal",
    [
        np.array([]),
        np.empty((0, 2)),
    ],
)
def test_calculate_peak_dbfs_empty(signal: npt.NDArray[Any]):
    with pytest.raises(ValueError, match="Input signal is empty."):
        calculate_peak_dbfs(signal)


@pytest.mark.parametrize(
    "signal",
    [
        np.zeros((2, 2, 2)),  # 3D
        np.array(1.0),  # scalar
    ],
)
def test_calculate_peak_dbfs_invalid_shape(signal: npt.NDArray[Any]):
    with pytest.raises(ValueError):
        calculate_peak_dbfs(signal)


@pytest.mark.parametrize(
    "invalid_input",
    ["a", None, [], {}, object()],
)
def test_calculate_peak_dbfs_invalid_type(invalid_input: Any):
    with pytest.raises(AttributeError):
        calculate_peak_dbfs(invalid_input)
