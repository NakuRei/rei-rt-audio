from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from rei_rt_audio.calculate import calculate_rms


@pytest.mark.parametrize(
    "signal, expected",
    [
        (np.array([1.0, -1.0, 1.0, -1.0]), 1.0),  # positive and negative values
        (np.array([0.0, 0.0, 0.0]), 0.0),  # all zeros
        (np.array([2.0, 2.0, 2.0]), 2.0),  # all positive values
        (np.array([-3.0, -3.0]), 3.0),  # all negative values
    ],
)
def test_calculate_rms_1d(signal: npt.NDArray[Any], expected: float):
    result = calculate_rms(signal)
    np.testing.assert_allclose(result, expected, rtol=1e-9)


@pytest.mark.parametrize(
    "signal, expected",
    [
        (
            np.array([[1.0, -2.0], [1.0, -2.0]]),
            np.array([1.0, 2.0]),
        ),  # positive and negative values
        (
            np.zeros((3, 2)),
            np.array([0.0, 0.0]),
        ),  # all zeros
        (
            np.array([[3.0, 4.0], [3.0, 4.0]]),
            np.array([3.0, 4.0]),
        ),  # all positive values
        (
            np.array([[-5.0, -6.0], [-5.0, -6.0]]),
            np.array([5.0, 6.0]),
        ),  # all negative values
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.sqrt(np.array([5.0, 10.0])),
        ),  # mixed values
        (
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            np.sqrt(np.array([35 / 3, 56 / 3])),
        ),  # more rows than columns
    ],
)
def test_calculate_rms_2d(signal: npt.NDArray[Any], expected: npt.NDArray[Any]):
    result = calculate_rms(signal)
    np.testing.assert_allclose(result, expected, rtol=1e-9)


@pytest.mark.parametrize(
    "signal",
    [
        np.array([]),
        np.empty((0, 2)),
    ],
)
def test_calculate_rms_empty(signal: npt.NDArray[Any]):
    with pytest.raises(ValueError, match="Input signal is empty."):
        calculate_rms(signal)


@pytest.mark.parametrize(
    "signal",
    [
        np.zeros((2, 2, 2)),  # 3D
        np.array(5.0),  # scalar
    ],
)
def test_calculate_rms_invalid_shape(signal: npt.NDArray[Any]):
    with pytest.raises(ValueError, match="Input signal must be 1D or 2D."):
        calculate_rms(signal)


@pytest.mark.parametrize(
    "invalid_input",
    ["a", None, [], {}, object()],
)
def test_calculate_rms_invalid_type(invalid_input: Any):
    with pytest.raises(AttributeError, match="object has no attribute 'size'"):
        calculate_rms(invalid_input)
