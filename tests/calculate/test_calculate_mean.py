from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from rei_rt_audio.calculate import calculate_mean_square


@pytest.mark.parametrize(
    "signal, expected",
    [
        (np.array([1.0, -1.0, 1.0, -1.0]), 1.0),  # positive and negative values
        (np.array([0.0, 0.0, 0.0]), 0.0),  # all zeros
        (np.array([2.0, 2.0, 2.0]), 4.0),  # all positive values
        (np.array([-3.0, -3.0]), 9.0),  # all negative values
    ],
)
def test_calculate_mean_square_1d(signal: npt.NDArray[Any], expected: float):
    result = calculate_mean_square(signal)
    np.testing.assert_allclose(result, expected, rtol=1e-9)


@pytest.mark.parametrize(
    "signal, expected",
    [
        (
            np.array([[1.0, -2.0], [1.0, -2.0]]),
            np.array([1.0, 4.0]),
        ),  # positive and negative values
        (
            np.zeros((3, 2)),
            np.array([0.0, 0.0]),
        ),  # all zeros
        (
            np.array([[2.0, 3.0], [2.0, 3.0]]),
            np.array([4.0, 9.0]),
        ),  # all positive values
        (
            np.array([[-3.0, -4.0], [-3.0, -4.0]]),
            np.array([9.0, 16.0]),
        ),  # all negative values
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([5.0, 10.0]),
        ),  # mixed values
        (
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            np.array([35 / 3, 56 / 3]),
        ),  # more rows than columns
    ],
)
def test_calculate_mean_square_2d(signal: npt.NDArray[Any], expected: float):
    result = calculate_mean_square(signal)
    np.testing.assert_allclose(result, expected, rtol=1e-9)


@pytest.mark.parametrize(
    "signal",
    [
        np.array([]),  # 1D empty
        np.empty((0, 2)),  # 2D empty rows
    ],
)
def test_calculate_mean_square_empty(signal: Any):
    with pytest.raises(ValueError, match="Input signal is empty."):
        calculate_mean_square(signal)


@pytest.mark.parametrize(
    "signal",
    [
        np.zeros((2, 2, 2)),  # 3D signal
        np.array(5.0),  # Scalar
    ],
)
def test_calculate_mean_square_invalid_input(signal: Any):
    with pytest.raises(ValueError, match="Input signal must be 1D or 2D."):
        calculate_mean_square(signal)


@pytest.mark.parametrize(
    "invalid_input",
    ["a", None, [], {}, object()],
)
def test_calculate_mean_square_invalid_type(invalid_input: Any):
    with pytest.raises(AttributeError, match="object has no attribute 'size'"):
        calculate_mean_square(invalid_input)
