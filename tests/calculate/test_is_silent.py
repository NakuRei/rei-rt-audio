from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from rei_rt_audio.calculate import is_silent


@pytest.mark.parametrize(
    "signal, threshold_db, expected",
    [
        (
            np.array([0.0, 0.0, 0.0]),
            -50.0,
            True,
        ),  # RMS = 0 -> True
        (
            np.array([0.01, 0.01, 0.01, 0.01]),
            -40.0,
            False,
        ),  # (RMS == threshold → False)
        (
            np.array([0.01, 0.01, 0.01, 0.01]),
            -39.0,
            True,
        ),  # RMS < threshold
        (
            np.array([0.1, 0.1, 0.1, 0.1]),
            -40.0,
            False,
        ),  # RMS > threshold
        (
            np.array([0.1, 0.1, 0.1, 0.1]),
            -20.0,
            False,
        ),  # (RMS == threshold → False)
        (
            np.array([0.1, 0.1, 0.1, 0.1]),
            -19.0,
            True,
        ),  # RMS < threshold
    ],
)
def test_is_silent_1d(
    signal: npt.NDArray[Any],
    threshold_db: float,
    expected: bool,
):
    assert is_silent(signal, threshold_db=threshold_db) is expected


@pytest.mark.parametrize(
    "signal, threshold_db, expected",
    [
        (
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            -50.0,
            True,
        ),  # RMS = 0 -> True
        (
            np.array([[0.01, 0.01], [0.01, 0.01]]),
            -40.0,
            False,
        ),  # (RMS == threshold → False)
        (
            np.array([[0.01, 0.01], [0.01, 0.01]]),
            -39.0,
            True,
        ),  # RMS < threshold
        (
            np.array([[0.1, 0.01], [0.1, 0.01]]),
            -20.0,
            False,
        ),  # max RMS > threshold
        (
            np.array([[0.01, 0.01], [0.01, 0.01]]),
            -20.0,
            True,
        ),  # max RMS < threshold
        (
            np.array([[0.01, 0.1], [0.01, 0.1]]),
            -20.0,
            False,
        ),  # One channel exceeds the threshold
    ],
)
def test_is_silent_2d(
    signal: npt.NDArray[Any],
    threshold_db: float,
    expected: bool,
):
    assert is_silent(signal, threshold_db=threshold_db) is expected


@pytest.mark.parametrize(
    "signal",
    [
        np.array([]),
        np.empty((0, 2)),
    ],
)
def test_is_silent_empty(signal: npt.NDArray[Any]):
    with pytest.raises(ValueError, match="empty"):
        is_silent(signal)


@pytest.mark.parametrize(
    "signal",
    [
        np.zeros((2, 2, 2)),  # 3D
        np.array(1.0),  # scalar
    ],
)
def test_is_silent_invalid_shape(signal: npt.NDArray[Any]):
    with pytest.raises(ValueError):
        is_silent(signal)


@pytest.mark.parametrize(
    "invalid_input",
    ["a", None, [], {}, object()],
)
def test_is_silent_invalid_type(invalid_input: Any):
    with pytest.raises(AttributeError):
        is_silent(invalid_input)


def test_is_silent_threshold_inf():
    signal = np.array([1.0, 1.0, 1.0])
    result = is_silent(signal, threshold_db=-np.inf)
    # thresholdが-infなら絶対にFalseになる
    assert result is False
