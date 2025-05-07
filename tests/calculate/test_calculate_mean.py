import numpy as np
import pytest

from rei_rt_audio.calculate import calculate_mean_square


def test_1d_signal_positive_and_negative():
    signal = np.array([1.0, -1.0, 1.0, -1.0])
    assert calculate_mean_square(signal) == 1.0


def test_1d_signal_all_zeros():
    signal = np.array([0.0, 0.0, 0.0])
    assert calculate_mean_square(signal) == 0.0


def test_1d_signal_all_positive():
    signal = np.array([2.0, 2.0, 2.0])
    assert calculate_mean_square(signal) == 4.0


def test_1d_signal_all_negative():
    signal = np.array([-3.0, -3.0])
    assert calculate_mean_square(signal) == 9.0


def test_1d_signal_empty():
    signal = np.array([])
    with pytest.raises(ValueError):
        calculate_mean_square(signal)


def test_2d_signal_basic():
    signal = np.array([[1.0, -2.0], [1.0, -2.0]])
    result = calculate_mean_square(signal)
    expected = np.array([1.0, 4.0])
    np.testing.assert_array_equal(result, expected)


def test_2d_signal_all_zeros():
    signal = np.zeros((3, 2))
    result = calculate_mean_square(signal)
    expected = np.array([0.0, 0.0])
    np.testing.assert_array_equal(result, expected)


def test_2d_signal_all_positive():
    signal = np.array([[2.0, 3.0], [2.0, 3.0]])
    result = calculate_mean_square(signal)
    expected = np.array([4.0, 9.0])
    np.testing.assert_array_equal(result, expected)


def test_2d_signal_empty_rows():
    signal = np.empty((0, 2))
    with pytest.raises(ValueError):
        calculate_mean_square(signal)


def test_invalid_dimension_3d():
    signal = np.zeros((2, 2, 2))
    with pytest.raises(ValueError):
        calculate_mean_square(signal)


def test_invalid_dimension_scalar():
    signal = np.array(5.0)
    with pytest.raises(ValueError):
        calculate_mean_square(signal)
