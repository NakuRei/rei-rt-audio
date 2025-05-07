import numpy as np
import pytest

from rei_rt_audio.calculate import linear_to_gain_db


@pytest.mark.parametrize(
    "gain, expected",
    [
        (1.0, 0.0),  # 1 -> 0dB
        (10.0, 20.0),  # 10 -> 20dB
        (0.1, -20.0),  # 0.1 -> -20dB
        (100.0, 40.0),  # 100 -> 40dB
    ],
)
def test_linear_to_gain_db_basic(gain: float, expected: float):
    result = linear_to_gain_db(gain)
    assert np.isclose(result, expected, rtol=1e-9)


def test_linear_to_gain_db_zero_raise():
    with pytest.raises(ValueError):
        linear_to_gain_db(0.0)


def test_linear_to_gain_db_negative_raise():
    with pytest.raises(ValueError):
        linear_to_gain_db(-1.0)


def test_linear_to_gain_db_inf():
    result = linear_to_gain_db(float("inf"))
    assert result == float("inf")


def test_linear_to_gain_db_nan():
    result = linear_to_gain_db(float("nan"))
    assert np.isnan(result)
