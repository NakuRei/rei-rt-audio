import math

import pytest

from rei_rt_audio.calculate import gain_db_to_linear


@pytest.mark.parametrize(
    "gain_db, expected",
    [
        (0, 1.0),  # 0dB -> 1.0
        (6, 10 ** (6 / 20)),  # 正の値
        (-6, 10 ** (-6 / 20)),  # 負の値
        (0.333, 10 ** (0.333 / 20)),  # 正の小数点以下の値
        (-0.333, 10 ** (-0.333 / 20)),  # 負の小数点以下の値
        (20, 10.0),  # 20dB -> 10.0
        (-20, 0.1),  # -20dB -> 0.1
        (100, 10 ** (100 / 20)),  # 非常に大きい値
        (-100, 10 ** (-100 / 20)),  # 非常に小さい値
    ],
)
def test_gain_db_to_linear_basic(gain_db: float, expected: float):
    result = gain_db_to_linear(gain_db)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_gain_db_to_linear_inf():
    # 正の無限大 -> 無限大
    result = gain_db_to_linear(float("inf"))
    assert result == float("inf")

    # 負の無限大 -> 0.0
    result = gain_db_to_linear(float("-inf"))
    assert result == 0.0


def test_gain_db_to_linear_nan():
    # NaN -> NaN (仕様によってはエラーでも良い)
    result = gain_db_to_linear(float("nan"))
    assert math.isnan(result)
