import math
from typing import Any

import pytest

from rei_rt_audio.calculate import gain_db_to_linear


@pytest.mark.parametrize(
    "gain_db, expected",
    [
        (0, 1.0),  # 0dB -> 1.0
        (6, 10 ** (6 / 20)),  # Positive value
        (-6, 10 ** (-6 / 20)),  # Negative value
        (0.333, 10 ** (0.333 / 20)),  # Positive fractional value
        (-0.333, 10 ** (-0.333 / 20)),  # Negative fractional value
        (20, 10.0),  # 20dB -> 10.0
        (-20, 0.1),  # -20dB -> 0.1
        (100, 10 ** (100 / 20)),  # Very large value
        (-100, 10 ** (-100 / 20)),  # Very small value
    ],
)
def test_gain_db_to_linear_basic(gain_db: float, expected: float):
    result = gain_db_to_linear(gain_db)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_gain_db_to_linear_inf():
    # Positive infinity -> infinity
    assert gain_db_to_linear(float("inf")) == float("inf")

    # Negative infinity -> 0.0 (by mathematical definition: 10 ** -inf == 0.0)
    assert gain_db_to_linear(float("-inf")) == 0.0


def test_gain_db_to_linear_nan():
    # NaN -> NaN
    result = gain_db_to_linear(float("nan"))
    assert math.isnan(result)


@pytest.mark.parametrize(
    "invalid_input",
    ["a", None, [], {}, object()],
)
def test_gain_db_to_linear_invalid_type(invalid_input: Any):
    with pytest.raises(TypeError):
        gain_db_to_linear(invalid_input)
