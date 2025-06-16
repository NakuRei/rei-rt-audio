import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from rei_rt_audio.preprocessor import SilenceGatePreprocessor


@pytest.fixture
def silence_gate():
    threshold = -40.0
    return SilenceGatePreprocessor(threshold)


@patch("rei_rt_audio.calculate.is_silent")
def test_preprocess_for_frame_silent(
    mock_is_silent: MagicMock,
    silence_gate: SilenceGatePreprocessor,
):
    mock_is_silent.return_value = True
    data = np.array([0.0, 0.0, 0.0])

    result = silence_gate.preprocess_for_frame(data)

    assert result is None
    mock_is_silent.assert_called_once_with(
        data, threshold_db=silence_gate.threshold_db
    )


@patch("rei_rt_audio.calculate.is_silent")
def test_preprocess_for_frame_non_silent(
    mock_is_silent: MagicMock,
    silence_gate: SilenceGatePreprocessor,
):
    mock_is_silent.return_value = False
    data = np.array([0.1, 0.2, 0.3])

    result = silence_gate.preprocess_for_frame(data)

    np.testing.assert_array_equal(result, data)
    mock_is_silent.assert_called_once_with(
        data, threshold_db=silence_gate.threshold_db
    )
