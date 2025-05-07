import numpy as np
from unittest.mock import MagicMock
import sounddevice as sd

from rei_rt_audio.callback import Passthrough


def test_callback_copies_data():
    """Normal case: data is copied without status"""
    in_data = np.array([1, 2, 3, 4], dtype=np.float32)
    out_data = np.zeros_like(in_data)

    passthrough = Passthrough()
    passthrough.callback(
        in_data,
        out_data,
        frames=len(in_data),
        time={},
        status=sd.CallbackFlags(0),
    )

    np.testing.assert_array_equal(out_data, in_data)


def test_callback_status_warning_and_copy():
    """In the case of status, data is copied and a warning is called"""
    in_data = np.array([10, 20, 30], dtype=np.float32)
    out_data = np.zeros_like(in_data)

    # Mock the logger to capture the warning
    mock_logger = MagicMock()

    passthrough = Passthrough(logger=mock_logger)
    passthrough.callback(
        in_data,
        out_data,
        frames=len(in_data),
        time={"input": 123},
        status=sd.CallbackFlags(1),
    )

    np.testing.assert_array_equal(out_data, in_data)

    mock_logger.warning.assert_called_once()
    args, _ = mock_logger.warning.call_args
    assert "callback status" in args[0]


def test_callback_zero_frames():
    """Edge case: No data is copied when frames is 0"""
    in_data = np.array([], dtype=np.float32)
    out_data = np.zeros_like(in_data)

    passthrough = Passthrough()
    passthrough.callback(
        in_data,
        out_data,
        frames=0,
        time={},
        status=sd.CallbackFlags(0),
    )

    np.testing.assert_array_equal(out_data, in_data)
