from typing import Any

import numpy as np
import numpy.typing as npt


def gain_db_to_linear(gain_db: float) -> float:
    """Convert gain in dB to linear gain.

    Args:
        gain_db: Gain in dB.

    Returns:
        Linear gain.
    """
    return 10 ** (gain_db / 20)


def linear_to_gain_db(gain: float) -> float:
    """Convert linear gain to dB.

    Args:
        gain: Linear gain.

    Returns:
        Gain in dB.

    Raises:
        ValueError: If gain is not positive.
    """
    if gain <= 0:
        raise ValueError("Gain must be positive.")
    return 20 * np.log10(gain)


def calculate_mean_square(
    signal: npt.NDArray[Any],
) -> npt.NDArray[Any] | float:
    """Compute the mean square value of a 1D or 2D signal.

    The mean square is the average of the squared amplitudes.
    Single-channel (1D) and multi-channel (2D) signals are supported.

    Args:
        signal (npt.NDArray[Any]): Input signal array.
            - 1D: Shape (n,), where n is the number of samples.
            - 2D: Shape (n, m), where n is the number of samples,
              and m is the number of channels.

    Returns:
        npt.NDArray[Any] | float:
            - If input is 1D, returns a float representing the mean square.
            - If input is 2D, returns a 1D array of shape (m,), the mean square
              per channel.

    Raises:
        ValueError: If the input signal is empty.
        ValueError: If the input signal is neither a 1D nor a 2D array.

    Example:
        >>> import numpy as np
        >>> signal = np.array([1.0, -1.0, 1.0, -1.0])
        >>> calculate_mean_square(signal)
        1.0

        >>> multi_channel_signal = np.array([[1.0, -2.0], [1.0, -2.0]])
        >>> calculate_mean_square(multi_channel_signal)
        array([1.0, 4.0])
    """

    if signal.size == 0:
        raise ValueError("Input signal is empty.")

    if signal.ndim not in [1, 2]:
        raise ValueError(
            f"Invalid signal shape: {signal.shape}. "
            "Input signal must be 1D or 2D numpy array."
        )

    return np.mean(np.square(signal), axis=0)


def calculate_rms(
    signal: npt.NDArray[Any],
) -> npt.NDArray[Any] | float:
    """Calculate the root mean square (RMS) of a 1D or 2D signal.

    The RMS is the square root of the mean square value. Single-channel (1D)
    and multi-channel (2D) signals are supported.

    Args:
        signal (npt.NDArray[Any]): Input signal array.
            - 1D: Shape (n,), where n is the number of samples.
            - 2D: Shape (n, m), where n is the number of samples,
              and m is the number of channels.

    Returns:
        npt.NDArray[Any] | float:
            - If input is 1D, returns a float representing the RMS.
            - If input is 2D, returns a 1D array of shape (m,), the RMS
              per channel.

    Raises:
        ValueError: If the input signal is neither a 1D nor a 2D array.

    Example:
        >>> import numpy as np
        >>> signal = np.array([1.0, -1.0, 1.0, -1.0])
        >>> calculate_rms(signal)
        1.0

        >>> multi_channel_signal = np.array([[1.0, -2.0], [1.0, -2.0]])
        >>> calculate_rms(multi_channel_signal)
        array([1.0, 2.0])
    """

    return np.sqrt(calculate_mean_square(signal))


def calculate_rms_dbfs(
    signal: npt.NDArray[Any],
    full_scale: float = 1.0,
    eps: float = 1e-10,
) -> npt.NDArray[Any] | float:
    """Calculate the RMS in decibels relative to full scale (dBFS).

    The RMS value is converted to dBFS using the formula:
        dBFS = 20 * log10(RMS / full_scale)
    where full_scale is the maximum possible amplitude of the signal.

    Args:
        signal (npt.NDArray[Any]): Input signal array.
            - 1D: Shape (n,), where n is the number of samples.
            - 2D: Shape (n, m), where n is the number of samples,
              and m is the number of channels.
        full_scale (float, optional): Full scale value. Defaults to 1.0.
        eps (float, optional):
            Small value to avoid division by zero. Defaults to 1e-10.

    Returns:
        npt.NDArray[Any] | float:
            - If input is 1D, returns a float representing the RMS in dBFS.
            - If input is 2D, returns a 1D array of shape (m,), the RMS in dBFS
              per channel.

    Raises:
        ValueError: If the input signal is neither a 1D nor a 2D array.

    Example:
        >>> import numpy as np
        >>> signal = np.array([1.0, -1.0, 1.0, -1.0])
        >>> calculate_rms_dbfs(signal)
        0.0

        >>> multi_channel_signal = np.array([[1.0, -2.0], [1.0, -2.0]])
        >>> calculate_rms_dbfs(multi_channel_signal)
        array([ 0., -6.02059991])
    """

    rms = calculate_rms(signal)
    return 20 * np.log10(np.maximum(rms / full_scale, eps))


def calculate_peak_dbfs(
    signal: npt.NDArray[Any],
    full_scale: float = 1.0,
    eps: float = 1e-10,
):
    """Calculate the peak value in decibels relative to full scale (dBFS).

    The peak value is converted to dBFS using the formula:
        dBFS = 20 * log10(peak / full_scale)
    where full_scale is the maximum possible amplitude of the signal.

    Args:
        signal (npt.NDArray[Any]): Input signal array.
            - 1D: Shape (n,), where n is the number of samples.
            - 2D: Shape (n, m), where n is the number of samples,
              and m is the number of channels.
        full_scale (float, optional): Full scale value. Defaults to 1.0.
        eps (float, optional):
            Small value to avoid division by zero. Defaults to 1e-10.

    Returns:
        npt.NDArray[Any] | float:
            - If input is 1D, returns a float representing the peak in dBFS.
            - If input is 2D, returns a 1D array of shape (m,), the peak in dBFS
              per channel.

    Raises:
        ValueError: If the input signal is neither a 1D nor a 2D array.

    Example:
        >>> import numpy as np
        >>> signal = np.array([1.0, -1.0, 1.0, -1.0])
        >>> calculate_peak_dbfs(signal)
        0.0

        >>> multi_channel_signal = np.array([[1.0, -2.0], [1.0, -2.0]])
        >>> calculate_peak_dbfs(multi_channel_signal)
        array([ 0., -6.02059991])
    """

    if signal.size == 0:
        raise ValueError("Input signal is empty.")

    if signal.ndim not in [1, 2]:
        raise ValueError(
            f"Invalid signal shape: {signal.shape}. "
            "Input signal must be 1D or 2D numpy array."
        )

    peak = np.max(np.abs(signal), axis=0)
    return 20 * np.log10(np.maximum(peak / full_scale, eps))


def is_silent(
    signal: npt.NDArray[Any],
    threshold_db: float = -50.0,
) -> bool:
    """Determine if a signal is silent based on its RMS value.

    This function calculates the RMS of the input signal and compares it
    to a threshold in decibels (dB). If the RMS is below the threshold,
    the signal is considered silent.

    Args:
        signal (npt.NDArray[Any]): Input signal array.
            - 1D: Shape (n,), where n is the number of samples.
            - 2D: Shape (n, m), where n is the number of samples,
              and m is the number of channels.
        threshold_db (float, optional): Silence threshold in decibels (dB).
            Defaults to -50.0 dB.

    Returns:
        bool: True if the signal is silent, False otherwise.

    Raises:
        ValueError: If the input signal is neither a 1D nor a 2D array.

    Example:
        >>> import numpy as np
        >>> signal = np.array([0.01, 0.01, 0.01, 0.01])
        >>> is_silent(signal, threshold_db=-40.0)
        True

        >>> multi_channel_signal = np.array([[0.1, 0.01], [0.1, 0.01]])
        >>> is_silent(multi_channel_signal, threshold_db=-20.0)
        False
    """
    rms = calculate_rms(signal)
    # Use the maximum RMS value if multiple channels exist
    if isinstance(rms, np.ndarray):
        rms = rms.max()
    if rms == 0:
        return True
    threshold = 10 ** (threshold_db / 20)
    return bool(rms < threshold)
