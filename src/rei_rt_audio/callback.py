import abc
from typing import Any, Final, Protocol

import numpy.typing as npt
import sounddevice as sd

from . import logger as my_logger


class InOutStreamCallbackProtocol(Protocol):
    """Protocol for a callback handler that processes input and output data."""

    def callback(
        self,
        in_data: npt.NDArray[Any],
        out_data: npt.NDArray[Any],
        frames: int,
        time: dict[str, Any],
        status: sd.CallbackFlags,
    ) -> None: ...


class InOutStreamCallbackHandler(abc.ABC):
    """Abstract base class for handling input and output stream callbacks.
    This class provides a framework for processing audio data in real-time
    using a callback mechanism. It defines the interface for subclasses to
    implement their own callback logic.
    """

    def __init__(
        self,
        logger: my_logger.LoggerProtocol | None = None,
        is_log_callback_info: bool = False,
    ) -> None:
        """Initialize the callback handler.

        Args:
            logger: A logger object. If None, a default logger is used.
            is_log_callback_info: Flag to indicate whether to log
                callback information.
        """
        super().__init__()

        if logger is None:
            logger = my_logger.PrintLogger()
        self._logger: Final = logger

        self._is_log_callback_info: Final = is_log_callback_info

    @abc.abstractmethod
    def callback(
        self,
        in_data: npt.NDArray[Any],
        out_data: npt.NDArray[Any],
        frames: int,
        time: dict[str, Any],
        status: sd.CallbackFlags,
    ) -> None:
        """Callback method to process input and output data.

        Args:
            in_data: Input data to process.
            out_data: Output data to write the processed result to.
            frames: Number of frames in the input data.
            time: A dictionary containing time information.
            status: Callback status flags.
        """
        pass

    def _format_log(self, base: str, frames: int, time: dict[str, Any]) -> str:
        """Format log message with callback information.

        Args:
            base: The base log message.
            frames: The number of frames processed.
            time: A dictionary containing time information.

        Returns:
            A formatted log message with callback information.
        """
        if self._is_log_callback_info:
            return f"{base}, frames: {frames}, time: {time}"
        return base

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"logger={repr(self._logger)}, "
            f"is_log_callback_info={self._is_log_callback_info})"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


class Passthrough(InOutStreamCallbackHandler):
    """Passthrough callback that simply copies input to output."""

    def callback(
        self,
        in_data: npt.NDArray[Any],
        out_data: npt.NDArray[Any],
        frames: int,
        time: dict[str, Any],
        status: sd.CallbackFlags,
    ) -> None:
        """Passthrough callback that simply copies input to output.

        Args:
            in_data: Input data to process.
            out_data: Output data to write the processed result to.
            frames: Number of frames in the input data.
            time: A dictionary containing time information.
            status: Callback status flags.
        """
        if status:
            self._logger.warning(
                self._format_log(
                    f"{self.__class__.__name__} callback status: {status}",
                    frames,
                    time,
                )
            )
        out_data[:] = in_data
