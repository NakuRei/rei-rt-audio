import abc
from typing import Any, Final, Protocol

import numpy.typing as npt

from . import calculate as my_calculate
from . import logger as my_logger


class ProcessorProtocol(Protocol):
    """Protocol for a processor that processes audio data."""

    def process(
        self,
        in_data: npt.NDArray[Any],
        out_data: npt.NDArray[Any],
        frames: int,
    ) -> None: ...


class Processor(abc.ABC):
    """Base class for a processor that processes audio data."""

    def __init__(
        self,
        logger: my_logger.LoggerProtocol | None = None,
    ) -> None:
        """Initialize the processor.

        Args:
            logger: A logger object. If None, a default logger is used.
        """
        super().__init__()

        if logger is None:
            logger = my_logger.PrintLogger()
        self._logger: Final = logger

    @abc.abstractmethod
    def process(
        self,
        in_data: npt.NDArray[Any],
        out_data: npt.NDArray[Any],
        frames: int,
    ) -> None:
        """Process the input data and write to output data.
        Args:
            in_data: Input data to process.
            out_data: Output data to write the processed result to.
            frames: Number of frames in the input data.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(logger={repr(self._logger)})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


class GainLinearProcessor(Processor):
    """Processor that applies gain in linear scale to audio data."""

    def __init__(
        self,
        gain: float,
        logger: my_logger.LoggerProtocol | None = None,
    ) -> None:
        """Initialize the GainLinearProcessor.

        Args:
            gain: Gain in linear scale.
            logger: A logger object. If None, a default logger is used.
        """
        super().__init__(logger=logger)
        self.gain = gain

    @property
    def gain(self) -> float:
        """Get the gain in linear scale."""
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        """Set the gain in linear scale.

        Args:
            value: Gain in linear scale.
        """
        self._gain = value

    def process(
        self,
        in_data: npt.NDArray[Any],
        out_data: npt.NDArray[Any],
        frames: int,
    ) -> None:
        """Process the input data and write to output data.

        Args:
            in_data: Input data to process.
            out_data: Output data to write the processed result to.
            frames: Number of frames in the input data.
        """
        out_data[:] = in_data * self._gain

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"gain={repr(self._gain)}, "
            f"logger={repr(self._logger)})"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(gain={self._gain})"


class GainDbProcessor(Processor):
    """Processor that applies gain in dB to audio data."""

    def __init__(
        self,
        gain_db: float,
        logger: my_logger.LoggerProtocol | None = None,
    ) -> None:
        """Initialize the GainDbProcessor.

        Args:
            gain_db: Gain in dB.
            logger: A logger object. If None, a default logger is used.
        """
        super().__init__(logger=logger)
        self.gain_db = gain_db

    @property
    def gain_db(self) -> float:
        """Get the gain in dB."""
        return self._gain_db

    @gain_db.setter
    def gain_db(self, value: float) -> None:
        """Set the gain in dB.

        Args:
            value: Gain in dB.
        """
        self._gain_db: float = value
        self._gain: float = my_calculate.gain_db_to_linear(value)

    @property
    def gain(self) -> float:
        """Get the gain in linear scale."""
        return self._gain

    def process(
        self,
        in_data: npt.NDArray[Any],
        out_data: npt.NDArray[Any],
        frames: int,
    ) -> None:
        """Process the input data and write to output data.

        Args:
            in_data: Input data to process.
            out_data: Output data to write the processed result to.
            frames: Number of frames in the input data.
        """
        out_data[:] = in_data * self._gain

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"gain_db={repr(self._gain_db)}, "
            f"logger={repr(self._logger)})"
        )

    def __str__(self):
        return f"{self.__class__.__name__}(gain_db={self._gain_db})"
