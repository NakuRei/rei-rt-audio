import abc
from typing import Any, Callable, Final, Protocol

import numpy as np
import numpy.typing as npt

from . import consumer as my_consumer
from . import logger as my_logger


class FrameHandlingStrategyProtocol(Protocol):
    def handle(self, frame: npt.NDArray[Any]) -> None: ...


PreprocessType = Callable[[npt.NDArray[Any]], npt.NDArray[Any] | None]


class FrameHandlingStrategy(abc.ABC):

    # To make it pickleable in a multi-process context,
    # we define it as a static method
    @staticmethod
    def default_preprocess(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return x

    def __init__(
        self,
        consumer: my_consumer.ConsumerProtocol,
        logger: my_logger.LoggerProtocol | None = None,
        preprocess_for_frame: PreprocessType | None = None,
    ) -> None:
        self._consumer: Final = consumer

        if logger is None:
            logger = my_logger.PrintLogger()
        self._logger: Final = logger

        if preprocess_for_frame is None:
            preprocess_for_frame = self.default_preprocess
        self._preprocess_for_frame: Final = preprocess_for_frame

    @abc.abstractmethod
    def handle(self, frame: npt.NDArray[Any]) -> None:
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"consumer={repr(self._consumer)}, "
            f"logger={repr(self._logger)})"
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__} ("
            f"consumer={self._consumer.__class__.__name__})"
        )


class FrameProcessingStrategy(FrameHandlingStrategy):
    def handle(self, frame: npt.NDArray[Any]) -> None:
        data = self._preprocess_for_frame(frame)
        if data is None:
            return

        self._consumer.consume(data)


class SlidingBufferProcessingStrategy(FrameHandlingStrategy):
    def __init__(
        self,
        n_buffer_samples: int,
        consumer: my_consumer.ConsumerProtocol,
        logger: my_logger.LoggerProtocol | None = None,
        preprocess_for_frame: PreprocessType | None = None,
    ):
        super().__init__(
            consumer=consumer,
            logger=logger,
            preprocess_for_frame=preprocess_for_frame,
        )

        if n_buffer_samples <= 0:
            raise ValueError("n_buffer_samples must be greater than 0.")
        self._n_buffer_samples: Final[int] = n_buffer_samples

        self._buffer: npt.NDArray[Any] | None = None
        self._n_channels: int | None = None

        self._name = self.__class__.__name__

    def _initialize_buffer(self, data: npt.NDArray[Any]) -> None:
        if self._buffer is not None:
            raise ValueError(f"{self._name}: Buffer is already initialized")

        if data.ndim == 1:
            self._n_channels = 1
            data = data.reshape(-1, 1)
        elif data.ndim == 2:
            self._n_channels = data.shape[1]
        else:
            raise ValueError(
                f"{self._name}: Invalid data shape: {data.shape}. Expected "
                "1D or 2D array."
            )

        buffer_dtype = data.dtype
        # Initialize buffer with zeros
        self._buffer = np.zeros(
            (self._n_buffer_samples, self._n_channels),
            dtype=buffer_dtype,
        )

    def handle(self, frame: npt.NDArray[Any]) -> None:
        data = self._preprocess_for_frame(frame)
        if data is None:
            return

        if self._buffer is None:
            self._initialize_buffer(data)
        assert self._buffer is not None
        assert self._n_channels is not None

        if data.ndim == 1:
            if self._n_channels != 1:
                raise ValueError(
                    f"{self._name}: n_channels mismatch. Expected "
                    f"{self._n_channels}, got 1."
                )
            data = data.reshape(-1, 1)

        n_incoming_samples: Final[int] = data.shape[0]

        if n_incoming_samples >= self._n_buffer_samples:
            # If incoming data is too large, keep only the latest part
            self._buffer[:] = data[-self._n_buffer_samples :]
        else:
            # Shift existing data to make space
            self._buffer[:-n_incoming_samples] = self._buffer[
                n_incoming_samples:
            ]
            self._buffer[-n_incoming_samples:] = data

        self._consumer.consume(self._buffer)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_buffer_samples={self._n_buffer_samples}, "
            f"consumer={repr(self._consumer)}, "
            f"logger={repr(self._logger)}, "
            f"preprocess_for_frame={repr(self._preprocess_for_frame)})"
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__} ("
            f"n_buffer_samples={self._n_buffer_samples}, "
            f"consumer={self._consumer.__class__.__name__})"
        )


class AccumulatingBufferProcessingStrategy(FrameHandlingStrategy):
    def __init__(
        self,
        n_buffer_samples: int,
        consumer: my_consumer.ConsumerProtocol,
        logger: my_logger.LoggerProtocol | None = None,
        preprocess_for_frame: PreprocessType | None = None,
    ):
        super().__init__(
            consumer=consumer,
            logger=logger,
            preprocess_for_frame=preprocess_for_frame,
        )

        if n_buffer_samples <= 0:
            raise ValueError("n_buffer_samples must be greater than 0.")
        self._n_buffer_samples: Final[int] = n_buffer_samples

        self._buffer: list[npt.NDArray[Any]] = []
        self._n_channels: int | None = None
        self._total_samples = 0

        self._name = self.__class__.__name__

    def handle(self, frame: npt.NDArray[Any]) -> None:
        data = self._preprocess_for_frame(frame)
        if data is None:
            return

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if self._n_channels is None:
            self._n_channels = data.shape[1]
        elif data.shape[1] != self._n_channels:
            raise ValueError(
                f"AccumulatingBufferProcessingStrategy: n_channels mismatch. "
                f"Expected {self._n_channels}, got {data.shape[1]}."
            )

        # Add to buffer
        self._buffer.append(data)
        self._total_samples += data.shape[0]

        if self._total_samples >= self._n_buffer_samples:
            accumulated_data = np.concatenate(self._buffer, axis=0)
            self._logger.debug(
                f"{self._name} processing buffer of size "
                f"{accumulated_data.shape}."
            )
            self._consumer.consume(accumulated_data)

            # Reset buffer
            self._buffer.clear()
            self._total_samples = 0

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_buffer_samples={self._n_buffer_samples}, "
            f"consumer={repr(self._consumer)}, "
            f"logger={repr(self._logger)}, "
            f"preprocess_for_frame={repr(self._preprocess_for_frame)})"
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__} ("
            f"n_buffer_samples={self._n_buffer_samples}, "
            f"consumer={self._consumer.__class__.__name__})"
        )
