import abc
import multiprocessing
import queue
import traceback
import threading
from typing import Any, Protocol, Final, Self
import sys

import numpy as np
import numpy.typing as npt

from . import controller as my_controller
from . import logger as my_logger
from . import strategy as my_strategy


class SubscriberProtocol(Protocol):
    """Protocol for a subscriber that receives data and processes it."""

    def update(self, data: npt.NDArray[Any]) -> None: ...


class Subscriber(abc.ABC):
    """Base class for subscribers that receive data and process it.

    This class provides a common interface for different types of subscribers,
    such as synchronous, asynchronous, and threaded subscribers.
    It also provides a mechanism for handling data frames using a strategy
    pattern.
    """

    def __init__(
        self,
        strategy: my_strategy.FrameHandlingStrategyProtocol,
        logger: my_logger.LoggerProtocol | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the subscriber with a strategy and optional logger.

        Args:
            strategy: The strategy to handle incoming data frames.
            logger: Optional logger for logging messages.
            name: Optional name for the subscriber.
        """
        self._strategy: Final = strategy

        if logger is None:
            logger = my_logger.PrintLogger()
        self._logger: Final = logger

        if name is None:
            name = self.__class__.__name__
        self._name: Final[str] = name

    def start(self) -> None:
        """Start the subscriber.

        This method should be overridden by subclasses to implement
        specific start behavior.
        """
        pass

    def stop(self) -> None:
        """Stop the subscriber.

        This method should be overridden by subclasses to implement
        specific stop behavior.
        """
        pass

    def __enter__(self) -> Self:
        """Enter the runtime context related to this subscriber.

        This method is called when the subscriber is used in a
        context manager (with statement).
        It starts the subscriber.
        """
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Exit the runtime context related to this subscriber.

        This method is called when the subscriber is used in a
        context manager (with statement).
        It stops the subscriber and handles any exceptions that occurred
        during the context.
        """
        self.stop()

    @abc.abstractmethod
    def update(self, data: npt.NDArray[Any]) -> None:
        """Update the subscriber with new data.

        This method should be overridden by subclasses to implement
        specific update behavior.

        Args:
            data: The incoming data to be processed.
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"strategy={repr(self._strategy)}, "
            f"logger={repr(self._logger)}, "
            f"name={repr(self._name)})"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} ("
            f"name={self._name}, "
            f"strategy={self._strategy.__class__.__name__})"
        )


class SyncSubscriber(Subscriber):
    """Synchronous subscriber that processes data immediately.

    This subscriber is intended for use in a synchronous context,
    where data is processed as soon as it is received.
    """

    def update(self, data: npt.NDArray[Any]) -> None:
        """Update the subscriber with new data.

        This method processes the incoming data immediately.

        Args:
            data: The incoming data to be processed.
        """
        try:
            self._strategy.handle(data)
        except Exception as e:
            self._logger.error(
                f"{self._name} (update): process error: {e}\n"
                f"{traceback.format_exc()}"
            )
            # Exit the thread with a non-zero status
            sys.exit(1)


class SlidingBufferThreadSubscriber(Subscriber):
    """Threaded subscriber that uses a sliding buffer to process data.

    This subscriber maintains a buffer of the most recent data samples
    and processes them in a separate thread. It is useful for scenarios
    where data arrives continuously and needs to be processed in chunks.
    """

    def __init__(
        self,
        strategy: my_strategy.FrameHandlingStrategyProtocol,
        n_buffer_samples: int,
        n_frame_samples: int | None,
        logger: my_logger.LoggerProtocol | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the sliding buffer subscriber.

        Args:
            strategy: The strategy to handle incoming data frames.
            n_buffer_samples: The size of the sliding buffer in samples.
            n_frame_samples: The number of samples to process in each frame.
            logger: Optional logger for logging messages.
            name: Optional name for the subscriber.

        Raises:
            ValueError: If n_buffer_samples or n_frame_samples is invalid.
        """
        super().__init__(
            strategy=strategy,
            logger=logger,
            name=name,
        )

        if n_buffer_samples <= 0:
            raise ValueError("n_buffer_samples must be greater than 0.")

        if n_frame_samples is None:
            self._logger.warning(
                (
                    "n_frame_samples is None. This will cause the "
                    "n_frame_samples to be equal to the first data samples."
                )
            )
        elif n_frame_samples <= 0:
            raise ValueError(
                "n_frame_samples must be greater than or equal to 0."
            )

        self._n_buffer_samples: Final[int] = n_buffer_samples
        self._n_frame_samples: int | None = n_frame_samples

        self._buffer: npt.NDArray[Any] | None = None
        self._n_channels: int | None = None

        self._thread_controller: Final = my_controller.ThreadController(
            target=self._run,
            logger=self._logger,
            name=f"{self._name} Runner",
        )

        # Synchronization
        self._buffer_lock: Final = threading.Lock()
        self._new_data_event: Final = threading.Event()

    def _initialize_buffer(self, data: npt.NDArray[Any]) -> None:
        """Initialize the sliding buffer with the incoming data.

        This method is called when the first data frame is received.
        It sets the buffer size and data type based on the incoming data.

        Args:
            data: The incoming data to initialize the buffer with.
        """
        if self._buffer is not None:
            # Safety check: should never happen
            self._logger.error(f"{self._name}: Buffer is already initialized")
            return

        if data.ndim == 1:
            self._n_channels = 1
            data = data.reshape(-1, 1)
        elif data.ndim == 2:
            self._n_channels = data.shape[1]
        else:
            # Safety check: should never happen
            self._logger.error(
                f"{self._name}: "
                f"Invalid data shape: {data.shape}. Expected 1D or 2D array."
            )
            return

        if self._n_frame_samples is None:
            self._n_frame_samples = data.shape[0]

        buffer_dtype = data.dtype
        # Initialize buffer with zeros
        self._buffer = np.zeros(
            (self._n_buffer_samples, self._n_channels),
            dtype=buffer_dtype,
        )

    def update(self, data: npt.NDArray[np.float32]) -> None:
        """Update the subscriber with new data.

        This method processes the incoming data and updates the sliding
        buffer. It also notifies the processing thread that new data is
        available.

        Args:
            data: The incoming data to be processed.
        """
        try:
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            n_incoming_samples: Final[int] = data.shape[0]

            with self._buffer_lock:
                if self._buffer is None:
                    self._initialize_buffer(data)

                if self._buffer is None:
                    # Safety check: should never happen
                    self._logger.error(
                        f"{self._name}: Buffer is not initialized"
                    )
                    return

                if n_incoming_samples >= self._n_buffer_samples:
                    # If incoming data is too large, keep only the latest part
                    self._buffer[:] = data[-self._n_buffer_samples :]
                else:
                    # Shift existing data to make space
                    self._buffer[:-n_incoming_samples] = self._buffer[
                        n_incoming_samples:
                    ]
                    self._buffer[-n_incoming_samples:] = data

                # Notify the processing thread that new data is available
                self._new_data_event.set()
        except Exception as e:
            self._logger.error(
                f"{self._name} (update): process error: {e}\n"
                f"{traceback.format_exc()}"
            )

    def start(self) -> None:
        """Start the processing thread."""
        self._thread_controller.start()
        self._logger.info(f"{self._name} started.")

    def stop(self) -> None:
        """Stop the processing thread and clean up."""
        self._thread_controller.stop()
        self._logger.info(f"{self._name} closed.")

    def _run(self) -> None:
        """Thread function to process the buffered data.

        This method runs in a separate thread and processes the buffered
        data as it becomes available. It handles any exceptions that occur
        during processing.

        Raises:
            Exception: If an error occurs while processing data.
        """
        self._logger.info(f"{self._name} running.")
        while self._thread_controller.is_running:
            try:
                # Wait for new data to be available
                if not self._new_data_event.wait(timeout=0.1):
                    continue
                with self._buffer_lock:
                    if self._buffer is None or self._n_frame_samples is None:
                        # Safety check: should never happen
                        self._logger.error(
                            f"{self._name}: Buffer is not initialized"
                        )
                        continue

                    # Process the buffered data
                    chunk = self._buffer[-self._n_frame_samples :].copy()
                    self._new_data_event.clear()

                # Process the buffered data
                self._strategy.handle(chunk)

            except Exception as e:
                self._logger.error(
                    f"{self._name} (_run): process error: {e}\n"
                    f"{traceback.format_exc()}"
                )
                # Exit the thread with a non-zero status
                sys.exit(1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"strategy={repr(self._strategy)}, "
            f"n_buffer_samples={self._n_buffer_samples}, "
            f"n_frame_samples={self._n_frame_samples}, "
            f"logger={repr(self._logger)}, "
            f"name={repr(self._name)})"
        )

    def __str__(self) -> str:
        dtype = None if self._buffer is None else self._buffer.dtype
        shape = None if self._buffer is None else self._buffer.shape
        return (
            f"{self.__class__.__name__} ("
            f"name={self._name}, "
            f"strategy={self._strategy.__class__.__name__}, "
            f"buffer_shape={shape}, "
            f"buffer_dtype={dtype}, "
            f"n_frame_samples={self._n_frame_samples})"
        )


class QueueThreadSubscriber(Subscriber):
    """Threaded subscriber that uses a queue to process data.

    This subscriber uses a queue to buffer incoming data and processes
    it in a separate thread. It is useful for scenarios where data
    arrives continuously and needs to be processed asynchronously.
    """

    def __init__(
        self,
        strategy: my_strategy.FrameHandlingStrategyProtocol,
        queue_max_size: int = 0,
        queue_timeout: float = 0.1,
        logger: my_logger.LoggerProtocol | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the queue subscriber.

        Args:
            strategy: The strategy to handle incoming data frames.
            queue_max_size: The maximum size of the queue.
            queue_timeout: The timeout for getting data from the queue.
            logger: Optional logger for logging messages.
            name: Optional name for the subscriber.

        Raises:
            ValueError: If queue_max_size or queue_timeout is invalid.
        """
        super().__init__(
            strategy=strategy,
            logger=logger,
            name=name,
        )

        if queue_max_size <= 0:
            raise ValueError("queue_max_size must be greater than 0.")
        self._queue_max_size: Final = queue_max_size

        if queue_timeout <= 0:
            raise ValueError("queue_timeout must be greater than 0.")
        self._queue_timeout: Final = queue_timeout

        self._queue: Final[queue.Queue[npt.NDArray[Any] | None]] = queue.Queue(
            maxsize=self._queue_max_size
        )

        self._thread_controller: Final = my_controller.ThreadController(
            target=self._run,
            logger=self._logger,
            name=f"{self._name} Runner",
        )

    @property
    def queue(self):
        """Get the queue used for buffering data.

        This property allows access to the queue for external
        monitoring or manipulation.
        """
        return self._queue

    def update(self, data: npt.NDArray[Any]) -> None:
        """Update the subscriber with new data.

        This method adds the incoming data to the queue for processing
        in the separate thread.

        Args:
            data: The incoming data to be processed.
        """
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            self._logger.warning(
                f"{self._name} (update): "
                f"Queue is full. Dropping frame with shape: {data.shape}",
            )
        except Exception as e:
            self._logger.error(
                f"{self._name} (update): process error: {e}\n"
                f"{traceback.format_exc()}"
            )

    def start(self) -> None:
        """Start the processing thread.

        This method starts the thread that will process data from the queue.
        """
        self._thread_controller.start()
        self._logger.info(f"{self._name} started.")

    def stop(self) -> None:
        """Stop the processing thread and clean up.

        This method stops the thread and cleans up any remaining items
        in the queue.
        """
        self._queue.put(None)  # Signal to stop the thread

        remaining_items = self._queue.qsize()
        self._logger.info(
            f"{self._name} stopped with {remaining_items} "
            f"unprocessed item(s) left in the queue."
        )

        self._thread_controller.stop()

        self._logger.info(f"{self._name} closed.")

    def _run(self) -> None:
        """Thread function to process data from the queue.

        This method runs in a separate thread and processes data
        from the queue as it becomes available. It handles any
        exceptions that occur during processing.

        Raises:
            Exception: If an error occurs while processing data.
        """
        self._logger.info(f"{self._name} running.")
        while True:
            try:
                data = self._queue.get(timeout=self._queue_timeout)
                if data is None:
                    break
                self._strategy.handle(data)
            except queue.Empty:
                if not self._thread_controller.is_running:
                    # If the thread is not running, exit the loop
                    break
                continue
            except Exception as e:
                self._logger.error(
                    f"{self._name} (_run): process error: {e}\n"
                    f"{traceback.format_exc()}"
                )
                # Exit the thread with a non-zero status
                sys.exit(1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"strategy={repr(self._strategy)}, "
            f"queue_max_size={self._queue_max_size}, "
            f"queue_timeout={self._queue_timeout}, "
            f"logger={repr(self._logger)}, "
            f"name={repr(self._name)})"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} ("
            f"name={self._name}, "
            f"strategy={self._strategy.__class__.__name__}, "
            f"queue size: {self._queue.qsize()})"
        )


class QueueProcessSubscriber(Subscriber):
    """Multiprocessing subscriber that uses a queue to process data.

    This subscriber uses a queue to buffer incoming data and processes
    it in a separate process. It is useful for scenarios where data
    arrives continuously and needs to be processed asynchronously.
    """

    def __init__(
        self,
        strategy: my_strategy.FrameHandlingStrategyProtocol,
        queue_max_size: int = 0,
        queue_timeout: float = 0.1,
        logger: my_logger.LoggerProtocol | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the queue subscriber.

        Args:
            strategy: The strategy to handle incoming data frames.
            queue_max_size: The maximum size of the queue.
            queue_timeout: The timeout for getting data from the queue.
            logger: Optional logger for logging messages.
            name: Optional name for the subscriber.

        Raises:
            ValueError: If queue_max_size or queue_timeout is invalid.
        """
        super().__init__(
            strategy=strategy,
            logger=logger,
            name=name,
        )
        if queue_max_size <= 0:
            raise ValueError("queue_max_size must be greater than 0.")
        self._queue_max_size: Final = queue_max_size

        if queue_timeout <= 0:
            raise ValueError("queue_timeout must be greater than 0.")
        self._queue_timeout: Final = queue_timeout

        self._queue: Final[multiprocessing.Queue[npt.NDArray[Any] | None]] = (
            multiprocessing.Queue(maxsize=self._queue_max_size)
        )

        self._process_controller = my_controller.ProcessController(
            target=self._run,
            logger=my_logger.PrintLogger(),
            name=f"{self._name} Runner",
        )

    @property
    def queue(self):
        """Get the queue used for buffering data.

        This property allows access to the queue for external
        monitoring or manipulation.
        """
        return self._queue

    def update(self, data: npt.NDArray[Any]) -> None:
        """Update the subscriber with new data.

        This method adds the incoming data to the queue for processing
        in the separate process.

        Args:
            data: The incoming data to be processed.
        """
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            self._logger.warning(
                f"{self._name} (update): "
                f"Queue is full. Dropping frame with shape: {data.shape}"
            )
        except Exception as e:
            self._logger.error(
                f"{self._name} (update): process error: {e}\n"
                f"{traceback.format_exc()}"
            )

    def start(self) -> None:
        """Start the processing process.

        This method starts the process that will process data from the queue.
        """
        self._process_controller.start()
        self._logger.info(f"{self._name} started.")

    def stop(self) -> None:
        """Stop the processing process and clean up.

        This method stops the process and cleans up any remaining items
        in the queue.
        """
        self._queue.put(None)  # Signal to stop the process

        try:
            remaining_items = self._queue.qsize()
            self._logger.info(
                f"{self._name} stopped with {remaining_items} "
                f"unprocessed item(s) left in the queue."
            )
        except NotImplementedError:
            self._logger.warning(
                "Queue.qsize() is not implemented on this platform."
            )

        self._process_controller.stop()

        self._logger.info(f"{self._name} closed.")

    def _run(self) -> None:
        """Process function to process data from the queue.

        This method runs in a separate process and processes data
        from the queue as it becomes available. It handles any
        exceptions that occur during processing.
        """
        self._logger.info(f"{self._name} running.")
        while True:
            try:
                data = self._queue.get(timeout=self._queue_timeout)
                if data is None:
                    break
                self._strategy.handle(data)
            except queue.Empty:
                if not self._process_controller.is_running:
                    # If the thread is not running, exit the loop
                    break
                continue
            except Exception as e:
                self._logger.error(
                    f"{self._name} (_run): fatal error: {e}\n"
                    f"{traceback.format_exc()}"
                )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"strategy={repr(self._strategy)}, "
            f"queue_max_size={self._queue_max_size}, "
            f"queue_timeout={self._queue_timeout}, "
            f"logger={repr(self._logger)}, "
            f"name={repr(self._name)})"
        )

    def __str__(self) -> str:
        try:
            queue_size_str = str(self._queue.qsize())
        except NotImplementedError:
            queue_size_str = "unknown on this platform"

        return (
            f"{self.__class__.__name__} ("
            f"name={self._name}, "
            f"strategy={self._strategy.__class__.__name__}, "
            f"queue size: {queue_size_str})"
        )
