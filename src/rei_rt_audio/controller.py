import multiprocessing
import threading
import traceback
from typing import Callable, Final, Self

from . import logger as my_logger


class ThreadController:
    """A class to control a thread.

    This class provides methods to start and stop a thread, and
    check if the thread is running. It also provides a context
    manager interface for easy usage in a with statement.

    Attributes:
        target: The target function to run in the thread.
        logger: The logger to use for logging messages.
        name: The name of the thread.
    """

    def __init__(
        self,
        target: Callable[[], None],
        logger: my_logger.LoggerProtocol,
        name: str = "ThreadController",
    ) -> None:
        """Initialize the ThreadController.

        Args:
            target: The target function to run in the thread.
            logger: The logger to use for logging messages.
            name: The name of the thread.
        """
        self._target = target
        self._logger = logger
        self._name = name

        self._ready_event: Final = threading.Event()
        self._running_event: Final = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        """Check if the thread is running.

        Returns:
            bool: True if the thread is running, False otherwise.
        """
        return self._running_event.is_set()

    def _target_with_event(self) -> None:
        self._running_event.set()
        self._ready_event.set()
        try:
            self._target()
        except Exception as e:
            self._logger.error(
                f"{self._name} thread error: {e}\n{traceback.format_exc()}"
            )
        finally:
            self._running_event.clear()

    def start(self) -> None:
        """Start the thread.

        If the thread is already running, a warning is logged.
        """
        if self._thread is not None and self._thread.is_alive():
            self._logger.warning(f"{self._name} thread is already running.")
            return

        self._logger.debug(f"{self._name} thread starting...")

        self._running_event.clear()
        self._ready_event.clear()
        self._thread = threading.Thread(
            target=self._target_with_event,
            name=self._name,
            daemon=True,
        )
        self._thread.start()
        self._ready_event.wait()

        self._logger.debug(f"{self._name} thread started.")

    def stop(self) -> None:
        """Stop the thread.

        If the thread is not running, a warning is logged.
        """
        if self._thread is None:
            self._logger.warning(f"{self._name} thread is not running.")
            return

        self._logger.debug(f"{self._name} thread stopping...")
        self._running_event.clear()
        self._thread.join()

        self._thread = None  # Clear thread explicitly
        self._logger.debug(f"{self._name} thread stopped.")

    def __enter__(self) -> Self:
        """Start the thread when entering the context manager."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Stop the thread when exiting the context manager."""
        self.stop()

    def __repr__(self) -> str:
        return (
            f"{self._name}("
            f"target={repr(self._target)}, "
            f"logger={repr(self._logger)}, "
            f"name={repr(self._name)})"
        )

    def __str__(self) -> str:
        return (
            f"{self._name} ("
            f"name={self._name}, "
            f"is_running={self.is_running}, "
            f"target={self._target.__name__})"
        )


class ProcessController:
    """A class to control a process.

    This class provides methods to start and stop a process, and
    check if the process is running. It also provides a context
    manager interface for easy usage in a with statement.

    Attributes:
        target: The target function to run in the process.
        logger: The logger to use for logging messages.
        name: The name of the process.
    """

    def __init__(
        self,
        target: Callable[[], None],
        logger: my_logger.LoggerProtocol,
        name: str = "ProcessController",
        timeout: float = 5.0,
    ) -> None:
        """Initialize the ProcessController.

        Args:
            target: The target function to run in the process.
            logger: The logger to use for logging messages.
            name: The name of the process.
        """
        self._target: Final = target
        self._logger: Final = logger
        self._name: Final[str] = name
        self._timeout: Final[float] = timeout

        self._ready_event: Final = multiprocessing.Event()
        manager: Final = multiprocessing.Manager()
        self._running_event: Final = manager.Event()
        self._process: multiprocessing.Process | None = None

    @property
    def is_running(self) -> bool:
        """Check if the process is running.

        Returns:
            bool: True if the process is running, False otherwise.
        """
        return self._running_event.is_set()

    def _target_with_event(self) -> None:
        self._running_event.set()
        self._ready_event.set()
        try:
            self._target()
        except Exception as e:
            self._logger.error(
                f"{self._name} process error: {e}\n{traceback.format_exc()}"
            )
        finally:
            self._running_event.clear()

    def start(self) -> None:
        """Start the process.

        If the process is already running, a warning is logged.
        """
        if self._process is not None and self._process.is_alive():
            self._logger.warning(f"{self._name} process is already running.")
            return

        self._logger.debug(f"{self._name} process starting...")

        self._running_event.clear()
        self._ready_event.clear()
        self._process = multiprocessing.Process(
            target=self._target_with_event,
            name=self._name,
            daemon=True,
        )
        self._process.start()
        self._ready_event.wait()

        self._logger.debug(f"{self._name} process started.")

    def stop(self) -> None:
        """Stop the process.

        If the process is not running, a warning is logged.
        """
        if self._process is None:
            self._logger.warning(f"{self._name} process is not running.")
            return

        self._logger.debug(f"{self._name} process stopping...")
        self._running_event.clear()
        self._process.join(timeout=self._timeout)

        if self._process.is_alive():
            self._logger.warning(
                f"{self._name} process did not stop in {self._timeout} "
                f"seconds. Terminating..."
            )
            self._process.terminate()
            self._process.join()

        if self._process.exitcode not in (0, None):
            self._logger.error(
                f"{self._name} process exited with error. "
                f"Exit code: {self._process.exitcode}"
            )

        self._process = None  # Clear process explicitly
        self._logger.debug(f"{self._name} process stopped.")

    def __enter__(self) -> Self:
        """Start the process when entering the context manager."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Stop the process when exiting the context manager."""
        self.stop()

    def __repr__(self) -> str:
        return (
            f"{self._name}("
            f"target={repr(self._target)}, "
            f"logger={repr(self._logger)}, "
            f"name={repr(self._name)})"
        )

    def __str__(self) -> str:
        return (
            f"{self._name} ("
            f"name={self._name}, "
            f"is_running={self.is_running}, "
            f"target={self._target.__name__})"
        )
