import abc
import datetime
import enum
import multiprocessing
import os
import traceback
import threading
from typing import Any, Final, Protocol, Sequence


class LoggerProtocol(Protocol):
    """
    A protocol that defines the methods for a logger.
    This is used to ensure that any logger passed to the LoggerAdapter
    implements the required methods.
    """

    def debug(self, msg: Any) -> None: ...
    def info(self, msg: Any) -> None: ...
    def warning(self, msg: Any) -> None: ...
    def error(self, msg: Any) -> None: ...


class LoggerAdapter:
    """
    A class that adapts a logger to the LoggerProtocol.
    This is useful for integrating with existing logging libraries
    that may not implement the LoggerProtocol directly.
    """

    def __init__(self, logger: Any) -> None:
        super().__init__()
        self._logger = logger

    def debug(self, msg: str) -> None:
        """Log a debug message.

        Args:
            msg (str): The message to log.
        """
        self._logger.debug(msg)

    def info(self, msg: str) -> None:
        """Log an info message.

        Args:
            msg (str): The message to log.
        """
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log a warning message.

        Args:
            msg (str): The message to log.
        """
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log an error message.

        Args:
            msg (str): The message to log.
        """
        self._logger.error(msg)


class LogLevel(enum.Enum):
    """
    Enum for log levels.
    Each log level has a corresponding integer value.
    The higher the value, the more severe the log level.
    """

    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4


class Logger(abc.ABC):
    """
    Abstract base class for logging
    """

    default_log_level: Final[LogLevel] = LogLevel.INFO

    def __init__(
        self,
        log_level: LogLevel | None = None,
        name: str = "Logger",
    ) -> None:
        """Initialize the logger with a log level and name.

        Args:
            log_level (LogLevel | None): The log level for the logger.
                If None, the default log level is used.
            name (str): The name of the logger.
        """
        super().__init__()
        self.log_level = log_level or self.default_log_level
        self.name = name

    def debug(self, msg: str) -> None:
        """
        Log a debug message if the log level is set to DEBUG or lower.

        Args:
            msg (str): The message to log.
        """
        if self._should_log(LogLevel.DEBUG):
            self._debug(msg)

    def info(self, msg: str) -> None:
        """
        Log an info message if the log level is set to INFO or lower.

        Args:
            msg (str): The message to log.
        """
        if self._should_log(LogLevel.INFO):
            self._info(msg)

    def warning(self, msg: str) -> None:
        """
        Log a warning message if the log level is set to WARNING or lower.

        Args:
            msg (str): The message to log.
        """
        if self._should_log(LogLevel.WARNING):
            self._warning(msg)

    def error(self, msg: str) -> None:
        """
        Log an error message if the log level is set to ERROR or lower.

        Args:
            msg (str): The message to log.
        """
        if self._should_log(LogLevel.ERROR):
            self._error(msg)

    def _should_log(self, level: LogLevel) -> bool:
        """
        Check if the message should be logged based on the current log
        level.

        Args:
            level (LogLevel): The log level of the message.
        Returns:
            bool: True if the message should be logged, False otherwise.
        """
        return level.value >= self.log_level.value

    def _format_message(self, level: LogLevel, msg: str) -> str:
        """
        Format the log message with a timestamp, thread name, and log level.

        Args:
            level (LogLevel): The log level of the message.
            msg (str): The message to log.

        Returns:
            str: The formatted log message.
        """
        thread = threading.current_thread()
        process = multiprocessing.current_process()
        process_name = f"[{process.name} (pid={os.getpid()})]"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"[{timestamp}] [{level.name}] "
            f"[Process: {process_name}] [Thread: {thread.name}] "
            f"[{self.name}] {msg}"
        )

    @abc.abstractmethod
    def _debug(self, msg: str) -> None:
        """Log a debug message."""
        pass

    @abc.abstractmethod
    def _info(self, msg: str) -> None:
        """Log an info message."""
        pass

    @abc.abstractmethod
    def _warning(self, msg: str) -> None:
        """Log a warning message."""
        pass

    @abc.abstractmethod
    def _error(self, msg: str) -> None:
        """Log an error message."""
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"log_level=LogLevel.{self.log_level.name}, "
            f"name={repr(self.name)})"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__} (level={self.log_level.name})"


class PrintLogger(Logger):
    """
    A logger that prints log messages to the console.
    """

    COLORS = {
        "DEBUG": "\033[94m",  # 青
        "INFO": "\033[92m",  # 緑
        "WARNING": "\033[93m",  # 黄
        "ERROR": "\033[91m",  # 赤
        "RESET": "\033[0m",  # リセット
    }

    def __init__(
        self,
        log_level: LogLevel | None = None,
        name: str = "PrintLogger",
        use_color: bool = True,
    ) -> None:
        """
        Initialize the logger with a log level, name, and color option.

        Args:
            log_level (LogLevel | None): The log level for the logger.
                If None, the default log level is used.
            name (str): The name of the logger.
            use_color (bool): Whether to use color in the console output.
        """
        super().__init__(log_level=log_level, name=name)
        self.use_color = use_color

    def _log(self, level: LogLevel, msg: str) -> None:
        """Log a message to the console with color formatting.

        Args:
            level (LogLevel): The log level of the message.
            msg (str): The message to log.
        """
        formatted_msg = self._format_message(level, msg)
        color = self.COLORS.get(level.name, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        if self.use_color:
            print(f"{color}{formatted_msg}{reset}", flush=True)
        else:
            print(formatted_msg, flush=True)

    def _debug(self, msg: str) -> None:
        self._log(LogLevel.DEBUG, msg)

    def _info(self, msg: str) -> None:
        self._log(LogLevel.INFO, msg)

    def _warning(self, msg: str) -> None:
        self._log(LogLevel.WARNING, msg)

    def _error(self, msg: str) -> None:
        self._log(LogLevel.ERROR, msg)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"log_level=LogLevel.{self.log_level.name}, "
            f"name={repr(self.name)}, "
            f"use_color={self.use_color})"
        )

    def __str__(self) -> str:
        color_flag = "color" if self.use_color else "no color"
        return f"PrintLogger (level={self.log_level.name}, {color_flag})"


class FileLogger(Logger):
    """
    A logger that writes log messages to a file.
    """

    def __init__(
        self,
        log_file: str,
        log_level: LogLevel | None = None,
        name: str = "FileLogger",
    ) -> None:
        super().__init__(log_level=log_level, name=name)
        self.log_file = log_file
        self._lock: Final[threading.Lock] = threading.Lock()

    def _log(self, level: LogLevel, msg: str) -> None:
        with self._lock:
            formatted_msg = self._format_message(level, msg)
            with open(self.log_file, "a") as f:
                f.write(f"{formatted_msg}\n")
                f.flush()

    def _debug(self, msg: str) -> None:
        self._log(LogLevel.DEBUG, msg)

    def _info(self, msg: str) -> None:
        self._log(LogLevel.INFO, msg)

    def _warning(self, msg: str) -> None:
        self._log(LogLevel.WARNING, msg)

    def _error(self, msg: str) -> None:
        self._log(LogLevel.ERROR, msg)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"log_file={repr(self.log_file)}, "
            f"log_level=LogLevel.{self.log_level.name}, "
            f"name={repr(self.name)})"
        )

    def __str__(self) -> str:
        return (
            f"FileLogger (file='{self.log_file}', level={self.log_level.name})"
        )


class MultiLogger(Logger):
    """
    A logger that sends log messages to multiple loggers.
    """

    def __init__(
        self,
        loggers: Sequence[Logger],
        log_level: LogLevel | None = None,
        name: str = "MultiLogger",
    ) -> None:
        """
        Initialize the logger with a list of loggers.

        Args:
            loggers (Sequence[Logger]): A list of loggers to send messages to.
            log_level (LogLevel | None): The log level for the logger.
                If None, the default log level is used.
            name (str): The name of the logger.
        """
        super().__init__(log_level=log_level, name=name)
        self.loggers = loggers

    def _debug(self, msg: str) -> None:
        for logger in self.loggers:
            try:
                logger.debug(msg)
            except Exception as e:
                print(
                    f"Error logging to {logger.name} in MultiLogger: "
                    f"{e}\n{traceback.format_exc()}"
                )

    def _info(self, msg: str) -> None:
        for logger in self.loggers:
            try:
                logger.info(msg)
            except Exception as e:
                print(
                    f"Error logging to {logger.name} in MultiLogger: "
                    f"{e}\n{traceback.format_exc()}"
                )

    def _warning(self, msg: str) -> None:
        for logger in self.loggers:
            try:
                logger.warning(msg)
            except Exception as e:
                print(
                    f"Error logging to {logger.name} in MultiLogger: "
                    f"{e}\n{traceback.format_exc()}"
                )

    def _error(self, msg: str) -> None:
        for logger in self.loggers:
            try:
                logger.error(msg)
            except Exception as e:
                print(
                    f"Error logging to {logger.name} in MultiLogger: "
                    f"{e}\n{traceback.format_exc()}"
                )

    def __repr__(self) -> str:
        loggers_repr = (
            "[" + ", ".join(repr(logger) for logger in self.loggers) + "]"
        )
        return (
            f"{self.__class__.__name__}("
            f"loggers={loggers_repr}, "
            f"log_level=LogLevel.{self.log_level.name}, "
            f"name={repr(self.name)})"
        )

    def __str__(self) -> str:
        names = ", ".join(str(logger) for logger in self.loggers)
        return f"MultiLogger [{names}]"
