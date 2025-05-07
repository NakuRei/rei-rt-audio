import abc
from typing import Any, Protocol

import numpy.typing as npt

from . import calculate as my_calculate
from . import logger as my_logger


class ConsumerProtocol(Protocol):
    def consume(self, data: npt.NDArray[Any]) -> None: ...


class Consumer(abc.ABC):
    def __init__(
        self,
        logger: my_logger.LoggerProtocol | None = None,
    ):
        if logger is None:
            logger = my_logger.PrintLogger()
        self._logger = logger

    @abc.abstractmethod
    def consume(self, data: npt.NDArray[Any]) -> None:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(logger={repr(self._logger)})"

    def __str__(self):
        return f"{self.__class__.__name__}"


class MaxAmpPrinter(Consumer):
    def consume(self, data: npt.NDArray[Any]) -> None:
        max_amp = data.max()
        print(f"Max amplitude: {max_amp}", flush=True)


class RmsPrinter(Consumer):
    def consume(self, data: npt.NDArray[Any]) -> None:
        rms = my_calculate.calculate_rms_dbfs(data)
        print(f"RMS: {rms}", flush=True)
