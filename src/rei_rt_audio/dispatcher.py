from typing import Any, Final, Sequence

import numpy as np

import numpy.typing as npt

from . import logger as my_logger
from . import publisher as my_publisher
from . import consumer as my_consumer


class InOutStreamDispatcher(my_consumer.Consumer):
    def __init__(
        self,
        publishers: Sequence[my_publisher.PublisherProtocol],
        logger: my_logger.LoggerProtocol | None = None,
    ) -> None:
        super().__init__(logger=logger)

        self._publishers: Final = publishers
        self._logger.debug(
            f"{self.__class__.__name__} initialized with "
            f"{len(publishers)} publisher(s)."
        )

    def consume(self, data: npt.NDArray[Any]) -> None:
        for pub in self._publishers:
            try:
                pub.publish(data=np.copy(data))
            except Exception as e:
                self._logger.error(f"Error in publisher: {e}")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"publishers=[{', '.join(repr(p) for p in self._publishers)}], "
            f"logger={repr(self._logger)})"
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__} with {len(self._publishers)} "
            "publisher(s)"
        )
