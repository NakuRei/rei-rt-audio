import abc
from typing import Any, Final, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from . import logger as my_logger
from . import subscriber as my_subscriber


class PublisherProtocol(Protocol):
    """Protocol for a publisher.
    This protocol defines the interface for a publisher that can
    publish data to them.
    """

    def publish(self, data: npt.NDArray[Any]) -> None: ...


class Publisher(abc.ABC):
    def __init__(
        self,
        logger: my_logger.LoggerProtocol | None = None,
        subscribers: Sequence[my_subscriber.SubscriberProtocol] | None = None,
    ) -> None:
        """Initialize the publisher.

        Args:
            logger: The logger to use. If None, a default logger is used.
            subscribers: A list of subscribers to register. If None, an
                empty list is used.
        """
        super().__init__()

        if logger is None:
            logger = my_logger.PrintLogger()
        self._logger: Final = logger

        if subscribers is None:
            subscribers = []
        self._subscribers: Final = list(subscribers)

    def register(
        self,
        subscriber: my_subscriber.SubscriberProtocol,
    ) -> None:
        """Register a subscriber.

        If the subscriber is already registered, a warning is logged.

        Args:
            subscriber: The subscriber to register.

        Raises:
            TypeError: If the subscriber is not a subclass of
                SubscriberProtocol.
        """
        if subscriber in self._subscribers:
            self._logger.warning(
                f"Subscriber of type {type(subscriber).__name__} is already "
                f"registered."
            )
            return

        self._subscribers.append(subscriber)
        self._logger.debug(
            f"Registered subscriber of type {type(subscriber).__name__}."
        )

    def unregister(
        self,
        subscriber: my_subscriber.SubscriberProtocol,
    ) -> None:
        """Unregister a subscriber.

        If the subscriber is not registered, a warning is logged.

        Args:
            subscriber: The subscriber to unregister.

        Raises:
            TypeError: If the subscriber is not a subclass of
                SubscriberProtocol.
        """
        if subscriber not in self._subscribers:
            self._logger.warning(
                f"Subscriber of type {type(subscriber).__name__} is not "
                f"registered."
            )
            return

        self._subscribers.remove(subscriber)
        self._logger.debug(
            f"Unregistered subscriber of type {type(subscriber).__name__}."
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"subscribers=[{', '.join(repr(s) for s in self._subscribers)}], "
            f"logger={repr(self._logger)})"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} with {len(self._subscribers)} "
            "subscriber(s)"
        )


class FramePublisher(Publisher):
    """Publisher for frame data.

    This publisher is used to publish frame data to subscribers.
    """

    def publish(self, data: npt.NDArray[Any]) -> None:
        """Publish frame data to subscribers.
        This method is called to publish frame data to all registered
        subscribers. It makes a copy of the data to ensure that the
        subscribers receive a snapshot of the data at the time of
        publishing.

        Args:
            data: The frame data to publish. This should be a numpy array
                of shape (n, m, c), where n is the height, m is the width,
                and c is the number of channels.
        """
        for sub in self._subscribers:
            try:
                sub.update(np.copy(data))
            except Exception as e:
                self._logger.error(
                    f"Error in subscriber {type(sub).__name__}: {e}"
                )
