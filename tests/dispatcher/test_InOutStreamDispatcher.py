import pytest
from unittest.mock import MagicMock
from typing import Tuple
import numpy as np

from rei_rt_audio.dispatcher import InOutStreamDispatcher
from rei_rt_audio.publisher import PublisherProtocol
from rei_rt_audio.logger import LoggerProtocol


@pytest.fixture
def mock_logger() -> MagicMock:
    """Return a mock logger implementing LoggerProtocol."""
    return MagicMock(spec=LoggerProtocol)


@pytest.fixture
def dispatcher(
    mock_logger: MagicMock,
) -> Tuple[InOutStreamDispatcher, list[MagicMock], MagicMock]:
    """Create and return an InOutStreamDispatcher instance with mock
    publishers and a mock logger."""
    publisher1: MagicMock = MagicMock(spec=PublisherProtocol)
    publisher2: MagicMock = MagicMock(spec=PublisherProtocol)

    dispatcher_instance = InOutStreamDispatcher(
        publishers=[publisher1, publisher2], logger=mock_logger
    )
    return dispatcher_instance, [publisher1, publisher2], mock_logger


def test_init_dispatcher_calls_logger_debug() -> None:
    """Ensure that logger.debug is called during initialization."""
    publishers = [
        MagicMock(spec=PublisherProtocol),
        MagicMock(spec=PublisherProtocol),
    ]
    logger = MagicMock(spec=LoggerProtocol)

    _ = InOutStreamDispatcher(publishers=publishers, logger=logger)

    logger.debug.assert_called_once()
    args, _ = logger.debug.call_args
    assert "initialized with" in args[0]


def test_consume_calls_publish(
    dispatcher: Tuple[InOutStreamDispatcher, list[MagicMock], MagicMock],
) -> None:
    """Ensure that the consume method calls publish on all publishers."""
    dispatcher_instance, publishers, logger = dispatcher
    data = np.array([1, 2, 3])

    dispatcher_instance.consume(data)

    for pub in publishers:
        pub.publish.assert_called_once()
        np.testing.assert_array_equal(pub.publish.call_args[1]["data"], data)

    logger.error.assert_not_called()


def test_consume_handles_exception_and_calls_error(
    dispatcher: Tuple[InOutStreamDispatcher, list[MagicMock], MagicMock],
) -> None:
    """Ensure that exceptions in publish do not stop other publishers and
    logger.error is called."""
    dispatcher_instance, publishers, logger = dispatcher
    data = np.array([4, 5, 6])

    # Simulate an exception in the first publisher's publish method
    publishers[0].publish.side_effect = Exception("Test error")

    dispatcher_instance.consume(data)

    publishers[0].publish.assert_called_once()
    publishers[1].publish.assert_called_once()

    logger.error.assert_called_once()
    args, _ = logger.error.call_args
    assert "Error in publisher" in args[0]


def test_repr(
    dispatcher: Tuple[InOutStreamDispatcher, list[MagicMock], MagicMock],
) -> None:
    """Ensure that __repr__ returns the correct string representation."""
    dispatcher_instance, publishers, logger = dispatcher

    result = repr(dispatcher_instance)

    assert "InOutStreamDispatcher" in result
    for pub in publishers:
        assert repr(pub) in result
    assert repr(logger) in result


def test_str(
    dispatcher: Tuple[InOutStreamDispatcher, list[MagicMock], MagicMock],
) -> None:
    """Ensure that __str__ returns the correct string representation."""
    dispatcher_instance, publishers, _ = dispatcher

    result = str(dispatcher_instance)

    assert "InOutStreamDispatcher" in result
    assert f"{len(publishers)} publisher(s)" in result
