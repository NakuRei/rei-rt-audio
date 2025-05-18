import abc
from typing import Any, Callable, Final, Protocol


import numpy.typing as npt

from . import calculate as my_calculate

PreprocessType = Callable[[npt.NDArray[Any]], npt.NDArray[Any] | None]


class PreprocessorProtocol(Protocol):
    """Protocol for a preprocessor that preprocesses audio data."""

    def preprocess_for_frame(
        self,
        data: npt.NDArray[Any],
    ) -> npt.NDArray[Any] | None: ...


class Preprocessor(abc.ABC):
    """Base class for a preprocessor that preprocesses audio data."""

    def __init__(self) -> None:
        """Initialize the preprocessor."""
        super().__init__()


class SilenceGatePreprocessor(Preprocessor):
    """Preprocessor that removes silent frames from audio data."""

    def __init__(
        self,
        threshold_db: float,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            silence_threshold_db: The threshold in decibels below which
                the audio is considered silent.
        """
        super().__init__()
        self._threshold_db: Final = threshold_db

    @property
    def threshold_db(self):
        """
        The threshold in decibels below which the audio is considered silent.
        """
        return self._threshold_db

    def preprocess_for_frame(
        self,
        data: npt.NDArray[Any],
    ) -> npt.NDArray[Any] | None:
        """Preprocess the audio data for a single frame.

        Args:
            data: The audio data to preprocess.

        Returns:
            The preprocessed audio data, or None if the audio is silent.
        """
        if my_calculate.is_silent(
            data,
            threshold_db=self._threshold_db,
        ):
            return None

        return data
