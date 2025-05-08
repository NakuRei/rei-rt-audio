import abc
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib.lines

from . import calculate as my_calculate
from . import consumer as my_consumer
from . import logger as my_logger


class Plotter(my_consumer.Consumer, abc.ABC):
    def __init__(
        self,
        channel_numbers: list[int],
        max_points: int,
        y_limit: tuple[float, float],
        initial_y_value: float,
        plotting_interval_ms: int = 100,
        logger: my_logger.LoggerProtocol | None = None,
        is_debug_mode: bool = False,
    ):
        super().__init__(logger=logger)

        self._channel_numbers: Final[list[int]] = channel_numbers
        self._max_points: Final[int] = max_points
        self._is_debug_mode: Final[bool] = is_debug_mode

        self._initialize_plot(
            y_limit=y_limit,
            initial_y_value=initial_y_value,
            plotting_interval_ms=plotting_interval_ms,
        )

        self._ch_numb_idx_map: Final = {
            ch_numb: ch_idx
            for ch_idx, ch_numb in enumerate(self._channel_numbers)
        }

    def _initialize_plot(
        self,
        y_limit: tuple[float, float],
        initial_y_value: float,
        plotting_interval_ms: int,
    ) -> None:
        self._lines: list[matplotlib.lines.Line2D] = []
        self._data_values: list[list[Any]] = []

        plt.ion()  # Enable interactive mode
        self._fig, self._ax = plt.subplots()
        for ch_numb in self._channel_numbers:
            (line,) = self._ax.plot([], [], label=f"Ch{ch_numb}")
            line.set_xdata(np.arange(self._max_points))
            line.set_ydata([initial_y_value] * self._max_points)
            self._lines.append(line)
            self._data_values.append([initial_y_value] * self._max_points)

        self._ax.set_xlim(0, self._max_points)
        self._ax.set_ylim(y_limit)
        self._ax.legend(loc="upper right")

        self.ani = matplotlib.animation.FuncAnimation(
            self._fig,
            self._update_plot,
            interval=plotting_interval_ms,
            blit=False,
            cache_frame_data=False,
        )

    def _update_plot(
        self,
        frame: int | None = None,
    ) -> list[matplotlib.lines.Line2D]:
        ys = [np.array(values) for values in self._data_values]

        for ch, line in enumerate(self._lines):
            line.set_ydata(ys[ch])

        return self._lines

    @abc.abstractmethod
    def _process_for_plotting(self, data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        pass

    def consume(self, data: npt.NDArray[Any]) -> None:
        if data.ndim != 2:
            raise ValueError("Data must be a 2D array")
        if data.shape[1] < max(self._channel_numbers):
            raise ValueError(
                f"Data must have at least {max(self._channel_numbers)} "
                f"channels"
            )

        data_for_plotting = self._process_for_plotting(data)

        for ch_numb in self._channel_numbers:
            if ch_numb in self._ch_numb_idx_map:
                ch_idx: int = self._ch_numb_idx_map[ch_numb]
                self._data_values[ch_idx].extend(
                    data_for_plotting[:, ch_numb - 1].tolist()
                )
                if len(self._data_values[ch_idx]) > self._max_points:
                    self._data_values[ch_idx] = self._data_values[ch_idx][
                        -self._max_points :
                    ]

            else:
                if self._is_debug_mode:
                    self._logger.debug(
                        (
                            f"Channel number {ch_numb} is not in the specified "
                            f"channel numbers."
                        )
                    )


class RmsDbfsPlotter(Plotter):
    def __init__(
        self,
        channel_numbers: list[int],
        max_points: int,
        y_limit: tuple[float, float] = (-100, 0),
        initial_y_value: float = -100,
        plotting_interval_ms: int = 100,
        logger: my_logger.LoggerProtocol | None = None,
        thresholds: tuple[float, ...] | None = None,
        is_debug_mode: bool = False,
    ):
        super().__init__(
            channel_numbers=channel_numbers,
            max_points=max_points,
            y_limit=y_limit,
            initial_y_value=initial_y_value,
            plotting_interval_ms=plotting_interval_ms,
            logger=logger,
            is_debug_mode=is_debug_mode,
        )

        # NOTE: thresholdsのsetterで_threshold_linesを使うので先に定義しておく
        self._threshold_lines: list[matplotlib.lines.Line2D] = []
        self.thresholds = thresholds

        self._ax.set_xlabel("Samples")
        self._ax.set_ylabel("RMS (dBFS)")
        self._ax.set_title("RMS Level")

    @property
    def thresholds(self) -> tuple[float, ...] | None:
        return self._thresholds

    @thresholds.setter
    def thresholds(self, value: tuple[float, ...] | None) -> None:
        self._thresholds = value
        self._plot_thresholds()

    def _plot_thresholds(self) -> None:
        # すでにある線を削除する
        for line in self._threshold_lines:
            line.remove()
        self._threshold_lines.clear()

        if self.thresholds is not None:
            for threshold in self.thresholds:
                line = self._ax.axhline(
                    y=threshold,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                )
                self._threshold_lines.append(line)

        # 再描画
        self._fig.canvas.draw_idle()

    def _process_for_plotting(self, data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        rms_dbfs = my_calculate.calculate_rms_dbfs(data)

        if not isinstance(rms_dbfs, np.ndarray):
            rms_dbfs = np.array([rms_dbfs])

        return rms_dbfs.reshape(1, -1)


class WaveformPlotter(Plotter):
    def __init__(
        self,
        channel_numbers: list[int],
        max_points: int,
        y_limit: tuple[float, float] = (-1.0, 1.0),
        initial_y_value: float = 0.0,
        plotting_interval_ms: int = 100,
        logger: my_logger.LoggerProtocol | None = None,
        is_debug_mode: bool = False,
    ):
        super().__init__(
            channel_numbers=channel_numbers,
            max_points=max_points,
            y_limit=y_limit,
            initial_y_value=initial_y_value,
            plotting_interval_ms=plotting_interval_ms,
            logger=logger,
            is_debug_mode=is_debug_mode,
        )

        self._ax.set_xlabel("Samples")
        self._ax.set_ylabel("Amplitude")
        self._ax.set_title("Waveform")

    def _process_for_plotting(self, data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return data
