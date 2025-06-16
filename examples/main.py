import os
import tkinter as tk
from typing import Any, Final, Sequence

import numpy.typing as npt

# Set environment variable before importing sounddevice.
os.environ["SD_ENABLE_ASIO"] = "1"
import sounddevice as sd  # noqa:E402

import rei_rt_audio  # noqa:E402


class AudioSystemFacade(rei_rt_audio.callback.InOutStreamCallbackHandler):
    def __init__(
        self,
        process_seconds: float,
        sample_rate: int,
        blocksize: int,
        logger: rei_rt_audio.logger.Logger,
    ) -> None:
        super().__init__(logger=logger)

        # Dispatcher
        self._subscribers = [
            rei_rt_audio.subscriber.QueueThreadSubscriber(
                strategy=rei_rt_audio.strategy.FrameProcessingStrategy(
                    consumer=rei_rt_audio.plotter.RmsDbfsPlotter(
                        max_points=256,
                        channel_numbers=[1, 2],
                        logger=self._logger,
                    ),
                ),
                queue_max_size=blocksize * 2,
                name="For RMS Plot Thread",
            ),
            rei_rt_audio.subscriber.QueueThreadSubscriber(
                strategy=rei_rt_audio.strategy.FrameProcessingStrategy(
                    consumer=rei_rt_audio.plotter.WaveformPlotter(
                        max_points=int(sample_rate * process_seconds),
                        channel_numbers=[1, 2],
                        logger=self._logger,
                    ),
                ),
                queue_max_size=blocksize * 2,
                name="For Waveform Plot Thread",
            ),
        ]
        publisher = rei_rt_audio.publisher.FramePublisher(
            logger=self._logger,
            subscribers=self._subscribers,
        )
        self._dispatcher = rei_rt_audio.dispatcher.InOutStreamDispatcher(
            publishers=[publisher],
            logger=self._logger,
        )

        # Processors
        self._gain_processor = rei_rt_audio.processor.GainDbProcessor(
            gain_db=0.0,
            logger=self._logger,
        )

    @property
    def subscribers(self) -> Sequence[rei_rt_audio.subscriber.Subscriber]:
        return self._subscribers

    @property
    def gain_db(self) -> float:
        return self._gain_processor.gain_db

    @gain_db.setter
    def gain_db(self, value: float) -> None:
        self._gain_processor.gain_db = value

    def callback(
        self,
        in_data: npt.NDArray[Any],
        out_data: npt.NDArray[Any],
        frames: int,
        time: dict[str, Any],
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            self._logger.warning(
                self._format_log(
                    f"{self.__class__.__name__} callback status: {status}",
                    frames,
                    time,
                )
            )

        self._gain_processor.process(
            in_data=in_data,
            out_data=out_data,
            frames=frames,
        )

        self._dispatcher.consume(data=out_data)

    def start(self) -> None:
        for subscriber in self._subscribers:
            subscriber.start()

    def stop(self) -> None:
        for subscriber in self._subscribers:
            try:
                subscriber.stop()
            except Exception as e:
                self._logger.error(f"Error stopping subscriber: {e}")


class GuiApp:
    def __init__(
        self,
        audio_system_facade: AudioSystemFacade,
        logger: rei_rt_audio.logger.Logger,
    ) -> None:
        self._logger: Final = logger
        self._audio_system_facade: Final = audio_system_facade

    def _build_gui(self) -> None:
        self._root = tk.Tk()
        self._root.title("Silent Threshold Controller")

        self._gain_db_slider = tk.Scale(
            self._root,
            from_=-100,
            to=12,
            resolution=1,
            orient="horizontal",
            label="Silence Threshold (dB)",
            command=self._update_gain_db,
            length=300,
        )
        self._gain_db_slider.set(self._audio_system_facade.gain_db)
        self._gain_db_slider.pack(padx=10, pady=10)

    def _update_gain_db(self, value: str) -> None:
        try:
            gain_db = float(value)
            self._audio_system_facade.gain_db = gain_db
            self._logger.info(f"Gain updated to {gain_db} dB")
        except ValueError:
            self._logger.error(f"Invalid gain value: {value}")

    def run_gui(self):
        try:
            self._build_gui()
            self._logger.info(f"{self.__class__.__name__} started (GUI open).")
            self._root.mainloop()
        finally:
            self._logger.info(
                f"{self.__class__.__name__} stopped (GUI closed)."
            )


if __name__ == "__main__":
    # Define parameters for the stream.

    in_device = "Line (2- Yamaha AG06MK2), MME"
    out_device = "VoiceMeeter Aux Input (VB-Audio, MME"
    device_sample_rate = 48000
    device_blocksize = 1024
    device_dtype = None
    device_latency = "low"
    device_channels = 2

    process_seconds = 1.5

    logger = rei_rt_audio.logger.PrintLogger(rei_rt_audio.logger.LogLevel.INFO)
    audio_system_facade = AudioSystemFacade(
        process_seconds=process_seconds,
        sample_rate=device_sample_rate,
        blocksize=device_blocksize,
        logger=logger,
    )
    app = GuiApp(
        audio_system_facade=audio_system_facade,
        logger=logger,
    )

    try:
        audio_system_facade.start()
        with sd.Stream(
            device=(in_device, out_device),
            channels=device_channels,
            samplerate=device_sample_rate,
            blocksize=device_blocksize,
            dtype=device_dtype,
            latency=device_latency,
            callback=audio_system_facade.callback,
        ) as stream:
            print("Stream started.")
            print("Stream signal type:", stream.dtype)  # type: ignore
            print("Stream blocksize:", stream.blocksize)  # type: ignore
            print("Stream latency:", stream.latency)  # type: ignore
            print("Stream channels:", stream.channels)  # type: ignore
            print("Stream sample rate:", stream.samplerate)  # type: ignore
            print("Stream device:", stream.device)  # type: ignore
            print("#" * 20)
            print("Close GUI to stop the stream.")
            print("#" * 20)
            app.run_gui()
    except KeyboardInterrupt:
        print("Stream stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        audio_system_facade.stop()
        print("Stream stopped.")

    print("Finished.")
