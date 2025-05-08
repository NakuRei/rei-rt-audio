import os
import tkinter as tk
from typing import Any

import numpy.typing as npt

# Set environment variable before importing sounddevice.
os.environ["SD_ENABLE_ASIO"] = "1"
import sounddevice as sd  # noqa:E402

import rei_rt_audio  # noqa:E402


class GuiApp(rei_rt_audio.callback.InOutStreamCallbackHandler):
    def __init__(
        self,
        dispatcher: rei_rt_audio.dispatcher.InOutStreamDispatcher,
        logger: rei_rt_audio.logger.Logger,
    ) -> None:
        super().__init__(logger=logger)
        self._dispatcher = dispatcher

        self._gain_processor = rei_rt_audio.processor.GainDbProcessor(
            gain_db=0.0,
            logger=self._logger,
        )

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
        self._gain_db_slider.set(self._gain_processor.gain_db)  # type: ignore
        self._gain_db_slider.pack(padx=10, pady=10)

    def _update_gain_db(self, value: str) -> None:
        try:
            gain_db = float(value)
            self._gain_processor.gain_db = gain_db
            self._logger.info(f"Gain updated to {gain_db} dB")
        except ValueError:
            self._logger.error(f"Invalid gain value: {value}")

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
    subscribers = [
        rei_rt_audio.subscriber.QueueThreadSubscriber(
            strategy=rei_rt_audio.strategy.AccumulatingBufferProcessingStrategy(
                n_buffer_samples=int(device_sample_rate * process_seconds),
                consumer=rei_rt_audio.consumer.MaxAmpPrinter(logger=logger),
                logger=logger,
            ),
            queue_max_size=device_blocksize * 2,
            logger=logger,
            name="For Max Amp Thread",
        ),
        rei_rt_audio.subscriber.QueueProcessSubscriber(
            strategy=rei_rt_audio.strategy.AccumulatingBufferProcessingStrategy(
                n_buffer_samples=int(device_sample_rate * process_seconds),
                consumer=rei_rt_audio.consumer.RmsPrinter(logger=logger),
                logger=logger,
            ),
            queue_max_size=device_blocksize * 2,
            logger=logger,
            name="For RMS Process",
        ),
        rei_rt_audio.subscriber.QueueThreadSubscriber(
            strategy=rei_rt_audio.strategy.FrameProcessingStrategy(
                consumer=rei_rt_audio.plotter.RmsDbfsPlotter(
                    max_points=256,
                    channel_numbers=[1, 2],
                    logger=logger,
                ),
                logger=logger,
            ),
            queue_max_size=device_blocksize * 2,
            logger=logger,
            name="For RMS Plot Thread",
        ),
        rei_rt_audio.subscriber.QueueThreadSubscriber(
            strategy=rei_rt_audio.strategy.FrameProcessingStrategy(
                consumer=rei_rt_audio.plotter.WaveformPlotter(
                    max_points=int(device_sample_rate * process_seconds),
                    channel_numbers=[1, 2],
                    logger=logger,
                ),
                logger=logger,
            ),
            queue_max_size=device_blocksize * 2,
            logger=logger,
            name="For Waveform Plot Thread",
        ),
    ]
    publisher = rei_rt_audio.publisher.FramePublisher(
        logger=logger,
        subscribers=subscribers,
    )
    dispatcher = rei_rt_audio.dispatcher.InOutStreamDispatcher(
        publishers=[publisher],
        logger=logger,
    )

    app = GuiApp(logger=logger, dispatcher=dispatcher)

    try:
        for subscriber in subscribers:
            subscriber.start()

        with sd.Stream(
            device=(in_device, out_device),
            channels=device_channels,
            samplerate=device_sample_rate,
            blocksize=device_blocksize,
            dtype=device_dtype,
            latency=device_latency,
            callback=app.callback,
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
        print("Stream stopped.")
        for subscriber in subscribers:
            try:
                subscriber.stop()
            except Exception as e:
                print(f"Error stopping subscriber: {e}")

    print("Finished.")
