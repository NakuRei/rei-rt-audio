import os
from typing import Any

import numpy.typing as npt

# Set environment variable before importing sounddevice.
os.environ["SD_ENABLE_ASIO"] = "1"
import sounddevice as sd  # noqa:E402

import rei_rt_audio  # noqa:E402


class Passthrough(rei_rt_audio.callback.InOutStreamCallbackHandler):
    """Passthrough callback that simply copies input to output."""

    def callback(
        self,
        in_data: npt.NDArray[Any],
        out_data: npt.NDArray[Any],
        frames: int,
        time: dict[str, Any],
        status: sd.CallbackFlags,
    ) -> None:
        """Passthrough callback that simply copies input to output.

        Args:
            in_data: Input data to process.
            out_data: Output data to write the processed result to.
            frames: Number of frames in the input data.
            time: A dictionary containing time information.
            status: Callback status flags.
        """
        if status:
            self._logger.warning(
                self._format_log(
                    f"{self.__class__.__name__} callback status: {status}",
                    frames,
                    time,
                )
            )
        out_data[:] = in_data


if __name__ == "__main__":
    # Define parameters for the stream.

    in_device = "VoiceMeeter Output (VB-Audio Vo, MME"
    out_device = "VoiceMeeter Aux Input (VB-Audio, MME"
    device_sample_rate = 48000
    device_blocksize = 1024
    device_dtype = None
    device_latency = "low"
    device_channels = 2

    callback_obj = Passthrough()
    try:
        with sd.Stream(
            device=(in_device, out_device),
            channels=device_channels,
            samplerate=device_sample_rate,
            blocksize=device_blocksize,
            dtype=device_dtype,
            latency=device_latency,
            callback=callback_obj.callback,
        ) as stream:
            print("Stream started.")
            print("Stream signal type:", stream.dtype)  # type: ignore
            print("Stream blocksize:", stream.blocksize)  # type: ignore
            print("Stream latency:", stream.latency)  # type: ignore
            print("Stream channels:", stream.channels)  # type: ignore
            print("Stream sample rate:", stream.samplerate)  # type: ignore
            print("Stream device:", stream.device)  # type: ignore
            print("#" * 20)
            print("Press Enter to stop the stream...")
            print("#" * 20)
            input()
    except KeyboardInterrupt:
        print("Stream stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Stream stopped.")

    print("Finished.")
