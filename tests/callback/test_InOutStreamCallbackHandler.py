import pytest
from unittest.mock import MagicMock

from rei_rt_audio.callback import InOutStreamCallbackHandler


# テスト用に抽象メソッドを実装したサブクラス
class ConcreteHandler(InOutStreamCallbackHandler):
    def callback(self, in_data, out_data, frames, time, status):  # type: ignore
        pass


def test_init_default_logger():
    handler = ConcreteHandler()
    assert handler._logger is not None  # type: ignore
    assert handler._is_log_callback_info is False  # type: ignore


def test_init_custom_logger():
    logger = MagicMock()
    handler = ConcreteHandler(logger=logger, is_log_callback_info=True)
    assert handler._logger is logger  # type: ignore
    assert handler._is_log_callback_info is True  # type: ignore


def test_format_log_with_info():
    handler = ConcreteHandler(is_log_callback_info=True)
    base_msg = "Processing"
    result = handler._format_log(  # type: ignore
        base_msg,
        frames=512,
        time={"input": 123},
    )
    assert result == "Processing, frames: 512, time: {'input': 123}"


def test_format_log_without_info():
    handler = ConcreteHandler(is_log_callback_info=False)
    base_msg = "Processing"
    result = handler._format_log(  # type: ignore
        base_msg,
        frames=512,
        time={"input": 123},
    )
    assert result == "Processing"


def test_repr():
    logger = MagicMock()
    handler = ConcreteHandler(logger=logger, is_log_callback_info=True)
    result = repr(handler)
    assert "ConcreteHandler" in result
    assert "is_log_callback_info=True" in result


def test_str():
    handler = ConcreteHandler()
    assert str(handler) == "ConcreteHandler"


def test_abstract_method_instantiation_error():
    with pytest.raises(TypeError):
        InOutStreamCallbackHandler()  # type: ignore
