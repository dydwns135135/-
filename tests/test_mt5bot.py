import pytest

from mt5bot.indicators import rsi
from mt5bot.strategy import entry_signal, should_exit, sl_tp_prices


def test_rsi_extremes():
    assert rsi(list(range(1, 40)))[-1] == 100.0
    assert rsi(list(range(40, 1, -1)))[-1] == 0.0
    assert rsi([1, 2, 3])[-1] is None


def test_entry_signal():
    assert entry_signal(28, 32) == "buy"
    assert entry_signal(72, 68) == "sell"
    assert entry_signal(40, 45) is None
    assert entry_signal(None, 50) is None


def test_exit():
    assert should_exit("buy", 55) and not should_exit("buy", 45)
    assert should_exit("sell", 45) and not should_exit("sell", 55)


def test_sl_tp():
    assert sl_tp_prices("buy", 2000.0, 5, 10, 2) == (1995.0, 2010.0)
    assert sl_tp_prices("sell", 2000.0, 5, 10, 2) == (2005.0, 1990.0)
    with pytest.raises(ValueError):
        sl_tp_prices("buy", 2000.0, 0, 10, 2)
