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


from mt5bot.indicators import atr, ema
from mt5bot.strategy import daily_loss_hit, trend_ok


def test_ema():
    e = ema([1.0] * 10, 5)
    assert e[3] is None and e[4] == 1.0 and e[-1] == 1.0
    assert ema(list(range(1, 11)), 3)[-1] > ema(list(range(1, 11)), 5)[-1]


def test_atr_constant_range():
    n = 30
    highs, lows, closes = [101.0] * n, [99.0] * n, [100.0] * n
    a = atr(highs, lows, closes, 14)
    assert a[13] is None and abs(a[-1] - 2.0) < 1e-9


def test_trend_ok():
    assert trend_ok("buy", 101, 100) and not trend_ok("buy", 99, 100)
    assert trend_ok("sell", 99, 100) and not trend_ok("sell", 101, 100)
    assert not trend_ok("buy", 101, None)


def test_daily_loss():
    assert daily_loss_hit(1000, 979, 2.0)
    assert not daily_loss_hit(1000, 985, 2.0)
    assert not daily_loss_hit(1000, 500, 0)


from mt5bot.strategy import virtual_exit


def test_virtual_exit():
    assert virtual_exit("buy", 95, 110, 105, 94, 40)[0].startswith("손절")
    assert virtual_exit("buy", 95, 110, 111, 99, 40)[0].startswith("익절")
    assert virtual_exit("buy", 95, 110, 111, 94, 40)[0].startswith("손절")  # 동시 도달은 SL
    assert virtual_exit("sell", 105, 90, 101, 89, 60)[0].startswith("익절")
    assert virtual_exit("buy", 95, 110, 100, 99, 55)[0].startswith("RSI")
    assert virtual_exit("buy", 95, 110, 100, 99, 45) is None
