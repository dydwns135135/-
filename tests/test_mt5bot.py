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


import random

from mt5bot import backtest


def test_simulate_tp_sl_and_cost():
    # 봉0 종가 100 에서 매수, ATR=2, SL=1.0*2→98, TP=2.0*2→104. 봉1 고가 105 → TP, 비용 0.5 차감
    h, l, c = [100, 105], [100, 99], [100, 101]
    t = backtest.simulate(h, l, c, ["buy", None], [2.0, 2.0], 1.0, 2.0, 0.5)
    assert len(t) == 1 and abs(t[0]["pnl"] - 3.5) < 1e-9
    # 같은 봉에 SL/TP 동시 도달 → SL 우선
    h, l, c = [100, 105], [100, 97], [100, 101]
    t = backtest.simulate(h, l, c, ["buy", None], [2.0, 2.0], 1.0, 2.0, 0.0)
    assert abs(t[0]["pnl"] - (-2.0)) < 1e-9
    # 숏
    h, l, c = [100, 101], [100, 95], [100, 99]
    t = backtest.simulate(h, l, c, ["sell", None], [2.0, 2.0], 1.0, 2.0, 0.0)
    assert abs(t[0]["pnl"] - 4.0) < 1e-9


def test_stats():
    s = backtest.stats([{"i": 0, "pnl": 3}, {"i": 1, "pnl": -1}, {"i": 2, "pnl": -1}])
    assert s["n"] == 3 and abs(s["win"] - 100 / 3) < 1e-9 and s["total"] == 1 and s["pf"] == 1.5 and s["maxdd"] == 2


def test_grid_runs_on_random_walk():
    random.seed(1)
    c, p = [], 2000.0
    for _ in range(3000):
        p += random.gauss(0, 1.0)
        c.append(p)
    h = [x + 0.5 for x in c]
    l = [x - 0.5 for x in c]
    rows = backtest.run_grid(h, l, c, 0.3)
    assert len(rows) == 45 and all(r["all"]["n"] >= 0 for r in rows)


def test_telegram_file_config(tmp_path, monkeypatch):
    from mt5bot import notify
    f = tmp_path / "telegram.txt"
    f.write_text("123:ABC\n999\n", encoding="utf-8")
    monkeypatch.setattr(notify, "CONFIG_FILE", f)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notify._config() == ("123:ABC", "999")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "555")
    assert notify._config() == ("123:ABC", "555")
