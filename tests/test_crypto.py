import numpy as np
import pandas as pd

from crypto.backtest import FuturesConfig, backtest_futures
from crypto.bot import run_once
from crypto.strategy import compute_ls_signals
from trader.backtest import synthetic_prices


def test_short_signals_mirror_and_exclusive():
    s = compute_ls_signals(synthetic_prices(1500, seed=1, vol=0.02))
    assert not (s["long_entry"] & s["short_entry"]).any()


def test_backtest_runs():
    r = backtest_futures(synthetic_prices(2000, seed=2, vol=0.01), FuturesConfig(leverage=3))
    assert (r["equity"] > 0).all() or r["청산횟수"] > 0
    assert r["거래횟수"] == len(r["trades"])


def _flat_then(price_path):
    """지표가 준비된 뒤 가격이 주어진 경로로 움직이는 OHLC."""
    n = 200
    c = np.r_[np.linspace(100, 160, n), price_path]
    idx = pd.date_range("2024-01-01", periods=len(c), freq="4h")
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c}, index=idx)


def test_liquidation_before_stop_with_high_leverage():
    # 스톱(10%)이 청산(20x → ~4.5%)보다 멀면 청산이 먼저
    df = synthetic_prices(1500, seed=5, vol=0.03)
    r = backtest_futures(df, FuturesConfig(leverage=20, stop_loss=0.10, margin_pct=0.5))
    assert r["청산횟수"] > 0
    assert (r["trades"].query("reason=='liquidation'")["pnl"] < 0).all()


def test_stop_loss_limits_loss():
    df = synthetic_prices(3000, seed=2, vol=0.01)
    r = backtest_futures(df, FuturesConfig(leverage=2, stop_loss=0.03, margin_pct=0.25))
    stops = r["trades"].query("reason=='stop'")
    assert len(stops) > 0 and (stops["pnl"] < 0).all()
    assert (stops["pnl"] > -1000 * 0.25 * 2 * 0.03 * 1.5).all()  # 손실은 대략 거래규모×손절%


class FakeBroker:
    def __init__(self, df, pos=None):
        self.df, self._pos, self.calls = df, pos, []

    def closed_candles(self, tf, limit=300): return self.df
    def price(self): return float(self.df["close"].iloc[-1])
    def balance(self): return 1000.0
    def position(self): return self._pos
    def open(self, side, amount, stop): self.calls.append(("open", side, amount, stop))
    def close(self, side, contracts): self.calls.append(("close", side, contracts))


def test_bot_closes_long_on_exit_signal():
    # 상승 후 급락 → 롱 청산 신호
    df = _flat_then(np.linspace(160, 100, 40))
    b = FakeBroker(df, {"side": "long", "contracts": 0.1, "entry": 150.0})
    run_once(b, "4h", FuturesConfig())
    assert b.calls == [("close", "long", 0.1)]


def test_bot_does_not_double_open():
    df = synthetic_prices(400)
    b = FakeBroker(df, {"side": "long", "contracts": 0.1, "entry": 1.0})
    run_once(b, "4h", FuturesConfig())
    assert not any(c[0] == "open" for c in b.calls)


def test_test_open_and_close():
    from crypto.bot import test_close, test_open

    class B(FakeBroker):
        def test_amount(self): return 0.001
        def stop_info(self): return {"trigger_orders": []}

    df = synthetic_prices(300)
    b = B(df)
    notes = test_open(b, FuturesConfig())
    assert b.calls[0][:3] == ("open", "buy", 0.001) and b.calls[0][3] < b.price()
    assert any("손절" in n for n in notes)
    b2 = B(df, {"side": "long", "contracts": 0.001, "entry": 1.0})
    assert test_open(b2, FuturesConfig())[0].startswith("이미 포지션")
    test_close(b2)
    assert b2.calls == [("close", "long", 0.001)]
