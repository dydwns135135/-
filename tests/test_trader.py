import numpy as np
import pandas as pd

from trader.backtest import backtest, synthetic_prices
from trader.broker import PaperBroker, Position
from trader.bot import run_once
from trader.indicators import bollinger, ichimoku
from trader.risk import RiskLimits, RiskManager
from trader.strategy import compute_signals


def test_bollinger_pctb_bounds():
    df = synthetic_prices(200)
    b = bollinger(df["close"])
    assert b["bb_upper"].iloc[-1] > b["bb_lower"].iloc[-1]
    assert b["pct_b"].dropna().between(-1, 2).all()


def test_ichimoku_no_lookahead():
    """미래 데이터를 바꿔도 과거 구름 값은 변하지 않아야 한다."""
    df = synthetic_prices(300)
    a = ichimoku(df)
    df2 = df.copy()
    df2.iloc[250:, :] *= 3
    b = ichimoku(df2)
    pd.testing.assert_frame_equal(a.iloc[:250], b.iloc[:250])


def test_signals_only_after_warmup():
    sig = compute_signals(synthetic_prices(300))
    assert not sig["entry"].iloc[:77].any()


def test_backtest_runs_and_equity_positive():
    r = backtest(synthetic_prices(1500, seed=3))
    assert (r.equity > 0).all()
    assert r.metrics["거래횟수"] == len(r.trades)


def test_backtest_no_fill_before_next_open():
    """체결 가격은 신호 당일 종가가 아니라 다음날 시가(+슬리피지)여야 한다."""
    r = backtest(synthetic_prices(1500, seed=3), slippage=0.0)
    df = synthetic_prices(1500, seed=3)
    for _, t in r.trades.iterrows():
        assert np.isclose(t.entry_px, df.loc[t.entry_date, "open"])


def test_risk_limits(tmp_path):
    rm = RiskManager(RiskLimits(max_order_usd=500, max_position_pct=0.2), str(tmp_path / "s.json"))
    assert rm.buy_qty(100, 10_000, 10_000, 0, 0) == 5  # 주문 한도 500
    assert rm.buy_qty(100, 10_000, 10_000, 1_900, 0) == 1  # 비중 한도 2000-1900
    assert rm.buy_qty(100, 10_000, 10_000, 0, 5) == 0  # 종목 수 한도
    assert rm.halted(10_000) is False
    assert rm.halted(9_000) is True  # 시작 자산 대비 -10%


class FakeSource:
    def __init__(self, df):
        self.df = df

    def candles(self, s, n):
        return self.df

    def price(self, s):
        return float(self.df["close"].iloc[-1])


def test_bot_paper_sells_on_stop_loss(tmp_path):
    df = synthetic_prices(300)
    b = PaperBroker(FakeSource(df), cash=10_000)
    px = b.price("X")
    b._pos["X"] = Position("X", 10, px * 2)  # 평단의 절반 → 손절선 이탈
    run_once(b, ["X"], RiskManager(RiskLimits(), str(tmp_path / "s.json")))
    assert "X" not in b.positions()
