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


def test_order_retries_hedged_on_25236():
    import ccxt
    from crypto.exchange import BitgetFutures

    class Ex:
        def __init__(self): self.calls = []
        def load_markets(self): pass
        def create_order(self, sym, typ, side, amt, params=None):
            self.calls.append(params["hedged"])
            if not params["hedged"]:
                raise ccxt.ExchangeError('bitget {"code":"25236","msg":"Incorrect position open type"}')
            return {"id": "1"}

    ex = Ex()
    b = BitgetFutures(ex, "BTC/USDT:USDT", 2)
    b._order("buy", 0.001, {})
    assert ex.calls == [False, True] and b.hedged is True
    b._order("sell", 0.001, {"reduceOnly": True})
    assert ex.calls == [False, True, True]  # 이후엔 학습한 모드로 바로 주문


def test_other_errors_not_retried():
    import ccxt
    import pytest
    from crypto.exchange import BitgetFutures

    class Ex:
        calls = 0
        def load_markets(self): pass
        def create_order(self, *a, **k):
            Ex.calls += 1
            raise ccxt.ExchangeError("bitget 40001 other")

    b = BitgetFutures(Ex(), "BTC/USDT:USDT", 2)
    with pytest.raises(ccxt.ExchangeError):
        b._order("buy", 1, {})
    assert Ex.calls == 1


def test_test_open_logs_diagnosis_on_failure(caplog):
    import pytest
    from crypto.bot import test_open

    class B(FakeBroker):
        def test_amount(self): return 0.001
        def open(self, *a): raise RuntimeError("25203 Insufficient margin")
        def diagnose(self): return {"balances": {"SUSDT": 10000}}

    with caplog.at_level("ERROR", logger="crypto"), pytest.raises(RuntimeError):
        test_open(B(synthetic_prices(300)), FuturesConfig())
    assert "SUSDT" in caplog.text and "Insufficient" in caplog.text


def test_live_aborts_on_permission_error_but_demo_continues():
    import ccxt
    import pytest
    from crypto.exchange import BitgetFutures

    class Ex:
        def __init__(self): self.orders = 0
        def load_markets(self): pass
        def amount_to_precision(self, s, a): return a
        def set_margin_mode(self, *a): raise ccxt.ExchangeError('{"code":"40014","msg":"need future pos write permissions"}')
        def set_leverage(self, *a): pass
        def create_order(self, *a, **k): self.orders += 1; return {}

    live = BitgetFutures(Ex(), "BTC/USDT:USDT", 2)
    with pytest.raises(ccxt.ExchangeError):
        live.open("buy", 0.001, 1.0)
    assert live.ex.orders == 0
    demo = BitgetFutures(Ex(), "BTC/USDT:USDT", 2, demo=True)
    demo.open("buy", 0.001, 1.0)
    assert demo.ex.orders == 1 and demo.setup_errors


def test_demo_always_sends_paptrading_header_even_for_susdt_products():
    import ccxt
    from crypto.exchange import configure_demo

    ex = ccxt.bitget({"apiKey": "k", "secret": "s", "password": "p"})
    configure_demo(ex)
    assert ex.options["uta"] is True  # 데모 계정은 통합 계정(UTA)
    # 실제 요청 직전에 합쳐지는 헤더 (S 상품 요청이라 ccxt 가 자체 헤더를 빼도 남아야 함)
    assert ex.prepare_request_headers({})["PAPTRADING"] == "1"
