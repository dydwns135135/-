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


def test_fetch_history_paginates_and_drops_unfinished_bar():
    from crypto.backtest_report import fetch_history

    step = 4 * 3600 * 1000
    t0 = 1_600_000_000_000
    allrows = [[t0 + i * step, 1, 2, 0.5, 1.5, 10] for i in range(450)]

    class Ex:
        calls = 0
        def fetch_ohlcv(self, sym, tf, since=None, limit=200):
            Ex.calls += 1
            return [r for r in allrows if r[0] >= since][:limit]

    df = fetch_history(Ex(), "BTC/USDT:USDT", years=100)
    assert len(df) == 449 and Ex.calls >= 3          # 200+200+50 → 마지막(진행 중) 봉 제외
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()


def test_run_report_shape_and_sharpe():
    from crypto.backtest_report import fmt, run_report, sharpe

    df = synthetic_prices(3000, seed=4, vol=0.01)
    rep = run_report(df)
    assert len(rep) == 12 and set(rep["구간"]) == {"전체", "앞60%", "뒤40%"}
    assert "롱+숏 x2" in fmt(rep)
    assert sharpe(pd.Series([100, 101, 100.5, 102.0, 101.0])) == sharpe(pd.Series([100, 101, 100.5, 102.0, 101.0]))


def test_strategies_causal_and_in_range():
    """신호는 과거만 사용: 미래 데이터를 바꿔도 과거 신호가 변하지 않고, 값은 {-1,0,1}."""
    from crypto import strategies as S

    df = synthetic_prices(900, seed=11, vol=0.02)
    df2 = df.copy()
    df2.iloc[700:] *= 2
    for fn in (S.sma_cross, S.momentum, S.donchian, S.rsi_revert):
        a, b = fn(df), fn(df2)
        pd.testing.assert_series_equal(a.iloc[:700], b.iloc[:700], check_names=False)
        assert set(a.unique()) <= {-1.0, 0.0, 1.0}


def test_simulate_buy_hold_matches_open_to_open_and_costs():
    from crypto import strategies as S

    df = synthetic_prices(300, seed=2, vol=0.01)
    pnl = S.simulate(df, S.buy_hold(df), fee=0, slip=0, funding_per_8h=0)
    exp = df["open"].iloc[-1] / df["open"].iloc[1] - 1  # 첫 진입은 2번째 봉 시가, 마지막 시가까지 보유
    assert abs((1 + pnl).prod() - 1 - exp) < 1e-9
    costly = S.simulate(df, S.buy_hold(df))
    assert (1 + costly).prod() < (1 + pnl).prod()  # 비용이 수익을 깎는다


def test_strategy_compare_shape_and_verdict():
    from crypto.strategy_compare import compare, fmt, verdict

    rep = compare(synthetic_prices(1500, seed=5, vol=0.015))
    assert len(rep) == 30 and set(rep["구간"]) == {"전체", "앞60%", "뒤40%"}
    assert "SMA롱만+변동성타깃" in fmt(rep) and len(verdict(rep)) == 10


def test_momentum_decide_and_signal():
    from crypto.momentum_bot import decide, momentum_positive

    up = pd.Series(np.linspace(100, 200, 120))
    down = pd.Series(np.linspace(200, 100, 120))
    assert momentum_positive(up, 90) is True and momentum_positive(down, 90) is False
    assert momentum_positive(up.iloc[:50], 90) is None  # 데이터 부족
    assert [decide(True, False), decide(False, True), decide(True, True), decide(False, False)] == \
        ["open", "close", "hold", "hold"]


class _MomBroker:
    def __init__(self, closes, pos=None, bal=10000.0):
        idx = pd.date_range("2025-01-01", periods=len(closes), freq="D", tz="UTC")
        self.df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes}, index=idx)
        self._pos, self.bal, self.calls = pos, bal, []
    def closed_candles(self, tf, limit=300): return self.df
    def price(self): return float(self.df["close"].iloc[-1])
    def balance(self): return self.bal
    def position(self): return self._pos
    def open(self, side, amount, stop=None): self.calls.append(("open", side, round(amount, 5), stop))
    def close(self, side, contracts): self.calls.append(("close", side, contracts))


def test_momentum_bot_opens_closes_holds_without_leverage():
    from crypto.momentum_bot import run_once

    up = list(np.linspace(100, 200, 140))
    down = list(np.linspace(200, 100, 140))
    b = _MomBroker(up)
    run_once(b, 90, 0.5)
    assert b.calls == [("open", "buy", round(10000 * 0.5 / 200, 5), None)]  # 잔고 50%, 1배
    b2 = _MomBroker(down, pos={"side": "long", "contracts": 0.05, "entry": 150.0})
    run_once(b2, 90, 0.5)
    assert b2.calls == [("close", "long", 0.05)]
    b3 = _MomBroker(up, pos={"side": "long", "contracts": 0.05, "entry": 150.0})
    run_once(b3, 90, 0.5)
    assert b3.calls == []  # 이미 보유 → 중복 진입 없음
    b4 = _MomBroker(up[:50])
    assert "부족" in run_once(b4, 90, 0.5)[0] and b4.calls == []


def test_closed_candles_uses_clock_not_row_position():
    from crypto.exchange import BitgetFutures

    day = 86_400_000
    class Ex:
        def load_markets(self): pass
        def parse_timeframe(self, tf): return 86400
        def fetch_ohlcv(self, s, tf, limit=300):
            return [[i * day, 1, 2, 0.5, 1.5, 1] for i in range(10)]  # 마지막 봉(9일)의 시작=9*day
    b = BitgetFutures(Ex(), "BTC/USDT:USDT", 1)
    b._now_ms = staticmethod(lambda: 9 * day + 1000)        # 9일 봉 진행 중 → 제외(0~8일 9개)
    assert len(b.closed_candles("1d")) == 9
    b._now_ms = staticmethod(lambda: 10 * day + 1000)       # 새 날이 되고 아직 새 봉이 없어도 9일 봉은 마감됨 → 10개
    assert len(b.closed_candles("1d")) == 10


def test_sensitivity_report_shape():
    from crypto.momentum_sensitivity import fmt, sensitivity, summarize

    rep = sensitivity(synthetic_prices(1500, seed=6, vol=0.02))
    assert len(rep) == 27 and "250일 모멘텀" in fmt(rep) and "개 중" in summarize(rep)[-1]


def test_telegram_find_chat_ids_hides_personal_info():
    from trader.telegram_setup import find_chat_ids

    updates = [
        {"update_id": 1, "message": {"chat": {"id": 111, "type": "private", "first_name": "비밀", "username": "u"}, "text": "hi"}},
        {"update_id": 2, "message": {"chat": {"id": -222, "type": "group", "title": "그룹"}}},
        {"update_id": 3, "my_chat_member": {}},
    ]
    assert find_chat_ids(updates) == {111: "private", -222: "group"}


def test_telegram_setup_never_prints_token(monkeypatch, capsys):
    import requests
    from trader import telegram_setup as T

    TOKEN = "999:SECRET_TOKEN_VALUE"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    class R:
        def __init__(self, j): self._j, self.status_code = j, 200
        def json(self): return self._j

    def fake_get(url, params=None, timeout=0):
        assert TOKEN in url  # 요청 주소에는 토큰이 들어가지만 출력에는 나오면 안 됨
        if url.endswith("getMe"): return R({"ok": True, "result": {"username": "my_bot"}})
        return R({"ok": True, "result": [{"message": {"chat": {"id": 123, "type": "private", "first_name": "x"}}}]})

    monkeypatch.setattr(requests, "get", fake_get)
    assert T.main() == 0
    out = capsys.readouterr().out
    assert "123" in out and "private" in out and TOKEN not in out and "SECRET_TOKEN_VALUE" not in out


def test_telegram_setup_reports_bad_token_without_leaking(monkeypatch, capsys):
    import requests
    from trader import telegram_setup as T

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bad:TOKEN_XYZ")
    def boom(url, params=None, timeout=0): raise requests.ConnectionError("failed for " + url)
    monkeypatch.setattr(requests, "get", boom)
    assert T.main() == 1
    out = capsys.readouterr().out
    assert "TOKEN_XYZ" not in out and "네트워크 오류" in out
