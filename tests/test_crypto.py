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
    def closed_candles(self, tf, limit=300, deep=False): return self.df
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
    assert len(rep) == 30 and "250일 모멘텀" in fmt(rep)  # 보유 + 8개 + 앙상블 = 10 전략 × 3 구간 and "개 중" in summarize(rep)[-1]


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


def test_deep_candles_paginate_past_default_90_bar_limit():
    """기본 조회는 ~90개만 주지만 since 로 이어 받으면 필요한 만큼 받는다(run: 89개 → 판단 보류)."""
    from crypto.exchange import BitgetFutures

    day = 86_400_000
    now = 1_000 * day + 5_000
    allrows = [[i * day, 1, 2, 0.5, 1.5, 1] for i in range(0, 1001)]  # 1001번째(1000*day)는 진행 중인 봉

    class Ex:
        calls = []
        def load_markets(self): pass
        def parse_timeframe(self, tf): return 86400
        def fetch_ohlcv(self, s, tf, since=None, limit=300):
            Ex.calls.append((since, limit))
            if since is None:  # 기본 조회: 최근 89개만
                return allrows[-89:]
            return [r for r in allrows if r[0] >= since][:limit]

    b = BitgetFutures(Ex(), "BTC/USDT:USDT", 1)
    b._now_ms = staticmethod(lambda: now)
    assert len(b.closed_candles("1d", limit=130)) == 88          # 기존 방식: 90일 모멘텀 불가
    df = b.closed_candles("1d", limit=130, deep=True)
    assert len(df) >= 91 and df.index.is_monotonic_increasing
    assert df.index[-1] == pd.Timestamp(999 * day, unit="ms", tz="UTC")  # 마지막은 마감된 봉
    assert all(lim <= 200 for _, lim in Ex.calls if _ is not None)


def test_momentum_ensemble_strategy_fraction_causal():
    from crypto import strategies as S

    df = synthetic_prices(900, seed=21, vol=0.02)
    w = S.momentum_ensemble(df)
    assert w.between(0, 1).all() and (w.iloc[:250] == 0).all()   # 250일 쌓이기 전엔 현금
    assert w.iloc[250:].nunique() > 2                              # 0/1 이 아닌 중간 비중도 나온다
    df2 = df.copy(); df2.iloc[700:] *= 3
    pd.testing.assert_series_equal(w.iloc[:700], S.momentum_ensemble(df2).iloc[:700])  # 인과성


def test_ensemble_weight_and_rebalance_plan():
    from crypto.momentum_bot import ensemble_weight, plan_rebalance

    up = pd.Series(np.linspace(100, 300, 300)); down = pd.Series(np.linspace(300, 100, 300))
    assert ensemble_weight(up) == (1.0, 8) and ensemble_weight(down) == (0.0, 0)
    assert ensemble_weight(up.iloc[:200]) is None
    # equity 10000, alloc 0.5, price 100 → 비중 100% = 50 개
    assert plan_rebalance(1.0, 0.5, 10000, 100, 0)[0] == "buy" and plan_rebalance(1.0, 0.5, 10000, 100, 0)[1] == 50
    assert plan_rebalance(0.5, 0.5, 10000, 100, 50) == ("sell", 25)          # 절반으로 축소
    assert plan_rebalance(0.0, 0.5, 10000, 100, 50) == ("sell", 50)          # 전량
    assert plan_rebalance(0.875, 0.5, 10000, 100, 50) == ("sell", 6.25)      # 12.5%p 차이: 조정 기준(10%p) 이상 → 축소
    assert plan_rebalance(1.0, 0.5, 10000, 100, 48)[0] == "hold"             # 4%p 차이는 무시
    assert plan_rebalance(0.0, 0.5, 10000, 0.0001, 0.00001)[0] == "hold"     # 최소 주문 미만


class _EnsBroker(_MomBroker):
    def __init__(self, closes, pos=None, bal=10000.0):
        super().__init__(closes, pos, bal)


def test_run_once_ensemble_buys_partial_and_full_close():
    from crypto.momentum_bot import run_once_ensemble

    up = list(np.linspace(100, 400, 320))
    b = _EnsBroker(up)                      # 8개 모두 + → 비중 100% = 잔고 50%
    run_once_ensemble(b, 0.5)
    assert b.calls[0][:2] == ("open", "buy") and abs(b.calls[0][2] - 10000 * 0.5 / 400) < 1e-4
    down = list(np.linspace(400, 100, 320))
    b2 = _EnsBroker(down, pos={"side": "long", "contracts": 12.5, "entry": 300.0})
    run_once_ensemble(b2, 0.5)
    assert b2.calls == [("close", "long", 12.5)]   # 모두 − → 전량 청산
    b3 = _EnsBroker(up, pos={"side": "long", "contracts": 12.5, "entry": 300.0})
    assert "조정 없음" in run_once_ensemble(b3, 0.5)[1] and b3.calls == []  # 이미 목표 비중
    b4 = _EnsBroker(up[:100])
    assert "부족" in run_once_ensemble(b4, 0.5)[0]


def test_sensitivity_includes_ensemble_and_summary():
    from crypto.momentum_sensitivity import ENSEMBLE, fmt, sensitivity, summarize

    rep = sensitivity(synthetic_prices(2000, seed=8, vol=0.02))
    assert (rep["전략"] == ENSEMBLE).sum() == 3 and ENSEMBLE in fmt(rep)
    txt = "\n".join(summarize(rep))
    assert "[앙상블] 샤프" in txt and "[앙상블] 최대낙폭" in txt and "8개 중" in txt


def _multi(n=900, seeds=(1, 2, 3, 4)):
    names = ("BTC", "ETH", "SOL", "XRP")
    return {nm: synthetic_prices(n, seed=s, vol=0.02) for nm, s in zip(names, seeds)}


def test_portfolio_align_uses_common_dates():
    from crypto.portfolio_compare import align

    d = _multi()
    d["SOL"] = d["SOL"].iloc[200:]  # SOL 은 늦게 상장
    out = align(d)
    assert all(len(v) == 700 for v in out.values())
    assert all(v.index.equals(out["BTC"].index) for v in out.values())


def test_portfolio_equal_weight_pnl_is_mean_of_coin_pnls_and_same_window():
    from crypto import portfolio_compare as P
    from crypto import strategies as S

    d = _multi()
    pnls = P.build(d)
    n = len(d["BTC"]) - 1 - P.WARM  # simulate 가 마지막 봉을 제외, 앞 WARM 제외
    assert all(len(v) == n for v in pnls.values())
    ens = pd.concat([S.simulate(x, S.momentum_ensemble(x)) for x in d.values()], axis=1).mean(axis=1).iloc[P.WARM:]
    pd.testing.assert_series_equal(pnls["4코인 앙상블(등분)"], ens, check_names=False)
    # 등분 포트폴리오의 총 비중은 1배를 넘지 않는다 → 코인별 손익의 평균이므로 단일 코인 손익 범위 안
    singles = pd.concat([pnls["BTC 앙상블"], pnls["ETH 앙상블"], pnls["SOL 앙상블"], pnls["XRP 앙상블"]], axis=1)
    assert (pnls["4코인 앙상블(등분)"] <= singles.max(axis=1) + 1e-12).all()


def test_portfolio_compare_shape_and_summary():
    from crypto import portfolio_compare as P

    d = _multi()
    rep, corr = P.compare(d), P.correlations(d)
    assert len(rep) == 7 * 3 and corr.shape == (4, 4)
    txt = "\n".join(P.summary(rep, corr, P.COINS))
    assert "평균 상관" in txt and "세 구간 모두 낙폭 감소" in txt


def test_load_data_retries_shorter_period_and_reports_all_failures(monkeypatch):
    import pytest
    from crypto import backtest_report as R

    seen = []
    def fake_fetch(ex, symbol, timeframe="4h", years=4.0, **k):
        seen.append(years)
        n = 0 if years >= 5 else 1000   # 먼 과거는 빈 결과(상장 전), 4년 이하는 충분
        return synthetic_prices(max(n, 1), seed=1) if n else synthetic_prices(1, seed=1).iloc[0:0]
    monkeypatch.setattr(R, "fetch_history", fake_fetch)
    ex_id, df = R.load_data(7, "SOL/USDT:USDT", min_bars=800, timeframe="1d")
    assert ex_id == "bitget" and len(df) == 1000 and seen == [7, 5, 4]   # 7년·5년은 비어 4년에 성공

    monkeypatch.setattr(R, "fetch_history", lambda *a, **k: synthetic_prices(1, seed=1).iloc[0:0])
    with pytest.raises(SystemExit) as e:
        R.load_data(7, "XRP/USDT:USDT", min_bars=800, timeframe="1d")
    msg = str(e.value)
    assert "XRP/USDT:USDT" in msg and "bitget/7년: 봉 0개" in msg and "okx/2년: 봉 0개" in msg


def test_load_data_first_success_unchanged_for_btc(monkeypatch):
    from crypto import backtest_report as R

    seen = []
    monkeypatch.setattr(R, "fetch_history", lambda ex, sym, timeframe="4h", years=4.0, **k: (seen.append(years), synthetic_prices(1500, seed=2))[1])
    ex_id, df = R.load_data(7, "BTC/USDT:USDT", min_bars=800, timeframe="1d")
    assert seen == [7] and ex_id == "bitget"   # 기존 BTC 경로는 첫 시도에서 그대로 성공


def _mk(base, quote="USDT", swap=True, linear=True, active=True):
    return {"base": base, "quote": quote, "swap": swap, "linear": linear, "active": active, "symbol": f"{base}/{quote}:{quote}", "info": {}}


def test_market_survey_filters_and_finds_noncrypto_candidates():
    from crypto.market_survey import candidates, usdt_perps

    markets = {m["symbol"]: m for m in (
        _mk("BTC"), _mk("SP500"), _mk("XAU"), _mk("NAS100"), _mk("TSLA"), _mk("DOGE"),
        _mk("BTC", quote="USDC"), _mk("ETH", swap=False), _mk("OLD", active=False),
    )}
    perps = usdt_perps(markets)
    assert sorted(m["base"] for m in perps) == ["BTC", "DOGE", "NAS100", "SP500", "TSLA", "XAU"]
    assert sorted(m["base"] for m in candidates(perps)) == ["NAS100", "SP500", "TSLA", "XAU"]


def test_market_survey_probe_shortens_period_until_data(monkeypatch):
    from crypto import market_survey as M

    seen = []
    def fake_fetch(ex, sym, timeframe="4h", years=4.0, **k):
        seen.append(years)
        return synthetic_prices(300, seed=3) if years <= 2 else synthetic_prices(1, seed=3).iloc[0:0]
    monkeypatch.setattr(M, "fetch_history", fake_fetch)
    r = M.probe(object(), "SP500/USDT:USDT")
    assert r["bars"] == 300 and r["req_years"] == 2 and seen == [8, 5, 3, 2]
    monkeypatch.setattr(M, "fetch_history", lambda *a, **k: synthetic_prices(1, seed=3).iloc[0:0])
    assert M.probe(object(), "X/USDT:USDT")["bars"] == 0


def test_asset_mix_parse_yahoo_adjusts_open_and_drops_nulls():
    from crypto.asset_mix import parse_ohlc

    ts = [1_600_000_000 + i * 86400 for i in range(4)]
    df = parse_ohlc(ts, [10.0, 20.0, None, 40.0], [10.0, 20.0, 30.0, 40.0], adjclose=[5.0, 10.0, 15.0, 40.0])
    assert len(df) == 3                                   # None 행 제거
    assert df["close"].iloc[0] == 5.0 and df["open"].iloc[0] == 5.0   # 조정비율 0.5 가 시가에도 적용
    assert df["close"].iloc[-1] == 40.0 and df["open"].iloc[-1] == 40.0


def test_asset_mix_calendar_fills_weekends_with_zero_pnl():
    from crypto.asset_mix import to_calendar

    idx = pd.to_datetime(["2024-01-05", "2024-01-08"], utc=True)      # 금 → 월
    df = to_calendar(pd.DataFrame({"open": [10.0, 11.0], "close": [10.5, 11.5]}, index=idx))
    assert len(df) == 4 and list(df["close"].round(2)) == [10.5, 10.5, 10.5, 11.5]
    assert df["open"].iloc[1] == 10.5 and df["open"].iloc[2] == 10.5   # 주말 시가=종가=직전 종가


def test_asset_mix_build_equal_weight_and_funding_effect():
    from crypto import asset_mix as A

    d = {"BTC": synthetic_prices(900, seed=31, vol=0.02), "QQQ": synthetic_prices(900, seed=32, vol=0.01), "GLD": synthetic_prices(900, seed=33, vol=0.008)}
    d = {k: v[["open", "close"]] for k, v in d.items()}
    p0, p1 = A.build(d, 0.0), A.build(d, 0.0001)
    assert "3자산 앙상블(등분)" in p0 and all(len(v) == len(p0["BTC 보유"]) for v in p0.values())
    assert ((1 + p1["BTC 보유"]).prod()) < ((1 + p0["BTC 보유"]).prod())       # 펀딩비가 보유 수익을 깎음
    mean = (p0["BTC 앙상블"] + p0["QQQ 앙상블"] + p0["GLD 앙상블"]) / 3
    pd.testing.assert_series_equal(p0["3자산 앙상블(등분)"], mean, check_names=False)


def test_asset_mix_load_asset_reports_all_failures(monkeypatch):
    import pytest
    from crypto import asset_mix as A

    def boom(*a, **k): raise RuntimeError("HTTP 429")
    monkeypatch.setattr(A, "fetch_yahoo", boom)
    monkeypatch.setattr(A, "fetch_stooq", boom)
    with pytest.raises(SystemExit) as e:
        A.load_asset("QQQ")
    assert "yahoo" in str(e.value) and "stooq" in str(e.value) and "429" in str(e.value)
    daily = pd.DataFrame({"open": 1.0, "close": 1.0}, index=pd.date_range("2005-01-03", periods=1500, freq="D", tz="UTC"))
    monkeypatch.setattr(A, "fetch_stooq", lambda c: daily)
    assert A.load_asset("GLD")[0] == "stooq"                                  # 야후 실패 → 스투크 폴백(일봉 검사 통과)


def test_asset_mix_check_daily_rejects_monthly_and_short_data():
    import pytest
    from crypto.asset_mix import check_daily

    daily = pd.DataFrame({"open": 1.0, "close": 1.0}, index=pd.date_range("2015-01-01", periods=1500, freq="D", tz="UTC"))
    assert len(check_daily(daily)) == 1500
    weekdays = daily[daily.index.dayofweek < 5]                       # 평일만 있는 일봉(주식)도 통과
    assert len(check_daily(weekdays)) == len(weekdays)
    monthly = pd.DataFrame({"open": 1.0, "close": 1.0}, index=pd.date_range("2000-01-01", periods=400, freq="MS", tz="UTC"))
    with pytest.raises(ValueError, match="일봉이 아님"):                # run #1: 월봉이 조용히 통과했던 문제
        check_daily(monthly)
    with pytest.raises(ValueError, match="너무 적음"):
        check_daily(daily.iloc[:100])


def test_asset_mix_load_asset_skips_monthly_yahoo_and_falls_back(monkeypatch):
    from crypto import asset_mix as A

    monthly = pd.DataFrame({"open": 1.0, "close": 1.0}, index=pd.date_range("2000-01-01", periods=400, freq="MS", tz="UTC"))
    daily = pd.DataFrame({"open": 1.0, "close": 1.0}, index=pd.date_range("2005-01-03", periods=1500, freq="D", tz="UTC"))
    monkeypatch.setattr(A, "fetch_yahoo", lambda t: monthly)
    monkeypatch.setattr(A, "fetch_stooq", lambda c: daily)
    src, df = A.load_asset("QQQ")
    assert src == "stooq" and len(df) == 1500                          # 야후가 월봉이면 거부하고 스투크로


def test_asset_mix_yahoo_request_pins_period_for_daily_data(monkeypatch):
    from crypto import asset_mix as A

    seen = {}
    class R:
        status_code = 200
        text = ""
        def json(self):
            ts = [1_600_000_000 + i * 86400 for i in range(3)]
            return {"chart": {"result": [{"timestamp": ts, "indicators": {"quote": [{"open": [1, 2, 3], "close": [1, 2, 3]}], "adjclose": [{"adjclose": [1, 2, 3]}]}}]}}
    def fake_get(url, params=None, headers=None, timeout=0):
        seen.update(params); return R()
    monkeypatch.setattr(A.requests, "get", fake_get)
    A.fetch_yahoo("QQQ")
    assert "range" not in seen and seen["interval"] == "1d" and seen["period1"] == 0 and seen["period2"] > 1_700_000_000


def test_funding_summarize_interval_and_annualization():
    from crypto.funding_check import summarize

    h = 3_600_000
    rows8 = [{"timestamp": 1_700_000_000_000 + i * 8 * h, "fundingRate": 0.0001} for i in range(30)]   # 8시간마다 0.01%
    s = summarize(rows8)
    assert s["정산간격(h)"] == 8.0 and abs(s["연환산(%)"] - 0.0001 * 3 * 365 * 100) < 1e-6 and s["양수비율(%)"] == 100
    rows1 = [{"timestamp": 1_700_000_000_000 + i * h, "fundingRate": -0.00002} for i in range(50)]      # 1시간마다 -0.002%
    s1 = summarize(rows1)
    assert s1["정산간격(h)"] == 1.0 and s1["연환산(%)"] < 0 and abs(s1["연환산(%)"] + 0.00002 * 8760 * 100) < 1e-6   # 롱이 받는다
    import pytest
    with pytest.raises(ValueError, match="너무 적음"):
        summarize(rows8[:2])


def test_funding_history_paginates_backwards_without_duplicates():
    from crypto.funding_check import history

    h = 8 * 3_600_000
    allrows = [{"timestamp": 1_700_000_000_000 + i * h, "fundingRate": 0.0001} for i in range(250)]

    class Ex:
        calls = 0
        def fetch_funding_rate_history(self, sym, limit=100, params=None):
            Ex.calls += 1
            until = (params or {}).get("until")
            pool = [r for r in allrows if until is None or r["timestamp"] <= until]
            return pool[-limit:]

    rows = history(Ex(), "X/USDT:USDT", pages=6)
    ts = [r["timestamp"] for r in rows]
    assert len(rows) == 250 and ts == sorted(set(ts)) and Ex.calls == 3
