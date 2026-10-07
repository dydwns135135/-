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


def test_asset_mix_per_asset_funding_only_hits_that_asset():
    from crypto import asset_mix as A

    d = {k: v[["open", "close"]] for k, v in {"BTC": synthetic_prices(700, seed=41, vol=0.02), "QQQ": synthetic_prices(700, seed=42, vol=0.01)}.items()}
    base = A.build(d, 0.0)
    only_btc = A.build(d, {"BTC": A.per8h(10.0), "QQQ": 0.0})
    pd.testing.assert_series_equal(only_btc["QQQ 보유"], base["QQQ 보유"])           # QQQ 는 펀딩비 영향 없음
    assert (1 + only_btc["BTC 보유"]).prod() < (1 + base["BTC 보유"]).prod()         # BTC 만 깎임
    assert abs(A.per8h(10.95) - 0.0001) < 1e-12                                       # 연 10.95% = 8시간 0.01%
    assert set(A.MEASURED) == {"BTC", "QQQ", "GLD"} and all(0 < v < 0.0001 for v in A.MEASURED.values())


def test_position_budget_and_plan_respect_max_notional():
    from crypto.momentum_bot import plan_rebalance, position_budget

    assert position_budget(10_000, 0.5) == 5_000 and position_budget(10_000, 0.5, 300) == 300
    assert position_budget(100, 0.5, 300) == 50                                  # 상한이 더 크면 잔고 기준
    # 잔고 1만, 알로크 0.5, 상한 300, 가격 100 → 비중 100% = 3개
    assert plan_rebalance(1.0, 0.5, 10_000, 100, 0, max_notional=300) == ("buy", 3.0)
    assert plan_rebalance(1.0, 0.5, 10_000, 100, 50, max_notional=300) == ("sell", 47.0)   # 이미 상한 초과 → 줄임


def test_kill_switch_flattens_and_blocks_new_entries_in_both_modes():
    from crypto.momentum_bot import run_once, run_once_ensemble

    up = list(np.linspace(100, 400, 320))
    b = _EnsBroker(up, pos={"side": "long", "contracts": 12.5, "entry": 300.0}, bal=700.0)
    notes = run_once_ensemble(b, 0.5, min_equity=1000.0)
    assert b.calls == [("close", "long", 12.5)] and "차단 장치" in notes[0] and "자동 재개되지 않습니다" in notes[-1]
    b2 = _EnsBroker(up, bal=700.0)                                                # 포지션 없음 + 강세여도 진입하지 않는다
    notes2 = run_once_ensemble(b2, 0.5, min_equity=1000.0)
    assert b2.calls == [] and "보유 포지션 없음" in notes2[1]
    b3 = _MomBroker(up, bal=700.0)
    assert "차단 장치" in run_once(b3, 90, 0.5, min_equity=1000.0)[0] and b3.calls == []
    b4 = _EnsBroker(up, bal=1500.0)                                               # 잔고가 충분하면 정상 진입
    run_once_ensemble(b4, 0.5, min_equity=1000.0)
    assert b4.calls and b4.calls[0][:2] == ("open", "buy")


def test_single_mode_entry_respects_max_notional():
    from crypto.momentum_bot import run_once

    up = list(np.linspace(100, 400, 320))
    b = _MomBroker(up, bal=10_000.0)
    run_once(b, 90, 0.5, max_notional=400.0)
    assert b.calls == [("open", "buy", round(400.0 / 400, 5), None)]            # 잔고 50%(5000)가 아니라 상한 400 → 1개


def _bars(opens, highs, lows, closes):
    idx = pd.date_range("2024-01-01", periods=len(opens), freq="D", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)


def _flat_w(n, v=1.0):
    return pd.Series([v] * n, index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"))


def test_stress_1x_no_stop_tracks_price_and_never_liquidates():
    from crypto.leverage_stress import simulate_leveraged

    p = [100, 100, 90, 60, 80, 120]                                        # -40% 까지 내렸다 반등
    df = _bars(p, [x * 1.01 for x in p], [x * 0.99 for x in p], p)
    r = simulate_leveraged(df, _flat_w(6), 1, None, fee=0, slip=0, fund_daily=0, start=0)
    assert r["liqs"] == 0 and r["stops"] == 0 and r["ruin"] is None
    assert abs(r["equity"].iloc[-1] - 120 / 100) < 1e-9                    # 첫날 시가 매수(100) → 마지막 종가 120


def test_stress_10x_is_liquidated_by_ordinary_drawdown_and_stays_ruined():
    from crypto.leverage_stress import simulate_leveraged

    o = [100, 100, 100, 100, 100, 100]
    lo = [100, 100, 88, 100, 100, 100]                                     # 3일째 장중 -12%
    df = _bars(o, [101] * 6, lo, [100, 100, 95, 100, 100, 100])
    r = simulate_leveraged(df, _flat_w(6), 10, None, fee=0, slip=0, fund_daily=0, start=0)
    assert r["liqs"] == 1 and r["ruin"] == df.index[2]
    assert (r["equity"].iloc[2:] == 0).all()                               # 파산 뒤 회복해도 0
    r2 = simulate_leveraged(df, _flat_w(6), 1, None, fee=0, slip=0, fund_daily=0, start=0)
    assert r2["liqs"] == 0 and r2["equity"].iloc[-1] > 0.9                 # 같은 가격 경로가 1배에서는 문제없음


def test_stress_stop_precedes_liquidation_and_costs_leverage_times_stop():
    from crypto.leverage_stress import simulate_leveraged

    lo = [100, 100, 96, 100, 100]                                          # 장중 -4%
    df = _bars([100] * 5, [101] * 5, lo, [100, 100, 99, 100, 100])
    r = simulate_leveraged(df, _flat_w(5), 10, 0.03, fee=0, slip=0, fund_daily=0, start=0)
    assert r["stops"] == 1 and r["liqs"] == 0 and r["ruin"] is None        # 손절(3%)이 청산(~9.5%)보다 먼저
    assert abs(r["equity"].iloc[2] - 0.70) < 1e-9                          # 10배 × 3% = 잔고의 30% 손실
    assert r["trades"] >= 1 and r["equity"].iloc[-1] <= 0.71               # 다음 날 재진입하지만 횡보라 회복 없음


def test_stress_wide_stop_loses_to_liquidation_and_gap_fills_at_open():
    from crypto.leverage_stress import simulate_leveraged

    df = _bars([100, 100, 100, 100], [101] * 4, [100, 100, 80, 100], [100, 100, 85, 100])
    r = simulate_leveraged(df, _flat_w(4), 10, 0.30, fee=0, slip=0, fund_daily=0, start=0)
    assert r["liqs"] == 1 and r["stops"] == 0                              # 손절 30% > 청산 거리 ~9.5% → 청산이 먼저
    gap = _bars([100, 100, 90, 100], [101, 101, 91, 101], [100, 100, 89, 100], [100, 100, 90, 100])  # 시가부터 -10% 갭
    g = simulate_leveraged(gap, _flat_w(4), 2, 0.05, fee=0, slip=0, fund_daily=0, start=0)
    assert g["stops"] == 1 and abs(g["equity"].iloc[2] - (1 + 2 * (90 / 100 - 1))) < 1e-9   # 손절가 95 가 아니라 시가 90 에 체결


def test_stress_funding_and_fees_reduce_equity_and_warmup_is_flat():
    from crypto.leverage_stress import simulate_leveraged

    df = _bars([100] * 10, [100] * 10, [100] * 10, [100] * 10)
    free = simulate_leveraged(df, _flat_w(10), 1, None, fee=0, slip=0, fund_daily=0, start=0)["equity"].iloc[-1]
    paid = simulate_leveraged(df, _flat_w(10), 1, None, fee=0.001, slip=0, fund_daily=0.0005, start=0)["equity"].iloc[-1]
    assert abs(free - 1.0) < 1e-12 and paid < 1.0
    late = simulate_leveraged(df, _flat_w(10), 1, None, fee=0, slip=0, fund_daily=0, start=5)
    assert len(late["equity"]) == 5 and (late["equity"] == 1.0).all() and late["trades"] == 1   # start 이전은 거래하지 않고, 첫 진입은 start+1 시가


def test_stress_grid_shape_and_stats_on_synthetic_data():
    from crypto.leverage_stress import fmt, grid, stats, simulate_leveraged

    df = synthetic_prices(1200, seed=51, vol=0.03)
    t = grid(df, start=250)
    assert len(t) == 25 and set(t["레버리지"]) == {"1배", "2배", "3배", "5배", "10배"} and "없음" in set(t["손절폭"])
    assert set(t["손절폭"]) == {"없음", "3%", "10%", "20%", "30%"} and "손절횟수" in t and "청산횟수" in t   # 설정과 횟수 칸이 겹치지 않는다
    assert (t["최종잔고(배)"] >= 0).all() and "파산" in fmt(t)
    one = t[(t["레버리지"] == "1배") & (t["손절폭"] == "없음")].iloc[0]
    assert one["청산횟수"] == 0 and one["손절횟수"] == 0                          # 1배·손절 없음·롱만은 청산·손절이 없다


def _upbit_rows(n, start="2024-01-01"):
    """업비트 응답 형태(최신순)의 가짜 일봉 JSON."""
    days = pd.date_range(start, periods=n, freq="D")
    rows = [{"candle_date_time_utc": d.strftime("%Y-%m-%dT00:00:00"), "opening_price": 100.0 + i, "high_price": 102.0 + i,
             "low_price": 98.0 + i, "trade_price": 101.0 + i} for i, d in enumerate(days)]
    return rows[::-1]


def test_upbit_parse_and_paginate_backwards_without_gaps():
    from crypto.upbit_data import fetch_upbit_daily, parse_upbit

    allrows = _upbit_rows(450)
    df = parse_upbit(allrows)
    assert len(df) == 450 and df.index.is_monotonic_increasing and df["close"].iloc[0] == 101.0 and df["open"].iloc[-1] == 100.0 + 449

    calls = []
    class R:
        status_code = 200
        text = ""
        def __init__(self, j): self._j = j
        def json(self): return self._j
    def fake_get(url, params=None, headers=None, timeout=0):
        calls.append(dict(params))
        pool = allrows if "to" not in params else [r for r in allrows if r["candle_date_time_utc"] + "Z" < params["to"]]
        return R(pool[:params["count"]])
    out = fetch_upbit_daily("KRW-BTC", total=450, get=fake_get, sleep=lambda s: None)
    assert len(out) == 450 and out.index.is_monotonic_increasing and not out.index.duplicated().any()
    assert "to" not in calls[0] and "to" in calls[1] and len(calls) == 3   # 200 + 200 + 50

    import pytest
    class Bad(R):
        status_code = 429
        text = "Too Many"
    with pytest.raises(RuntimeError, match="429"):
        fetch_upbit_daily("KRW-BTC", get=lambda *a, **k: Bad([]), sleep=lambda s: None)


def test_upbit_closed_only_drops_in_progress_candle():
    from crypto.upbit_data import closed_only, parse_upbit

    df = parse_upbit(_upbit_rows(5, "2024-01-01"))                       # 1/1 ~ 1/5
    now = pd.Timestamp("2024-01-05T12:00:00", tz="UTC")                  # 1/5 봉은 아직 진행 중
    assert closed_only(df, now).index[-1] == pd.Timestamp("2024-01-04", tz="UTC")
    assert closed_only(df, pd.Timestamp("2024-01-06T00:00:00", tz="UTC")).index[-1] == pd.Timestamp("2024-01-05", tz="UTC")


def test_upbit_backtest_compare_shape_and_fewer_adjustments_with_larger_step():
    from crypto.upbit_backtest import compare, fmt

    df = synthetic_prices(1600, seed=61, vol=0.025)
    t = compare(df, None)
    assert len(t) == 4 and list(t["전략"])[0] == "BTC 보유(원화)"
    steps = t.iloc[1:]["조정횟수"].tolist()
    assert steps[0] >= steps[1] >= steps[2] and steps[0] > steps[2]       # 임계가 클수록 주문이 줄어든다
    assert t.iloc[0]["조정횟수"] == 1 and "연평균조정" in fmt(t)             # 보유는 첫 진입 1번


def test_signal_alert_decisions_state_and_message(tmp_path):
    from crypto.signal_alert import build_message, decide_alert, load_state, save_state

    assert decide_alert(0.5, None) == "init" and decide_alert(0.0, None) == "hold"
    assert decide_alert(0.75, 0.50) == "adjust" and decide_alert(0.625, 0.50) == "hold"      # 25%p 이상일 때만
    assert decide_alert(0.0, 0.125) == "adjust"                                             # 0 으로 돌아가면 작은 차이도 조정
    assert decide_alert(0.375, 0.625) == "adjust" and decide_alert(1.0, 1.0) == "hold"
    p = tmp_path / "s.json"
    assert load_state(p) is None
    save_state(p, 0.75)
    assert load_state(p) == 0.75
    p.write_text("not json")
    assert load_state(p) is None                                                            # 손상된 상태 파일은 무시

    closes = pd.Series([1.0])
    m = build_message(0.5, 4, 0.75, "adjust", 100_000_000, closes, 1_000_000, 1.0)
    assert "조정 필요: 일부 매도 약 250,000원" in m and "목표 보유 금액 500,000원" in m and "0.005000 BTC" in m
    m0 = build_message(0.0, 0, 0.25, "adjust", 100_000_000, closes, None, 1.0)
    assert "전량 매도" in m0 and "비중 25%p" in m0
    mh = build_message(0.75, 6, 0.75, "hold", 100_000_000, closes, 2_000_000, 0.5)
    assert mh.startswith("조정 없음") and "750,000원" in mh                                  # 2,000,000 × 50% × 75%


def _scalp_df(n=600, drift=0.0005, seed=3):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999, "close": c}, index=idx)


def test_scalp_simulate_cost_and_timing():
    import pandas as pd
    from crypto import upbit_scalp as sc
    df = _scalp_df()
    always = pd.Series(1.0, index=df.index)
    # 항상 보유: 첫 진입 비용 1번만, 비용 0이면 시가→시가 누적수익과 같다
    free = sc.simulate(df, always, 0.0)
    paid = sc.simulate(df, always, 0.001)
    assert paid.iloc[-1] < free.iloc[-1]
    assert abs(paid.iloc[-1] / free.iloc[-1] - (1 - 0.001)) < 1e-4
    # 신호가 마감 봉에서 나와도 같은 봉 수익은 먹지 못한다(다음 봉부터)
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[10] = 1.0
    eq = sc.simulate(df, sig, 0.0)
    assert eq.iloc[10] == 1.0 and eq.iloc[11] != 1.0


def test_scalp_strategies_signals_binary_and_trade_costs_hurt():
    from crypto import upbit_scalp as sc
    df = _scalp_df(n=800, drift=0.0)
    tbl = sc.evaluate(df)
    assert len(tbl) == len(sc.STRATEGIES) + 1
    for name, fn in sc.STRATEGIES.items():
        s = fn(df)
        assert set(s.unique()) <= {0.0, 1.0}
    rows = tbl[tbl["전략"] != "BTC 보유"]
    assert (rows["비용0 연환산"] >= rows["연환산"] - 1e-9).all()
    assert "하루평균" in sc.fmt(tbl)


def test_scalp_breakout_enters_after_high_and_exits_after_low():
    import pandas as pd
    from crypto import upbit_scalp as sc
    idx = pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC")
    close = [10, 10, 10, 10, 12, 13, 13, 9, 9, 9, 9, 9]
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=idx)
    s = sc.breakout(df, 3, 2)
    assert s.iloc[3] == 0.0 and s.iloc[4] == 1.0 and s.iloc[6] == 1.0 and s.iloc[7] == 0.0


def test_upbit_minutes_fetch_uses_minutes_url():
    from crypto.upbit_data import fetch_upbit_minutes
    seen = []

    class R:
        status_code = 200

        def __init__(self, rows):
            self.rows = rows

        def json(self):
            return self.rows

    def fake(url, params=None, headers=None, timeout=None):
        seen.append(url)
        rows = [{"opening_price": 1, "high_price": 1, "low_price": 1, "trade_price": 1,
                 "candle_date_time_utc": f"2024-01-01T{23 - i // 60:02d}:{59 - i % 60:02d}:00"} for i in range(5)]
        return R(rows)

    df = fetch_upbit_minutes(60, total=5, get=fake, sleep=lambda s: None)
    assert seen and seen[0].endswith("/candles/minutes/60") and len(df) == 5


def _rot_panel(n=420):
    import numpy as np
    import pandas as pd
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    t = np.arange(n)
    c = pd.DataFrame({
        "KRW-BTC": 100 * np.exp(0.002 * t), "KRW-UP": 50 * np.exp(0.004 * t),
        "KRW-DOWN": 80 * np.exp(-0.003 * t), "KRW-FLAT": 70 + np.sin(t / 7), "KRW-NEW": 30 * np.exp(0.003 * t),
        "KRW-UP2": 40 * np.exp(0.0035 * t)}, index=idx)
    c.loc[idx[:300], "KRW-NEW"] = np.nan   # 늦게 상장
    o = c.shift(1).fillna(c)
    v = pd.DataFrame({"KRW-BTC": 1000.0, "KRW-UP": 800.0, "KRW-DOWN": 600.0, "KRW-FLAT": 500.0, "KRW-NEW": 900.0, "KRW-UP2": 700.0}, index=idx)
    return o, c, v


def test_rotation_scores_nan_until_history_and_flag_trend():
    from crypto import upbit_rotation as ro
    o, c, v = _rot_panel()
    sc = ro.scores(c)
    assert sc["KRW-BTC"].iloc[:250].isna().all() and sc["KRW-BTC"].iloc[-1] == 1.0
    assert sc["KRW-DOWN"].iloc[-1] == 0.0
    assert sc["KRW-NEW"].iloc[-1] != sc["KRW-NEW"].iloc[-1]   # 상장 250일 안 됨 → 후보 제외(NaN)


def test_rotation_picks_strongest_trending_and_goes_cash_when_all_down():
    import pandas as pd
    from crypto import upbit_rotation as ro
    o, c, v = _rot_panel()
    w = ro.target_weights(c, v, "rotation", n_univ=5, k=2, rebal=1)
    last = w.iloc[-1]
    assert last["KRW-UP"] > 0 and last["KRW-UP2"] > 0 and last["KRW-DOWN"] == 0 and last["KRW-NEW"] == 0
    assert abs(last.sum() - 1.0) < 1e-9
    down = c.copy()
    for col in down.columns:
        down[col] = 100 * (0.998 ** pd.Series(range(len(down)), index=down.index).values)
    wd = ro.target_weights(down, v, "rotation", n_univ=5, k=2, rebal=1)
    assert wd.iloc[-1].sum() == 0.0


def test_rotation_rebalance_holds_weights_and_costs_reduce_equity():
    from crypto import upbit_rotation as ro
    o, c, v = _rot_panel()
    w7 = ro.target_weights(c, v, "equal", n_univ=3, rebal=7)
    assert (w7.diff().abs().sum(axis=1) > 0).sum() <= len(w7) // 7 + 2
    free = ro.simulate(o, w7, 0.0)
    paid = ro.simulate(o, w7, 0.003)
    assert ((1 + paid["ret"]).prod()) < ((1 + free["ret"]).prod())
    assert paid["turn"].sum() > 0


def test_rotation_compare_and_today_table_shapes():
    from crypto import upbit_rotation as ro
    panel = _rot_panel(n=700)
    res = ro.compare(panel)
    assert set(res) == {"전체", "뒤쪽 절반", "최근 2년"}
    assert len(res["전체"]) == 4 and "연환산" in res["전체"].columns
    t = ro.today_table(panel[1], panel[2])
    assert "KRW-BTC".replace("KRW-", "") in set(t["종목"]) and "오늘 목표비중" in t.columns
    assert "연환산" in ro.fmt_compare(res["전체"]) and "종목" in ro.fmt_today(t)
