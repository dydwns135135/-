from stock.analyze import analyze_stock, analyze_themes, market_regime
from stock.data import synthetic_market
from stock.themes import THEMES


def test_demo_pipeline():
    m = synthetic_market()
    reg = market_regime(m["KOSPI"])
    assert 0 <= reg["points"] <= 3
    t, s = analyze_themes(m, THEMES)
    assert set(t["theme"]) == set(THEMES)
    assert t["score"].is_monotonic_decreasing
    assert s["score"].between(0, 100).all()


def test_overheat_flagged_and_no_flow_ok():
    m = synthetic_market()
    hot = analyze_stock(m["373220"])
    assert hot["stage"].startswith("과열") or hot["ret20"] > 0.2
    r = analyze_stock(m["000660"][["open", "high", "low", "close", "volume"]])
    assert r["flow_ratio"] is None and 0 <= r["score"] <= 100


def test_short_history_returns_none():
    assert analyze_stock(synthetic_market()["000660"].iloc[:30]) is None


def test_dart_classify_and_decide():
    from stock.analyze import decide
    from stock.dart import classify_title, summarize
    assert classify_title("단일판매ㆍ공급계약체결") == 1
    assert classify_title("유상증자결정") == -1
    assert classify_title("분기보고서") == 0
    m = synthetic_market()
    r = analyze_stock(m["000660"])
    reg = {"points": 3, "label": "강세(적극)"}
    assert decide({**r, "stage": "과열(추격금지)"}, reg)[0] == "낮음"
    assert decide({**r, "stage": "선점 후보"}, reg, summarize(["유상증자결정"]))[0] == "낮음"
    strong = {**r, "stage": "선점 후보", "score": 80, "ret20": 0.05, "both_buy": True}
    assert decide(strong, reg, summarize(["공급계약체결"]))[0] == "높음"
    assert decide(strong, {"points": 1, "label": "약세(보수)"}, summarize(["공급계약체결"]))[0] == "보통"


def test_naver_parsers_and_flow_merge():
    import pandas as pd
    from stock.naver import add_flow, parse_chart, parse_flow
    xml = ('<chartdata><chart><item data="20261027|100|110|90|105|1000" />'
           '<item data="20261028|105|120|100|115|2000" /></chart></chartdata>')
    df = parse_chart(xml)
    assert list(df.index.strftime("%Y%m%d")) == ["20261027", "20261028"] and df["close"].iloc[-1] == 115
    html = ("<table><tr><th>날짜</th><th>종가</th></tr>"
            "<tr><td><span>2026.10.28</span></td><td>115</td><td>5</td><td>+1%</td><td>2,000</td><td>+1,000</td><td>-200</td><td>9</td><td>5%</td></tr>"
            "<tr><td><span>2026.10.27</span></td><td>105</td><td>5</td><td>+1%</td><td>1,000</td><td>500</td><td>300</td><td>9</td><td>5%</td></tr></table>")
    fl = parse_flow(html)
    assert fl.loc["2026-10-28", "inst_qty"] == 1000 and fl.loc["2026-10-28", "foreign_qty"] == -200
    j = add_flow(df, fl)
    assert j["inst"].iloc[-1] == 1000 * 115 and j["individual"].iloc[-1] == -(1000 * 115 - 200 * 115)
    assert "inst" not in add_flow(df, fl.iloc[:0]).columns  # 수급이 비면 가격만


def test_naver_flow_json_prefers_real_individual():
    from stock.naver import add_flow, parse_chart, parse_flow_json
    df = parse_chart('<item data="20261027|100|110|90|105|1000" /><item data="20261028|105|120|100|115|2000" />')
    data = [{"bizdate": "20261028", "organPureBuyQuant": "+1,000", "foreignerPureBuyQuant": "-200", "individualPureBuyQuant": "-700"},
            {"bizdate": "20261027", "organPureBuyQuant": "500", "foreignerPureBuyQuant": "300", "individualPureBuyQuant": "-900"}]
    j = add_flow(df, parse_flow_json(data))
    assert j["inst"].iloc[-1] == 1000 * 115 and j["foreign"].iloc[-1] == -200 * 115
    assert j["individual"].iloc[-1] == -700 * 115  # 근사(-(기관+외국인))가 아니라 실제 개인 값
    import pytest
    with pytest.raises(ValueError):
        parse_flow_json([{"bizdate": "x"}])


def test_add_flow_tolerates_missing_today():
    import pandas as pd
    from stock.naver import add_flow, parse_chart
    days = pd.bdate_range("2026-10-19", periods=8)
    df = parse_chart("".join(f'<item data="{d:%Y%m%d}|100|110|90|105|1000" />' for d in days))
    flow = pd.DataFrame({"inst_qty": 10.0, "foreign_qty": 5.0, "indiv_qty": -15.0}, index=days[:-1])  # 오늘(마지막) 수급 없음
    j = add_flow(df, flow)
    assert "foreign" in j and j["foreign"].iloc[-1] == 0 and j["foreign"].iloc[-2] == 5 * 105
    assert "foreign" not in add_flow(df, flow.iloc[:-5])  # 최근 수급이 너무 적으면 사용 안 함


def test_watchlist_section_in_report(capsys):
    from stock.__main__ import main
    main(["--demo"])
    out = capsys.readouterr().out
    assert "[관심 종목 현황]" in out and "삼성전자" in out


def test_lean_labels():
    from stock.analyze import lean
    up = lean(True, True, True, 0.05, 1.5, True, False, False)
    assert up[0] == "상승 우세" and up[1] == 6
    dn = lean(False, False, False, -0.05, 1.5, False, True, False)
    assert dn[0] == "하락 우세" and dn[1] == -6
    assert lean(True, False, True, 0.0, 1.0, False, False, False)[0] == "중립"
    assert lean(True, True, True, 0.3, 2.0, True, False, True)[0].startswith("상승 흐름(과열")


def test_yahoo_parser_and_us_demo(capsys):
    from stock.__main__ import main
    from stock.yahoo import parse_chart as yparse
    data = {"chart": {"result": [{"meta": {"gmtoffset": -14400},
                                  "timestamp": [1761660600, 1761747000, 1761833400],
                                  "indicators": {"quote": [{"open": [1, 2, None], "high": [2, 3, None], "low": [1, 1, None],
                                                            "close": [2, 3, None], "volume": [10, 20, None]}]}}]}}
    df = yparse(data)
    assert len(df) == 2 and df["close"].iloc[-1] == 3  # 값이 비는 날은 제외
    import pytest
    with pytest.raises(Exception):
        yparse({"chart": {"result": [{"timestamp": [1], "indicators": {"quote": [{"open": [None], "high": [None], "low": [None], "close": [None], "volume": [None]}]}}]}})
    main(["--demo", "--us"])
    out = capsys.readouterr().out
    assert "S&P500" in out and "엔비디아" in out and "미국은 투자자별 수급" in out


def test_us_sources_order_and_parsers(monkeypatch):
    import pytest
    import stock.yahoo as y

    rows = [{"localDate": "20261027", "openPrice": "1", "highPrice": "2", "lowPrice": "1", "closePrice": "2", "accumulatedTradingVolume": "10"},
            {"localDate": "20261028", "openPrice": "2", "highPrice": "3", "lowPrice": "1", "closePrice": "3", "accumulatedTradingVolume": "20"},
            {"localDate": "bad"}]
    df = y.parse_naver_us(rows)
    assert len(df) == 2 and df["close"].iloc[-1] == 3
    with pytest.raises(ValueError):
        y.parse_naver_us([{}])

    # 접미사: .O 는 404, .N 에서 성공 / 네이버가 되면 야후는 부르지 않는다
    class R:
        def __init__(self, code, js=None): self.status_code, self._js = code, js
        def raise_for_status(self):
            if self.status_code >= 400: raise y.requests.HTTPError(str(self.status_code))
        def json(self): return self._js
    urls = []
    def fake_get(url, **kw):
        urls.append(url)
        return R(200, rows) if ".N/" in url else R(404)
    monkeypatch.setattr(y.requests, "get", fake_get)
    monkeypatch.setattr(y, "fetch_yahoo", lambda *a, **k: (_ for _ in ()).throw(AssertionError("야후는 부르면 안 됨")))
    assert y.fetch("LMT", log=lambda m: None)["close"].iloc[-1] == 3
    assert any(".O/" in u for u in urls) and any(".N/" in u for u in urls)

    # 네이버·야후가 모두 실패하면 Stooq
    csv = "Date,Open,High,Low,Close,Volume\n2026-10-27,1,2,1,2,10\n2026-10-28,2,3,1,3,20\n"
    monkeypatch.setattr(y, "fetch_naver_us", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("네이버 x")))
    monkeypatch.setattr(y, "fetch_yahoo", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("야후 실패(HTTP 429)")))
    monkeypatch.setattr(y, "fetch_stooq", lambda *a, **k: y.parse_stooq(csv))
    logs = []
    assert y.fetch("NVDA", log=logs.append)["close"].iloc[-1] == 3 and "Stooq" in logs[0]
    with pytest.raises(ValueError):
        y.parse_stooq("No data")


def test_yahoo_retry_switches_host(monkeypatch):
    import stock.yahoo as y

    class R:
        def __init__(self, code, js=None): self.status_code, self._js = code, js
        def raise_for_status(self):
            if self.status_code >= 400: raise y.requests.HTTPError(str(self.status_code))
        def json(self): return self._js
    chart = {"chart": {"result": [{"meta": {}, "timestamp": [1761660600, 1761747000],
                                   "indicators": {"quote": [{"open": [1, 2], "high": [2, 3], "low": [1, 1], "close": [2, 3], "volume": [10, 20]}]}}]}}
    calls = []
    def fake_get(url, **kw):
        calls.append(url)
        return R(429) if len(calls) < 3 else R(200, chart)
    monkeypatch.setattr(y.requests, "get", fake_get)
    assert len(y.fetch_yahoo("NVDA", sleep=lambda s: None)) == 2 and "query2" in calls[1]


def test_naver_us_parses_nested_json_and_reports_errors(monkeypatch):
    import pytest
    import stock.yahoo as y
    nested = {"isSuccess": True, "result": {"priceInfos": [
        {"localDate": "20261027", "openPrice": 1, "highPrice": 2, "lowPrice": 1, "closePrice": 2, "accumulatedTradingVolume": 10},
        {"localDate": "20261028", "openPrice": 2, "highPrice": 3, "lowPrice": 1, "closePrice": 3, "accumulatedTradingVolume": 20}]}}
    assert y.parse_naver_us(nested)["close"].iloc[-1] == 3

    class R:
        status_code, text = 404, "Not Found page"
    monkeypatch.setattr(y.requests, "get", lambda *a, **k: R())
    with pytest.raises(RuntimeError) as e:
        y.fetch_naver_us("NVDA")
    msg = str(e.value)
    assert ".O#1 HTTP404" in msg and ".K#4 HTTP404" in msg  # 어느 후보가 어떤 이유로 실패했는지 남는다


def test_load_us_uses_proxy_index_and_aborts_early(monkeypatch):
    import pandas as pd
    import pytest
    import stock.yahoo as y
    monkeypatch.setattr(y.time, "sleep", lambda s: None)
    days = pd.bdate_range("2026-07-01", periods=80)
    def mk(base):
        c = pd.Series(range(80), index=days, dtype=float) + base
        return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 100.0})
    def fake(sym, days="1y", log=print):
        if sym in ("SPY", "QQQ"):
            raise RuntimeError("지수 없음")
        return mk(100 if sym == "AAA" else 200)
    monkeypatch.setattr(y, "fetch", fake)
    logs = []
    out = y.load_yahoo(["AAA", "BBB"], log=logs.append)
    assert set(out) == {"AAA", "BBB", "SPX"} and any("평균" in m for m in logs)
    assert out["SPX"]["close"].iloc[0] == pytest.approx(100.0)

    def always_fail(sym, days="1y", log=print):
        raise RuntimeError("막힘")
    monkeypatch.setattr(y, "fetch", always_fail)
    with pytest.raises(RuntimeError, match="미국 데이터를 받지 못함"):
        y.load_yahoo(["A", "B", "C", "D", "E"], log=lambda m: None)


def test_report_marks_missing_and_proxy(capsys, monkeypatch):
    import stock.__main__ as m
    from stock.data import synthetic_market
    from stock.us_themes import US_THEMES
    mk = synthetic_market(themes=US_THEMES, index="SPX", flow=False)
    del mk["LMT"], mk["RTX"]  # 방산·우주 일부 누락
    mk["SPX"].attrs["proxy"] = True
    monkeypatch.setattr(m, "synthetic_market", lambda *a, **k: mk)
    m.main(["--demo", "--us"])
    out = capsys.readouterr().out
    total = sum(len(v) for v in US_THEMES.values())
    assert f"[수집] {total - 2}/{total}종목 — 누락: LMT, RTX" in out and "⚠️종목 3/5" in out and "S&P500 대용" in out


def test_backtest_forward_returns_and_no_lookahead():
    import numpy as np
    import pandas as pd
    from stock.backtest import collect, format_report, verdict
    days = pd.bdate_range("2024-01-01", periods=200)
    c = 100 * 1.01 ** np.arange(200)
    df = pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1000.0}, index=days)
    tm = {"T": {"AAA": "에이"}}
    full = collect({"AAA": df}, tm)
    assert abs(full["r5"].iloc[0] - (1.01 ** 5 - 1)) < 1e-9 and abs(full["r10"].iloc[0] - (1.01 ** 10 - 1)) < 1e-9
    assert full["date"].max() == days[-11]  # 마지막 10일은 이후 수익률을 알 수 없어 제외

    # 미래 데이터를 바꿔도 과거 시점의 신호는 변하지 않아야 한다(룩어헤드 없음)
    df2 = df.copy()
    df2.iloc[120:, df2.columns.get_loc("close")] *= 0.5
    for col in ("open", "high", "low"):
        df2.iloc[120:, df2.columns.get_loc(col)] *= 0.5
    a, b = collect({"AAA": df}, tm), collect({"AAA": df2}, tm)
    cut = days[100]
    pd.testing.assert_frame_equal(a[a["date"] <= cut][["date", "stage", "bias", "score"]].reset_index(drop=True),
                                  b[b["date"] <= cut][["date", "stage", "bias", "score"]].reset_index(drop=True))
    assert "[성과 확인]" in format_report(full) and "표본 부족" in verdict(full, full["stage"] == "과열")
    assert "평가할 데이터가 부족" in format_report(collect({"AAA": df.iloc[:50]}, tm))


def test_backtest_cli_demo(capsys):
    from stock.__main__ import main
    main(["--demo", "--backtest"])
    assert "[성과 확인]" in capsys.readouterr().out
    main(["--demo", "--us", "--backtest"])
    assert "[성과 확인]" in capsys.readouterr().out


def test_rank_test_detects_signal_and_noise():
    import numpy as np
    import pandas as pd
    from stock.backtest import format_report, rank_test
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.bdate_range("2024-01-01", periods=300):
        for i in range(12):
            score = float(i)  # 점수가 높을수록 이후 수익률이 높은 세상
            rows.append({"code": f"S{i}", "name": f"S{i}", "theme": "T", "date": d, "stage": "관망", "bias": "중립",
                         "score": score, "bias_pts": int(rng.integers(-3, 4)),
                         "r5": 0.002 * i + rng.normal(0, 0.01), "r10": 0.004 * i + rng.normal(0, 0.01)})
    df = pd.DataFrame(rows)
    r = rank_test(df, "score")
    assert r[5]["top"][0] > 0.005 and abs(r[5]["top"][2]) >= 2  # 진짜 관계는 잡힌다
    assert r[5]["bottom"][0] < -0.005
    noise = rank_test(df.assign(score=rng.permutation(df["score"].to_numpy())), "score")
    assert abs(noise[5]["top"][2]) < 3  # 점수를 섞으면 관계가 사라진다
    assert rank_test(df.iloc[:20], "score")[5]["top"] is None  # 날짜 수가 모자라면 판단 보류
    assert "[날마다 순위 시험]" in format_report(df) and "점수 상위 3" in format_report(df)


def test_report_wording_updated(capsys):
    from stock.__main__ import main
    main(["--demo"])
    out = capsys.readouterr().out
    assert "근거는 확인되지 않았습니다" in out and "추격 금지" not in out


def test_snapshot_roundtrip_and_flow_last(tmp_path):
    import pandas as pd
    from stock.analyze import analyze_themes
    from stock.data import synthetic_market
    from stock.naver import add_flow, parse_chart
    from stock.snapshot import COLS, build_snapshot, load_snapshots
    from stock.themes import THEMES
    m = synthetic_market()
    _, stocks = analyze_themes(m, THEMES)
    snap = build_snapshot(stocks, m)
    assert list(snap.columns) == COLS and snap["foreign5"].notna().all() and snap["date"].str.len().eq(10).all()
    (tmp_path / "kr").mkdir()
    snap.to_csv(tmp_path / "kr" / "2026-10-28.csv", index=False)
    snap.to_csv(tmp_path / "kr" / "2026-10-29.csv", index=False)  # 같은 (date, code)는 하나만 남는다
    back = load_snapshots(tmp_path)
    assert len(back) == len(snap) and back["code"].str.len().eq(6).all()  # 종목코드 앞의 0 유지
    # 당일 수급이 비어 있으면 flow_last 는 마지막으로 실제 수급이 있던 날
    days = pd.bdate_range("2026-10-19", periods=8)
    df = parse_chart("".join(f'<item data="{d:%Y%m%d}|100|110|90|105|1000" />' for d in days))
    flow = pd.DataFrame({"inst_qty": 10.0, "foreign_qty": 5.0, "indiv_qty": -15.0}, index=days[:-1])
    assert add_flow(df, flow).attrs["flow_last"] == f"{days[-2]:%Y-%m-%d}"


def test_evaluate_snapshots_detects_planted_signal_and_waits_when_short():
    import numpy as np
    import pandas as pd
    from stock.snapshot import attach_returns, evaluate
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2025-01-01", periods=400)
    market, rows = {}, []
    for i in range(12):
        c = 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))
        market[f"{i:06d}"] = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=days)
    # 스냅샷: 수급 점수에 '이후 10일 수익률'을 심어 둔 세계 → 시험이 이를 잡아내야 한다
    for t in range(60, 300):
        for code, df in market.items():
            c = df["close"].to_numpy()
            fut = c[t + 10] / c[t] - 1
            rows.append({"date": f"{days[t]:%Y-%m-%d}", "code": code, "name": code, "theme": "T", "close": c[t], "score": rng.random(),
                         "stage": "관망", "bias": "중립", "bias_pts": 0, "ret5": 0, "ret20": 0, "vol_ratio": 1, "prox60": 1,
                         "flow_ratio": fut + rng.normal(0, 0.005), "indiv_ratio": 0, "both_buy": fut > 0,
                         "foreign5": 0, "inst5": 0, "indiv5": 0, "flow_last": f"{days[t]:%Y-%m-%d}"})
    snap = pd.DataFrame(rows)
    got = attach_returns(snap.iloc[:24], market)
    first = got.iloc[0]
    c0 = market[first["code"]]["close"]
    p = c0.index.get_loc(first["date"])
    assert abs(first["r5"] - (c0.iloc[p + 5] / c0.iloc[p] - 1)) < 1e-12  # 날짜 정렬이 맞다
    out = evaluate(snap, market)
    assert "수급 상위 3" in out and "의미 있음" in out
    assert "아직 부족" in evaluate(snap[snap["date"] <= f"{days[70]:%Y-%m-%d}"], market)  # 짧으면 판단 보류
    assert "아직 없습니다" in evaluate(snap.iloc[:0], market)


def test_evaluate_ignores_snapshots_with_lagging_flow():
    import numpy as np
    import pandas as pd
    from stock.snapshot import evaluate
    rng = np.random.default_rng(2)
    days = pd.bdate_range("2025-01-01", periods=400)
    market = {}
    for i in range(12):
        c = 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))
        market[f"{i:06d}"] = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=days)
    rows = []
    for t in range(60, 300):
        for code, df in market.items():
            c = df["close"].to_numpy()
            rows.append({"date": f"{days[t]:%Y-%m-%d}", "code": code, "name": code, "theme": "T", "close": c[t], "score": rng.random(),
                         "stage": "관망", "bias": "중립", "bias_pts": 0, "ret5": 0, "ret20": 0, "vol_ratio": 1, "prox60": 1,
                         "flow_ratio": c[t + 10] / c[t] - 1, "indiv_ratio": 0, "both_buy": True, "foreign5": 0, "inst5": 0, "indiv5": 0,
                         "flow_last": f"{days[t - 3]:%Y-%m-%d}"})  # 전부 이틀 이상 늦은 수급
    out = evaluate(pd.DataFrame(rows), market)
    assert "수급 상위 3" not in out and "확인된 날이 아직 0일" in out and "이틀 이상 늦은" in out
    one_day = [dict(r, flow_last=f"{days[int(np.where(days == pd.Timestamp(r['date']))[0][0]) - 1]:%Y-%m-%d}") for r in rows]  # 하루 늦은 수급은 정상
    assert "수급 상위 3" in evaluate(pd.DataFrame(one_day), market)
    rows2 = [dict(r, flow_last=r["date"]) for r in rows]  # 그날 수급이 있으면 더 좋다
    assert "수급 상위 3" in evaluate(pd.DataFrame(rows2), market)
    assert "수급 상위 3" not in evaluate(pd.DataFrame([dict(r, flow_last=None) for r in rows]), market)  # 수급 기록이 없으면 제외


def _stock(code, name, kind="stock"):
    return {"stockType": "domestic", "stockEndType": kind, "itemCode": code, "stockName": name, "closePrice": "1,000"}


def _fake_get_factory(member_ok=True):
    calls = []

    def fake_get(path, **p):
        calls.append((path, p))
        if path == "/api/stocks/theme" and "no" not in p:
            if p["page"] == 1:
                return {"stockListSortType": "THEME", "groups": [
                    {"no": 614, "name": "S7", "totalCount": 7, "changeRate": "10.26", "riseCount": 7, "fallCount": 0, "steadyCount": 0},
                    {"no": 155, "name": "반도체 대표주", "totalCount": 3, "changeRate": "9.19", "riseCount": 3, "fallCount": 0, "steadyCount": 0},
                    {"no": 700, "name": "하락 테마", "totalCount": 5, "changeRate": "-3.0", "riseCount": 0, "fallCount": 5, "steadyCount": 0},
                    {"no": 701, "name": "작은 테마", "totalCount": 2, "changeRate": "20.0", "riseCount": 2, "fallCount": 0, "steadyCount": 0}]}
            return {"stockListSortType": "THEME", "groups": []}
        if path.startswith("/api/stocks/theme/614") and member_ok:
            return {"stocks": [_stock("000660", "SK하이닉스"), _stock("0198D0", "ETF", "etf"), _stock("042700", "한미반도체"),
                               _stock("000660", "SK하이닉스"), _stock("058470", "리노공업"), _stock("240810", "원익IPS")]}
        if path.startswith("/api/stocks/theme/155") and member_ok:
            return {"result": {"stocks": [_stock("000660", "SK하이닉스"), _stock("005930", "삼성전자"), _stock("066570", "LG전자")]}}
        if path.startswith("/api/stocks/priceTop/") or path.startswith("/api/stocks/up/"):
            cat = path.rsplit("/", 1)[1]
            return {"stocks": [_stock("9" + cat[-1] * 5, f"{cat}1"), _stock("8" + cat[-1] * 5, f"{cat}2", "etf"), _stock("000660", "SK하이닉스")]}
        raise RuntimeError("HTTP404")
    fake_get.calls = calls
    return fake_get


def test_scan_parsers_and_picking():
    from stock.scan import find_stocks, parse_groups, pick_themes
    g = parse_groups({"groups": [{"no": 1, "name": "A", "totalCount": 5, "changeRate": "+1,2", "riseCount": 4},
                                 {"no": 1, "name": "중복", "totalCount": 5}, {"x": 1}]})
    assert [x["no"] for x in g] == [1] and g[0]["rate"] == 12.0 and g[0]["rise"] == 4  # 중복 제거, 숫자 파싱
    got = pick_themes([{"no": 1, "name": "a", "total": 5, "rate": 3.0, "rise": 4}, {"no": 2, "name": "b", "total": 2, "rate": 9.0, "rise": 2},
                       {"no": 3, "name": "c", "total": 5, "rate": 8.0, "rise": 1}, {"no": 4, "name": "d", "total": 5, "rate": None, "rise": 5}])
    assert [x["no"] for x in got] == [1]  # 작은 테마·상승 비율 낮은 테마·등락률 없는 테마 제외
    j = {"result": {"stocks": [_stock("005930", "삼성전자"), _stock("069500", "KODEX", "etf"), {"itemCode": "12", "stockName": "짧은코드"},
                               _stock("005930", "삼성전자")]}}
    assert find_stocks(j) == [("005930", "삼성전자")]  # ETF·6자리 아님·중복 제외, 구조 위치와 무관


def test_build_scan_themes_dedupes_caps_and_filters():
    from stock.scan import build_scan
    tm, st = build_scan(n_themes=5, per_theme=4, delay=0, get=_fake_get_factory(), log=lambda m: None)
    assert list(tm) == ["S7", "반도체 대표주"]  # 등락률 순, 하락·작은 테마 제외
    assert list(tm["S7"]) == ["000660", "042700", "058470", "240810"]  # ETF·중복 제외, 4개 제한
    assert st["source"] == "theme" and st["themes_total"] == 4 and st["themes_picked"] == 2 and st["stocks"] == 6  # 000660 은 두 테마에 겹침
    assert st["member_errors"]["/api/stocks/theme/{no}"] == "OK"


def test_build_scan_falls_back_to_ranking_and_reports_tried_urls():
    import pytest
    from stock.scan import build_scan
    tm, st = build_scan(delay=0, get=_fake_get_factory(member_ok=False), log=lambda m: None)
    assert list(tm) == ["시장 상위(거래대금·상승률)"] and st["source"] == "ranking"
    assert "000660" in tm["시장 상위(거래대금·상승률)"] and len(tm["시장 상위(거래대금·상승률)"]) >= 3
    assert "ETF" not in "".join(tm["시장 상위(거래대금·상승률)"].values())  # ETF 제외
    assert st["member_errors"]["/api/stocks/theme/{no}"] == "HTTP404"  # 시도한 주소와 상태가 남는다
    with pytest.raises(RuntimeError, match="테마 목록을 읽지 못함"):
        build_scan(delay=0, get=lambda path, **p: (_ for _ in ()).throw(RuntimeError("HTTP500")), log=lambda m: None)


def test_scan_cli_end_to_end(capsys, monkeypatch, tmp_path):
    import stock.__main__ as m
    import stock.scan as scan
    from stock.data import synthetic_market
    themes = {"테마A": {"000001": "가", "000002": "나", "000003": "다", "000004": "라"},
              "테마B": {"000003": "다", "000005": "마", "000006": "바"}}   # 000003 은 두 테마에 걸침
    mk = synthetic_market(themes=themes, index="KOSPI")
    monkeypatch.setattr(scan, "build_scan", lambda **k: (themes, {"themes_total": 120, "themes_picked": 2, "themes_used": 2, "source": "theme", "member_errors": {}, "stocks": 6}))
    monkeypatch.setattr(m, "load_naver", lambda codes, **k: mk)
    m.main(["--scan", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    import pandas as pd
    snap = pd.read_csv(tmp_path / "scan_snapshot.csv", dtype={"code": str})
    assert snap["code"].is_unique and len(snap) == 6 and {"themes", "best_theme", "theme_rank", "theme_score"} <= set(snap.columns)
    assert (tmp_path / "themes.csv").exists() and not (tmp_path / "snapshot.csv").exists()  # 훑기는 scan_snapshot 만 저장
    assert "[훑기] 네이버 테마 120개 중 오늘 오르는 2개" in out and "테마A" in out and "데이터 수집 실패 0개" in out
    import re
    names = re.findall(r"^  (\S+)\(테마[AB]\) ", out, flags=re.M)  # 선점 후보·관심 종목 줄
    assert len(names) == len(set(names))  # 여러 테마에 걸친 종목도 한 번만
    import pytest
    with pytest.raises(SystemExit):
        m.main(["--scan", "--demo"])
    with pytest.raises(SystemExit):
        m.main([])


def test_decode_prefers_utf8():
    from stock.scan import decode
    assert decode("증권".encode("utf-8")) == "증권" and decode("증권".encode("euc-kr")) == "증권"


def _scan_universe():
    from stock.data import synthetic_market
    themes = {"테마A": {"000001": "가", "000002": "나", "000003": "다", "000004": "라"},
              "테마B": {"000003": "다", "000005": "마", "000006": "바"},      # 000003 은 두 테마에 걸침
              "테마C": {"000007": "사", "000008": "아", "000009": "자", "000010": "차"}}
    return themes, synthetic_market(themes=themes, index="KOSPI")


def test_build_scan_snapshot_membership_and_best_theme(tmp_path):
    from stock.analyze import analyze_themes
    from stock.snapshot import SCAN_COLS, build_scan_snapshot, load_scan_snapshots, scan_codes
    themes, mk = _scan_universe()
    th, rows = analyze_themes(mk, themes)
    snap = build_scan_snapshot(rows, th, mk)
    assert list(snap.columns) == SCAN_COLS and snap["code"].is_unique and len(snap) == 10  # 겹친 종목도 한 줄
    shared = snap[snap["code"] == "000003"].iloc[0]
    assert set(shared["themes"].split("|")) == {"테마A", "테마B"}
    rank = {t: i + 1 for i, t in enumerate(th["theme"])}
    assert shared["best_theme"] == min(["테마A", "테마B"], key=rank.get) and shared["theme_rank"] == rank[shared["best_theme"]]
    (tmp_path / "scan" / "themes").mkdir(parents=True)
    snap.to_csv(tmp_path / "scan" / "2026-10-28.csv", index=False)
    th.to_csv(tmp_path / "scan" / "themes" / "2026-10-28.csv", index=False)  # 테마 순위 파일은 읽지 않는다
    back = load_scan_snapshots(tmp_path)
    assert len(back) == 10 and back["code"].str.len().eq(6).all() and scan_codes(tmp_path) == sorted(snap["code"])
    assert scan_codes(tmp_path / "없음") == []


def test_group_excess_and_evaluate_scan_detect_planted_signal():
    import numpy as np
    import pandas as pd
    from stock.snapshot import evaluate_scan, group_excess
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2025-01-01", periods=420)
    market, rows = {}, []
    for i in range(16):
        c = 100 * np.cumprod(1 + rng.normal(0, 0.01, 420))
        market[f"{i:06d}"] = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=days)
    for t in range(60, 300):
        fut = {code: df["close"].iloc[t + 10] / df["close"].iloc[t] - 1 for code, df in market.items()}
        order = sorted(fut, key=fut.get, reverse=True)           # 훑기가 '미래를 아는' 세계: 상위 종목이 실제로 더 오른다
        for rank_, code in enumerate(order):
            rows.append({"date": f"{days[t]:%Y-%m-%d}", "code": code, "name": code, "theme": "T", "close": 1.0,
                         "score": 100 - rank_ + rng.normal(0, 0.1), "stage": "선점 후보" if rank_ < 3 else "관망", "bias": "중립", "bias_pts": 0,
                         "ret5": 0, "ret20": 0, "vol_ratio": 1, "prox60": 1, "flow_ratio": 0.0, "indiv_ratio": 0, "both_buy": False,
                         "foreign5": 0, "inst5": 0, "indiv5": 0, "flow_last": f"{days[t]:%Y-%m-%d}", "themes": "T",
                         "best_theme": "T", "theme_rank": 1 + rank_ // 4, "theme_score": 100 - rank_ // 4})
    snap = pd.DataFrame(rows)
    out = evaluate_scan(snap, market)
    assert "점수 상위 3" in out and "선점 후보" in out and "상위 3테마" in out and "의미 있음" in out
    from stock.snapshot import attach_returns
    d = attach_returns(snap, market)
    ex = group_excess(d, d["stage"] == "선점 후보", 10)
    assert ex is not None and ex[0] > 0.01 and ex[2] >= 2
    short = d[d["date"] < "2025-04-01"]
    assert group_excess(short, short["stage"] == "선점 후보", 10) is None  # 표본이 짧으면 None
    assert "아직 부족" in evaluate_scan(snap[snap["date"] <= f"{days[70]:%Y-%m-%d}"], market)
    assert "아직 없습니다" in evaluate_scan(snap.iloc[:0], market)
    stale = snap.assign(flow_last="2000-01-01")                      # 수급이 늦은 날은 모두 제외
    assert "확인된 날이 아직 0일" in evaluate_scan(stale, market)


def test_flowcheck_allows_one_business_day_lag(tmp_path, capsys):
    import pandas as pd
    from stock.flowcheck import lag_ok, main
    s = pd.DataFrame({"date": ["2026-10-26", "2026-10-28", "2026-10-28", "2026-10-28", "2026-10-28"],
                      "flow_last": ["2026-10-23", "2026-10-28", "2026-10-27", "2026-10-26", None]})  # 월(금→월 OK), 그날, 하루 전, 이틀 전, 없음
    assert lag_ok(s).tolist() == [True, True, True, False, False]
    p = tmp_path / "snap.csv"
    pd.DataFrame({"date": ["2026-10-28"] * 5, "flow_last": ["2026-10-27"] * 5}).to_csv(p, index=False)
    main(str(p), "수급 스냅샷")
    assert "직전 거래일까지 확인된 종목 5/5" in capsys.readouterr().out  # 모두 하루 늦어도 경고 없음


def test_lag_flow_unifies_flow_definition_across_stocks():
    import numpy as np
    import pandas as pd
    from stock.snapshot import lag_flow
    days = pd.bdate_range("2026-10-19", periods=8)
    def mk(today_flow, flow_last):
        df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
                           "foreign": np.arange(8, dtype=float) * 10, "inst": 1.0, "individual": -1.0}, index=days)
        df.loc[days[-1], ["foreign", "inst", "individual"]] = today_flow   # 오늘 수급: 있는 종목은 실제 값, 없는 종목은 0 으로 채워진 값
        df.attrs["flow_last"] = flow_last
        return df
    fresh, stale = mk(999.0, f"{days[-1]:%Y-%m-%d}"), mk(0.0, f"{days[-2]:%Y-%m-%d}")   # 오늘 수급이 있는 종목 / 없는 종목
    out = lag_flow({"A": fresh, "B": stale, "KOSPI": pd.DataFrame({"close": [1.0, 2.0]})})
    prev = f"{days[-2]:%Y-%m-%d}"
    assert out["A"]["foreign"].iloc[-5:].sum() == out["B"]["foreign"].iloc[-5:].sum()   # 오늘 수급 유무와 상관없이 같은 정의
    assert out["A"]["foreign"].iloc[-1] == 60.0 and out["A"]["foreign"].iloc[0] == 0.0  # 한 거래일 뒤로 밀림(오늘 값 999 는 버려짐)
    assert out["A"].attrs["flow_last"] == out["B"].attrs["flow_last"] == prev
    assert fresh["foreign"].iloc[-1] == 999.0 and fresh.attrs["flow_last"] == f"{days[-1]:%Y-%m-%d}"  # 원본은 건드리지 않는다
    assert "foreign" not in out["KOSPI"].columns
    old = mk(0.0, f"{days[-4]:%Y-%m-%d}")                              # 더 오래된 수급 기록은 그대로 두어 시험에서 제외된다
    assert lag_flow({"C": old})["C"].attrs["flow_last"] == f"{days[-4]:%Y-%m-%d}"
    none = mk(0.0, None)
    assert lag_flow({"D": none})["D"].attrs["flow_last"] is None


def test_report_and_saved_snapshot_share_lagged_flow(capsys, monkeypatch, tmp_path):
    import pandas as pd
    import stock.__main__ as m
    import stock.scan as scan
    from stock.data import synthetic_market
    themes = {"테마A": {f"{i:06d}": f"종목{i}" for i in range(1, 6)}, "테마B": {f"{i:06d}": f"종목{i}" for i in range(6, 11)}}
    mk = synthetic_market(themes=themes, index="KOSPI")
    monkeypatch.setattr(scan, "build_scan", lambda **k: (themes, {"themes_total": 9, "themes_picked": 2, "themes_used": 2, "source": "theme",
                                                                    "member_errors": {}, "stocks": 10}))
    monkeypatch.setattr(m, "load_naver", lambda codes, **k: mk)
    m.main(["--scan", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "가격은 오늘 종가, 수급은 직전 거래일까지" in out
    snap = pd.read_csv(tmp_path / "scan_snapshot.csv", dtype={"code": str})
    st = pd.read_csv(tmp_path / "stocks.csv", dtype={"code": str})
    for code in ("000001", "000007"):
        want = mk[code]["foreign"].iloc[-6:-1].sum()                       # 직전 5거래일 수급(오늘 제외)
        assert abs(snap[snap["code"] == code].iloc[0]["foreign5"] - want) < 1e-3
        assert abs(st[st["code"] == code].iloc[0]["foreign5"] - want) < 1e-3  # 리포트와 저장분이 같은 정의(이중으로 밀리지 않음)
    th, ths = pd.read_csv(tmp_path / "themes.csv"), pd.read_csv(tmp_path / "scan_themes.csv")
    assert th["theme"].tolist() == ths["theme"].tolist()                      # 저장 테마 순위 = 리포트 테마 순위
    assert snap["theme_rank"].min() == 1


def test_ts_pro_series_matches_pine_ema_and_crossover_definition():
    import numpy as np
    import pandas as pd
    from stock.macdtest import crosses, ts_pro_series
    rng = np.random.default_rng(5)
    c = pd.Series(100 + np.cumsum(rng.normal(0, 1, 120)), index=pd.bdate_range("2025-01-01", periods=120))

    def pine_ema(x, n):               # Pine ta.ema: 첫 값으로 시작, alpha = 2/(n+1)
        out, a = [x[0]], 2 / (n + 1)
        for v in x[1:]:
            out.append(a * v + (1 - a) * out[-1])
        return np.array(out)
    ref_macd = pine_ema(c.to_numpy(), 12) - pine_ema(c.to_numpy(), 26)
    macd, sig, pct = ts_pro_series(c)
    assert np.allclose(macd, ref_macd) and np.allclose(sig, pine_ema(ref_macd, 9)) and np.allclose(pct, macd / c * 100)
    m = pd.Series([0, -1, -2, 1, 2, 3, 1, 0, -1.0]); s = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0.0])
    bull, bear = crosses(m, s)   # ta.crossover: 지금 위 & 직전 같거나 아래, crossunder 는 반대
    assert bull.tolist() == [False, False, False, True, False, False, False, False, False]
    assert bear.tolist() == [False, True, False, False, False, False, False, False, True]


def test_collect_events_finds_planted_edge_dedupes_and_ignores_noise():
    import numpy as np
    import pandas as pd
    from stock.macdtest import collect_events, verdict
    rng = np.random.default_rng(6)
    n, days = 400, pd.bdate_range("2024-01-01", periods=400)
    flags, market = {}, {}
    for k in range(14):
        ret = rng.normal(0, 0.01, n)
        pick = np.sort(rng.choice(np.arange(80, n - 15), 8, replace=False))
        pick = np.array([p for i, p in enumerate(pick) if i == 0 or p - pick[i - 1] > 12])   # 신호끼리 12일 이상 떨어지게
        ret[pick + 1] += 0.04                                                    # 신호 다음 날 +4%가 심어진 세상
        c = 100 * np.cumprod(1 + ret)
        market[f"S{k}"] = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=days)
        f = np.zeros(n, dtype=bool); f[pick] = True; flags[f"S{k}"] = f
    fn = lambda df: {"BUY 심음": pd.Series(flags[df.attrs["code"]] if "code" in df.attrs else False, index=df.index)}
    for k, df in market.items():
        df.attrs["code"] = k
    ev, base = collect_events(market, fn)
    x = ev[ev["variant"] == "BUY 심음"]
    assert len(x) > 40 and x["ex5"].mean() > 0.03 and x["ex10"].mean() > 0.03
    assert "높음" in verdict(ev, "BUY 심음")
    # 겹침 제거: 같은 종목에서 10거래일 안의 재신호는 센 번호로 안 잡힌다
    close = market["S0"]["close"]
    dup = {"BUY 겹침": pd.Series([i in (100, 103, 105, 140) for i in range(n)], index=days)}
    one = pd.DataFrame({"x": close})
    ev2, _ = collect_events({"S0": market["S0"], "S1": market["S1"]}, lambda df: dup)
    assert sorted(ev2[ev2["code"] == "S0"]["date"].dt.strftime("%Y-%m-%d").tolist()) == [f"{days[100]:%Y-%m-%d}", f"{days[140]:%Y-%m-%d}"]
    noise = {"BUY 무작위": pd.Series(rng.random(n) < 0.05, index=days)}       # 신호가 아무 정보도 없으면 초과수익은 0 근처
    ev3, _ = collect_events(market, lambda df: noise)
    assert abs(ev3["ex10"].mean()) < 0.01


def test_round_trips_cost_hold_and_comparison():
    import pandas as pd
    from stock.macdtest import round_trips
    days = pd.bdate_range("2025-01-01", periods=10)
    a = pd.DataFrame({"close": [100, 100, 100, 110, 110, 110, 121, 121, 121, 121.0]}, index=days)
    b = pd.DataFrame({"close": [100.0] * 10}, index=days)
    fl = lambda df: {"B": pd.Series([False, False, True, False, False, False, False, False, False, False] if df["close"].iloc[3] == 110 else [False] * 10, index=df.index),
                     "S": pd.Series([False] * 6 + [True] + [False] * 3 if df["close"].iloc[3] == 110 else [False] * 10, index=df.index)}
    t = round_trips({"A": a, "B": b}, fl, buy="B", sell="S", cost=0.01, warmup=0)
    assert len(t) == 1 and t["hold"].iloc[0] == 4
    assert abs(t["ret"].iloc[0] - (121 / 100 - 1 - 0.01)) < 1e-12
    assert abs(t["ex"].iloc[0] - (0.21 - 0.01 - 0.105)) < 1e-12              # 같은 기간 두 종목 평균(+10.5%)과 비교


def test_cap_universe_pages_filters_etf_and_survives_errors():
    from stock.scan import cap_universe
    def fake_get(path, **p):
        if "KOSPI" in path:
            if p["page"] == 1:
                return {"stocks": [{"stockEndType": "stock", "itemCode": "005930", "stockName": "삼성전자"},
                                   {"stockEndType": "etf", "itemCode": "069500", "stockName": "KODEX"},
                                   {"stockEndType": "stock", "itemCode": "000660", "stockName": "SK하이닉스"}]}
            if p["page"] == 2:
                return {"stocks": [{"stockEndType": "stock", "itemCode": "000660", "stockName": "SK하이닉스"},
                                   {"stockEndType": "stock", "itemCode": "035420", "stockName": "NAVER"}]}
            return {"stocks": []}
        raise RuntimeError("HTTP500")                                          # 코스닥은 실패해도 코스피 결과는 남는다
    got = cap_universe(n_kospi=3, n_kosdaq=2, get=fake_get)
    assert list(got) == ["005930", "000660", "035420"] and "069500" not in got
    assert cap_universe(get=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))) == {}


def test_macd_cli_demo(capsys):
    from stock.__main__ import main
    main(["--demo", "--macd"])
    out = capsys.readouterr().out
    assert "[MACD 신호 시험]" in out and "왕복 매매" in out and "TS Pro(0선+1.0%)" in out and "(한국)" in out
    main(["--demo", "--us", "--macd"])
    assert "(미국)" in capsys.readouterr().out


def test_open_entry_uses_next_day_open_and_same_basis_baseline():
    import pandas as pd
    from stock.macdtest import collect_events
    days = pd.bdate_range("2025-01-01", periods=30)
    close = [100.0 + i for i in range(30)]
    opn = [c + 7 for c in close]                         # 시가가 종가와 다르게 (갭)
    a = pd.DataFrame({"open": opn, "high": close, "low": close, "close": close, "volume": 1.0}, index=days)
    b = pd.DataFrame({"open": [50.0] * 30, "high": 50.0, "low": 50.0, "close": [50.0] * 30, "volume": 1.0}, index=days)
    flag = pd.Series([i == 12 for i in range(30)], index=days)
    fn = lambda df: {"BUY t": flag}
    ev_c, base_c = collect_events({"A": a, "B": b}, fn, warmup=0)
    ev_o, base_o = collect_events({"A": a, "B": b}, fn, warmup=0, entry="open")
    ra = ev_c[(ev_c.code == "A")].iloc[0]; ro = ev_o[(ev_o.code == "A")].iloc[0]
    assert abs(ra["r5"] - (close[17] / close[12] - 1)) < 1e-12                       # 신호일 종가 → 5일 뒤 종가
    assert abs(ro["r5"] - (close[17] / opn[13] - 1)) < 1e-12                         # 다음 날 시가 → 신호일 기준 5일 뒤 종가
    assert abs(ro["r10"] - (close[22] / opn[13] - 1)) < 1e-12
    day = days[12]
    want = ((close[17] / opn[13] - 1) + (50 / 50 - 1)) / 2                           # 기준선도 같은 방식(전체 종목 평균)
    assert abs((ro["r5"] - ro["ex5"]) - want) < 1e-12 and ro["date"] == day


def test_persist_verdict_and_robust_report_separates_surviving_from_vanishing_edge(capsys):
    import numpy as np
    import pandas as pd
    from stock.macdtest import persist_verdict, run_macd_robust
    ex = lambda m, sd=0.01, n=60, seed=0: pd.DataFrame({"ex5": np.random.default_rng(seed).normal(m, sd, n)})
    assert "유지됨" in persist_verdict(ex(0.03), ex(0.03, seed=1))
    assert "약함" in persist_verdict(ex(0.03), ex(0.001, 0.02, seed=2))
    assert "사라지거나" in persist_verdict(ex(0.03), ex(-0.02, seed=3))
    assert "표본 부족" in persist_verdict(ex(0.03), ex(0.03, n=5))

    n, days = 760, pd.bdate_range("2023-06-01", periods=760)   # 학습 약 1.8년 + 확인 1년

    def build(plant_until):
        rng = np.random.default_rng(11)
        market, flags = {}, {}
        for k in range(14):
            ret = rng.normal(0, 0.01, n)
            pick = np.sort(rng.choice(np.arange(80, n - 15), 24, replace=False))
            pick = np.array([p for i, p in enumerate(pick) if i == 0 or p - pick[i - 1] > 12])
            ret[pick[pick < plant_until] + 1] += 0.04                  # 심은 효과는 plant_until 이전 신호에만
            c = 100 * np.cumprod(1 + ret)
            df = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c, "low": c, "close": c, "volume": 1.0}, index=days)  # 시가 = 전날 종가(갭 없음)
            df.attrs["code"] = f"S{k}"
            market[f"S{k}"] = df
            f = np.zeros(n, dtype=bool); f[pick] = True; flags[f"S{k}"] = f
        return market, (lambda df: {"BUY 심음": pd.Series(flags[df.attrs["code"]], index=df.index)})
    m_all, fn_all = build(plant_until=n)        # 효과가 기간 내내 있는 세상
    m_is, fn_is = build(plant_until=480)        # 학습 구간에만 있고 확인 구간에서는 사라지는 세상
    out_all = run_macd_robust(m_all, "(시험)", fn_all)
    out_is = run_macd_robust(m_is, "(시험)", fn_is)
    assert out_all.count("확인 구간에서도 유지됨") == 2                                # ①②모두
    assert "확인 구간에서도 유지됨" not in out_is and ("사라지거나" in out_is or "약함" in out_is)
    assert "학습 구간" in out_all and "확인 구간(마지막 1년)" in out_all


def test_macd_robust_cli_demo(capsys):
    from stock.__main__ import main
    main(["--demo", "--macd-robust"])
    out = capsys.readouterr().out
    assert "[MACD 재확인]" in out and "① 신호일 종가에 진입" in out and "② 다음 날 시가에 진입" in out and "(한국)" in out
    main(["--demo", "--us", "--macd-robust"])
    assert "(미국)" in capsys.readouterr().out


def test_open_entry_ignores_zero_open_prices_instead_of_inf():
    import numpy as np
    import pandas as pd
    from stock.macdtest import collect_events
    days = pd.bdate_range("2025-01-01", periods=40)
    close = [100.0 + i for i in range(40)]
    a = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=days)
    b = a.copy()
    b.loc[days[13], "open"] = 0.0                                   # 시가가 0 으로 온 날(데이터 누락)
    flag = pd.Series([i in (12, 25) for i in range(40)], index=days)
    ev, base = collect_events({"A": a, "B": b}, lambda df: {"BUY t": flag}, warmup=0, entry="open")
    assert np.isfinite(ev[["r5", "r10", "ex5", "ex10"]].to_numpy()).all() and all(np.isfinite(v) for v in base.values())
    # 0 시가로 진입하는 신호(B 종목의 12일째 신호)는 진입가가 없어 제외되고, 정상 신호는 남는다
    assert not ((ev["code"] == "B") & (ev["date"] == days[12])).any() and ((ev["code"] == "A") & (ev["date"] == days[12])).any()


def test_score_components_sum_to_score_and_open_entry_in_collect():
    import numpy as np
    import pandas as pd
    from stock.analyze import analyze_stock
    from stock.backtest import collect
    from stock.data import synthetic_market
    m = synthetic_market()
    r = analyze_stock(m["000660"][["open", "high", "low", "close", "volume"]])    # 수급 열이 없으면 만점 80
    total = r["c_trend"] + r["c_near"] + r["c_volume"] + r["c_squeeze"] + r["c_flow"]
    assert r["c_flow"] == 0 and abs(total / 80 * 100 - r["score"]) < 0.06
    # collect: 구성 요소가 담기고, 다음 날 시가 진입 수익률이 시가와 종가를 구분해 계산된다
    rng = np.random.default_rng(8)
    days = pd.bdate_range("2025-01-01", periods=120)
    c = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 120))
    o = c * 1.01                                                                  # 시가가 종가와 다르게
    df = pd.DataFrame({"open": o, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1000.0}, index=days)
    ev = collect({"A": df}, {"T": {"A": "a"}})
    row = ev.iloc[0]
    t = list(days).index(row["date"])
    assert {"c_trend", "c_near", "c_volume", "c_squeeze"} <= set(ev.columns)
    assert abs(row["r5"] - (c[t + 5] / c[t] - 1)) < 1e-12 and abs(row["ro5"] - (c[t + 5] / o[t + 1] - 1)) < 1e-12
    df2 = df.copy(); df2.iloc[t + 1, df2.columns.get_loc("open")] = 0.0              # 시가 0 이면 그 신호만 제외(NaN)
    assert np.isnan(collect({"A": df2}, {"T": {"A": "a"}}).query("date == @row.date").iloc[0]["ro5"])


def test_score_robust_report_separates_lasting_from_vanishing_relationship():
    import numpy as np
    import pandas as pd
    from stock.backtest import score_robust_report
    days = pd.bdate_range("2023-06-01", periods=760)

    def build(plant_until):
        rng = np.random.default_rng(21)
        rows = []
        for d_i, d in enumerate(days):
            for k in range(14):
                score = float(rng.normal(50, 15))
                beta = 0.0008 if d_i < plant_until else 0.0       # 점수가 높을수록 이후가 좋은 관계: 심은 구간에서만
                rec = {"code": f"S{k}", "name": f"S{k}", "theme": "T", "date": d, "score": score, "bias_pts": int(rng.integers(-3, 4)),
                       "stage": "선점 후보" if score > 65 else "관망", "c_trend": float(rng.uniform(0, 25)), "c_near": float(rng.uniform(0, 25)),
                       "c_volume": float(rng.uniform(0, 20)), "c_squeeze": float(rng.uniform(0, 10))}
                for h in (5, 10):
                    rec[f"r{h}"] = beta * (score - 50) * h / 5 + rng.normal(0, 0.01)
                    rec[f"ro{h}"] = rec[f"r{h}"] - 0.001
                rows.append(rec)
        return pd.DataFrame(rows)
    out_all = score_robust_report(build(760), "(시험)")
    out_is = score_robust_report(build(480), "(시험)")                                    # 관계가 학습 구간에만 있는 세상
    assert out_all.count("확인 구간에서도 유지됨") == 2                                    # ①②모두, 학습에서 고른 기준 = 점수
    assert "고른 기준: 점수(가격 기반)" in out_all
    assert "확인 구간에서도 유지됨" not in out_is
    assert "① 신호일 종가에 진입" in out_all and "② 다음 날 시가에 진입" in out_all and "선점 후보" in out_all


def test_score_robust_cli_wiring(monkeypatch, capsys):
    import stock.__main__ as m
    import stock.backtest as bt
    seen = {}
    monkeypatch.setattr(bt, "run_score_robust", lambda market, label="": seen.update(codes=sorted(market), label=label) or "[점수 재확인] 스텁")
    m.main(["--demo", "--score-robust"])
    assert "[점수 재확인]" in capsys.readouterr().out and "KOSPI" not in seen["codes"] and seen["label"] == "(한국)" and len(seen["codes"]) == 21


def test_run_score_robust_small_market_runs():
    import stock.backtest as bt
    from stock.data import synthetic_market
    full = synthetic_market(n=420, themes={"A": {f"{i:06d}": str(i) for i in range(1, 9)}}, index="KOSPI")
    mk = {c: d for c, d in full.items() if c != "KOSPI"}
    out = bt.run_score_robust(mk, "(소형)")
    assert "[점수 재확인]" in out and "학습 구간" in out and "② 다음 날 시가" in out


def test_report_wording_shows_watch_candidate_not_preempt(capsys):
    from stock.__main__ import DISCLAIMER, main
    main(["--demo"])
    out = capsys.readouterr().out
    assert "[선점 후보]" not in out and "근거는 확인되지 않았습니다" in DISCLAIMER


def test_score_walk_report_and_cli_wiring(monkeypatch, capsys):
    import numpy as np
    import pandas as pd
    import stock.__main__ as m
    import stock.backtest as bt
    days = pd.bdate_range("2018-01-01", periods=1700)
    rng = np.random.default_rng(5)
    rows = []
    for i, d in enumerate(days):
        if i % 5:
            continue
        for k in range(14):
            score = float(rng.normal(50, 15))
            rec = {"code": f"S{k}", "name": f"S{k}", "theme": "T", "date": d, "score": score, "bias_pts": int(rng.integers(-3, 4)),
                   "stage": "선점 후보" if score > 65 else "관망", "c_trend": float(rng.uniform(0, 25)), "c_near": float(rng.uniform(0, 25)),
                   "c_volume": float(rng.uniform(0, 20)), "c_squeeze": float(rng.uniform(0, 10))}
            for h in bt.HORIZONS:
                rec[f"r{h}"] = rng.normal(0, 0.01)
                rec[f"ro{h}"] = rng.normal(0, 0.01)
            rows.append(rec)
    out = bt.walk_forward_report(pd.DataFrame(rows), "(시험)", 5)
    assert "[점수 워크포워드]" in out and "관찰 후보" in out and "연도별 요약" in out
    assert "[점수 워크포워드]" in bt.walk_forward_report(pd.DataFrame(), "", 5)
    seen = {}
    monkeypatch.setattr(bt, "run_score_walk", lambda market, label="": seen.update(codes=sorted(market), label=label) or "[점수 워크포워드] 스텁")
    m.main(["--demo", "--score-walk"])
    assert "스텁" in capsys.readouterr().out and "KOSPI" not in seen["codes"] and seen["label"] == "(한국)"


def test_dart_fetch_summaries_reports_reason_and_caches(monkeypatch):
    import stock.dart as d
    d._CC.clear()
    calls = {"cc": 0}

    def fake_cc(key):
        calls["cc"] += 1
        d._CC[key] = {"005930": "00126380"}
        return d._CC[key]
    monkeypatch.setattr(d, "corp_codes", fake_cc)
    monkeypatch.setattr(d, "recent_titles", lambda key, cc, days=90: ["유상증자결정", "분기보고서"])
    out = d.fetch_summaries(["005930"], key="k")
    assert out["005930"]["count"] == 2 and d.LAST_ERROR == ""
    assert d.fetch_summaries([], key="k") == {} and calls["cc"] == 1       # 대상이 없으면 목록을 받지 않는다

    def bad(key, cc, days=90):
        raise RuntimeError("DART 응답 010: 등록되지 않은 키")
    monkeypatch.setattr(d, "recent_titles", bad)
    assert d.fetch_summaries(["005930"], key="k") == {} and "010" in d.LAST_ERROR
    monkeypatch.delenv("DART_API_KEY", raising=False)
    assert d.fetch_summaries(["005930"]) == {} and "없음" in d.LAST_ERROR


FORM4 = """<ownershipDocument><nonDerivativeTable>
<nonDerivativeTransaction><transactionCoding><transactionCode>P</transactionCode></transactionCoding>
<transactionAmounts><transactionShares><value>1000</value></transactionShares><transactionPricePerShare><value>50.0</value></transactionPricePerShare></transactionAmounts></nonDerivativeTransaction>
<nonDerivativeTransaction><transactionCoding><transactionCode>S</transactionCode></transactionCoding>
<transactionAmounts><transactionShares><value>200</value></transactionShares><transactionPricePerShare><value>60.0</value></transactionPricePerShare></transactionAmounts></nonDerivativeTransaction>
<nonDerivativeTransaction><transactionCoding><transactionCode>M</transactionCode></transactionCoding>
<transactionAmounts><transactionShares><value>999</value></transactionShares><transactionPricePerShare><value>1.0</value></transactionPricePerShare></transactionAmounts></nonDerivativeTransaction>
</nonDerivativeTable></ownershipDocument>"""


def test_edgar_parse_summarize_and_fetch(monkeypatch, capsys):
    import stock.edgar as e
    from stock.__main__ import us_disclosure_lines
    assert e.parse_form4(FORM4) == [("P", 1000.0, 50.0), ("S", 200.0, 60.0)]      # 옵션 행사(M)는 제외
    fl = [{"form": "8-K", "date": "2026-10-01", "items": "2.02,9.01", "acc": "0001-26-000001", "doc": "a.htm"},
          {"form": "8-K", "date": "2026-10-02", "items": "4.02", "acc": "0001-26-000002", "doc": "b.htm"},
          {"form": "4", "date": "2026-10-03", "items": "", "acc": "0001-26-000003", "doc": "xslF345X05/f4.xml"}]
    s = e.summarize_8k(fl)
    assert s["count"] == 2 and s["negative"] == ["재무제표 신뢰 불가(정정 필요)(2026-10-02)"] and s["notable"] == ["실적 발표"]
    assert e.summarize_8k([{"form": "6-K", "date": "2026-10-01", "items": "", "acc": "x", "doc": "y"}])["foreign"] is True
    urls = []

    class R:
        text = FORM4
    monkeypatch.setattr(e, "_get", lambda url, ua=None: urls.append(url) or R())
    ins = e.insider_trades(123, fl)
    assert ins["buys"] == 1 and ins["sells"] == 1 and ins["buy_value"] == 50000.0 and ins["sell_value"] == 12000.0
    assert urls == ["https://www.sec.gov/Archives/edgar/data/123/000126000003/f4.xml"]      # xsl 폴더 없는 원본 주소
    monkeypatch.setattr(e, "ticker_ciks", lambda: {"LLY": 123})
    monkeypatch.setattr(e, "recent_filings", lambda cik, days=90, ticker=None: fl)
    out = e.fetch_summaries(["LLY", "ZZZZ"])
    assert out["LLY"]["insider"]["buys"] == 1 and "ZZZZ" not in out and "ZZZZ" in e.LAST_ERROR
    lines = us_disclosure_lines(out["LLY"])
    assert "8-K 2건" in lines[0] and "주의 항목" in lines[0] and "매수 1건" in lines[1]
    assert "미확인" in us_disclosure_lines(None, "접속 실패")[0]


def test_edgar_blocked_reports_reason_and_uses_fallback(monkeypatch):
    import stock.edgar as e

    class Resp:
        def __init__(self, code, text="", js=None):
            self.status_code, self.text, self._js = code, text, js

        def json(self):
            return self._js
    monkeypatch.setattr(e.time, "sleep", lambda s: None)
    monkeypatch.setattr(e.requests, "get", lambda url, headers=None, timeout=0: Resp(403, "<html><body>Undeclared Automated Tool</body></html>"))
    try:
        e._get("https://www.sec.gov/files/company_tickers.json")
        assert False
    except RuntimeError as ex:
        assert "HTTP 403 www.sec.gov" in str(ex) and "Undeclared Automated Tool" in str(ex)
    e._TICKERS.clear()
    out = e.fetch_summaries(["AAPL", "NVDA"])
    assert out == {} and "티커 목록 실패" in e.LAST_ERROR and "data.sec.gov" in e.LAST_ERROR     # 내장 CIK로 data.sec.gov까지 시도했다는 뜻

    def sub(url, headers=None, timeout=0):
        if "company_tickers" in url:
            return Resp(403, "blocked")
        return Resp(200, js={"tickers": ["AAPL"], "filings": {"recent": {"form": ["8-K"], "filingDate": [e.date.today().isoformat()],
                                                                       "items": ["2.02"], "accessionNumber": ["1-1-1"], "primaryDocument": ["a.htm"]}}})
    monkeypatch.setattr(e.requests, "get", sub)
    out = e.fetch_summaries(["AAPL", "NVDA"], insiders=False)
    assert out["AAPL"]["count"] == 1 and "NVDA" not in out and "다름" in e.LAST_ERROR   # 응답 티커와 다르면 버린다


def test_dart_check_report_lists_dated_titles(monkeypatch):
    import stock.dart as d
    monkeypatch.setattr(d, "corp_codes", lambda key: {"036460": "00001"})
    monkeypatch.setattr(d, "recent_items", lambda key, cc, days=90: [("20261028", "단일판매ㆍ공급계약체결"), ("20261001", "유상증자결정"), ("20260930", "분기보고서")])
    out = d.check_report(["036460", "999999"], key="k")
    assert "[036460] 최근 90일 공시 3건" in out and "2026-10-28 단일판매ㆍ공급계약체결 ＋호재성" in out
    assert "2026-10-01 유상증자결정 －악재성" in out and "[999999] DART 회사 코드 없음" in out


def test_paper_trading_step_entry_exit_and_stats():
    import numpy as np
    import pandas as pd
    from stock.paper import COLS, COST, HOLD, MAX_NEW, report, stats, step
    days = pd.bdate_range("2026-10-01", periods=14)

    def px(open_, close):
        return pd.DataFrame({"open": open_, "close": close}, index=days)
    a = px([100.0] * 14, [100.0 + i for i in range(14)])                    # 신호 다음 날 시가 100, 이후 종가 상승
    prices = {"AAA": a, "KOSPI": pd.DataFrame({"close": np.linspace(1000, 1010, 14)}, index=days)}
    sd = f"{days[2]:%Y-%m-%d}"
    scan = pd.DataFrame({"date": [sd] * 5, "code": ["AAA", "B", "C", "D", "E"], "name": ["AAA", "B", "C", "D", "E"], "stage": ["선점 후보"] * 4 + ["관망"],
                         "score": [70, 90, 80, 60, 99], "close": [100.0] * 5})
    L, ev = step(pd.DataFrame(columns=COLS), scan, prices)
    assert [n for n, _ in ev["new"]] == ["B", "C", "AAA"] and len(L) == MAX_NEW        # 점수 순 상위 3, 관망은 제외
    L2, ev2 = step(L, scan, prices)
    assert len(L2) == MAX_NEW and ev2["new"] == []                                   # 같은 날 다시 돌려도 중복 신호 없음
    L3, ev3 = step(L2, scan.iloc[0:0], prices)
    row = L3[L3["code"] == "AAA"].iloc[0]
    assert row["status"] == "closed" and row["entry_date"] == f"{days[3]:%Y-%m-%d}" and row["entry_px"] == 100.0
    assert row["exit_date"] == f"{days[2 + HOLD]:%Y-%m-%d}"                          # 신호일로부터 HOLD 거래일째
    assert abs(row["ret_net"] - ((100.0 + 2 + HOLD) / 100.0 - 1 - COST)) < 1e-9
    assert row["kospi_ret"] == row["kospi_ret"] and L3[L3["code"] == "B"].iloc[0]["status"] == "pending"   # 가격이 없는 종목은 그대로 대기
    s = stats(L3)
    assert s["n"] == 1 and s["open"] == 2 and "아직 적어" in report(L3, ev3) and "실제 매매·주문 없음" in report(L3, ev3)


def test_paper_us_runs_with_own_ledger(tmp_path, monkeypatch):
    import stock.paper as pp
    import stock.yahoo as y
    from stock.data import synthetic_market
    from stock.us_themes import US_THEMES
    mk = synthetic_market(themes=US_THEMES, index="SPX", flow=False, n=300)
    monkeypatch.setattr(y, "load_yahoo", lambda tickers, log=print, **k: mk)
    out = pp.run_us(str(tmp_path))
    assert "[모의투자·미국]" in out and "실제 매매·주문 없음" in out
    assert (tmp_path / "paper" / "ledger_us.csv").exists() and not (tmp_path / "paper" / "ledger.csv").exists()
    n1 = len(pp.load_ledger(str(tmp_path), "ledger_us.csv"))
    assert "누적" in pp.run_us(str(tmp_path))                                  # 같은 날 다시 돌려도 장부가 늘지 않는다
    assert len(pp.load_ledger(str(tmp_path), "ledger_us.csv")) == n1
