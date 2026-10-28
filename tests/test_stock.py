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
    assert "[수집] 21/23종목 — 누락: LMT, RTX" in out and "⚠️종목 3/5" in out and "S&P500 대용" in out


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
                         "flow_last": f"{days[t - 1]:%Y-%m-%d}"})  # 전부 하루 늦은 수급
    out = evaluate(pd.DataFrame(rows), market)
    assert "수급 상위 3" not in out and "확정된 날이 아직 0일" in out and "하루 늦은" in out
    rows2 = [dict(r, flow_last=r["date"]) for r in rows]  # 전부 그날 수급이면 시험이 진행된다
    assert "수급 상위 3" in evaluate(pd.DataFrame(rows2), market)


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
    assert "확정된 날이 아직 0일" in evaluate_scan(stale, market)
