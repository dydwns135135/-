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


THEME_LIST_HTML = """<table><tr><th>테마명</th><th>전일대비</th><th>최근3일</th></tr>
<tr><td class="col_type1"><a href="/sise/sise_group_detail.naver?type=theme&no=511">HBM(High Bandwidth Memory)</a></td>
<td><span>+1.20%</span></td><td><span>+6.50%</span></td><td>10</td><td>1</td><td>2</td>
<td><a href="/item/main.naver?code=000660">SK하이닉스</a></td><td><a href="/item/main.naver?code=042700">한미반도체</a></td></tr>
<tr><td><a href="/sise/sise_group_detail.naver?type=theme&no=303">방위산업</a></td><td><span>-0.80%</span></td><td><span>+2.10%</span></td>
<td><a href="/item/main.naver?code=012450">한화에어로스페이스</a></td></tr>
<tr><td><a href="/sise/sise_group_detail.naver?type=theme&no=88">조선</a></td><td><span>+3.00%</span></td><td><span>-1.00%</span></td></tr>
<tr><td><a href="/sise/sise_group_detail.naver?type=theme&no=511">HBM(High Bandwidth Memory)</a></td><td>중복 줄</td></tr></table>"""


def _detail_html(codes):
    rows = "".join(f'<tr><td><div class="name_area"><a href="/item/main.naver?code={c}">N{c}</a></div></td><td>1,000</td></tr>' for c in codes)
    return f"<table>{rows}<tr><td><a href=\"/item/main.naver?code={codes[0]}\">N{codes[0]}</a></td></tr></table>"  # 같은 종목이 다시 나옴


def test_scan_parsers_and_picking():
    from stock.scan import parse_theme_list, parse_theme_members, pick_themes
    th = parse_theme_list(THEME_LIST_HTML)
    assert [t["no"] for t in th] == ["511", "303", "88"]  # 중복 줄 제거
    assert th[0]["d1"] == 1.2 and th[0]["d3"] == 6.5 and th[0]["leaders"] == [("000660", "SK하이닉스"), ("042700", "한미반도체")]
    assert th[2]["d1"] == 3.0 and th[2]["d3"] == -1.0
    got = pick_themes(th, n3=1, n1=1)  # 3일 1위(HBM) + 전일 1위(조선)
    assert [t["no"] for t in got] == ["511", "88"]
    assert parse_theme_members(_detail_html(["005930", "000660", "035420"])) == [("005930", "N005930"), ("000660", "N000660"), ("035420", "N035420")]
    assert parse_theme_list("<html>점검 중입니다</html>") == []


def test_build_scan_merges_leaders_dedupes_and_reports_failure():
    import pytest
    from stock.scan import DETAIL_URL, LIST_URL, build_scan
    def fake_get(url, **p):
        if url == LIST_URL:
            return THEME_LIST_HTML if p.get("page") == 1 else "<table></table>"
        no = p["no"]
        return {"511": _detail_html(["000660", "058470", "240810", "036930"]),   # 000660 은 주도주와 겹침
                "303": _detail_html(["012450", "079550", "047810"]),
                "88": _detail_html(["009540"])}[no]                               # 종목이 3개 미만 → 제외
    tm, stats = build_scan(n3=5, n1=5, per_theme=4, delay=0, get=fake_get, log=lambda m: None)
    assert list(tm["HBM(High Bandwidth Memory)"]) == ["000660", "042700", "058470", "240810"]  # 주도주 먼저, 중복 제거, 4개 제한
    assert "조선" not in tm and stats["themes_thin"] == 1 and stats["themes_used"] == 2 and stats["stocks"] == 7
    with pytest.raises(RuntimeError, match="점검 중"):
        build_scan(delay=0, get=lambda url, **p: "<html>점검 중입니다</html>", log=lambda m: None)


def test_scan_cli_end_to_end(capsys, monkeypatch):
    import stock.__main__ as m
    import stock.scan as scan
    from stock.data import synthetic_market
    themes = {"테마A": {"000001": "가", "000002": "나", "000003": "다", "000004": "라"},
              "테마B": {"000003": "다", "000005": "마", "000006": "바"}}   # 000003 은 두 테마에 걸침
    mk = synthetic_market(themes=themes, index="KOSPI")
    monkeypatch.setattr(scan, "build_scan", lambda **k: (themes, {"themes_total": 120, "themes_picked": 2, "themes_used": 2, "themes_thin": 0, "stocks": 6}))
    monkeypatch.setattr(m, "load_naver", lambda codes, **k: mk)
    m.main(["--scan"])
    out = capsys.readouterr().out
    assert "[훑기] 네이버 테마 120개 중 움직이는 2개" in out and "테마A" in out and "데이터 수집 실패 0개" in out
    import re
    names = re.findall(r"^  (\S+)\(테마[AB]\) ", out, flags=re.M)  # 선점 후보·관심 종목 줄
    assert len(names) == len(set(names))  # 여러 테마에 걸친 종목도 한 번만
    import pytest
    with pytest.raises(SystemExit):
        m.main(["--scan", "--demo"])
    with pytest.raises(SystemExit):
        m.main([])


def test_decode_prefers_utf8_and_describe_flags_js_pages():
    from stock.diag_naver import describe
    from stock.scan import decode
    assert decode("증권".encode("utf-8")) == "증권" and decode("증권".encode("euc-kr")) == "증권"

    class R:
        status_code, url = 200, "https://x/y"
        headers = {"content-type": "text/html; charset=UTF-8"}
        content = "<html><title>Npay 증권</title><script>self.__next_s=[]</script></html>".encode("utf-8")
    out = describe(R())
    assert "Npay 증권" in out and "<tr': 0" in out and "item/main.naver': 0" in out
    R.headers = {"content-type": "application/json"}
    R.content = '{"stocks": [{"itemCode": "005930"}]}'.encode("utf-8")
    assert "005930" in describe(R())
