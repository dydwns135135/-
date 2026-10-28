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
    assert decide({**r, "stage": "과열(추격금지)"}, reg)[0] == "관망(패스)"
    assert decide({**r, "stage": "선점 후보"}, reg, summarize(["유상증자결정"]))[0] == "관망(패스)"
    strong = {**r, "stage": "선점 후보", "score": 80, "ret20": 0.05, "both_buy": True}
    assert decide(strong, reg, summarize(["공급계약체결"]))[0] == "즉시 진입"
    assert decide(strong, {"points": 1, "label": "약세(보수)"}, summarize(["공급계약체결"]))[0] == "분할 진입"


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
    monkeypatch.setattr(y, "fetch_naver_us", lambda s: (_ for _ in ()).throw(RuntimeError("네이버 x")))
    monkeypatch.setattr(y, "fetch_yahoo", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("야후 실패(HTTP 429)")))
    monkeypatch.setattr(y, "fetch_stooq", lambda s: y.parse_stooq(csv))
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
