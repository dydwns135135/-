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
