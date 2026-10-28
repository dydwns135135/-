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
