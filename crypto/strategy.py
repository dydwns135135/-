"""롱/숏 대칭 전략: 일목균형표(추세) + 볼린저밴드(눌림/반등 진입).

롱  진입: 종가 > 구름 상단 & 전환선 > 기준선 & %B <= pullback
    청산: 종가 < 기준선 | 종가 < 구름 상단 | %B >= overheat
숏  진입: 종가 < 구름 하단 & 전환선 < 기준선 & %B >= 1 - pullback
    청산: 종가 > 기준선 | 종가 > 구름 하단 | %B <= 1 - overheat
"""
from __future__ import annotations

import pandas as pd

from trader.strategy import StrategyParams, compute_signals


def compute_ls_signals(df: pd.DataFrame, p: StrategyParams = StrategyParams()) -> pd.DataFrame:
    s = compute_signals(df, p)  # 지표 + 롱 신호(entry/exit)
    c = s["close"]
    ready = s[["pct_b", "kijun", "cloud_bottom"]].notna().all(axis=1)
    bearish = (c < s["cloud_bottom"]) & (s["tenkan"] < s["kijun"])
    out = s.rename(columns={"entry": "long_entry", "exit": "long_exit"})
    out["short_entry"] = bearish & (s["pct_b"] >= 1 - p.pullback) & ready
    out["short_exit"] = ((c > s["kijun"]) | (c > s["cloud_bottom"]) | (s["pct_b"] <= 1 - p.overheat)) & ready
    return out


def latest_action(df: pd.DataFrame, p: StrategyParams = StrategyParams()) -> dict:
    """마지막 *마감된* 봉 기준 신호."""
    r = compute_ls_signals(df, p).iloc[-1]
    return {k: bool(r[k]) for k in ("long_entry", "long_exit", "short_entry", "short_exit")}
