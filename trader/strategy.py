"""일목균형표(추세) + 볼린저밴드(진입 타이밍) 롱 온리 전략.

진입: 일목 강세(종가 > 구름 상단, 전환선 > 기준선)인 상태에서
      %B 가 pullback 이하로 내려온 눌림목.
청산: 종가가 기준선 아래로 내려가거나 구름 상단 아래로 이탈, 또는 %B 가 overheat 이상(과열).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators import bollinger, ichimoku


@dataclass
class StrategyParams:
    bb_window: int = 20
    bb_k: float = 2.0
    pullback: float = 0.35  # %B 가 이 값 이하면 눌림목
    overheat: float = 1.0  # %B 가 이 값 이상이면 과열 청산
    tenkan: int = 9
    kijun: int = 26
    senkou_b: int = 52
    shift: int = 26


def compute_signals(df: pd.DataFrame, p: StrategyParams = StrategyParams()) -> pd.DataFrame:
    """df: index=날짜, columns=open/high/low/close. 각 행 종가 시점에 알 수 있는 정보만 사용.
    반환: 지표 + `entry`, `exit` 불리언 컬럼."""
    out = df.copy()
    out = out.join(bollinger(df["close"], p.bb_window, p.bb_k))
    out = out.join(ichimoku(df, p.tenkan, p.kijun, p.senkou_b, p.shift))
    c = out["close"]
    bullish = (c > out["cloud_top"]) & (out["tenkan"] > out["kijun"])
    out["entry"] = bullish & (out["pct_b"] <= p.pullback)
    out["exit"] = (c < out["kijun"]) | (c < out["cloud_top"]) | (out["pct_b"] >= p.overheat)
    # 지표 계산이 안 되는 초기 구간은 신호 없음
    ready = out[["pct_b", "kijun", "cloud_top"]].notna().all(axis=1)
    out["entry"] &= ready
    out["exit"] &= ready
    return out


def latest_signal(df: pd.DataFrame, p: StrategyParams = StrategyParams()) -> str:
    """'buy' | 'sell' | 'hold' (마지막 봉 기준)."""
    last = compute_signals(df, p).iloc[-1]
    if last["exit"]:
        return "sell"
    if last["entry"]:
        return "buy"
    return "hold"
