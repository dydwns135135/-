"""교과서적인 대표 매매 규칙들. 각 함수는 일봉 DataFrame(open/high/low/close)을 받아
'봉 종가 시점에 알 수 있는 정보만으로' 목표 포지션(+1 롱, 0 현금, -1 숏) 시리즈를 돌려준다.
(체결은 다음 봉 시가 — simulate() 가 한 봉 지연시킨다.) 변수는 흔히 쓰는 값으로 고정하고 튜닝하지 않는다."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader.indicators import rsi


def buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def sma_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200, short: bool = False) -> pd.Series:
    f, s = df["close"].rolling(fast).mean(), df["close"].rolling(slow).mean()
    ready = s.notna()
    pos = pd.Series(np.where(f > s, 1.0, -1.0 if short else 0.0), index=df.index)
    return pos.where(ready, 0.0)


def momentum(df: pd.DataFrame, lookback: int = 90, short: bool = False) -> pd.Series:
    ret = df["close"].pct_change(lookback)
    pos = pd.Series(np.where(ret > 0, 1.0, -1.0 if short else 0.0), index=df.index)
    return pos.where(ret.notna(), 0.0)


def donchian(df: pd.DataFrame, entry: int = 20, exit_: int = 10, short: bool = False) -> pd.Series:
    """직전 entry 일 최고가를 종가가 돌파하면 롱, 직전 exit_ 일 최저가를 이탈하면 청산(숏이면 반대)."""
    c = df["close"].to_numpy()
    hi_e = df["high"].rolling(entry).max().shift(1).to_numpy()
    lo_e = df["low"].rolling(entry).min().shift(1).to_numpy()
    hi_x = df["high"].rolling(exit_).max().shift(1).to_numpy()
    lo_x = df["low"].rolling(exit_).min().shift(1).to_numpy()
    pos, cur = np.zeros(len(c)), 0.0
    for i in range(len(c)):
        if np.isnan(hi_e[i]) or np.isnan(lo_x[i]):
            pos[i] = 0.0
            continue
        if cur <= 0 and c[i] > hi_e[i]:
            cur = 1.0
        elif cur >= 0 and short and c[i] < lo_e[i]:
            cur = -1.0
        elif cur > 0 and c[i] < lo_x[i]:
            cur = 0.0
        elif cur < 0 and c[i] > hi_x[i]:
            cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=df.index)


def rsi_revert(df: pd.DataFrame, window: int = 14, lo: float = 30, hi: float = 50) -> pd.Series:
    """과매도(RSI<lo)에서 사서 RSI>hi 에서 정리하는 평균회귀 (롱만)."""
    r = rsi(df["close"], window).to_numpy()
    pos, cur = np.zeros(len(r)), 0.0
    for i in range(len(r)):
        if np.isnan(r[i]):
            continue
        if cur == 0 and r[i] < lo:
            cur = 1.0
        elif cur == 1 and r[i] > hi:
            cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=df.index)


def vol_target_size(df: pd.DataFrame, target: float = 0.20, max_lev: float = 2.0, window: int = 30,
                    bars_per_year: float = 365) -> pd.Series:
    """최근 변동성이 크면 포지션을 줄이고 작으면 늘린다(연 target 변동성 목표, 최대 max_lev 배)."""
    vol = df["close"].pct_change().rolling(window).std() * np.sqrt(bars_per_year)
    return (target / vol).clip(upper=max_lev).fillna(0.0)


def simulate(df: pd.DataFrame, pos: pd.Series, size: pd.Series | None = None, fee: float = 0.0006,
             slip: float = 0.0005, funding_per_8h: float = 0.0001, bar_hours: float = 24.0) -> pd.Series:
    """봉별 손익률(자본 대비). 봉 i 종가에 정한 포지션은 i+1 봉 시가에 체결되어 다음 시가까지 보유.
    수수료·슬리피지는 비중 변화량에 부과, 펀딩비는 롱이 내고 숏이 받는 상수로 가정."""
    o = df["open"].to_numpy()
    r_oo = pd.Series(np.r_[o[1:] / o[:-1] - 1, 0.0], index=df.index)  # open_i → open_{i+1}
    w_dec = pos * (size if size is not None else 1.0)       # 종가 i 에 정한 목표 비중
    w = w_dec.shift(1).fillna(0.0)                           # 봉 i 에 실제로 들고 있는 비중
    turnover = (w - w.shift(1).fillna(0.0)).abs()
    funding = w * funding_per_8h * bar_hours / 8
    pnl = w * r_oo - turnover * (fee + slip) - funding
    return pnl.iloc[:-1]  # 마지막 봉은 다음 시가가 없어 제외
