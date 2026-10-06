"""기술적 지표: 볼린저밴드, 일목균형표, RSI."""
from __future__ import annotations

import pandas as pd


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    pct_b = (close - lower) / (upper - lower)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "pct_b": pct_b})


def ichimoku(
    df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52, shift: int = 26
) -> pd.DataFrame:
    """일목균형표. 선행스팬은 `shift`일 앞으로 그려지므로, t 시점의 구름은
    t-shift 시점까지의 데이터로 계산한 값이다(미래 데이터를 쓰지 않음).
    후행스팬(chikou)은 신호에 쓰지 않으므로 계산하지 않는다."""
    hi, lo = df["high"], df["low"]

    def mid(n: int) -> pd.Series:
        return (hi.rolling(n).max() + lo.rolling(n).min()) / 2

    t, k = mid(tenkan), mid(kijun)
    span_a = ((t + k) / 2).shift(shift)
    span_b = mid(senkou_b).shift(shift)
    return pd.DataFrame(
        {
            "tenkan": t,
            "kijun": k,
            "span_a": span_a,
            "span_b": span_b,
            "cloud_top": pd.concat([span_a, span_b], axis=1).max(axis=1, skipna=False),
            "cloud_bottom": pd.concat([span_a, span_b], axis=1).min(axis=1, skipna=False),
        }
    )


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)
