"""일봉 백테스트. 신호는 t일 종가에 확정, 체결은 t+1일 시가(수수료·슬리피지 반영)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy import StrategyParams, compute_signals


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close"]].dropna()


def backtest(
    df: pd.DataFrame,
    params: StrategyParams = StrategyParams(),
    cash: float = 10_000.0,
    fee_rate: float = 0.0025,  # 편도 수수료율 (실제 토스증권 요율 확인 후 조정)
    slippage: float = 0.0005,
    stop_loss: float | None = 0.08,
) -> BacktestResult:
    sig = compute_signals(df, params)
    opens, closes = sig["open"].to_numpy(), sig["close"].to_numpy()
    entry, exit_ = sig["entry"].to_numpy(), sig["exit"].to_numpy()

    shares, entry_px, entry_i = 0, 0.0, 0
    equity, trades = [], []
    pending = None  # 'buy' | 'sell'  (다음 봉 시가에 체결)

    for i in range(len(sig)):
        # 1) 전일 신호를 오늘 시가에 체결
        if pending == "buy" and shares == 0:
            px = opens[i] * (1 + slippage)
            n = int(cash // (px * (1 + fee_rate)))
            if n > 0:
                cash -= n * px * (1 + fee_rate)
                shares, entry_px, entry_i = n, px, i
        elif pending == "sell" and shares > 0:
            px = opens[i] * (1 - slippage)
            cash += shares * px * (1 - fee_rate)
            trades.append((sig.index[entry_i], sig.index[i], entry_px, px, shares))
            shares = 0
        pending = None

        # 2) 장중 손절(시가/저가 기준 단순화: 종가가 손절선 아래면 다음 시가 청산)
        stop_hit = shares > 0 and stop_loss is not None and closes[i] <= entry_px * (1 - stop_loss)

        # 3) 오늘 종가 기준 신호 → 내일 시가 체결
        if shares > 0 and (exit_[i] or stop_hit):
            pending = "sell"
        elif shares == 0 and entry[i] and not exit_[i]:
            pending = "buy"

        equity.append(cash + shares * closes[i])

    equity_s = pd.Series(equity, index=sig.index, name="equity")
    tdf = pd.DataFrame(trades, columns=["entry_date", "exit_date", "entry_px", "exit_px", "shares"])
    if not tdf.empty:
        tdf["ret"] = tdf["exit_px"] / tdf["entry_px"] - 1
    return BacktestResult(equity_s, tdf, _metrics(equity_s, tdf, df["close"]))


def _metrics(eq: pd.Series, trades: pd.DataFrame, close: pd.Series) -> dict:
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    dd = (eq / eq.cummax() - 1).min()
    bh = close.iloc[-1] / close.iloc[0] - 1
    return {
        "총수익률": total,
        "연환산수익률": (1 + total) ** (1 / years) - 1 if total > -1 else -1.0,
        "최대낙폭(MDD)": dd,
        "거래횟수": len(trades),
        "승률": float((trades["ret"] > 0).mean()) if len(trades) else float("nan"),
        "단순보유수익률": bh,
    }


def synthetic_prices(n: int = 1500, seed: int = 0, drift: float = 0.0004, vol: float = 0.015) -> pd.DataFrame:
    """테스트·데모용 랜덤워크 일봉 (실제 시장 데이터가 아님)."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
    spread = np.abs(rng.normal(0, vol / 2, n))
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) * (1 + spread),
         "low": np.minimum(open_, close) * (1 - spread), "close": close},
        index=idx,
    )
