"""TradingView 지표 'EMA & Volume Breakout'(사용자 제공)을 그대로 백테스트한다.
python -m crypto.ema_volume [--years 3]   (Bitget 공개 시세만 사용, API 키·주문 없음)

규칙(Pine 스크립트와 동일, 값 변경 없음): 빠른 EMA 20 / 느린 EMA 50, 거래량 평균 20봉.
- 매수 신호: 빠른 EMA가 느린 EMA를 위로 교차 + 그 봉 거래량 > 평균 × 1.5
- 매도 신호: 아래로 교차 + 거래량 > 평균 × 1.5
신호는 봉 마감 때 확정, 다음 봉 시가에 체결. 포지션은 반대 신호가 나올 때까지 유지한다.
- 롱+숏: 매수 신호 → 롱, 매도 신호 → 숏(선물)    - 롱만: 매수 신호 → 롱, 매도 신호 → 청산(현물처럼)
비용: 체결당 0.11%(수수료 0.06% + 슬리피지 0.05%). 펀딩비는 반영하지 않음."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .upbit_scalp import simulate

FAST, SLOW, VOL_LEN, VOL_MULT = 20, 50, 20, 1.5
COST = 0.0011
BARS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365}
SYMBOLS = ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT")


def buy_sell(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    fast = df["close"].ewm(span=FAST, adjust=False).mean()
    slow = df["close"].ewm(span=SLOW, adjust=False).mean()
    vol_ok = df["volume"] > df["volume"].rolling(VOL_LEN).mean() * VOL_MULT
    up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    warm = pd.Series(np.arange(len(df)) >= SLOW + VOL_LEN, index=df.index)
    return (up & vol_ok & warm), (dn & vol_ok & warm)


def positions(buy: pd.Series, sell: pd.Series, long_short: bool) -> pd.Series:
    out, pos = np.zeros(len(buy)), 0.0
    b, s = buy.to_numpy(), sell.to_numpy()
    for i in range(len(b)):
        if b[i]:
            pos = 1.0
        elif s[i]:
            pos = -1.0 if long_short else 0.0
        out[i] = pos
    return pd.Series(out, index=buy.index)


def stats(eq: pd.Series, pos: pd.Series, bars_per_year: float) -> dict:
    r = eq.pct_change().fillna(eq.iloc[0] - 1)
    years = max(len(eq) / bars_per_year, 1e-9)
    final = float(eq.iloc[-1])
    p = pos.shift(1).fillna(0.0)
    return {"연환산": final ** (1 / years) - 1 if final > 0 else -1.0,
            "최대낙폭": float((eq / eq.cummax() - 1).min()),
            "샤프": float(r.mean() / r.std() * np.sqrt(bars_per_year)) if r.std() > 0 else float("nan"),
            "거래횟수": int((p.diff().abs() > 0).sum()), "평균투입": float(p.abs().mean())}


def compare(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    buy, sell = buy_sell(df)
    sigs = {"보유": pd.Series(1.0, index=df.index), "롱+숏": positions(buy, sell, True), "롱만": positions(buy, sell, False)}
    bpy = BARS_PER_YEAR[tf]
    half = len(df) // 2
    rows = {}
    for name, sig in sigs.items():
        eq = simulate(df, sig, COST)
        rows[f"{name} 전체"] = stats(eq, sig, bpy)
        e2, s2 = simulate(df.iloc[half:], sig.iloc[half:], COST), sig.iloc[half:]
        rows[f"{name} 뒤절반"] = stats(e2, s2, bpy)
    return pd.DataFrame(rows).T


def signal_counts(df: pd.DataFrame) -> tuple[int, int]:
    b, s = buy_sell(df)
    return int(b.sum()), int(s.sum())


def fmt(t: pd.DataFrame) -> str:
    x = t.copy()
    x["연환산"] = x["연환산"].map(lambda v: f"{v * 100:+.0f}%")
    x["최대낙폭"] = x["최대낙폭"].map(lambda v: f"{v * 100:+.0f}%")
    x["샤프"] = x["샤프"].map(lambda v: f"{v:.2f}")
    x["거래횟수"] = x["거래횟수"].astype(int)
    x["평균투입"] = x["평균투입"].map(lambda v: f"{v * 100:.0f}%")
    return x.to_string()


def main() -> None:
    from .backtest_report import load_data
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3.0)
    a = ap.parse_args()
    out, tabs = [], {"1h": [], "4h": []}
    for tf, min_bars in (("1h", 3000), ("4h", 1500)):
        for sym in SYMBOLS:
            ex, df = load_data(a.years, symbol=sym, min_bars=min_bars, timeframe=tf, fallback_years=(3, 2, 1), keep_volume=True)
            nb, ns = signal_counts(df)
            t = compare(df, tf)
            tabs[tf].append(t)
            out += [f"=== {sym.split('/')[0]} {tf}: {len(df)}봉 {df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d} ({ex}) 매수신호 {nb}·매도신호 {ns} ===", fmt(t), ""]
        out += [f"=== {tf} 코인 {len(tabs[tf])}개 평균 ===", fmt(sum(tabs[tf]) / len(tabs[tf])), ""]
    out += [f"규칙은 Pine 스크립트 그대로(EMA {FAST}/{SLOW}, 거래량 {VOL_LEN}봉 평균의 {VOL_MULT}배). 체결당 비용 {COST * 100:.2f}%, 펀딩비 미반영, 다음 봉 시가 체결.",
            "'롱+숏'은 선물, '롱만'은 현물처럼 매도 신호에 청산. 과거 데이터이며 수익을 보장하지 않습니다."]
    text = "\n".join(out)
    print(text)
    os.makedirs("data", exist_ok=True)
    open("data/ema_volume.txt", "w").write(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
