"""TradingView 지표 'ATR Trailing Stop (Chandelier)'(사용자 제공)를 추세 추종 규칙으로 백테스트한다.
python -m crypto.chandelier   (공개 시세만 사용, API 키·주문 없음)

지표 값은 그대로(ATR 22, 배수 3.0): 롱 스탑 = 최근 22봉 최고가 − 3×ATR, 숏 스탑 = 최근 22봉 최저가 + 3×ATR (ATR 은 Pine ta.atr 와 같은 RMA).
스크립트에는 진입 규칙이 없어, 샹들리에 청산 지표의 고전적인 방향 전환 규칙을 결과를 보기 전에 정해 쓴다:
- 방향 = 롱: 종가 > 직전 봉의 숏 스탑  /  숏: 종가 < 직전 봉의 롱 스탑  /  둘 다 아니면 직전 방향 유지
- 신호는 봉 마감 때 확정, 다음 봉 시가 체결(직전 봉 스탑을 써서 미래 정보 없음)
- 롱+숏: 방향을 그대로 따라 포지션 전환(선물)    롱만: 롱일 때만 보유, 숏 방향이면 현금(현물)
대상: 업비트 일봉 BTC·ETH·XRP(현물 롱만, 체결당 0.10%) / Bitget 4시간봉 BTC·ETH·SOL·XRP(선물 롱+숏·롱만, 체결당 0.11%, 펀딩비 미반영)."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .ema_volume import stats
from .upbit_scalp import simulate

ATR_LEN, MULT = 22, 3.0
UPBIT_MARKETS = ("KRW-BTC", "KRW-ETH", "KRW-XRP")
BITGET_SYMBOLS = ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT")
UPBIT_COST, BITGET_COST = 0.0010, 0.0011


def stops(df: pd.DataFrame, atr_len: int = ATR_LEN, mult: float = MULT) -> tuple[pd.Series, pd.Series]:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_len, adjust=False, min_periods=atr_len).mean()   # Pine ta.atr = RMA
    long_stop = df["high"].rolling(atr_len).max() - atr * mult
    short_stop = df["low"].rolling(atr_len).min() + atr * mult
    return long_stop, short_stop


def direction(df: pd.DataFrame, atr_len: int = ATR_LEN, mult: float = MULT) -> pd.Series:
    """+1 롱 / -1 숏 / 0(준비 구간). 직전 봉의 스탑과 이번 종가를 비교."""
    ls, ss = stops(df, atr_len, mult)
    c, lsp, ssp = df["close"].to_numpy(), ls.shift(1).to_numpy(), ss.shift(1).to_numpy()
    out, d = np.zeros(len(c)), 0.0
    for i in range(len(c)):
        if not (np.isnan(lsp[i]) or np.isnan(ssp[i])):
            if c[i] > ssp[i]:
                d = 1.0
            elif c[i] < lsp[i]:
                d = -1.0
        out[i] = d
    return pd.Series(out, index=df.index)


def compare(df: pd.DataFrame, bars_per_year: float, cost: float, long_short: bool) -> pd.DataFrame:
    d = direction(df)
    sigs = {"보유": pd.Series(1.0, index=df.index)}
    if long_short:
        sigs["롱+숏"] = d
    sigs["롱만"] = (d > 0).astype(float)
    half = len(df) // 2
    rows = {}
    for name, sig in sigs.items():
        rows[f"{name} 전체"] = stats(simulate(df, sig, cost), sig, bars_per_year)
        rows[f"{name} 뒤절반"] = stats(simulate(df.iloc[half:], sig.iloc[half:], cost), sig.iloc[half:], bars_per_year)
    return pd.DataFrame(rows).T


def average(tables: list[pd.DataFrame]) -> pd.DataFrame:
    return sum(tables) / len(tables)


def fmt(t: pd.DataFrame) -> str:
    x = t.copy()
    x["연환산"] = x["연환산"].map(lambda v: f"{v * 100:+.0f}%")
    x["최대낙폭"] = x["최대낙폭"].map(lambda v: f"{v * 100:+.0f}%")
    x["샤프"] = x["샤프"].map(lambda v: f"{v:.2f}")
    x["거래횟수"] = x["거래횟수"].map(lambda v: f"{v:.0f}")
    x["평균투입"] = x["평균투입"].map(lambda v: f"{v * 100:.0f}%")
    return x.to_string()


def main() -> None:
    from .backtest_report import load_data
    from .upbit_data import closed_only, fetch_upbit_daily
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3.0)
    a = ap.parse_args()
    out, tabs = [], []
    for m in UPBIT_MARKETS:
        df = closed_only(fetch_upbit_daily(m, total=2600))
        t = compare(df, 365, UPBIT_COST, long_short=False)
        tabs.append(t)
        out += [f"=== 업비트 일봉 {m[4:]}: {len(df)}봉 {df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d} (현물 롱만) ===", fmt(t), ""]
    out += [f"=== 업비트 일봉 코인 {len(tabs)}개 평균 ===", fmt(average(tabs)), ""]
    tabs = []
    for sym in BITGET_SYMBOLS:
        ex, df = load_data(a.years, symbol=sym, min_bars=1500, timeframe="4h", fallback_years=(3, 2, 1))
        t = compare(df, 6 * 365, BITGET_COST, long_short=True)
        tabs.append(t)
        out += [f"=== Bitget 4시간봉 {sym.split('/')[0]}: {len(df)}봉 {df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d} ({ex}) ===", fmt(t), ""]
    out += [f"=== Bitget 4시간봉 코인 {len(tabs)}개 평균 ===", fmt(average(tabs)), ""]
    out += [f"지표 값 그대로(ATR {ATR_LEN}, 배수 {MULT}). 진입은 방향 전환 규칙을 결과를 보기 전에 정해 사용. 다음 봉 시가 체결, 펀딩비 미반영.",
            "'롱+숏'은 선물, '롱만'은 현물처럼 숏 방향에서 현금. 과거 데이터이며 수익을 보장하지 않습니다."]
    text = "\n".join(out)
    print(text)
    os.makedirs("data", exist_ok=True)
    open("data/chandelier.txt", "w").write(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
