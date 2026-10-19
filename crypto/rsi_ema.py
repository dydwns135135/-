"""TradingView 전략 'RSI반전+EMA200필터'(사용자 제공)를 그대로 백테스트한다.
python -m crypto.rsi_ema [--years 3]   (Bitget 공개 시세만 사용, API 키·주문 없음)

규칙(Pine 스크립트와 동일, 값 변경 없음):
- 롱: RSI(14)가 30을 위로 돌파 + 종가 > EMA200 / 숏: RSI가 70을 아래로 돌파 + 종가 < EMA200 (포지션 없을 때만)
- 손절 = 신호 봉 종가 ∓ 2×ATR(14), 익절 = 신호 봉 종가 ± 3×ATR(14)
- 롱은 RSI ≥ 50, 숏은 RSI ≤ 50 이 되면 청산
체결 가정(Pine 기본과 같게): 신호 봉 마감 후 다음 봉 시가 진입, RSI 청산도 다음 봉 시가.
손절·익절이 같은 봉에서 모두 닿으면 손절이 먼저(보수적). 시가가 이미 손절·익절을 넘었으면 시가 체결.
비용: 체결당 수수료 0.06% + 슬리피지 0.05%(왕복 0.22%). 레버리지 1배·포지션 1개·전액 기준. 펀딩비 미반영."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .upbit_scalp import rsi

RSI_LEN, OS, OB, EMA_LEN, ATR_LEN = 14, 30.0, 70.0, 200, 14
SL_MULT, TP_MULT = 2.0, 3.0
COST = 0.0011                       # 체결당
SYMBOLS = ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT")
BARS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365}


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rsi"] = rsi(d["close"], RSI_LEN)
    d["ema"] = d["close"].ewm(span=EMA_LEN, adjust=False).mean()
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / ATR_LEN, adjust=False, min_periods=ATR_LEN).mean()   # Pine ta.atr = RMA
    return d


def signals(d: pd.DataFrame) -> pd.Series:
    r, rp = d["rsi"], d["rsi"].shift(1)
    long_ = (rp <= OS) & (r > OS) & (d["close"] > d["ema"])
    short = (rp >= OB) & (r < OB) & (d["close"] < d["ema"])
    out = pd.Series(0, index=d.index)
    out[long_.fillna(False)] = 1
    out[short.fillna(False)] = -1
    out.iloc[:EMA_LEN] = 0
    return out


def backtest(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    d = indicators(df)
    sig = signals(d).to_numpy()
    O, H, L, C = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    R, A = d["rsi"].to_numpy(float), d["atr"].to_numpy(float)
    n, i, rows = len(d), 0, []
    while i < n - 1:
        s = int(sig[i])
        if s == 0 or np.isnan(A[i]):
            i += 1
            continue
        entry = O[i + 1]
        sl, tp = C[i] - s * SL_MULT * A[i], C[i] + s * TP_MULT * A[i]
        j, exit_px, why = i + 1, None, None
        while j < n:
            if s > 0:
                if L[j] <= sl:
                    exit_px, why = min(O[j], sl), "손절"
                elif H[j] >= tp:
                    exit_px, why = max(O[j], tp), "익절"
            else:
                if H[j] >= sl:
                    exit_px, why = max(O[j], sl), "손절"
                elif L[j] <= tp:
                    exit_px, why = min(O[j], tp), "익절"
            if exit_px is not None:
                break
            if (s > 0 and R[j] >= 50) or (s < 0 and R[j] <= 50):
                if j + 1 < n:
                    exit_px, why, j = O[j + 1], "RSI50 청산", j + 1
                break
            j += 1
        if exit_px is None:                     # 데이터 끝까지 열린 포지션은 제외
            break
        ret = s * (exit_px / entry - 1) - 2 * COST
        rows.append({"symbol": symbol, "time": d.index[i], "exit": d.index[j], "side": s, "why": why,
                     "ret": ret, "held": j - i})
        i = j            # 청산된 봉의 마감부터 다시 신호 확인(Pine: 그 봉 마감에 포지션 0이면 진입 조건 평가)
    return pd.DataFrame(rows)


def stats(tr: pd.DataFrame, years: float | None = None) -> dict:
    if tr.empty:
        return {"거래수": 0, "승률": float("nan"), "평균": float("nan"), "최악": float("nan"),
                "연환산": float("nan"), "최대낙폭": 0.0, "손절/익절/RSI": "-"}
    t = tr.sort_values("exit")
    eq = (1 + t["ret"]).cumprod()
    final = float(eq.iloc[-1])
    cnt = t["why"].value_counts()
    return {"거래수": len(t), "승률": float((t["ret"] > 0).mean()), "평균": float(t["ret"].mean()),
            "최악": float(t["ret"].min()),
            "연환산": (final ** (1 / years) - 1 if final > 0 else -1.0) if years else float("nan"),
            "최대낙폭": float((eq / eq.cummax() - 1).min()),
            "손절/익절/RSI": f"{cnt.get('손절', 0)}/{cnt.get('익절', 0)}/{cnt.get('RSI50 청산', 0)}"}


def fmt(rows: dict[str, dict]) -> str:
    t = pd.DataFrame(rows).T
    t["거래수"] = t["거래수"].astype(int)
    t["승률"] = t["승률"].map(lambda x: "-" if x != x else f"{x * 100:.0f}%")
    for c in ("평균", "최악"):
        t[c] = t[c].map(lambda x: "-" if x != x else f"{x * 100:+.2f}%")
    for c in ("연환산", "최대낙폭"):
        t[c] = t[c].map(lambda x: "-" if x != x else f"{x * 100:+.0f}%")
    return t.to_string()


def main() -> None:
    from .backtest_report import load_data
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=float, default=3.0)
    a = ap.parse_args()
    out = []
    for tf, min_bars in (("1h", 3000), ("4h", 1500)):
        rows, all_tr, info = {}, [], []
        for sym in SYMBOLS:
            ex, df = load_data(a.years, symbol=sym, min_bars=min_bars, timeframe=tf, fallback_years=(3, 2, 1))
            tr = backtest(df, sym)
            all_tr.append(tr)
            coin = sym.split("/")[0]
            rows[coin] = stats(tr, len(df) / BARS_PER_YEAR[tf])
            info.append(f"{coin} {len(df)}봉 {df.index[0]:%Y-%m-%d}~")
        tr = pd.concat(all_tr, ignore_index=True)
        if len(tr):
            mid = tr["time"].min() + (tr["time"].max() - tr["time"].min()) / 2
            rows["합산 전체"] = stats(tr)
            rows["합산 앞절반"] = stats(tr[tr["time"] < mid])
            rows["합산 뒤절반"] = stats(tr[tr["time"] >= mid])
            rows["합산 롱만"] = stats(tr[tr["side"] > 0])
            rows["합산 숏만"] = stats(tr[tr["side"] < 0])
        out += [f"=== {tf} · " + " | ".join(info) + " ===", fmt(rows), ""]
    out += ["규칙은 Pine 스크립트 그대로(RSI 14, 30/70 돌파, EMA200 필터, 손절 2×ATR, 익절 3×ATR, RSI 50 청산).",
            "1배·전액·포지션 1개 기준 거래당 수익률(비용 왕복 0.22% 반영). 레버리지 N배면 수익·손실·낙폭이 대략 N배.",
            "'합산'의 연환산은 의미가 없어 비웠습니다. 펀딩비 미반영. 과거 데이터이며 수익을 보장하지 않습니다."]
    text = "\n".join(out)
    print(text)
    os.makedirs("data", exist_ok=True)
    with open("data/rsi_ema.txt", "w") as f:
        f.write(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
