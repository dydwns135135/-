"""TradingView 지표 'BB & RSI Reversal'(사용자 제공)을 그대로 백테스트한다.
python -m crypto.bb_rsi [--years 3]   (Bitget 공개 시세만 사용, API 키·주문 없음)

진입(Pine 스크립트와 동일, 값 변경 없음): 종가가 볼린저 하단(20봉, 2σ) 아래 + RSI(14) < 30  → 신호 봉 마감 후 다음 봉 시가에 롱 진입.
스크립트에는 청산 규칙이 없어서 결과를 보기 전에 아래 두 가지를 미리 정해 둘 다 돌린다:
- A. 중심선 청산: 종가가 볼린저 중심선(20봉 평균)을 넘으면 다음 봉 시가에 청산. 손절 없음(바닥 잡기의 기본형), 48봉이 지나면 강제 청산
- B. 손절·목표: 손절 1.5×ATR(14)=1R, 목표1 +1.5R(절반), 목표2 +3R, 48봉 후 정리 (futures_signal 과 같은 방식)
롱만 본다(스크립트에 매도 신호가 없음). 비용은 왕복 0.22%(체결당 수수료 0.06% + 슬리피지 0.05%), 펀딩비 미반영. 포지션은 코인당 하나."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import futures_signal as fs
from .upbit_scalp import rsi

BB_LEN, BB_MULT, RSI_LEN, RSI_MAX = 20, 2.0, 14, 30.0
TIMEOUT = 48
RT_COST = fs.RT_COST
SYMBOLS = fs.SYMBOLS
BARS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365}


def buy_condition(df: pd.DataFrame) -> pd.Series:
    mid = df["close"].rolling(BB_LEN).mean()
    lower = mid - BB_MULT * df["close"].rolling(BB_LEN).std(ddof=0)
    cond = (df["close"] < lower) & (rsi(df["close"], RSI_LEN) < RSI_MAX)
    cond &= pd.Series(np.arange(len(df)) >= max(BB_LEN, RSI_LEN * 3), index=df.index)
    return cond.fillna(False)


def backtest_mid_exit(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """A: 중심선 청산(손절 없음, 48봉 강제 청산). 수익률은 비용 반영 후."""
    mid = df["close"].rolling(BB_LEN).mean().to_numpy()
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    rows, free_at = [], 0
    for i in np.flatnonzero(buy_condition(df).to_numpy()):
        if i < free_at or i + 2 >= len(df):
            continue
        entry, j = o[i + 1], i + 1
        while True:
            if c[j] > mid[j] or j - i >= TIMEOUT or j + 1 >= len(df) - 1:
                break
            j += 1
        exit_px = o[min(j + 1, len(df) - 1)]
        ret = exit_px / entry - 1 - RT_COST
        free_at = j + 1
        rows.append({"symbol": symbol, "time": df.index[i], "exit": df.index[min(j + 1, len(df) - 1)],
                     "ret": ret, "held": j + 1 - (i + 1), "timeout": bool(j - i >= TIMEOUT)})
    return pd.DataFrame(rows)


def backtest_r(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """B: 손절 1.5×ATR, 목표 1.5R/3R."""
    d = fs.prepare(df.assign(volume=df["volume"] if "volume" in df else 1.0))
    sig = buy_condition(d)
    opens, atrs = d["open"].to_numpy(), d["atr"].to_numpy()
    rows, free_at = [], 0
    for i in np.flatnonzero((sig & d["atr"].notna()).to_numpy()):
        if i < free_at or i + 1 >= len(d):
            continue
        entry, r = opens[i + 1], fs.STOP_ATR * atrs[i]
        res = fs.resolve(1, entry, r, d.iloc[i + 1: i + 1 + fs.TIMEOUT])
        if res["status"] == "진행중":
            continue
        free_at = i + 1 + res["held"]
        rows.append({"symbol": symbol, "time": d.index[i], "exit": d.index[min(i + res["held"], len(d) - 1)], "side": 1,
                     "status": res["status"], "gross_r": res["r"], "net_r": fs.net_r(res["r"], r / entry), "held": res["held"]})
    return pd.DataFrame(rows)


def stats_mid(tr: pd.DataFrame, years: float) -> dict:
    if tr.empty:
        return {"신호수": 0, "승률": float("nan"), "평균": float("nan"), "중앙값": float("nan"), "최악": float("nan"),
                "평균보유(봉)": float("nan"), "연환산(복리)": float("nan"), "최대낙폭": 0.0}
    t = tr.sort_values("exit")
    eq = (1 + t["ret"]).cumprod()
    final = float(eq.iloc[-1])
    return {"신호수": len(t), "승률": float((t["ret"] > 0).mean()), "평균": float(t["ret"].mean()), "중앙값": float(t["ret"].median()),
            "최악": float(t["ret"].min()), "평균보유(봉)": float(t["held"].mean()),
            "연환산(복리)": final ** (1 / max(years, 1e-9)) - 1 if final > 0 else -1.0, "최대낙폭": float((eq / eq.cummax() - 1).min())}


def fmt_mid(rows: dict[str, dict]) -> str:
    t = pd.DataFrame(rows).T
    t["신호수"] = t["신호수"].astype(int)
    for c in ("승률", "연환산(복리)", "최대낙폭"):
        t[c] = t[c].map(lambda x: "-" if x != x else f"{x * 100:+.0f}%" if c != "승률" else f"{x * 100:.0f}%")
    for c in ("평균", "중앙값", "최악"):
        t[c] = t[c].map(lambda x: "-" if x != x else f"{x * 100:+.2f}%")
    t["평균보유(봉)"] = t["평균보유(봉)"].map(lambda x: "-" if x != x else f"{x:.0f}")
    return t.to_string()


def main() -> None:
    from .backtest_report import load_data
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3.0)
    a = ap.parse_args()
    out = []
    for tf, min_bars in (("1h", 3000), ("4h", 1500)):
        mids, rs, info = [], [], []
        mid_rows, r_rows = {}, {}
        for sym in SYMBOLS:
            ex, df = load_data(a.years, symbol=sym, min_bars=min_bars, timeframe=tf, fallback_years=(3, 2, 1), keep_volume=True)
            years = len(df) / BARS_PER_YEAR[tf]
            coin = sym.split("/")[0]
            ta, tb = backtest_mid_exit(df, sym), backtest_r(df, sym)
            mids.append(ta)
            rs.append(tb)
            mid_rows[coin] = stats_mid(ta, years)
            r_rows[coin] = fs.summarize(tb, years * 365)
            info.append(f"{coin} {len(df)}봉 {df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d}")
        ta_all, tb_all = pd.concat(mids, ignore_index=True), pd.concat(rs, ignore_index=True)
        mid_t = ta_all["time"].min() + (ta_all["time"].max() - ta_all["time"].min()) / 2 if len(ta_all) else None
        r_t = tb_all["time"].min() + (tb_all["time"].max() - tb_all["time"].min()) / 2 if len(tb_all) else None
        mid_rows["합산 전체"] = stats_mid(ta_all, float("nan")) | {"연환산(복리)": float("nan")}
        r_rows["합산 전체"] = fs.summarize(tb_all, 365)
        if mid_t is not None:
            mid_rows["합산 앞절반"] = stats_mid(ta_all[ta_all["time"] < mid_t], float("nan")) | {"연환산(복리)": float("nan")}
            mid_rows["합산 뒤절반"] = stats_mid(ta_all[ta_all["time"] >= mid_t], float("nan")) | {"연환산(복리)": float("nan")}
            r_rows["합산 앞절반"] = fs.summarize(tb_all[tb_all["time"] < r_t], 365)
            r_rows["합산 뒤절반"] = fs.summarize(tb_all[tb_all["time"] >= r_t], 365)
        out += [f"=== {tf} · 데이터: " + " | ".join(info) + " ===", "",
                f"[A. 중심선 청산 · 손절 없음 · 비용 반영 후 거래당 수익률]", fmt_mid(mid_rows), "",
                f"[B. 손절 1.5×ATR / 목표 1.5R·3R · 비용 반영 후 R]", fs.fmt_summary(r_rows), ""]
    out += [f"진입은 Pine 스크립트 그대로(BB {BB_LEN}/{BB_MULT}σ, RSI {RSI_LEN} < {RSI_MAX:.0f}). 청산 규칙은 스크립트에 없어 결과를 보기 전에 정한 2가지입니다.",
            "롱만, 코인당 포지션 1개, 왕복 비용 0.22%, 펀딩비 미반영. '합산'의 연환산은 의미가 없어 비웠습니다. 과거 데이터이며 수익을 보장하지 않습니다."]
    text = "\n".join(out)
    print(text)
    os.makedirs("data", exist_ok=True)
    open("data/bb_rsi.txt", "w").write(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
