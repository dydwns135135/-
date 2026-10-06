"""대표적인 매매 규칙들을 같은 조건(일봉, 수수료·슬리피지·펀딩비 반영, 다음 봉 시가 체결)에서 비교한다.
python -m crypto.strategy_compare [--years 7]    (공개 시세만 사용)

주의: 규칙을 여러 개 비교하면 '가장 좋아 보이는 것'은 우연일 수 있다(다중 비교 편향).
그래서 변수는 튜닝하지 않았고, 전체뿐 아니라 앞 60% / 뒤 40% 두 구간 모두에서 통하는지 본다."""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

from . import strategies as st
from .backtest_report import load_data


def build(df: pd.DataFrame) -> dict[str, pd.Series]:
    vt = st.vol_target_size(df)
    sma = st.sma_cross(df)
    return {
        "보유(1배)": st.simulate(df, st.buy_hold(df)),
        "SMA50/200 롱만": st.simulate(df, sma),
        "SMA50/200 롱+숏": st.simulate(df, st.sma_cross(df, short=True)),
        "돈치안20/10 롱만": st.simulate(df, st.donchian(df)),
        "돈치안20/10 롱+숏": st.simulate(df, st.donchian(df, short=True)),
        "90일모멘텀 롱만": st.simulate(df, st.momentum(df)),
        "90일모멘텀 롱+숏": st.simulate(df, st.momentum(df, short=True)),
        "RSI평균회귀 롱만": st.simulate(df, st.rsi_revert(df)),
        "보유+변동성타깃": st.simulate(df, st.buy_hold(df), vt),
        "SMA롱만+변동성타깃": st.simulate(df, sma, vt),
    }


def metrics(pnl: pd.Series, bars_per_year: float = 365) -> dict:
    eq = (1 + pnl).cumprod()
    years = max(len(pnl) / bars_per_year, 1e-9)
    total = float(eq.iloc[-1] - 1)
    sd = pnl.std()
    return {
        "수익률": total,
        "연환산": (1 + total) ** (1 / years) - 1 if total > -1 else -1.0,
        "MDD": float((eq / eq.cummax() - 1).min()),
        "샤프": float(pnl.mean() / sd * math.sqrt(bars_per_year)) if sd and sd > 0 else float("nan"),
        "노출": float((pnl != 0).mean()),
    }


def compare(df: pd.DataFrame) -> pd.DataFrame:
    pnls = build(df)  # 전체 구간에서 한 번 계산(신호는 과거만 사용) 후 구간별로 자른다
    n = len(next(iter(pnls.values())))
    cut = int(n * 0.6)
    segs = {"전체": slice(0, n), "앞60%": slice(0, cut), "뒤40%": slice(cut, n)}
    rows = []
    for name, pnl in pnls.items():
        for seg, sl in segs.items():
            part = pnl.iloc[sl]
            rows.append({"전략": name, "구간": seg, **metrics(part)})
    return pd.DataFrame(rows)


def verdict(rep: pd.DataFrame) -> list[str]:
    """두 구간 모두에서 샤프가 양수이고 단순보유보다 MDD 가 작은 규칙을 표시."""
    out = []
    for name, g in rep.groupby("전략", sort=False):
        a, b = g[g["구간"] == "앞60%"].iloc[0], g[g["구간"] == "뒤40%"].iloc[0]
        ok = (a["샤프"] > 0) and (b["샤프"] > 0)
        out.append(f"{'○' if ok else '×'} {name}: 앞60% 샤프 {a['샤프']:.2f} / 뒤40% 샤프 {b['샤프']:.2f}")
    return out


def fmt(rep: pd.DataFrame) -> str:
    d = rep.copy()
    for c in ("수익률", "연환산", "MDD", "노출"):
        d[c] = d[c].map(lambda v: f"{v:+.1%}" if c != "노출" else f"{v:.0%}")
    d["샤프"] = d["샤프"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
    return d.to_string(index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=float, default=7.0)
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    ex_id, df = load_data(a.years, a.symbol, min_bars=800, timeframe="1d")
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "btc_1d.csv"), index_label="date")
    print(f"데이터: {ex_id} {a.symbol} 1d, {len(df)}봉, {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}")
    rep = compare(df)
    rep.to_csv(os.path.join(a.out, "strategy_compare.csv"), index=False)
    print(fmt(rep))
    print("\n두 구간(앞60%·뒤40%) 모두 샤프>0 인가:")
    print("\n".join(verdict(rep)))
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a") as f:
            f.write(f"### BTC 일봉 전략 비교 ({ex_id}, {len(df)}봉)\n\n```\n{fmt(rep)}\n\n" + "\n".join(verdict(rep)) + "\n```\n")


if __name__ == "__main__":
    main()
