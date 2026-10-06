"""B 검증: 90일 모멘텀(롱만)이 우연히 잘 나온 값인지, 이웃 값(30~250일)에서도 비슷한지 확인한다.
최고 값을 고르려는 것이 아니라 '규칙이 한 점에서만 좋은지, 넓은 구간에서 안정적인지'를 보는 용도.
python -m crypto.momentum_sensitivity [--years 7]   (공개 시세만 사용)"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from . import strategies as st
from .backtest_report import load_data
from .strategy_compare import metrics

LOOKBACKS = (30, 45, 60, 90, 120, 150, 180, 250)
ENSEMBLE = "앙상블(8개 평균)"


def sensitivity(df: pd.DataFrame, lookbacks=LOOKBACKS) -> pd.DataFrame:
    pnls = {"보유(1배)": st.simulate(df, st.buy_hold(df))}
    for lb in lookbacks:
        pnls[f"{lb}일 모멘텀"] = st.simulate(df, st.momentum(df, lookback=lb))
    pnls[ENSEMBLE] = st.simulate(df, st.momentum_ensemble(df, tuple(lookbacks)))
    n = len(next(iter(pnls.values())))
    cut = int(n * 0.6)
    segs = {"전체": slice(0, n), "앞60%": slice(0, cut), "뒤40%": slice(cut, n)}
    rows = []
    for name, pnl in pnls.items():
        for seg, sl in segs.items():
            rows.append({"전략": name, "구간": seg, **metrics(pnl.iloc[sl])})
    return pd.DataFrame(rows)


def summarize(rep: pd.DataFrame) -> list[str]:
    bh = rep[rep["전략"] == "보유(1배)"].set_index("구간")
    lines, good = [], 0
    names = [n for n in rep["전략"].unique() if n not in ("보유(1배)", ENSEMBLE)]
    for n in names:
        g = rep[rep["전략"] == n].set_index("구간")
        pos_both = g.loc["앞60%", "샤프"] > 0 and g.loc["뒤40%", "샤프"] > 0
        beat_bh_full = g.loc["전체", "샤프"] > bh.loc["전체", "샤프"]
        smaller_dd = g.loc["전체", "MDD"] > bh.loc["전체", "MDD"]  # MDD 는 음수: 클수록 낙폭이 작음
        good += int(pos_both and beat_bh_full)
        lines.append(f"{n}: 두 구간 샤프>0 {'○' if pos_both else '×'} / 전체 샤프 보유보다 높음 {'○' if beat_bh_full else '×'} / "
                     f"낙폭 보유보다 작음 {'○' if smaller_dd else '×'}")
    lines.append(f"→ {len(names)}개 중 {good}개가 '두 구간 모두 양수 + 보유보다 높은 샤프'")
    if (rep["전략"] == ENSEMBLE).any():
        e = rep[rep["전략"] == ENSEMBLE].set_index("구간")
        ind = rep[rep["전략"].isin(names)]
        med = {seg: float(ind[ind["구간"] == seg]["샤프"].median()) for seg in ("전체", "앞60%", "뒤40%")}
        worst_dd = float(ind[ind["구간"] == "전체"]["MDD"].min())
        lines += [
            "",
            f"[앙상블] 샤프 전체 {e.loc['전체', '샤프']:.2f} / 앞60% {e.loc['앞60%', '샤프']:.2f} / 뒤40% {e.loc['뒤40%', '샤프']:.2f}"
            f"  (개별 8개 중앙값: {med['전체']:.2f} / {med['앞60%']:.2f} / {med['뒤40%']:.2f}, 보유: {bh.loc['전체', '샤프']:.2f})",
            f"[앙상블] 최대낙폭 {e.loc['전체', 'MDD']:+.1%} (개별 8개 중 최악 {worst_dd:+.1%}, 보유 {bh.loc['전체', 'MDD']:+.1%})",
        ]
    return lines


def fmt(rep: pd.DataFrame) -> str:
    d = rep.copy()
    for c in ("수익률", "연환산", "MDD"):
        d[c] = d[c].map(lambda v: f"{v:+.1%}")
    d["노출"] = d["노출"].map(lambda v: f"{v:.0%}")
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
    rep = sensitivity(df)
    rep.to_csv(os.path.join(a.out, "momentum_sensitivity.csv"), index=False)
    print(f"데이터: {ex_id} {a.symbol} 1d, {len(df)}봉, {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}")
    print(fmt(rep))
    print("\n".join(["", *summarize(rep)]))
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a") as f:
            f.write("### 모멘텀 이웃 값 검증\n\n```\n" + fmt(rep) + "\n\n" + "\n".join(summarize(rep)) + "\n```\n")


if __name__ == "__main__":
    main()
