"""분산 검증: BTC 하나 vs 여러 코인에 같은 앙상블 모멘텀을 적용해 자본을 등분한 포트폴리오.
python -m crypto.portfolio_compare [--years 7]   (공개 시세만 사용)

코인은 미리 정한 BTC·ETH·SOL·XRP 네 개(결과를 보고 바꾸지 않는다). 네 코인이 모두 거래된 기간만 쓰고,
앙상블에 필요한 초기 250일은 모든 전략에서 똑같이 제외해 같은 날짜 구간으로 비교한다.
포트폴리오 손익 = 코인별 손익의 평균(= 매일 등분 비중으로 재조정, 재조정 비용은 반영하지 않음)."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import strategies as st
from .backtest_report import load_data
from .strategy_compare import metrics

COINS = ("BTC", "ETH", "SOL", "XRP")
WARM = 250  # 앙상블 최장 기간. 이 기간은 모든 전략에서 제외


def align(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """모든 코인이 존재하는 날짜만 남긴다."""
    idx = None
    for df in dfs.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    return {k: v.loc[idx] for k, v in dfs.items()}


def build(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """전략별 일 손익률(워밍업 제외)."""
    ens = {k: st.simulate(d, st.momentum_ensemble(d)) for k, d in dfs.items()}
    hold = {k: st.simulate(d, st.buy_hold(d)) for k, d in dfs.items()}
    n = len(dfs)
    out: dict[str, pd.Series] = {}
    first = next(iter(dfs))
    out[f"{first} 보유"] = hold[first]
    out[f"{first} 앙상블"] = ens[first]
    for k in dfs:
        if k != first:
            out[f"{k} 앙상블"] = ens[k]
    out[f"{n}코인 보유(등분)"] = pd.concat(hold.values(), axis=1).mean(axis=1)
    out[f"{n}코인 앙상블(등분)"] = pd.concat(ens.values(), axis=1).mean(axis=1)
    return {k: v.iloc[WARM:] for k, v in out.items()}


def compare(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pnls = build(dfs)
    n = len(next(iter(pnls.values())))
    cut = int(n * 0.6)
    segs = {"전체": slice(0, n), "앞60%": slice(0, cut), "뒤40%": slice(cut, n)}
    rows = []
    for name, pnl in pnls.items():
        for seg, sl in segs.items():
            rows.append({"전략": name, "구간": seg, **metrics(pnl.iloc[sl])})
    return pd.DataFrame(rows)


def correlations(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({k: d["close"].pct_change() for k, d in dfs.items()}).iloc[WARM:].corr()


def summary(rep: pd.DataFrame, corr: pd.DataFrame, coins: tuple[str, ...]) -> list[str]:
    n = len(coins)
    port, base = f"{n}코인 앙상블(등분)", f"{coins[0]} 앙상블"
    p, b = rep[rep["전략"] == port].set_index("구간"), rep[rep["전략"] == base].set_index("구간")
    pair = corr.values[np.triu_indices(len(corr), 1)]
    lines = [f"코인 간 일수익률 평균 상관: {pair.mean():.2f} (1에 가까울수록 같이 움직여 분산 효과가 작음)", ""]
    for seg in ("전체", "앞60%", "뒤40%"):
        lines.append(f"[{seg}] {port} vs {base}: 샤프 {p.loc[seg, '샤프']:.2f} vs {b.loc[seg, '샤프']:.2f} / "
                     f"낙폭 {p.loc[seg, 'MDD']:+.1%} vs {b.loc[seg, 'MDD']:+.1%} / 수익 {p.loc[seg, '수익률']:+.0%} vs {b.loc[seg, '수익률']:+.0%}")
    better = all(p.loc[s, "MDD"] > b.loc[s, "MDD"] for s in ("전체", "앞60%", "뒤40%"))
    sharpe_ok = all(p.loc[s, "샤프"] >= b.loc[s, "샤프"] - 0.05 for s in ("전체", "앞60%", "뒤40%"))
    lines.append(f"→ 세 구간 모두 낙폭 감소: {'○' if better else '×'} / 세 구간 모두 샤프가 BTC 앙상블 대비 -0.05 이내 유지: {'○' if sharpe_ok else '×'}")
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
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    raw, srcs = {}, {}
    for c in COINS:
        ex_id, df = load_data(a.years, f"{c}/USDT:USDT", min_bars=800, timeframe="1d")
        raw[c], srcs[c] = df, ex_id
    dfs = align(raw)
    n = len(next(iter(dfs.values())))
    first = next(iter(dfs.values())).index
    print("데이터: " + ", ".join(f"{c}({srcs[c]})" for c in COINS) + f" | 공통 {n}일 {first[0]:%Y-%m-%d}~{first[-1]:%Y-%m-%d}, 앞 {WARM}일 제외")
    rep, corr = compare(dfs), correlations(dfs)
    os.makedirs(a.out, exist_ok=True)
    rep.to_csv(os.path.join(a.out, "portfolio_compare.csv"), index=False)
    text = fmt(rep) + "\n\n상관행렬:\n" + corr.round(2).to_string() + "\n\n" + "\n".join(summary(rep, corr, COINS))
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("### 코인 분산 검증\n\n```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
