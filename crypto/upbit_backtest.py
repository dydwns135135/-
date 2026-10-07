"""업비트 BTC/원화 일봉으로 앙상블 모멘텀(현물, 1배, 롱만)을 백테스트한다.
python -m crypto.upbit_backtest   (공개 시세만 사용, API 키·주문 없음)

조건: 업비트 원화마켓 수수료 0.05%(체결당) + 슬리피지 0.05%, 현물이라 펀딩비·청산 없음, 앙상블 비중 100% = 투입금 전액.
같은 모델로 '조정 임계치(몇 %p 이상 차이날 때만 조정)'별 조정 횟수도 비교해, 직접 주문하는 부담을 가늠한다."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import strategies as st
from .asset_mix import check_daily
from .leverage_stress import simulate_leveraged
from .upbit_data import closed_only, fetch_upbit_daily

FEE, SLIP = 0.0005, 0.0005
WARM = 250
STEPS = (0.10, 0.25, 0.50)


def run(df: pd.DataFrame, w: pd.Series, start: int, step: float = 0.10) -> dict:
    return simulate_leveraged(df, w, 1, None, fee=FEE, slip=SLIP, fund_daily=0.0, step=step, start=start)


def metrics(res: dict) -> dict:
    s = res["equity"]
    r = s.pct_change().dropna()
    years = max(len(s) / 365, 1e-9)
    final = float(s.iloc[-1])
    return {"최종잔고(배)": final, "연환산": final ** (1 / years) - 1 if final > 0 else -1.0,
            "최대낙폭": float((s / s.cummax() - 1).min()),
            "샤프": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else float("nan"),
            "조정횟수": res["trades"], "연평균조정": res["trades"] / years}


def start_for(df: pd.DataFrame, from_date: str | None) -> int:
    return WARM if not from_date else max(WARM, int(df.index.searchsorted(pd.Timestamp(from_date, tz="UTC"))))


def compare(df: pd.DataFrame, from_date: str | None) -> pd.DataFrame:
    s0 = start_for(df, from_date)
    w = st.momentum_ensemble(df)
    hold = pd.Series(1.0, index=df.index)
    rows = [{"전략": "BTC 보유(원화)", **metrics(run(df, hold, s0))}]
    for step in STEPS:
        rows.append({"전략": f"앙상블(조정 임계 {step:.0%}p)", **metrics(run(df, w, s0, step))})
    return pd.DataFrame(rows)


def fmt(t: pd.DataFrame) -> str:
    d = t.copy()
    d["최종잔고(배)"] = d["최종잔고(배)"].map(lambda v: f"{v:,.2f}")
    d["연환산"] = d["연환산"].map(lambda v: f"{v:+.0%}")
    d["최대낙폭"] = d["최대낙폭"].map(lambda v: f"{v:+.0%}")
    d["샤프"] = d["샤프"].map(lambda v: f"{v:.2f}")
    d["연평균조정"] = d["연평균조정"].map(lambda v: f"{v:.0f}")
    return d.to_string(index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    df = closed_only(check_daily(fetch_upbit_daily("KRW-BTC")))
    gap = df.index.to_series().diff().dt.total_seconds().median() / 86400
    print(f"데이터: upbit KRW-BTC 일봉 {len(df)}봉 {df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d} (봉 간격 중앙값 {gap:.1f}일), "
          f"종가 {df['close'].iloc[0]:,.0f}원 → {df['close'].iloc[-1]:,.0f}원")
    texts = []
    os.makedirs(a.out, exist_ok=True)
    for title, frm, tag in (("전체 기간", None, "all"), ("최근 약 4년(2022-01-01~)", "2022-01-01", "recent")):
        s0 = start_for(df, frm)
        t = compare(df, frm)
        t.to_csv(os.path.join(a.out, f"upbit_backtest_{tag}.csv"), index=False)
        texts.append(f"=== {title}: {df.index[s0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d} ({len(df) - s0}일) ===\n{fmt(t)}")
    text = "\n\n".join(texts) + ("\n\n읽는 법: 최종잔고 1.00 = 시작 금액 그대로. 조정 임계가 클수록 주문 횟수가 줄지만 목표 비중에서 더 벗어난 채 유지됩니다.\n"
                                "업비트 수수료 0.05% + 슬리피지 0.05% 가정, 펀딩비 없음. 과거 데이터이며 수익을 보장하지 않습니다.")
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("### 업비트 BTC/원화 앙상블 백테스트\n\n```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
