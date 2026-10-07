"""레버리지·손절 스트레스 테스트: BTC 앙상블 모멘텀을 레버리지 1·2·3·5·10배 × 손절 폭 조합으로
일봉(시가·고가·저가·종가)에서 시뮬레이션해 손절·청산이 몇 번 일어나고 최종 잔고가 어떻게 되는지 본다.
python -m crypto.leverage_stress   (공개 시세만 사용, API 키·주문 없음)

모델(단순화, 실제 거래소와 다를 수 있음):
- 계좌 하나, 포지션 하나(롱만). 레버리지 L = (목표 포지션 금액) / (잔고) 를 비중 100% 일 때 기준으로 하며,
  목표 금액 = L × 잔고 × 앙상블 비중(0~1). 신호는 종가에 정하고 다음 날 시가에 체결, 목표와 10%p 이상 차이날 때만 조정(봇과 동일).
- 교차 마진처럼 잔고 전체가 담보. 청산가 = (수량×진입가 − 지갑잔고) / (수량×(1−유지증거금률)). 하루 중 저가가 청산가 이하이면 청산 → 잔고 0(파산).
- 손절은 진입(평균)가 × (1−손절폭). 저가가 닿으면 손절가(갭이면 시가)에 체결. 손절가가 청산가보다 낮으면 청산이 먼저.
- 손절·청산 뒤에도 신호가 양(+)이면 다음 날 시가에 다시 진입한다(봇의 실제 동작).
- 비용: 수수료 0.06% + 슬리피지 0.05%(체결당), 펀딩비는 실측 BTC 연 6.6% 를 포지션 금액에 매일 부과."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import strategies as st
from .asset_mix import per8h
from .backtest_report import load_data

LEVERAGES = (1, 2, 3, 5, 10)
STOPS = (None, 0.03, 0.10, 0.20, 0.30)  # None = 손절 없음(백테스트와 동일)
MMR = 0.005          # 유지증거금률(단순화)
FEE, SLIP = 0.0006, 0.0005
FUND_DAILY = per8h(6.6) * 3
WARM = 250
STEP = 0.10


def simulate_leveraged(df: pd.DataFrame, w: pd.Series, lev: float, stop: float | None = None, fee: float = FEE,
                       slip: float = SLIP, fund_daily: float = FUND_DAILY, mmr: float = MMR, step: float = STEP,
                       start: int = 0) -> dict:
    """df: open/high/low/close, w: 종가에 정한 목표 비중(0~1). start 이전 구간은 거래하지 않는다(워밍업)."""
    O, H, L, C = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    wt = w.to_numpy(float)
    n = len(df)
    W, q, entry = 1.0, 0.0, 0.0          # 지갑잔고(정규화 1.0), 수량, 평균 진입가
    eq = np.full(n, np.nan)
    stops = liqs = trades = 0
    ruin_i = None
    for i in range(start, n):
        if ruin_i is not None:
            eq[i] = 0.0
            continue
        # 1) 시가: 전일 종가 신호에 맞춰 조정
        if i > start:
            E = W + q * (O[i] - entry)
            full_q = lev * E / O[i] if E > 0 else 0.0
            target = full_q * wt[i - 1]
            delta = target - q
            if wt[i - 1] <= 0:
                if q > 0:
                    fill = O[i] * (1 - slip)
                    W += q * (fill - entry) - q * fill * fee
                    q, trades = 0.0, trades + 1
            elif abs(delta) >= step * full_q and abs(delta) * O[i] > 1e-9:
                if delta > 0:
                    fill = O[i] * (1 + slip)
                    W -= delta * fill * fee
                    entry = (q * entry + delta * fill) / (q + delta)
                    q += delta
                else:
                    fill = O[i] * (1 - slip)
                    W += -delta * (fill - entry) - (-delta) * fill * fee
                    q += delta
                trades += 1
        # 2) 장중: 손절/청산 판정 (가격이 먼저 닿는 쪽)
        if q > 0:
            stop_px = entry * (1 - stop) if stop else 0.0
            liq_px = (q * entry - W) / (q * (1 - mmr)) if q * entry > W else 0.0
            hit_stop = stop is not None and L[i] <= stop_px
            hit_liq = liq_px > 0 and L[i] <= liq_px
            if hit_stop and (not hit_liq or stop_px >= liq_px):
                fill = min(stop_px, O[i]) * (1 - slip)
                W += q * (fill - entry) - q * fill * fee
                q, stops = 0.0, stops + 1
            elif hit_liq:
                W, q, liqs = 0.0, 0.0, liqs + 1
                ruin_i = i
        # 3) 펀딩비(포지션 금액 기준) 후 종가 평가
        if q > 0:
            W -= q * C[i] * fund_daily
        equity = W + q * (C[i] - entry) if q > 0 else W
        if equity <= 0 and q > 0:   # 펀딩비·종가 평가로 잔고가 바닥나도 파산 처리
            equity, W, q, ruin_i = 0.0, 0.0, 0.0, i
        eq[i] = max(equity, 0.0)
    s = pd.Series(eq, index=df.index).iloc[start:]
    return {"equity": s, "stops": stops, "liqs": liqs, "trades": trades, "ruin": df.index[ruin_i] if ruin_i is not None else None}


def stats(res: dict) -> dict:
    s = res["equity"]
    years = max(len(s) / 365, 1e-9)
    final = float(s.iloc[-1])
    cagr = final ** (1 / years) - 1 if final > 0 else -1.0
    mdd = float((s / s.cummax() - 1).min())
    return {"최종잔고(배)": final, "연환산": cagr, "최대낙폭": mdd, "손절횟수": res["stops"], "청산횟수": res["liqs"],
            "조정횟수": res["trades"], "파산": f"{res['ruin']:%Y-%m-%d}" if res["ruin"] is not None else "-"}


def grid(df: pd.DataFrame, start: int, levs=LEVERAGES, stops=STOPS) -> pd.DataFrame:
    w = st.momentum_ensemble(df)
    rows = []
    for lev in levs:
        for sp in stops:
            r = stats(simulate_leveraged(df, w, lev, sp, start=start))
            rows.append({"레버리지": f"{lev}배", "손절폭": "없음" if sp is None else f"{sp:.0%}", **r})
    return pd.DataFrame(rows)


def fmt(t: pd.DataFrame) -> str:
    d = t.copy()
    d["최종잔고(배)"] = d["최종잔고(배)"].map(lambda v: f"{v:,.2f}")
    d["연환산"] = d["연환산"].map(lambda v: f"{v:+.0%}")
    d["최대낙폭"] = d["최대낙폭"].map(lambda v: f"{v:+.0%}")
    return d.to_string(index=False)


def window_start(df: pd.DataFrame, from_date: str | None) -> int:
    if not from_date:
        return WARM
    return max(WARM, int(df.index.searchsorted(pd.Timestamp(from_date, tz="UTC"))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=float, default=7.0)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    ex_id, df = load_data(a.years, "BTC/USDT:USDT", min_bars=800, timeframe="1d")
    print(f"데이터: {ex_id} BTC/USDT 일봉 {len(df)}봉 {df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d}")
    texts = []
    for title, frm in (("전체 기간", None), ("최근 약 4년(2022-01-01~)", "2022-01-01")):
        st0 = window_start(df, frm)
        t = grid(df, st0)
        idx = df.index[st0]
        texts.append(f"=== {title}: {idx:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d} ({len(df) - st0}일) ===\n{fmt(t)}")
        os.makedirs(a.out, exist_ok=True)
        t.to_csv(os.path.join(a.out, f"leverage_stress_{'all' if frm is None else 'recent'}.csv"), index=False)
    text = "\n\n".join(texts) + ("\n\n읽는 법: 최종잔고 1.00 = 시작 금액 그대로. 파산 날짜가 있으면 그날 잔고가 0이 된 것(청산).\n"
                                "손절·청산 뒤에도 신호가 양(+)이면 다음 날 다시 진입합니다. 거래소 규칙·갭·점검 등은 단순화했습니다.")
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("### 레버리지·손절 스트레스 테스트\n\n```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
