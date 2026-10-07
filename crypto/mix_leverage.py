"""섞기(앙상블 × 일목) 전략을 롱만 / 롱·숏, 레버리지 1·3·5·10배로 BTC 일봉에서 시뮬레이션한다(청산 포함).
python -m crypto.mix_leverage [--years 7]   (공개 시세만 사용, API 키·주문 없음)

규칙(결과를 보기 전에 정함):
- 롱만:  종가가 일목 구름 위면 앙상블 비중 w(0~1), 아니면 0  (실거래·데모 봇의 mix 모드와 같음)
- 롱·숏: 구름 위면 +w(롱), 구름 아래(구름 하단보다 낮음)면 −(1−w)(숏, 하락 기간이 많을수록 크게), 구름 안이면 0
- 목표 비중이 25%p 이상 바뀌거나 방향이 바뀔 때만 조정. 신호는 종가, 체결은 다음 날 시가.
모델(단순화): 교차 마진, 지갑 전체가 담보. 포지션 금액 = 레버리지 × 잔고 × |비중|.
- 장중 저가(롱)·고가(숏)에서 잔고가 유지증거금(0.5%) 이하가 되면 청산 → 잔고 0(파산, 이후 거래 없음)
- 비용: 체결당 수수료 0.06% + 슬리피지 0.05%. 펀딩비 연 6.6%: 롱은 내고 숏은 받는다(실측 평균 가정)."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .asset_mix import per8h
from .strategies import momentum_ensemble

FEE, SLIP, MMR = 0.0006, 0.0005, 0.005
FUND_DAILY = per8h(6.6) * 3
WARM, STEP = 250, 0.25
LEVERAGES = (1, 3, 5, 10)


def cloud_bounds(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(구름 상단, 구름 하단). 선행스팬은 26봉 앞에 그리므로 그날 값은 과거 데이터로만 정해진다."""
    h, l = df["high"], df["low"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    a = ((tenkan + kijun) / 2).shift(26)
    b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    both = pd.concat([a, b], axis=1)
    return both.max(axis=1, skipna=False), both.min(axis=1, skipna=False)


def next_cloud_bounds(df: pd.DataFrame) -> tuple[float, float]:
    """다음 봉(진행 중인 봉)의 (구름 상단, 구름 하단). 선행스팬은 26봉 앞에 그리므로 이미 정해져 있다."""
    h, l = df["high"], df["low"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    a = float(((tenkan + kijun) / 2).shift(25).iloc[-1])
    b = float(((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(25).iloc[-1])
    return max(a, b), min(a, b)


def stepped_signed(w: pd.Series, step: float = STEP) -> pd.Series:
    """방향이 바뀌거나 0이 되거나, 기준과 step 이상 차이 날 때만 기준 비중을 갱신."""
    ref, out = 0.0, []
    for x in w.to_numpy(float):
        if (x == 0 and ref != 0) or (x * ref < 0) or abs(x - ref) >= step - 1e-12:
            ref = float(x)
        out.append(ref)
    return pd.Series(out, index=w.index)


def mix_weights(df: pd.DataFrame, long_short: bool) -> pd.Series:
    ens = momentum_ensemble(df)
    top, bottom = cloud_bounds(df)
    above = (df["close"] > top).fillna(False)
    below = (df["close"] < bottom).fillna(False)
    w = ens.where(above, 0.0)
    if long_short:
        w = w.where(~below, -(1 - ens))
    w.iloc[:WARM] = 0.0
    return stepped_signed(w)


def simulate(df: pd.DataFrame, w: pd.Series, lev: float, start: int = WARM) -> dict:
    """일별 잔고(시작 1.0), 거래 수, 파산일. 손익은 시가·장중·종가 순으로 평가(값 = 잔고)."""
    O, H, L, C = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    wt = w.to_numpy(float)
    n = len(df)
    E, q, mark, cur_w = 1.0, 0.0, O[start] if start < n else 0.0, 0.0
    eq = np.full(n, np.nan)
    trades, ruin = 0, None
    for i in range(start, n):
        if ruin is not None:
            eq[i] = 0.0
            continue
        E += q * (O[i] - mark)
        mark = O[i]
        if i > start and wt[i - 1] != cur_w:   # 목표 비중이 바뀐 날만 조정(그 외에는 수량 유지 = 봇과 같음)
            target = lev * E / O[i] * wt[i - 1] if E > 0 else 0.0
            E -= abs(target - q) * O[i] * (FEE + SLIP)
            q, cur_w, trades = target, wt[i - 1], trades + 1
        worst = L[i] if q > 0 else H[i]
        if q != 0 and E + q * (worst - mark) <= MMR * abs(q) * worst:
            E, q, ruin = 0.0, 0.0, df.index[i]
            eq[i] = 0.0
            continue
        E += q * (C[i] - mark) - q * C[i] * FUND_DAILY
        mark = C[i]
        eq[i] = E
    return {"equity": pd.Series(eq, index=df.index).iloc[start:], "trades": trades, "ruin": ruin}


def metrics(res: dict) -> dict:
    s = res["equity"]
    years = max(len(s) / 365, 1e-9)
    final = float(s.iloc[-1])
    r = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return {"최종잔고(배)": final, "연환산": final ** (1 / years) - 1 if final > 0 else -1.0,
            "최대낙폭": float((s / s.cummax() - 1).min()),
            "샤프": float(r.mean() / r.std() * np.sqrt(365)) if len(r) > 1 and r.std() > 0 else float("nan"),
            "연거래": res["trades"] / years, "파산": f"{res['ruin']:%Y-%m-%d}" if res["ruin"] is not None else "-"}


def compare(df: pd.DataFrame, from_date: str | None = None) -> pd.DataFrame:
    start = WARM if not from_date else max(WARM, int(df.index.searchsorted(pd.Timestamp(from_date, tz="UTC"))))
    rows = []
    for ls in (False, True):
        w = mix_weights(df, ls)
        for lev in LEVERAGES:
            rows.append({"방향": "롱·숏" if ls else "롱만", "레버리지": f"{lev}배", **metrics(simulate(df, w, lev, start))})
    return pd.DataFrame(rows)


def fmt(t: pd.DataFrame) -> str:
    x = t.copy()
    x["최종잔고(배)"] = x["최종잔고(배)"].map(lambda v: f"{v:.2f}")
    for c in ("연환산", "최대낙폭"):
        x[c] = x[c].map(lambda v: f"{v * 100:+.0f}%")
    x["샤프"] = x["샤프"].map(lambda v: "-" if v != v else f"{v:.2f}")
    x["연거래"] = x["연거래"].map(lambda v: f"{v:.1f}")
    return x.to_string(index=False)


def main() -> None:
    from .backtest_report import load_data
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=float, default=7.0)
    a = ap.parse_args()
    ex_id, df = load_data(a.years, "BTC/USDT:USDT", min_bars=800, timeframe="1d")
    lines = [f"데이터: {ex_id} BTC/USDT 일봉 {len(df)}봉 {df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d}"]
    for title, frm in (("전체 기간", None), ("최근 약 4년(2022-01-01~)", "2022-01-01")):
        st0 = WARM if not frm else max(WARM, int(df.index.searchsorted(pd.Timestamp(frm, tz="UTC"))))
        lines += ["", f"=== {title}: {df.index[st0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d} ===", fmt(compare(df, frm))]
    lines += ["", "읽는 법: 최종잔고 1.00 = 시작 금액 그대로, 0.00 = 전부 잃음. '파산'에 날짜가 있으면 그날 청산된 것.",
              "교차 마진·유지증거금 0.5%·펀딩 연 6.6%(롱 지불/숏 수취)로 단순화. 과거 데이터이며 수익을 보장하지 않습니다."]
    text = "\n".join(lines)
    print(text)
    os.makedirs("data", exist_ok=True)
    with open("data/mix_leverage.txt", "w") as f:
        f.write(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
