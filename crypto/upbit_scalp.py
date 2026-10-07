"""업비트 BTC/원화 단타(1시간봉·15분봉) 전략을 수수료 포함으로 백테스트한다.
python -m crypto.upbit_scalp   (공개 시세만 사용, API 키·주문 없음)

현물·롱만·전액 투입(비중 0 또는 100%). 신호는 봉 마감 때 확정하고 다음 봉 시가에 체결한다.
비용은 체결당 수수료 0.05% + 슬리피지 0.05%. '비용 0' 열은 같은 신호에서 수수료만 뺀 값으로,
단타가 비용 때문에 얼마나 깎이는지 보여 주려는 비교용이다."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .upbit_data import fetch_upbit_minutes

FEE, SLIP = 0.0005, 0.0005
COST = FEE + SLIP
BARS_PER_DAY = {60: 24, 15: 96}
TOTAL = {60: 26000, 15: 35000}


def breakout(df: pd.DataFrame, n: int, m: int) -> pd.Series:
    """종가가 직전 n봉 최고가를 넘으면 매수, 직전 m봉 최저가를 깨면 청산."""
    hi = df["high"].rolling(n).max().shift(1).to_numpy()
    lo = df["low"].rolling(m).min().shift(1).to_numpy()
    c = df["close"].to_numpy()
    out = np.zeros(len(c))
    on = 0.0
    for i in range(len(c)):
        if on == 0.0 and c[i] > hi[i]:
            on = 1.0
        elif on == 1.0 and c[i] < lo[i]:
            on = 0.0
        out[i] = on
    return pd.Series(out, index=df.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def rsi_revert(df: pd.DataFrame, entry: float = 30, exit_: float = 55, period: int = 14) -> pd.Series:
    """RSI가 entry 아래로 과매도면 매수, exit_ 위로 회복하면 청산."""
    r = rsi(df["close"], period).to_numpy()
    out = np.zeros(len(r))
    on = 0.0
    for i in range(len(r)):
        if np.isnan(r[i]):
            out[i] = 0.0
            continue
        if on == 0.0 and r[i] < entry:
            on = 1.0
        elif on == 1.0 and r[i] > exit_:
            on = 0.0
        out[i] = on
    return pd.Series(out, index=df.index)


def ema_cross(df: pd.DataFrame, fast: int = 12, slow: int = 48) -> pd.Series:
    f = df["close"].ewm(span=fast, adjust=False).mean()
    s = df["close"].ewm(span=slow, adjust=False).mean()
    out = (f > s).astype(float)
    out[:slow] = 0.0
    return out


STRATEGIES = {
    "돌파 24/12봉": lambda d: breakout(d, 24, 12),
    "돌파 72/24봉": lambda d: breakout(d, 72, 24),
    "RSI 과매도 반등": lambda d: rsi_revert(d),
    "이평 12/48 교차": lambda d: ema_cross(d),
}


def simulate(df: pd.DataFrame, signal: pd.Series, cost: float = COST) -> pd.Series:
    """신호(봉 t 마감 확정) → 봉 t+1 시가 체결. 포지션은 시가→다음 시가 구간 수익을 먹고, 포지션이 바뀔 때 cost 를 낸다."""
    pos = signal.shift(1).fillna(0.0)
    o = df["open"]
    ret = (o.shift(-1) / o - 1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    return (1 + pos * ret - turn * cost).cumprod()


def daily_returns(eq: pd.Series) -> pd.Series:
    return eq.resample("1D").last().pct_change().dropna()


def stats(eq: pd.Series, signal: pd.Series) -> dict:
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    final = float(eq.iloc[-1])
    d = daily_returns(eq)
    pos = signal.shift(1).fillna(0.0)
    entries = int(((pos.diff() > 0).sum()) + (1 if pos.iloc[0] > 0 else 0))
    return {
        "연환산": final ** (365 / days) - 1 if final > 0 else -1.0,
        "최대낙폭": float((eq / eq.cummax() - 1).min()),
        "하루평균": float(d.mean()),
        "5%이상 일수": int((d >= 0.05).sum()),
        "하루거래": entries / days,
        "최종(배)": final,
    }


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    half = len(df) // 2
    for name, fn in STRATEGIES.items():
        sig = fn(df)
        eq = simulate(df, sig)
        s = stats(eq, sig)
        free = stats(simulate(df, sig, 0.0), sig)["연환산"]
        eq2 = simulate(df.iloc[half:], sig.iloc[half:])
        s2 = stats(eq2, sig.iloc[half:])["연환산"]
        rows.append({"전략": name, "연환산": s["연환산"], "최대낙폭": s["최대낙폭"], "하루평균": s["하루평균"],
                     "5%이상 일수": s["5%이상 일수"], "하루거래": s["하루거래"],
                     "비용0 연환산": free, "뒤절반 연환산": s2})
    flat = pd.Series(1.0, index=df.index)
    bh = simulate(df, flat)
    b = stats(bh, flat)
    rows.append({"전략": "BTC 보유", "연환산": b["연환산"], "최대낙폭": b["최대낙폭"], "하루평균": b["하루평균"],
                 "5%이상 일수": b["5%이상 일수"], "하루거래": 0.0, "비용0 연환산": b["연환산"],
                 "뒤절반 연환산": stats(simulate(df.iloc[half:], flat.iloc[half:]), flat.iloc[half:])["연환산"]})
    return pd.DataFrame(rows)


def fmt(tbl: pd.DataFrame) -> str:
    t = tbl.copy()
    for c in ("연환산", "최대낙폭", "비용0 연환산", "뒤절반 연환산"):
        t[c] = t[c].map(lambda x: f"{x * 100:+.0f}%")
    t["하루평균"] = t["하루평균"].map(lambda x: f"{x * 100:+.3f}%")
    t["하루거래"] = t["하루거래"].map(lambda x: f"{x:.1f}")
    return t.to_string(index=False)


def check_bars(df: pd.DataFrame, unit: int) -> str:
    gap = df.index.to_series().diff().dropna()
    med = gap.median() / pd.Timedelta(minutes=1)
    big = float((gap > pd.Timedelta(minutes=unit * 1.5)).mean())
    return f"{len(df)}봉 {df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d} (간격 중앙값 {med:.0f}분, 빠진 구간 {big * 100:.1f}%)"


def main() -> None:
    lines = []
    now = pd.Timestamp.now(tz="UTC")
    for unit in (60, 15):
        df = fetch_upbit_minutes(unit, total=TOTAL[unit])
        df = df[df.index + pd.Timedelta(minutes=unit) <= now]
        info = check_bars(df, unit)
        if len(df) < BARS_PER_DAY[unit] * 200:
            raise SystemExit(f"{unit}분봉 데이터 부족: {info}")
        tbl = evaluate(df)
        lines += [f"=== {unit}분봉: {info} ===", fmt(tbl), ""]
    lines.append("현물·롱만·전액 투입, 체결당 수수료 0.05% + 슬리피지 0.05%. 신호는 마감 후 다음 봉 시가 체결.")
    lines.append("'비용0 연환산'은 수수료를 뺀 값, '뒤절반 연환산'은 기간 뒤쪽 절반만 본 값(꾸준한지 확인용). 과거 데이터이며 수익을 보장하지 않습니다.")
    text = "\n".join(lines)
    print(text)
    os.makedirs("data", exist_ok=True)
    open("data/upbit_scalp.txt", "w").write(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
