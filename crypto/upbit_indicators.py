"""일목균형표·볼린저밴드를 업비트 일봉(BTC·ETH·XRP)으로 백테스트하고 BTC 앙상블과 같은 조건으로 비교한다.
python -m crypto.upbit_indicators   (공개 시세만 사용, API 키·주문 없음)

현물·롱만·전액(또는 비율) 투입. 신호는 일봉 마감 때 확정, 다음 날 시가 체결. 체결당 수수료 0.05% + 슬리피지 0.05%.
- 일목: 종가가 구름(선행스팬 A·B 중 높은 값) 위면 보유 / 같은 조건 + 전환선>기준선이면 보유
- 볼린저(20일, 2σ): 상단 돌파 시 매수·중심선 이탈 시 청산(추세형) / 하단 이탈 시 매수·중심선 회복 시 청산(되돌림형)
- 앙상블: 8개 기간 중 오른 비율만큼 보유, 비중이 25%p 이상 달라질 때만 조정(기준 비교용)"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .strategies import momentum_ensemble
from .upbit_data import closed_only, fetch_upbit_daily
from .upbit_scalp import COST, simulate

WARM = 250
MARKETS = ("KRW-BTC", "KRW-ETH", "KRW-XRP")


def ichimoku_lines(df: pd.DataFrame) -> dict[str, pd.Series]:
    h, l = df["high"], df["low"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)                    # 26일 앞에 그리지만 계산은 과거 값 → 미래 정보 없음
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    return {"tenkan": tenkan, "kijun": kijun, "top": pd.concat([span_a, span_b], axis=1).max(axis=1, skipna=False)}


def next_cloud_top(df: pd.DataFrame) -> float:
    """다음 봉(지금 진행 중인 봉)의 구름 상단. 선행스팬은 26봉 앞에 그리므로 다음 봉 값은 이미 정해져 있다."""
    h, l = df["high"], df["low"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    raw_a = (tenkan + kijun) / 2
    raw_b = (h.rolling(52).max() + l.rolling(52).min()) / 2
    return float(max(raw_a.shift(25).iloc[-1], raw_b.shift(25).iloc[-1]))


def ichimoku_cloud(df: pd.DataFrame, confirm: bool = False) -> pd.Series:
    ln = ichimoku_lines(df)
    on = df["close"] > ln["top"]
    if confirm:
        on &= ln["tenkan"] > ln["kijun"]
    return on.fillna(False).astype(float)


def _state(enter: np.ndarray, exit_: np.ndarray, index) -> pd.Series:
    out, on = np.zeros(len(enter)), 0.0
    for i in range(len(enter)):
        if on == 0.0 and enter[i]:
            on = 1.0
        elif on == 1.0 and exit_[i]:
            on = 0.0
        out[i] = on
    return pd.Series(out, index=index)


def bollinger(df: pd.DataFrame, mode: str = "breakout", n: int = 20, k: float = 2.0) -> pd.Series:
    c = df["close"]
    mid = c.rolling(n).mean()
    sd = c.rolling(n).std(ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    if mode == "breakout":
        return _state((c > up).to_numpy(), (c < mid).to_numpy(), df.index)
    if mode == "revert":
        return _state((c < lo).to_numpy(), (c > mid).to_numpy(), df.index)
    raise ValueError(mode)


def stepped(w: pd.Series, step: float = 0.25) -> pd.Series:
    """목표 비중이 기준 비중과 step 이상 달라질 때만 기준을 갱신(0이 되면 즉시)."""
    ref, out = 0.0, []
    for x in w.to_numpy():
        if abs(x - ref) >= step - 1e-12 or (x <= 0 < ref):
            ref = float(x)
        out.append(ref)
    return pd.Series(out, index=w.index)


STRATEGIES = {
    "일목 구름 위": lambda d: ichimoku_cloud(d),
    "일목 구름+전환>기준": lambda d: ichimoku_cloud(d, confirm=True),
    "볼린저 상단돌파(추세)": lambda d: bollinger(d, "breakout"),
    "볼린저 하단반등(되돌림)": lambda d: bollinger(d, "revert"),
    "앙상블(25%p)": lambda d: stepped(momentum_ensemble(d)),
}


def stats(sim_eq: pd.Series, sig: pd.Series) -> dict:
    r = sim_eq.pct_change().fillna(sim_eq.iloc[0] - 1)
    years = max(len(sim_eq) / 365, 1e-9)
    final = float(sim_eq.iloc[-1])
    pos = sig.shift(1).fillna(0.0)
    return {"연환산": final ** (1 / years) - 1 if final > 0 else -1.0,
            "최대낙폭": float((sim_eq / sim_eq.cummax() - 1).min()),
            "샤프": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else float("nan"),
            "연거래": float(pos.diff().abs().sum() / 2 / years), "평균투입": float(pos.mean())}


def compare(df: pd.DataFrame, from_date: str | None = None) -> pd.DataFrame:
    start = WARM if not from_date else max(WARM, int(df.index.searchsorted(pd.Timestamp(from_date, tz="UTC"))))
    sigs = {"BTC 보유": pd.Series(1.0, index=df.index)}
    sigs.update({k: fn(df) for k, fn in STRATEGIES.items()})
    rows = {}
    for name, sig in sigs.items():
        eq = simulate(df, sig)
        rows[name] = stats(eq.iloc[start:] / eq.iloc[start - 1], sig.iloc[start:]) if start else stats(eq, sig)
    return pd.DataFrame(rows).T


def fmt(tbl: pd.DataFrame) -> str:
    t = tbl.copy()
    t["연환산"] = t["연환산"].map(lambda x: f"{x * 100:+.0f}%")
    t["최대낙폭"] = t["최대낙폭"].map(lambda x: f"{x * 100:+.0f}%")
    t["샤프"] = t["샤프"].map(lambda x: f"{x:.2f}")
    t["연거래"] = t["연거래"].map(lambda x: f"{x:.1f}")
    t["평균투입"] = t["평균투입"].map(lambda x: f"{x * 100:.0f}%")
    return t.to_string()


def main() -> None:
    lines = []
    for m in MARKETS:
        df = closed_only(fetch_upbit_daily(m, total=2600))
        gap = df.index.to_series().diff().dropna().median() / pd.Timedelta(days=1)
        if len(df) < WARM + 400 or gap > 1.5:
            lines.append(f"[{m}] 데이터 부족 또는 간격 이상: {len(df)}봉, 중앙값 {gap:.1f}일 → 건너뜀")
            continue
        coin = m[4:]
        for label, fd in (("전체", None), ("최근 약 4년(2022~)", "2022-01-01")):
            tbl = compare(df, fd)
            tbl = tbl.rename(index={"BTC 보유": f"{coin} 보유"})
            s0 = df.index[WARM if not fd else max(WARM, int(df.index.searchsorted(pd.Timestamp(fd, tz='UTC'))))]
            lines += [f"=== {coin} {label}: {s0:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d} ({len(df)}봉 중) ===", fmt(tbl), ""]
    lines += ["현물·롱만, 체결당 비용 0.10%(수수료 0.05% + 슬리피지 0.05%), 신호는 마감 후 다음 날 시가 체결.",
              "'연거래'는 1년 평균 매수 횟수. 과거 데이터이며 수익을 보장하지 않습니다. 지표 파라미터는 일반적인 기본값(일목 9/26/52, 볼린저 20일 2σ)이고 최적화하지 않았습니다."]
    text = "\n".join(lines)
    print(text)
    os.makedirs("data", exist_ok=True)
    open("data/upbit_indicators.txt", "w").write(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
