"""앙상블 모멘텀과 일목 구름을 섞은 방식을 업비트 일봉(BTC·ETH·XRP)으로 백테스트한다.
python -m crypto.upbit_mix   (공개 시세만 사용, API 키·주문 없음)

미리 정한 두 가지 섞기만 본다(결과를 보고 고르지 않으려고 변형을 늘리지 않음):
- 평균: (앙상블 비중 + 일목 구름 위 여부(0/1)) / 2
- 교집합: 앙상블 비중 × 일목 구름 위 여부 (구름 아래면 보유 안 함)
섞은 비중도 앙상블과 같이 25%p 이상 달라질 때만 조정한다. 조건은 upbit_indicators 와 같다(현물·롱만, 체결당 0.10%, 다음 날 시가 체결)."""
from __future__ import annotations

import os

import pandas as pd

from .strategies import momentum_ensemble
from .upbit_data import closed_only, fetch_upbit_daily
from .upbit_indicators import MARKETS, WARM, ichimoku_cloud, stats, stepped
from .upbit_scalp import simulate

NAMES = ("보유", "앙상블(25%p)", "일목 구름 위", "섞기: 평균", "섞기: 교집합")


def raw_weights(df: pd.DataFrame) -> dict[str, pd.Series]:
    ens = momentum_ensemble(df)
    ichi = ichimoku_cloud(df)
    return {"앙상블(25%p)": stepped(ens), "일목 구름 위": ichi,
            "섞기: 평균": stepped((ens + ichi) / 2), "섞기: 교집합": stepped(ens * ichi)}


def compare(df: pd.DataFrame, from_date: str | None = None) -> pd.DataFrame:
    start = WARM if not from_date else max(WARM, int(df.index.searchsorted(pd.Timestamp(from_date, tz="UTC"))))
    sigs = {"보유": pd.Series(1.0, index=df.index), **raw_weights(df)}
    rows = {}
    for name in NAMES:
        sig = sigs[name]
        eq = simulate(df, sig)
        rows[name] = stats(eq.iloc[start:] / eq.iloc[start - 1], sig.iloc[start:])
    return pd.DataFrame(rows).T


def average(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """코인별 표를 전략별로 단순 평균(우연히 한 코인에서만 좋게 나온 건 아닌지 보려는 요약)."""
    return sum(tables) / len(tables)


def fmt(tbl: pd.DataFrame) -> str:
    t = tbl.copy()
    t["연환산"] = t["연환산"].map(lambda x: f"{x * 100:+.0f}%")
    t["최대낙폭"] = t["최대낙폭"].map(lambda x: f"{x * 100:+.0f}%")
    t["샤프"] = t["샤프"].map(lambda x: f"{x:.2f}")
    t["연거래"] = t["연거래"].map(lambda x: f"{x:.1f}")
    t["평균투입"] = t["평균투입"].map(lambda x: f"{x * 100:.0f}%")
    return t.to_string()


def main() -> None:
    segs = {"전체": None, "최근 약 4년(2022~)": "2022-01-01"}
    per: dict[str, list[pd.DataFrame]] = {k: [] for k in segs}
    lines = []
    for m in MARKETS:
        df = closed_only(fetch_upbit_daily(m, total=2600))
        gap = df.index.to_series().diff().dropna().median() / pd.Timedelta(days=1)
        if len(df) < WARM + 400 or gap > 1.5:
            lines.append(f"[{m}] 데이터 부족 또는 간격 이상: {len(df)}봉, 중앙값 {gap:.1f}일 → 건너뜀")
            continue
        for label, fd in segs.items():
            tbl = compare(df, fd)
            per[label].append(tbl)
            s0 = df.index[WARM if not fd else max(WARM, int(df.index.searchsorted(pd.Timestamp(fd, tz='UTC'))))]
            lines += [f"=== {m[4:]} {label}: {s0:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d} ===", fmt(tbl), ""]
    for label, tabs in per.items():
        if tabs:
            lines += [f"=== 코인 {len(tabs)}개 평균 · {label} ===", fmt(average(tabs)), ""]
    lines += ["현물·롱만, 체결당 비용 0.10%, 신호는 마감 후 다음 날 시가 체결. 섞기 방식은 결과를 보기 전에 정한 2가지뿐입니다.",
              "과거 데이터이며 수익을 보장하지 않습니다."]
    text = "\n".join(lines)
    print(text)
    os.makedirs("data", exist_ok=True)
    open("data/upbit_mix.txt", "w").write(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
