"""실제 BTC 4시간봉을 거래소에서 내려받아 여러 설정으로 백테스트하고 표로 출력한다.
python -m crypto.backtest_report [--years 4] [--out data]   (공개 시세만 사용, API 키 불필요)

과적합을 줄이기 위해 설정을 많이 돌려 최고를 고르지 않는다: 미리 정한 소수의 설정을
전체 / 앞 60% / 뒤 40% 구간에서 같이 보여 준다. 구간별로 결과가 일관되지 않으면 신뢰하지 말 것."""
from __future__ import annotations

import argparse
import math
import os
import time

import ccxt
import pandas as pd

from .backtest import FuturesConfig, backtest_futures

TF_MS = 4 * 3600 * 1000


def fetch_history(ex, symbol: str, timeframe: str = "4h", years: float = 4.0, limit: int = 200,
                  max_calls: int = 400, keep_volume: bool = False) -> pd.DataFrame:
    """since 를 앞으로 밀며 과거부터 현재까지 캔들을 이어 받는다."""
    since = int(time.time() * 1000) - int(years * 365 * 24 * 3600 * 1000)
    rows, calls = [], 0
    while calls < max_calls:
        calls += 1
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch:
            break
        new = [r for r in batch if not rows or r[0] > rows[-1][0]]
        if not new:
            break
        rows += new
        since = rows[-1][0] + 1
        if len(batch) < 2:
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    df = df[~df.index.duplicated()].sort_index()
    cols = ["open", "high", "low", "close"] + (["volume"] if keep_volume else [])
    return df.iloc[:-1][cols]  # 마지막은 진행 중인 봉


def load_data(years: float, symbol: str = "BTC/USDT:USDT", min_bars: int = 2000, timeframe: str = "4h",
              fallback_years: tuple[float, ...] = (5, 4, 3, 2), keep_volume: bool = False):
    """거래소(Bitget → OKX) 순으로 시세를 받는다. 상장이 늦은 코인은 먼 과거를 요청하면 빈 결과가 올 수 있어
    요청 기간을 줄여(fallback_years) 다시 시도한다. 실패하면 시도별 이유를 모두 보여 준다."""
    errors: list[str] = []
    tries = [years] + [y for y in fallback_years if y < years]
    for ex_id in ("bitget", "okx"):
        ex = getattr(ccxt, ex_id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        for y in tries:
            try:
                df = fetch_history(ex, symbol, timeframe=timeframe, years=y, keep_volume=keep_volume)
            except Exception as e:  # 거래소 하나가 막혀도 다음으로
                errors.append(f"{ex_id}/{y:g}년: {type(e).__name__}: {str(e)[:100]}")
                continue
            if len(df) >= min_bars:
                return ex_id, df
            errors.append(f"{ex_id}/{y:g}년: 봉 {len(df)}개")
    raise SystemExit(f"{symbol} 시세를 충분히 내려받지 못함 → " + " | ".join(errors))


CONFIGS = [  # 미리 정한 소수의 설정 (결과를 보고 고르지 않음)
    ("롱만 x1", dict(leverage=1.0, allow_short=False)),
    ("롱만 x2", dict(leverage=2.0, allow_short=False)),
    ("롱+숏 x1", dict(leverage=1.0, allow_short=True)),
    ("롱+숏 x2", dict(leverage=2.0, allow_short=True)),
]


def sharpe(eq: pd.Series, bars_per_year: float = 6 * 365) -> float:
    r = eq.pct_change().dropna()
    return float(r.mean() / r.std() * math.sqrt(bars_per_year)) if len(r) > 2 and r.std() > 0 else float("nan")


def run_report(df: pd.DataFrame, stop_loss: float = 0.03) -> pd.DataFrame:
    n = len(df)
    segments = {"전체": df, "앞60%": df.iloc[: int(n * 0.6)], "뒤40%": df.iloc[int(n * 0.6):]}
    out = []
    for seg_name, seg in segments.items():
        for name, kw in CONFIGS:
            r = backtest_futures(seg, FuturesConfig(stop_loss=stop_loss, **kw))
            out.append({
                "구간": seg_name, "설정": name,
                "기간": f"{seg.index[0]:%Y-%m-%d}~{seg.index[-1]:%Y-%m-%d}",
                "수익률": r["총수익률"], "연환산": r["연환산수익률"], "MDD": r["최대낙폭(MDD)"],
                "샤프": sharpe(r["equity"]), "거래수": r["거래횟수"], "승률": r["승률"],
                "청산": r["청산횟수"], "단순보유": r["단순보유수익률"],
            })
    return pd.DataFrame(out)


def fmt(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in ("수익률", "연환산", "MDD", "승률", "단순보유"):
        d[c] = d[c].map(lambda v: "-" if pd.isna(v) else f"{v:+.1%}")
    d["샤프"] = d["샤프"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
    return d.to_string(index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=float, default=4.0)
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    ex_id, df = load_data(a.years, a.symbol)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "btc_4h.csv"), index_label="date")
    print(f"데이터: {ex_id} {a.symbol} 4h, {len(df)}봉, {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}")
    rep = run_report(df)
    rep.to_csv(os.path.join(a.out, "report.csv"), index=False)
    text = fmt(rep)
    print(text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a") as f:
            f.write(f"### BTC 4시간봉 백테스트 ({ex_id}, {len(df)}봉)\n\n```\n{text}\n```\n")


if __name__ == "__main__":
    main()
