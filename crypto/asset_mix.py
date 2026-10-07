"""장기 데이터로 '성격이 다른 자산' 분산 검증: BTC · 나스닥 100(QQQ) · 금(GLD)에 같은 앙상블 모멘텀을 적용.
python -m crypto.asset_mix   (공개 시세만 사용, API 키·주문 없음)

- 자산 3개는 미리 정한 값(결과를 보고 바꾸지 않는다). 데이터는 Yahoo(실패 시 Stooq, BTC 는 거래소)에서 받는다.
- ETF 는 평일만 거래하므로 달력 일자로 늘려 휴장일은 전일 가격을 유지(손익 0). 배당 반영(조정가) 사용.
- 비용 가정이 결과를 크게 좌우하므로 두 가지로 모두 본다:
  (a) 펀딩비 없음(현물·ETF 로 실행한다고 가정)  (b) 선물 상수 펀딩비 0.01%/8h(연 약 11%, 코인 기준)
- 구간: 전체 / 앞 60% / 뒤 40%. 앙상블 준비 250일은 모든 전략에서 제외."""
from __future__ import annotations

import argparse
import io
import os

import numpy as np
import pandas as pd
import requests

from . import strategies as st
from .backtest_report import load_data
from .strategy_compare import metrics

WARM = 250
UA = {"User-Agent": "Mozilla/5.0 (compatible; research-backtest)"}
ASSETS = {"BTC": "BTC-USD", "QQQ": "QQQ", "GLD": "GLD"}  # 이름 → Yahoo 티커
STOOQ = {"QQQ": "qqq.us", "GLD": "gld.us"}


def fetch_yahoo(ticker: str, timeout: int = 30) -> pd.DataFrame:
    """Yahoo 차트 API(일봉, 가능한 전 기간). 시가는 배당·분할 조정비율을 곱해 조정가로 맞춘다."""
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                     params={"range": "max", "interval": "1d", "events": "div,splits"}, headers=UA, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Yahoo HTTP {r.status_code}: {r.text[:80]!r}")
    res = r.json()["chart"]["result"][0]
    q, ts = res["indicators"]["quote"][0], res["timestamp"]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    return parse_ohlc(ts, q["open"], q["close"], adj)


def parse_ohlc(ts, open_, close, adjclose=None) -> pd.DataFrame:
    df = pd.DataFrame({"open": open_, "close": close}, index=pd.to_datetime(ts, unit="s", utc=True).normalize())
    if adjclose is not None:
        f = pd.Series(adjclose, index=df.index) / df["close"]
        df["open"], df["close"] = df["open"] * f, pd.Series(adjclose, index=df.index)
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[(df["open"] > 0) & (df["close"] > 0)]


def fetch_stooq(code: str, timeout: int = 30) -> pd.DataFrame:
    r = requests.get("https://stooq.com/q/d/l/", params={"s": code, "i": "d"}, headers=UA, timeout=timeout)
    if r.status_code != 200 or "Date" not in r.text[:40]:
        raise RuntimeError(f"Stooq HTTP {r.status_code}: {r.text[:80]!r}")
    d = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
    d.index = d.index.tz_localize("UTC")
    return d.rename(columns={"Open": "open", "Close": "close"})[["open", "close"]].dropna()


def load_asset(name: str) -> tuple[str, pd.DataFrame]:
    errors = []
    try:
        return "yahoo", fetch_yahoo(ASSETS[name])
    except Exception as e:
        errors.append(f"yahoo: {type(e).__name__}: {str(e)[:100]}")
    if name in STOOQ:
        try:
            return "stooq", fetch_stooq(STOOQ[name])
        except Exception as e:
            errors.append(f"stooq: {type(e).__name__}: {str(e)[:100]}")
    if name == "BTC":
        try:
            ex_id, d = load_data(8, "BTC/USDT:USDT", min_bars=800, timeframe="1d")
            return ex_id, d[["open", "close"]]
        except SystemExit as e:
            errors.append(f"거래소: {str(e)[:150]}")
    raise SystemExit(f"{name} 시세를 받지 못함 → " + " | ".join(errors))


def to_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """달력 일자로 확장: 휴장일은 직전 종가 유지(시가=종가=직전 종가 → 손익 0)."""
    idx = pd.date_range(df.index[0], df.index[-1], freq="D", tz="UTC")
    out = df.reindex(idx)
    out["close"] = out["close"].ffill()
    out["open"] = out["open"].fillna(out["close"])
    return out


def build(dfs: dict[str, pd.DataFrame], funding: float) -> dict[str, pd.Series]:
    """전략별 일 손익률(WARM 제외). dfs 는 같은 달력 일자 인덱스."""
    names = list(dfs)
    ens = {k: st.simulate(d.assign(high=d["close"], low=d["close"]), st.momentum_ensemble(d), funding_per_8h=funding) for k, d in dfs.items()}
    hold = {k: st.simulate(d.assign(high=d["close"], low=d["close"]), st.buy_hold(d), funding_per_8h=funding) for k, d in dfs.items()}
    out: dict[str, pd.Series] = {}
    for k in names:
        out[f"{k} 보유"] = hold[k]
        out[f"{k} 앙상블"] = ens[k]
    n = len(names)
    out[f"{n}자산 보유(등분)"] = pd.concat(hold.values(), axis=1).mean(axis=1)
    out[f"{n}자산 앙상블(등분)"] = pd.concat(ens.values(), axis=1).mean(axis=1)
    return {k: v.iloc[WARM:] for k, v in out.items()}


def compare(dfs: dict[str, pd.DataFrame], funding: float) -> pd.DataFrame:
    pnls = build(dfs, funding)
    n = len(next(iter(pnls.values())))
    cut = int(n * 0.6)
    segs = {"전체": slice(0, n), "앞60%": slice(0, cut), "뒤40%": slice(cut, n)}
    return pd.DataFrame([{"전략": k, "구간": s, **metrics(p.iloc[sl], bars_per_year=365)} for k, p in pnls.items() for s, sl in segs.items()])


def weekday_corr(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    r = pd.DataFrame({k: d["close"].pct_change() for k, d in dfs.items()})
    return r[r.index.dayofweek < 5].iloc[WARM:].corr()


def summary(rep: pd.DataFrame, names: list[str]) -> list[str]:
    n = len(names)
    port, base = f"{n}자산 앙상블(등분)", f"{names[0]} 앙상블"
    p, b = rep[rep["전략"] == port].set_index("구간"), rep[rep["전략"] == base].set_index("구간")
    lines = []
    for seg in ("전체", "앞60%", "뒤40%"):
        lines.append(f"[{seg}] {port} vs {base}: 샤프 {p.loc[seg, '샤프']:.2f} vs {b.loc[seg, '샤프']:.2f} / "
                     f"낙폭 {p.loc[seg, 'MDD']:+.1%} vs {b.loc[seg, 'MDD']:+.1%} / 수익 {p.loc[seg, '수익률']:+.0%} vs {b.loc[seg, '수익률']:+.0%}")
    ok_dd = all(p.loc[s, "MDD"] > b.loc[s, "MDD"] for s in ("전체", "앞60%", "뒤40%"))
    ok_sh = all(p.loc[s, "샤프"] >= b.loc[s, "샤프"] for s in ("전체", "앞60%", "뒤40%"))
    lines.append(f"→ 세 구간 모두 낙폭 감소: {'○' if ok_dd else '×'} / 세 구간 모두 샤프 개선: {'○' if ok_sh else '×'}")
    return lines


def fmt(rep: pd.DataFrame) -> str:
    d = rep.copy()
    for c in ("수익률", "연환산", "MDD"):
        d[c] = d[c].map(lambda v: f"{v:+.1%}")
    d["노출"] = d["노출"].map(lambda v: f"{v:.0%}")
    d["샤프"] = d["샤프"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
    return d.to_string(index=False)


def common(raw: dict[str, pd.DataFrame], names: list[str]) -> dict[str, pd.DataFrame]:
    cal = {k: to_calendar(raw[k]) for k in names}
    start = max(d.index[0] for d in cal.values())
    end = min(d.index[-1] for d in cal.values())
    return {k: cal[k].loc[start:end] for k in names}


def section(title: str, dfs: dict[str, pd.DataFrame], names: list[str]) -> str:
    idx = next(iter(dfs.values())).index
    out = [f"=== {title}: {', '.join(names)} | 공통 {len(idx)}일 {idx[0]:%Y-%m-%d}~{idx[-1]:%Y-%m-%d} (앞 {WARM}일 제외) ==="]
    for label, fund in (("(a) 펀딩비 없음(현물·ETF 가정)", 0.0), ("(b) 선물 펀딩비 연 ~11% 가정", 0.0001)):
        rep = compare(dfs, fund)
        out += ["", f"--- {label} ---", fmt(rep), "", *summary(rep, names)]
    c = weekday_corr(dfs)
    out += ["", "평일 일수익률 상관:", c.round(2).to_string()]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    raw, src = {}, {}
    for k in ASSETS:
        src[k], raw[k] = load_asset(k)
        print(f"  {k}: {src[k]} {len(raw[k])}봉 {raw[k].index[0]:%Y-%m-%d}~{raw[k].index[-1]:%Y-%m-%d}")
    texts = [section("A. BTC+나스닥100+금", common(raw, ["BTC", "QQQ", "GLD"]), ["BTC", "QQQ", "GLD"]),
             section("B. 전통자산만(장기)", common(raw, ["QQQ", "GLD"]), ["QQQ", "GLD"])]
    text = "\n\n".join(texts)
    print("\n" + text)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "asset_mix.txt"), "w").write(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("### 자산 분산 장기 검증\n\n```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
