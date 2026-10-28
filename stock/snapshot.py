"""한국 수급 스냅샷: 수급(외국인·기관·개인)은 과거 이력을 구할 수 없어 매일의 분석 결과를 저장해 두고,
시간이 지난 뒤 그 날 이후 수익률과 맞춰 '수급이 들어간 버전'이 쓸 만한지 시험한다."""
import glob
import os

import numpy as np
import pandas as pd

from stock.backtest import HORIZONS, _rank_line, rank_test

COLS = ["date", "code", "name", "theme", "close", "score", "stage", "bias", "bias_pts", "ret5", "ret20", "vol_ratio",
        "prox60", "flow_ratio", "indiv_ratio", "both_buy", "foreign5", "inst5", "indiv5", "flow_last"]


def build_snapshot(stocks, market):
    """analyze_themes 의 종목표 + 일봉 마지막 날짜·종가 → 저장용 표. date 는 각 종목 일봉의 마지막 날짜."""
    s = stocks.copy()
    s["date"] = [f"{market[c].index[-1]:%Y-%m-%d}" for c in s["code"]]
    s["close"] = [float(market[c]["close"].iloc[-1]) for c in s["code"]]
    for c in COLS:
        if c not in s:
            s[c] = np.nan
    return s[COLS]


def load_snapshots(path):
    """path/kr/*.csv (또는 path/*.csv) 를 합쳐 (date, code) 중복은 마지막 것만 남긴다."""
    files = sorted(glob.glob(os.path.join(path, "kr", "*.csv")) or glob.glob(os.path.join(path, "*.csv")))
    if not files:
        return pd.DataFrame(columns=COLS)
    df = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in files], ignore_index=True)
    return df.drop_duplicates(["date", "code"], keep="last").reset_index(drop=True)


def attach_returns(snap, market, horizons=HORIZONS):
    """스냅샷 날짜의 종가 대비 h거래일 뒤 수익률을 붙인다. 아직 h일이 지나지 않은 행은 빠진다."""
    rows = []
    for code, g in snap.groupby("code"):
        if code not in market:
            continue
        c = market[code]["close"]
        pos = c.index.get_indexer(pd.to_datetime(g["date"]))
        for (_, r), p in zip(g.iterrows(), pos):
            if p < 0 or p + max(horizons) >= len(c):
                continue
            rec = r.to_dict()
            rec["date"] = pd.Timestamp(r["date"])
            for h in horizons:
                rec[f"r{h}"] = c.iloc[p + h] / c.iloc[p] - 1
            rows.append(rec)
    return pd.DataFrame(rows)


def evaluate(snap, market, horizons=HORIZONS):
    if snap.empty:
        return "[수급 버전 시험] 저장된 스냅샷이 아직 없습니다. 매일 16:30 한국 분석이 돌 때마다 하나씩 쌓입니다."
    days = snap["date"].nunique()
    head = f"[수급 버전 시험] 스냅샷 {days}일치 ({snap['date'].min()} ~ {snap['date'].max()}) · {snap['code'].nunique()}종목"
    df = attach_returns(snap, market, horizons)
    need = {h: h * 10 + max(horizons) for h in horizons}  # h일 간격 표본 10개 + 마지막 max(h)일은 수익률 미확정
    if df.empty or df["date"].nunique() < 10:
        return (head + "\n이후 수익률이 확정된 날이 아직 부족합니다. " +
                " / ".join(f"{h}일 기준 약 {need[h]}거래일 이상 필요" for h in horizons) + " — 더 쌓인 뒤 시험합니다.")
    d = df.dropna(subset=["flow_ratio"])
    lines = [head, f"수익률이 확정된 날 {df['date'].nunique()}일 · 수급이 있는 종목-일 {len(d)}건",
             "매일 수급 상위 3종목 − 그날 전체 평균 (겹치지 않게 5·10일 간격 표본)"]
    for key, label in (("flow_ratio", "수급 상위 3"), ("score", "점수 상위 3(수급 포함)")):
        lines.append(_rank_line(label, rank_test(d, key), horizons))
    both = d["both_buy"].astype(str).str.lower().eq("true")
    for h in horizons:
        a, b = d[both][f"r{h}"], d[~both][f"r{h}"]
        if len(a) >= 30:
            lines.append(f"  외국인·기관 동반 순매수 {h}일: {a.mean():+.2%} (n={len(a)}) vs 그 외 {b.mean():+.2%} (n={len(b)})")
    short = [h for h in horizons if rank_test(d, "flow_ratio")[h]["top"] is None]
    if short:
        lines.append("  ※ " + ", ".join(f"{h}일" for h in short) + " 기준은 표본이 아직 부족해 판단 보류")
    lines.append("※ 표본이 쌓일수록 믿을 만해집니다. 여러 항목 중 하나가 우연히 좋게 나올 수 있으니 반복되는지 보세요.")
    return "\n".join(lines)
