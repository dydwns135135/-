"""한국 수급 스냅샷: 수급(외국인·기관·개인)은 과거 이력을 구할 수 없어 매일의 분석 결과를 저장해 두고,
시간이 지난 뒤 그 날 이후 수익률과 맞춰 '수급이 들어간 버전'이 쓸 만한지 시험한다."""
import glob
import os

import numpy as np
import pandas as pd

from stock.backtest import HORIZONS, _rank_line, rank_test

COLS = ["date", "code", "name", "theme", "close", "score", "stage", "bias", "bias_pts", "ret5", "ret20", "vol_ratio",
        "prox60", "flow_ratio", "indiv_ratio", "both_buy", "foreign5", "inst5", "indiv5", "flow_last"]


FLOW_COLS = ("foreign", "inst", "individual")


def lag_flow(market):
    """리포트·저장·시험 공통: 수급을 모든 종목에서 '직전 거래일까지'로 통일한다(가격·거래량은 그날 종가 그대로).

    네이버는 당일 수급을 저녁에 종목별로 순서대로 올려서(20:04에 0/182, 20:18에 90/182종목) 같은 날 스냅샷 안에서도 어떤 종목은
    오늘 수급이 있고 어떤 종목은 없다. 수급 열을 한 거래일 뒤로 밀어(shift) 오늘 수급이 있어도 쓰지 않으면 실행 시각과 상관없이
    모든 종목이 같은 정의(최근 5거래일 가격·거래량 + 직전 5거래일 수급)가 된다. 둘 다 그 시점에 알 수 있는 값이라 미래 정보는 없다.
    """
    out = {}
    for code, df in market.items():
        cols = [c for c in FLOW_COLS if c in df.columns]
        if not cols or len(df) < 2:
            out[code] = df
            continue
        d = df.copy()
        d[cols] = d[cols].shift(1).fillna(0.0)
        prev = f"{df.index[-2]:%Y-%m-%d}"
        fl = df.attrs.get("flow_last")
        d.attrs = dict(df.attrs)
        d.attrs["flow_last"] = fl if (fl is None or str(fl) < prev) else prev  # 직전 거래일보다 더 늦은 수급 기록은 그대로 둔다
        out[code] = d
    return out


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


def flow_fresh(df):
    """수급이 직전 거래일 이후까지 확인된 행(그날 수급이 있으면 더 좋고, 하루 늦은 것은 매일 똑같이 허용). 그보다 늦거나 없으면 제외."""
    return pd.to_datetime(df["flow_last"], errors="coerce") >= df["prev_date"]


def attach_returns(snap, market, horizons=HORIZONS):
    """스냅샷 날짜의 종가 대비 h거래일 뒤 수익률을 붙인다. 아직 h일이 지나지 않은 행은 빠진다."""
    rows = []
    for code, g in snap.groupby("code"):
        if code not in market:
            continue
        c = market[code]["close"]
        pos = c.index.get_indexer(pd.to_datetime(g["date"]))
        for (_, r), p in zip(g.iterrows(), pos):
            if p < 1 or p + max(horizons) >= len(c):
                continue
            rec = r.to_dict()
            rec["date"] = pd.Timestamp(r["date"])
            rec["prev_date"] = c.index[p - 1]  # 직전 거래일 — 수급이 여기까지 있으면 정상(당일 수급은 저녁에도 안 올라옴)
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
    fresh = d[flow_fresh(d)]  # 수급이 직전 거래일까지 확인된 행만
    skipped = len(d) - len(fresh)
    d = fresh
    if d["date"].nunique() < 10:
        return (head + f"\n수급이 직전 거래일까지 확인된 날이 아직 {d['date'].nunique()}일뿐입니다(이틀 이상 늦은 {skipped}건은 제외). "
                "더 쌓인 뒤 시험합니다.")
    lines = [head, f"수익률이 확정된 날 {df['date'].nunique()}일 · 수급이 직전 거래일까지 확인된 종목-일 {len(d)}건" +
             (f" (이틀 이상 늦은 {skipped}건 제외)" if skipped else ""),
             "가격은 그날 종가, 수급은 직전 거래일까지로 매일 똑같이 정의(둘 다 그 시점에 알 수 있는 값)",
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


# ---------------------------------------------------------------------------------------------
# 시장 훑기 스냅샷: 매일의 훑기 결과(종목별 점수·단계·수급 + 소속 테마)를 쌓아 두었다가, 시간이 지난 뒤
# '훑기에서 높게 나온 종목·테마가 실제로 더 올랐는지'를 시험한다. 과거 테마 구성은 구할 수 없어 지금부터 쌓아야 한다.
SCAN_COLS = COLS + ["themes", "best_theme", "theme_rank", "theme_score"]


def build_scan_snapshot(all_rows, themes, market):
    """all_rows: analyze_themes 종목표(같은 종목이 여러 테마에 있으면 여러 줄), themes: 테마 순위표(점수 내림차순).

    종목당 한 줄로 합치고, 소속 테마 전체(themes)와 그중 순위가 가장 높은 테마(best_theme·순위·점수)를 남긴다.
    """
    rank = {r.theme: (i + 1, r.score) for i, r in enumerate(themes.itertuples())}
    member = all_rows.groupby("code")["theme"].apply(list)
    best = {c: min(ts, key=lambda t: rank[t][0]) for c, ts in member.items()}
    uniq = all_rows.sort_values("score", ascending=False).drop_duplicates("code")
    s = build_snapshot(uniq, market)
    s["themes"] = [("|".join(member[c])) for c in s["code"]]
    s["best_theme"] = [best[c] for c in s["code"]]
    s["theme_rank"] = [rank[best[c]][0] for c in s["code"]]
    s["theme_score"] = [rank[best[c]][1] for c in s["code"]]
    return s[SCAN_COLS]


def load_scan_snapshots(path):
    """path/scan/*.csv (하위 폴더 scan/themes 는 제외)를 합쳐 (date, code) 중복은 마지막 것만 남긴다."""
    files = sorted(glob.glob(os.path.join(path, "scan", "*.csv")))
    if not files:
        return pd.DataFrame(columns=SCAN_COLS)
    df = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in files], ignore_index=True)
    return df.drop_duplicates(["date", "code"], keep="last").reset_index(drop=True)


def scan_codes(path):
    return sorted(load_scan_snapshots(path)["code"].unique()) if os.path.isdir(os.path.join(path, "scan")) else []


def group_excess(df, mask, h, step=1):
    """mask 에 든 종목의 이후 h일 수익률이 그날 훑은 전체 평균보다 얼마나 높았는지(날짜별 평균 → h일 간격 표본 → t값)."""
    base = df.groupby("date")[f"r{h}"].mean()
    g = df[mask].groupby("date")[f"r{h}"].mean()
    ex = (g - base.reindex(g.index)).sort_index().iloc[::max(1, h // step)]
    if len(ex) < 10 or ex.std() == 0:
        return None
    return ex.mean(), (ex > 0).mean(), ex.mean() / (ex.std() / np.sqrt(len(ex))), len(ex)


def _ex_line(label, df, mask, horizons=HORIZONS):
    parts = []
    for h in horizons:
        r = group_excess(df, mask, h)
        if r is None:
            parts.append(f"{h}일 표본 부족")
            continue
        mean, win, t, n = r
        parts.append(f"{h}일 {mean:+.2%}p (이긴 날 {win:.0%}, t={t:+.1f} → {'의미 있음' if abs(t) >= 2 else '구별 안 됨'}, {n}일)")
    return f"  {label:<10}" + " | ".join(parts)


def evaluate_scan(snap, market, horizons=HORIZONS):
    if snap.empty:
        return "[훑기 시험] 저장된 훑기 스냅샷이 아직 없습니다. 매일 저녁 훑기가 돌 때마다 하나씩 쌓입니다."
    head = f"[훑기 시험] 훑기 스냅샷 {snap['date'].nunique()}일치 ({snap['date'].min()} ~ {snap['date'].max()}) · 누적 {snap['code'].nunique()}종목"
    df = attach_returns(snap, market, horizons)
    need = " / ".join(f"{h}일 기준 약 {h * 10 + max(horizons)}거래일" for h in horizons)
    min_days = min(horizons) * 10  # 겹치지 않는 표본 10개가 나오는 최소 일수
    if df.empty or df["date"].nunique() < min_days:
        return head + f"\n이후 수익률이 확정된 날이 {0 if df.empty else df['date'].nunique()}일뿐이라 아직 부족합니다. {need} 이상 필요 — 더 쌓인 뒤 시험합니다."
    fresh = flow_fresh(df)  # 수급이 직전 거래일까지 확인된 행만
    skipped = int((~fresh).sum())
    d = df[fresh].copy()
    if d["date"].nunique() < min_days:
        return head + f"\n수급이 직전 거래일까지 확인된 날이 아직 {d['date'].nunique()}일뿐입니다(이틀 이상 늦은 {skipped}건 제외). 더 쌓인 뒤 시험합니다."
    d["stage"] = d["stage"].astype(str).str.split("(").str[0]
    lines = [head, f"수익률이 확정된 날 {d['date'].nunique()}일 · 종목-일 {len(d):,}건" + (f" (수급이 이틀 이상 늦은 {skipped}건 제외)" if skipped else ""),
             "가격은 그날 종가, 수급은 직전 거래일까지로 매일 똑같이 정의(둘 다 그 시점에 알 수 있는 값)",
             "그날 훑은 전체 평균과의 차이 (겹치지 않게 5·10일 간격 표본)"]
    for key, label in (("score", "점수 상위 3"), ("theme_score", "테마점수 상위 3")):
        lines.append(_rank_line(label, rank_test(d, key), horizons))
    lines.append("단계별 (해당 단계 종목 평균 − 전체 평균):")
    for st in ("선점 후보", "상승 진행", "과열", "관망"):
        if (d["stage"] == st).any():
            lines.append(_ex_line(st, d, d["stage"] == st, horizons))
    lines.append("테마 순위별:")
    lines.append(_ex_line("상위 3테마", d, d["theme_rank"] <= 3, horizons))
    lines.append("※ 표본이 쌓일수록 믿을 만해집니다. 여러 항목을 동시에 보면 우연히 하나쯤 좋게 나올 수 있으니 반복되는지 보세요.")
    return "\n".join(lines)
