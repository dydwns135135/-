"""성과 확인: 과거 각 날짜에 그날까지의 데이터만으로 분석을 다시 돌려, 신호가 뜬 뒤 5·10거래일 수익률을 잰다.

수급·공시는 과거 이력을 구하기 어려워 제외하고 가격·거래량 기반 점수만 평가한다. 종가에 진입한다고 가정(수수료·슬리피지 제외).
한계: 지금 유명한 종목만 대상이라 생존 편향이 있고, 같은 종목의 연속된 날 신호는 겹쳐서 표본이 실제보다 독립적이지 않다.
"""
import numpy as np
import pandas as pd

from stock.analyze import analyze_stock

OHLCV = ["open", "high", "low", "close", "volume"]
WINDOW = 130  # analyze_stock 이 보는 최대 구간(60일선+여유)만 잘라 속도를 높인다
HORIZONS = (5, 10)


def collect(market, themes_map, horizons=HORIZONS, min_hist=70, sample_dates=None):
    """종목·날짜별 (단계, 점수, 흐름, 이후 수익률) 표. sample_dates 를 주면 그 날짜만 계산한다(긴 기간 시험의 속도용, 모든 종목이 같은 날짜)."""
    rows = []
    for theme, members in themes_map.items():
        for code, name in members.items():
            if code not in market:
                continue
            df = market[code][OHLCV]
            c = df["close"].to_numpy()
            o = df["open"].to_numpy()
            for t in range(min_hist - 1, len(df) - max(horizons)):
                if sample_dates is not None and df.index[t] not in sample_dates:
                    continue
                r = analyze_stock(df.iloc[max(0, t + 1 - WINDOW):t + 1])
                if r is None:
                    continue
                rec = {"code": code, "name": name, "theme": theme, "date": df.index[t],
                       "stage": r["stage"].split("(")[0], "bias": r["bias"].split("(")[0], "score": r["score"],
                       "bias_pts": r["bias_pts"], "c_trend": r["c_trend"], "c_near": r["c_near"],
                       "c_volume": r["c_volume"], "c_squeeze": r["c_squeeze"]}
                for h in horizons:
                    rec[f"r{h}"] = c[t + h] / c[t] - 1                                  # 신호일 종가 진입
                    rec[f"ro{h}"] = c[t + h] / o[t + 1] - 1 if o[t + 1] > 0 else np.nan  # 다음 날 시가 진입(시가 0 인 날은 제외)
                rows.append(rec)
    return pd.DataFrame(rows)


def summarize(df, key, horizons=HORIZONS):
    out = []
    for k, g in df.groupby(key):
        row = {key: k, "n": len(g)}
        for h in horizons:
            row[f"mean{h}"], row[f"win{h}"] = g[f"r{h}"].mean(), (g[f"r{h}"] > 0).mean()
        out.append(row)
    return pd.DataFrame(out)


def rank_test(df, key="score", k=3, horizons=HORIZONS, min_stocks=10, step=1):
    """날마다 종목을 key 로 줄 세워 상위 k·하위 k 의 이후 수익률이 그날 전체 평균보다 얼마나 높았는지 본다.

    h일 간격으로만 표본을 뽑아 겹치는 기간을 없애므로(독립에 가깝다) 단순 신호 집계보다 오차 계산이 정직하다.
    step: 데이터가 이미 step 거래일 간격으로만 있을 때(긴 기간 시험) 간격을 h//step 으로 줄인다.
    """
    order = [key] if key == "score" else [key, "score"]
    out = {}
    for h in horizons:
        rows = []
        for date, g in df.groupby("date"):
            if len(g) < min_stocks:
                continue
            g = g.sort_values(order, ascending=False)
            m = g[f"r{h}"].mean()
            rows.append((date, g.head(k)[f"r{h}"].mean() - m, g.tail(k)[f"r{h}"].mean() - m))
        s = pd.DataFrame(rows, columns=["date", "top", "bottom"]).set_index("date").sort_index().iloc[::max(1, h // step)]
        res = {"days": len(s)}
        for col in ("top", "bottom"):
            x = s[col]
            if len(x) >= 10 and x.std() > 0:
                res[col] = (x.mean(), (x > 0).mean(), x.mean() / (x.std() / np.sqrt(len(x))))
            else:
                res[col] = None
        out[h] = res
    return out


def _rank_line(label, res, horizons=HORIZONS):
    parts = []
    for h in horizons:
        r = res[h]["top"]
        if r is None:
            parts.append(f"{h}일 표본 부족")
            continue
        mean, win, t = r
        tag = "의미 있음" if abs(t) >= 2 else "구별 안 됨"
        parts.append(f"{h}일 {mean:+.2%}p (이긴 날 {win:.0%}, t={t:+.1f} → {tag})")
    return f"  {label:<14}" + " | ".join(parts)


def verdict(df, mask, h=10, min_n=30):
    """신호 집단의 평균 수익률이 전체 평균(기준선)보다 높은지 대략 판정. 겹치는 기간 때문에 표본 오차는 과소평가된다."""
    g = df[mask][f"r{h}"]
    if len(g) < min_n:
        return f"표본 부족(n={len(g)} < {min_n}) — 판단 보류"
    edge = g.mean() - df[f"r{h}"].mean()
    se = g.std() / np.sqrt(len(g))
    if edge > 2 * se:
        return f"기준선보다 {edge:+.1%}p 높음 (오차 범위 밖, 다만 겹치는 신호 때문에 과대평가 가능)"
    if edge < -2 * se:
        return f"기준선보다 {edge:+.1%}p 낮음 (오차 범위 밖)"
    return f"기준선과 구별되지 않음 ({edge:+.1%}p, 오차 ±{2 * se:.1%}p)"


def format_report(df, horizons=HORIZONS):
    if df.empty:
        return "[성과 확인] 평가할 데이터가 부족합니다(종목당 최소 80거래일 필요)."
    d0, d1 = df["date"].min(), df["date"].max()
    lines = [f"[성과 확인] {d0:%Y-%m-%d} ~ {d1:%Y-%m-%d} · {df['code'].nunique()}종목 · 신호 {len(df):,}건 (수급·공시 제외, 가격 기반)",
             "신호가 뜬 날 종가에 샀다고 가정한 뒤 5·10거래일 후 수익률"]
    base = " / ".join(f"{h}일 {df[f'r{h}'].mean():+.1%} (승률 {(df[f'r{h}'] > 0).mean():.0%})" for h in horizons)
    lines.append(f"기준선(모든 날 평균): {base}")
    for key, title in (("stage", "단계별"), ("bias", "흐름별")):
        lines.append(f"\n{title}:")
        s = summarize(df, key, horizons).sort_values("n", ascending=False)
        for _, r in s.iterrows():
            lines.append(f"  {r[key]:<8} n={int(r['n']):>5} | " +
                         " | ".join(f"{h}일 {r[f'mean{h}']:+.1%} (승률 {r[f'win{h}']:.0%})" for h in horizons))
    n_days = rank_test(df, "score")[HORIZONS[0]]["days"]
    lines.append(f"\n[날마다 순위 시험] 매일 점수 상위 3종목 − 그날 전체 평균 (겹치지 않게 {HORIZONS[0]}·{HORIZONS[1]}일 간격 표본, 약 {n_days}일)")
    for key, label in (("score", "점수 상위 3"), ("bias_pts", "흐름 상위 3")):
        res = rank_test(df, key)
        lines.append(_rank_line(label, res))
    low = rank_test(df, "score")
    lines.append("  점수 하위 3   " + " | ".join(
        (f"{h}일 {low[h]['bottom'][0]:+.2%}p (t={low[h]['bottom'][2]:+.1f})" if low[h]["bottom"] else f"{h}일 표본 부족") for h in HORIZONS))
    lines.append("  |t|≥2 이면 우연이라 보기 어렵다는 뜻이지만, 여러 항목을 동시에 보면 우연히 하나쯤 높게 나올 수 있습니다.")
    lines.append("\n판정(10일 기준):")
    lines.append("  선점 후보: " + verdict(df, df["stage"] == "선점 후보"))
    lines.append("  상승 우세: " + verdict(df, df["bias"] == "상승 우세"))
    lines.append("  과열     : " + verdict(df, df["stage"] == "과열"))
    lines.append("\n※ 지금 유명한 종목만 본 결과라 실제보다 좋게 나올 수 있고, 과거 성과가 미래를 보장하지 않습니다.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# 점수 재확인: 앞 구간에서 가장 잘 맞는 기준을 고르고, 마지막 1년에서 유지되는지 확인한다(MACD 재확인과 같은 방식).
SCORE_KEYS = (("score", "점수(가격 기반)"), ("c_trend", "추세 항목"), ("c_near", "고점 근접 항목"),
              ("c_volume", "거래량 항목"), ("c_squeeze", "변동폭 수축 항목"), ("bias_pts", "흐름 점수"))
STAGES = ("선점 후보", "상승 진행", "과열", "관망")


def _cell_rank(res):
    out = []
    for h in HORIZONS:
        r = res[h]["top"]
        out.append(f"{h}일 " + ("표본 부족" if r is None else f"{r[0]:+.2%}p(t={r[2]:+.1f})"))
    return " ".join(out)


def _persist(res, h=5):
    r = res[h]["top"]
    if r is None:
        return "확인 구간 표본 부족 — 판단 보류"
    mean, _, t = r
    if mean > 0 and t >= 2:
        return f"확인 구간에서도 유지됨 ({h}일 {mean:+.2%}p, t={t:+.1f})"
    if mean > 0:
        return f"방향은 같지만 통계적으로 약함 ({h}일 {mean:+.2%}p, t={t:+.1f}) — 입증되지 않음"
    return f"확인 구간에서 사라지거나 반대 ({h}일 {mean:+.2%}p, t={t:+.1f})"


def score_robust_report(df, label="", oos_days=365, keys=SCORE_KEYS):
    """collect 결과 → 학습/확인 구간 분리 리포트. 진입은 ①신호일 종가 ②다음 날 시가."""
    from stock.snapshot import group_excess  # 지연 임포트: snapshot 이 backtest 를 임포트한다
    if df.empty:
        return "[점수 재확인] 평가할 데이터가 부족합니다."
    d0, d1 = df["date"].min(), df["date"].max()
    cutoff = d1 - pd.Timedelta(days=oos_days)
    lines = [f"[점수 재확인] 분석기 점수(수급·공시 제외, 가격 기반) {label} · 종목 {df['code'].nunique()}개",
             f"학습 구간 {d0:%Y-%m-%d} ~ {cutoff:%Y-%m-%d} / 확인 구간(마지막 {oos_days // 365}년) {cutoff:%Y-%m-%d} ~ {d1:%Y-%m-%d}",
             "매일 상위 3종목 − 그날 전체 평균 (겹치지 않게 5·10일 간격 표본). 학습 구간에서 5일 t값이 가장 높은 기준을 고르고 확인 구간에서 유지되는지 봅니다."]
    for title, rname in (("① 신호일 종가에 진입", "r"), ("② 다음 날 시가에 진입 (리포트를 보고 실제로 사는 경우)", "ro")):
        d = df.drop(columns=[f"r{h}" for h in HORIZONS]).rename(columns={f"ro{h}": f"r{h}" for h in HORIZONS}) if rname == "ro" else df
        d = d.dropna(subset=[f"r{h}" for h in HORIZONS])
        is_d, oos_d = d[d["date"] <= cutoff], d[d["date"] > cutoff]
        lines.append(f"\n{title}")
        cand = []
        for key, label_k in keys:
            a, b = rank_test(is_d, key), rank_test(oos_d, key)
            lines.append(f"  {label_k:<14} 학습 {_cell_rank(a)} │ 확인 {_cell_rank(b)}")
            if a[5]["top"] is not None:
                cand.append((a[5]["top"][2], key, label_k, b))
        if cand:
            t, key, label_k, b = max(cand, key=lambda x: x[0])
            lines.append(f"  → 학습에서 고른 기준: {label_k} (학습 5일 t={t:+.1f}) → {_persist(b)}")
        else:
            lines.append("  → 학습 구간 표본이 부족해 기준을 고르지 못함")
        lines.append("  단계별 (해당 단계 종목 평균 − 그날 전체 평균):")
        for st in STAGES:
            parts = []
            for tag, part in (("학습", is_d), ("확인", oos_d)):
                seg = []
                for h in HORIZONS:
                    r = group_excess(part, part["stage"] == st, h)
                    seg.append(f"{h}일 " + ("표본 부족" if r is None else f"{r[0]:+.2%}p(t={r[2]:+.1f})"))
                parts.append(f"{tag} " + " ".join(seg))
            lines.append(f"    {st:<6} " + " │ ".join(parts))
    lines.append("\n※ 지금 유명한 종목만 본 결과(생존 편향), 여러 기준을 함께 비교하므로 우연히 하나쯤 좋게 나올 수 있습니다. 수급·공시는 과거 이력이 없어 빠졌고, 세금·슬리피지는 단순화했습니다. 과거 성과가 미래를 보장하지 않습니다.")
    return "\n".join(lines)


def run_score_robust(market, label=""):
    tm = {"전체": {c: c for c in market}}
    return score_robust_report(collect(market, tm), label)


# ---------------------------------------------------------------------------------------------
# 워크포워드: 해마다 그 해 이전 데이터만으로 기준을 고르고 그 해에 확인한다(긴 기간, 여러 시장 국면).
def _wf_cell(r):
    return "표본 부족" if r is None else f"{r[0]:+.2%}p(t={r[2]:+.1f})"


def _wf_summary(label, vals):
    ok = [v for v in vals if v is not None]
    if len(ok) < 3:
        return f"  {label:<14} 해 수 부족"
    means = np.array([v[0] for v in ok])
    return f"  {label:<14} 평균 {means.mean():+.2%}p · 플러스였던 해 {int((means > 0).sum())}/{len(ok)}"


def walk_forward_report(df, label="", step=5, min_train_years=2):
    """collect 결과(다음 날 시가 진입 수익률 ro5/ro10 사용) → 연도별 워크포워드. 데이터는 step 거래일 간격 표본이어야 한다."""
    from stock.snapshot import group_excess
    if df.empty:
        return "[점수 워크포워드] 평가할 데이터가 부족합니다."
    d = df.drop(columns=[f"r{h}" for h in HORIZONS]).rename(columns={f"ro{h}": f"r{h}" for h in HORIZONS})
    d = d.dropna(subset=[f"r{h}" for h in HORIZONS])
    d0, d1 = d["date"].min(), d["date"].max()
    years = [y for y in sorted(d["date"].dt.year.unique()) if pd.Timestamp(year=int(y), month=1, day=1) - d0 >= pd.Timedelta(days=365 * min_train_years)]
    lines = [f"[점수 워크포워드] 분석기 점수(수급·공시 제외, 가격 기반) {label} · 종목 {d['code'].nunique()}개 · {d0:%Y-%m-%d} ~ {d1:%Y-%m-%d}",
             "해마다 그 해 이전 데이터만으로 5일 t값이 가장 높은 기준을 고르고(매일 상위 3종목 − 그날 평균), 그 해에 확인합니다. 진입은 다음 날 시가.",
             "표의 값은 확인한 해의 5일 초과수익(t값). 한 해 표본은 작아 t값이 2를 넘기 어렵습니다."]
    if not years:
        lines.append(f"\n학습에 필요한 {min_train_years}년 이상의 앞 데이터가 있는 해가 없습니다(데이터 {(d1 - d0).days // 365}년치).")
        return "\n".join(lines)
    chosen_v, score_v, cand_v, hot_v = [], [], [], []
    lines.append("\n해     고른 기준(학습)        고른 기준 확인   │ 점수 전체        │ 관찰 후보         │ 과열")
    for y in years:
        train, test = d[d["date"] < pd.Timestamp(year=int(y), month=1, day=1)], d[d["date"].dt.year == y]
        cand = []
        for key, name in SCORE_KEYS:
            r = rank_test(train, key, step=step)[5]["top"]
            if r is not None:
                cand.append((r[2], key, name))
        if not cand:
            lines.append(f"{y}   학습 표본 부족")
            continue
        _, key, name = max(cand)
        r_ch, r_sc = rank_test(test, key, step=step)[5]["top"], rank_test(test, "score", step=step)[5]["top"]
        r_cd, r_hot = group_excess(test, test["stage"] == "선점 후보", 5, step), group_excess(test, test["stage"] == "과열", 5, step)
        chosen_v.append(r_ch); score_v.append(r_sc); cand_v.append(r_cd); hot_v.append(r_hot)
        lines.append(f"{y}   {name:<14} {_wf_cell(r_ch):<20} │ {_wf_cell(r_sc):<18} │ {_wf_cell(r_cd):<18} │ {_wf_cell(r_hot)}")
    lines.append("\n연도별 요약 (확인한 해의 5일 초과수익):")
    lines += [_wf_summary("학습에서 고른 기준", chosen_v), _wf_summary("점수 전체", score_v),
              _wf_summary("관찰 후보 단계", cand_v), _wf_summary("과열 단계", hot_v)]
    ok = [v for v in chosen_v if v is not None]
    if len(ok) >= 3:
        pos = sum(1 for v in ok if v[0] > 0) / len(ok)
        mean = float(np.mean([v[0] for v in ok]))
        verdict = ("여러 해에 걸쳐 대체로 플러스를 유지" if pos >= 0.7 and mean > 0 else
                   "해마다 들쑥날쑥해 유지된다고 보기 어려움" if pos <= 0.55 or mean <= 0 else "애매함 — 더 지켜봐야 함")
        lines.append(f"\n판정: 학습에서 고른 기준은 {verdict} (플러스 {pos:.0%}, 평균 {mean:+.2%}p)")
    lines.append("※ 지금 유명한 종목만 본 결과(생존 편향), 기준 여러 개를 비교하므로 우연히 좋아 보일 수 있습니다. 수급·공시는 제외, 세금·슬리피지는 단순화했고 과거 성과가 미래를 보장하지 않습니다.")
    return "\n".join(lines)


def run_score_walk(market, label="", step=5):
    all_dates = sorted(set().union(*[set(d.index) for d in market.values()]))
    sample = set(all_dates[::step])
    df = collect(market, {"전체": {c: c for c in market}}, sample_dates=sample)
    return walk_forward_report(df, label, step)
