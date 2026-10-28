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


def collect(market, themes_map, horizons=HORIZONS, min_hist=70):
    """종목·날짜별 (단계, 점수, 흐름, 이후 수익률) 표."""
    rows = []
    for theme, members in themes_map.items():
        for code, name in members.items():
            if code not in market:
                continue
            df = market[code][OHLCV]
            c = df["close"].to_numpy()
            for t in range(min_hist - 1, len(df) - max(horizons)):
                r = analyze_stock(df.iloc[max(0, t + 1 - WINDOW):t + 1])
                if r is None:
                    continue
                rec = {"code": code, "name": name, "theme": theme, "date": df.index[t],
                       "stage": r["stage"].split("(")[0], "bias": r["bias"].split("(")[0], "score": r["score"]}
                for h in horizons:
                    rec[f"r{h}"] = c[t + h] / c[t] - 1
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
    lines.append("\n판정(10일 기준):")
    lines.append("  선점 후보: " + verdict(df, df["stage"] == "선점 후보"))
    lines.append("  상승 우세: " + verdict(df, df["bias"] == "상승 우세"))
    lines.append("  과열     : " + verdict(df, df["stage"] == "과열"))
    lines.append("\n※ 지금 유명한 종목만 본 결과라 실제보다 좋게 나올 수 있고, 과거 성과가 미래를 보장하지 않습니다.")
    return "\n".join(lines)
