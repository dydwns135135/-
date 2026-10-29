"""TS Pro(MACD 교차) 신호 시험: TradingView 지표 'Trend Standard Pro (Stock)' 의 일봉 로직을 그대로 옮겨 과거 데이터로 시험한다.

- 신호: MACD(12·26·9, EMA)가 시그널을 위로 교차(BUY)/아래로 교차(SELL). 필터: 0선(BUY는 MACD>0, SELL은 MACD<0), 퍼센트(|MACD÷종가×100| ≥ 기준, 일봉 기본 1.0).
- 이벤트 연구: 신호가 뜬 날 종가에 샀다고 보고 5·10거래일 뒤 수익률에서 그날 전체 종목 평균을 뺀 초과수익을 본다.
  같은 종목에서 10거래일 안에 다시 뜬 신호는 겹치므로 제외한다.
- 왕복 매매: BUY 에 사서 다음 SELL 까지 보유(비용 가정 포함), 같은 기간 전체 종목 평균과 비교한다.
한계: 지금 시점의 대형주만 대상이라 생존 편향이 있고, 종가 체결·세금/슬리피지 단순화.
"""
import numpy as np
import pandas as pd

HORIZONS = (5, 10)
WARMUP = 60          # EMA 시작값의 영향을 피하려고 첫 60봉은 건너뜀
GAP = max(HORIZONS)  # 같은 종목 신호 최소 간격(겹침 제거)
PCT_MAIN = 1.0       # 일봉 기본 퍼센트 필터(Pine 기본값)


def ts_pro_series(close, fast=12, slow=26, sig=9):
    """Pine 의 ta.ema(시작값=첫 값, alpha=2/(n+1))와 같은 계산. (macd, signal, macd%)"""
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd, signal, macd / close * 100


def crosses(macd, signal):
    """ta.crossover / ta.crossunder"""
    bull = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    bear = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    return bull, bear


def ts_pro_flags(df, pcts=(0.5, 1.0, 2.0)):
    """신호 변형별 불리언 시리즈 {이름: Series}. 'BUY …' 와 'SELL …' 로 시작한다."""
    macd, signal, pct = ts_pro_series(df["close"])
    bull, bear = crosses(macd, signal)
    ap = pct.abs()
    out = {"BUY 교차만": bull, "BUY +0선": bull & (macd > 0), f"BUY +퍼센트{PCT_MAIN}": bull & (ap >= PCT_MAIN)}
    for p in pcts:
        out[f"BUY TS Pro(0선+{p}%)"] = bull & (macd > 0) & (ap >= p)
    out["SELL 교차만"] = bear
    out[f"SELL TS Pro(0선+{PCT_MAIN}%)"] = bear & (macd < 0) & (ap >= PCT_MAIN)
    return out


def collect_events(market, flags_fn=ts_pro_flags, horizons=HORIZONS, warmup=WARMUP, gap=GAP, entry="close"):
    """신호 하나당 한 줄: (종목, 변형, 날짜, h일 수익률, h일 초과수익). 같은 종목·변형에서 gap 거래일 안의 재신호는 제외.

    entry="close": 신호가 뜬 날 종가에 진입(신호일 종가 → h거래일 뒤 종가).
    entry="open": 봉이 마감된 뒤 알림을 받고 실제로 사는 경우 — 다음 날 시가에 진입(다음 날 시가 → 신호일 기준 h거래일 뒤 종가).
    기준선(그날 전체 종목 평균)도 같은 방식으로 계산한다.
    """
    closes = pd.DataFrame({c: d["close"] for c, d in market.items()}).sort_index()
    if entry == "open":
        opens = pd.DataFrame({c: d["open"] for c, d in market.items()}).sort_index().reindex(closes.index)
        fwd = {h: closes.shift(-h) / opens.shift(-1) - 1 for h in horizons}
    else:
        fwd = {h: closes.shift(-h) / closes - 1 for h in horizons}
    base = {h: fwd[h].mean(axis=1) for h in horizons}   # 그날 전체 종목 평균 수익률
    rows = []
    for code, df in market.items():
        flags = flags_fn(df)
        r = {h: fwd[h][code].reindex(df.index).to_numpy() for h in horizons}
        b = {h: base[h].reindex(df.index).to_numpy() for h in horizons}
        for name, flag in flags.items():
            last = -10**9
            for i in np.flatnonzero(flag.fillna(False).to_numpy()):
                if i < warmup or i - last < gap or any(np.isnan(r[h][i]) or np.isnan(b[h][i]) for h in horizons):
                    continue
                last = i
                rec = {"code": code, "variant": name, "date": df.index[i]}
                for h in horizons:
                    rec[f"r{h}"], rec[f"ex{h}"] = r[h][i], r[h][i] - b[h][i]
                rows.append(rec)
    return pd.DataFrame(rows), {h: float(np.nanmean(fwd[h].to_numpy())) for h in horizons}


def _t(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) >= 10 and x.std() > 0 else None


def round_trips(market, flags_fn=ts_pro_flags, buy="BUY TS Pro(0선+1.0%)", sell="SELL TS Pro(0선+1.0%)", cost=0.003, warmup=WARMUP):
    """BUY 에 사서 다음 SELL 까지 보유(끝까지 SELL 이 없으면 제외). 같은 기간 전체 종목 평균 수익률과 비교."""
    closes = pd.DataFrame({c: d["close"] for c, d in market.items()}).sort_index()
    trades = []
    for code, df in market.items():
        fl = flags_fn(df)
        b, s = fl[buy].fillna(False).to_numpy(), fl[sell].fillna(False).to_numpy()
        c, i = df["close"].to_numpy(), warmup
        while i < len(df):
            if b[i]:
                j = next((k for k in range(i + 1, len(df)) if s[k]), None)
                if j is None:
                    break
                u = (closes.loc[df.index[j]] / closes.loc[df.index[i]] - 1).mean()
                trades.append({"code": code, "ret": c[j] / c[i] - 1 - cost, "hold": j - i, "ex": c[j] / c[i] - 1 - cost - u})
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(trades)


def _line(label, ev, name, horizons=HORIZONS):
    x = ev[ev["variant"] == name] if len(ev) else ev
    if len(x) == 0:
        return f"  {label:<22} 신호 없음"
    parts = []
    for h in horizons:
        t = _t(x[f"ex{h}"])
        parts.append(f"{h}일 {x[f'ex{h}'].mean():+.2%}p (이긴 {(x[f'ex{h}'] > 0).mean():.0%}, " + ("t 계산 불가" if t is None else f"t={t:+.1f}") + ")")
    return f"  {label:<22} n={len(x):>4} | " + " | ".join(parts)


def verdict(ev, name, h=10, min_n=30):
    x = ev[ev["variant"] == name][f"ex{h}"] if len(ev) else pd.Series(dtype=float)
    if len(x) < min_n:
        return f"표본 부족(n={len(x)} < {min_n}) — 판단 보류"
    t = _t(x)
    if t is not None and t >= 2:
        return f"전체 평균보다 {x.mean():+.2%}p 높음 (t={t:+.1f}, 오차 범위 밖)"
    if t is not None and t <= -2:
        return f"전체 평균보다 {x.mean():+.2%}p 낮음 (t={t:+.1f}, 오차 범위 밖)"
    return f"전체 평균과 구별되지 않음 ({x.mean():+.2%}p, t={0 if t is None else t:+.1f})"


def run_macd_test(market, label="", cost=0.003):
    if not market:
        return "[MACD 신호 시험] 데이터가 없습니다."
    ev, base = collect_events(market)
    d0 = min(d.index.min() for d in market.values())
    d1 = max(d.index.max() for d in market.values())
    lines = [f"[MACD 신호 시험] TS Pro(12·26·9, 일봉) {label} · 종목 {len(market)}개 · {d0:%Y-%m-%d} ~ {d1:%Y-%m-%d}",
             "신호가 뜬 날 종가에 샀다고 보고 5·10거래일 뒤 수익률 − 그날 전체 종목 평균 (같은 종목 10일 안 재신호 제외)",
             "기준선(모든 날 평균): " + " / ".join(f"{h}일 {base[h]:+.2%}" for h in HORIZONS), "", "BUY 신호 (초과수익이 +여야 도움):"]
    for name in [n for n in (ev["variant"].unique() if len(ev) else []) if n.startswith("BUY")]:
        lines.append(_line(name[4:], ev, name))
    lines.append("SELL 신호 (초과수익이 −여야 도움):")
    for name in [n for n in (ev["variant"].unique() if len(ev) else []) if n.startswith("SELL")]:
        lines.append(_line(name[5:], ev, name))
    rt = round_trips(market, cost=cost)
    lines.append(f"\n왕복 매매 (BUY→다음 SELL까지 보유, 비용 {cost:.1%} 가정):")
    if len(rt) < 10:
        lines.append(f"  왕복 {len(rt)}건 — 표본 부족")
    else:
        t = _t(rt["ex"])
        lines.append(f"  {len(rt)}건 | 평균 {rt['ret'].mean():+.2%} (승률 {(rt['ret'] > 0).mean():.0%}) | 평균 보유 {rt['hold'].mean():.0f}거래일 | "
                     f"같은 기간 전체 평균 대비 {rt['ex'].mean():+.2%}p (t={0 if t is None else t:+.1f})")
    lines += ["", "판정(10일 기준): " + verdict(ev, "BUY TS Pro(0선+1.0%)"),
              "※ 지금 유명한 종목만 본 결과라 실제보다 좋게 나올 수 있고, 여러 변형을 함께 보면 우연히 하나쯤 좋게 나올 수 있습니다. 과거 성과가 미래를 보장하지 않습니다."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# 재확인: (1) 앞 구간으로 고르고 마지막 1년으로 확인  (2) 다음 날 시가 진입
def _seg(x):
    if len(x) == 0:
        return "신호 없음"
    out = [f"n={len(x):>4}"]
    for h in HORIZONS:
        t = _t(x[f"ex{h}"])
        out.append(f"{h}일 {x[f'ex{h}'].mean():+.2%}p(t=" + ("–" if t is None else f"{t:+.1f}") + ")")
    return " ".join(out)


def persist_verdict(is_x, oos_x, h=5, min_n=30):
    """학습 구간에서 고른 변형이 확인 구간에서도 유지되는지."""
    if len(oos_x) < min_n:
        return f"확인 구간 표본 부족(n={len(oos_x)} < {min_n}) — 판단 보류"
    m, t = oos_x[f"ex{h}"].mean(), _t(oos_x[f"ex{h}"])
    ts = "–" if t is None else f"{t:+.1f}"
    if m > 0 and t is not None and t >= 2:
        return f"확인 구간에서도 유지됨 ({h}일 {m:+.2%}p, t={ts})"
    if m > 0:
        return f"방향은 같지만 통계적으로 약함 ({h}일 {m:+.2%}p, t={ts}) — 입증되지 않음"
    return f"확인 구간에서 사라지거나 반대 ({h}일 {m:+.2%}p, t={ts})"


def run_macd_robust(market, label="", flags_fn=ts_pro_flags, oos_days=365, min_n=30):
    if not market:
        return "[MACD 재확인] 데이터가 없습니다."
    d0 = min(d.index.min() for d in market.values())
    d1 = max(d.index.max() for d in market.values())
    cutoff = d1 - pd.Timedelta(days=oos_days)
    lines = [f"[MACD 재확인] TS Pro(12·26·9, 일봉) {label} · 종목 {len(market)}개",
             f"학습 구간 {d0:%Y-%m-%d} ~ {cutoff:%Y-%m-%d} / 확인 구간(마지막 {oos_days // 365}년) {cutoff:%Y-%m-%d} ~ {d1:%Y-%m-%d}",
             "초과수익 = 신호 뒤 수익률 − 그날 전체 종목 평균 (같은 종목 10일 안 재신호 제외). 학습 구간에서 5일 t값이 가장 높은 BUY 변형을 고르고, 확인 구간에서 유지되는지 봅니다."]
    for entry, title in (("close", "① 신호일 종가에 진입"), ("open", "② 다음 날 시가에 진입 (알림을 받고 실제로 사는 경우)")):
        ev, base = collect_events(market, flags_fn, entry=entry)
        lines.append(f"\n{title} — 기준선 5일 {base[5]:+.2%} / 10일 {base[10]:+.2%}")
        if len(ev) == 0:
            lines.append("  신호 없음")
            continue
        buys = [n for n in ev["variant"].unique() if n.startswith("BUY")]
        is_ev, oos_ev = ev[ev["date"] <= cutoff], ev[ev["date"] > cutoff]
        cand = []
        for name in buys:
            a, b = is_ev[is_ev["variant"] == name], oos_ev[oos_ev["variant"] == name]
            lines.append(f"  {name[4:]:<18} 학습 {_seg(a)} │ 확인 {_seg(b)}")
            t = _t(a["ex5"]) if len(a) >= min_n else None
            if t is not None:
                cand.append((t, name))
        if cand:
            t, name = max(cand)
            b = oos_ev[oos_ev["variant"] == name]
            lines.append(f"  → 학습에서 고른 변형: {name[4:]} (학습 5일 t={t:+.1f}) → {persist_verdict(is_ev[is_ev['variant'] == name], b, 5, min_n)}")
        else:
            lines.append("  → 학습 구간 표본이 부족해 변형을 고르지 못함")
    lines.append("\n※ 지금 유명한 종목만 본 결과(생존 편향), 변형을 여러 개 비교하므로 우연히 하나쯤 좋게 나올 수 있습니다. 세금·슬리피지는 단순화했고 과거 성과가 미래를 보장하지 않습니다.")
    return "\n".join(lines)
