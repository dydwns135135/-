"""종목/테마 점수화. 기준은 모두 객관적 수치이며 임계값은 상수로 조정 가능."""
import numpy as np
import pandas as pd

OVERHEAT_RET5 = 0.25      # 5일 +25% 이상이면 과열
OVERHEAT_EXT = 1.25       # 종가가 20일선 대비 25% 이상 위면 과열
EARLY_RET20 = 0.20        # 선점 후보: 20일 상승률 이 값 미만
PICK_SCORE = 60           # 선점 후보 최소 점수


def market_regime(idx_df):
    """시황: 지수가 20/60일선 위인지, 20일선 기울기."""
    c = idx_df["close"]
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    above20, above60 = c.iloc[-1] > ma20.iloc[-1], c.iloc[-1] > ma60.iloc[-1]
    rising = ma20.iloc[-1] > ma20.iloc[-6]
    pts = int(above20) + int(above60) + int(rising)
    label = {3: "강세(적극)", 2: "중립(선별)", 1: "약세(보수)", 0: "하락(관망)"}[pts]
    return {"label": label, "points": pts, "ret20": c.iloc[-1] / c.iloc[-21] - 1}


def lean(above20, aligned, slope_up, ret5, vol_ratio, both_buy, both_sell, overheated):
    """현재 흐름이 상승/하락 어느 쪽으로 기울었는지 요약(예측 아님). 6개 신호 각 ±1, 합 ±3 이상이면 한쪽 우세."""
    sig = [("종가>20일선" if above20 else "종가<20일선", 1 if above20 else -1),
           ("20일선>60일선" if aligned else "20일선<60일선", 1 if aligned else -1),
           ("20일선 상승" if slope_up else "20일선 하락", 1 if slope_up else -1),
           (f"5일 {ret5:+.1%}", 1 if ret5 > 0 else -1 if ret5 < 0 else 0)]
    if both_buy or both_sell:
        sig.append(("외국인·기관 동반 순매수" if both_buy else "외국인·기관 동반 순매도", 1 if both_buy else -1))
    if vol_ratio > 1.3 and ret5 != 0:
        up = ret5 > 0
        sig.append((f"거래량 x{vol_ratio:.1f} 동반 " + ("상승" if up else "하락"), 1 if up else -1))
    pts = sum(v for _, v in sig)
    label = "상승 우세" if pts >= 3 else "하락 우세" if pts <= -3 else "중립"
    if overheated and pts > 0:
        label = "상승 흐름(과열·조정 위험)"
    return label, pts, [n for n, _ in sig]


def analyze_stock(df):
    """한 종목의 지표·점수. 점수 100점 = 추세25 + 돌파근접25 + 거래량20 + 수축10 + 수급20."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    if len(df) < 70:
        return None
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    last = c.iloc[-1]

    trend = 0.0
    trend += 10 if last > ma20.iloc[-1] else 0
    trend += 8 if ma20.iloc[-1] > ma60.iloc[-1] else 0
    trend += 7 if ma20.iloc[-1] > ma20.iloc[-6] else 0

    hi60 = h.iloc[-61:-1].max()
    prox = last / hi60
    near = float(np.clip((prox - 0.85) / 0.15, 0, 1)) * 20 + (5 if last >= hi60 else 0)

    vol_ratio = v.iloc[-3:].mean() / v.iloc[-23:-3].mean()
    volume = float(np.clip((vol_ratio - 1) / 1.5, 0, 1)) * 20

    rng10 = (h.iloc[-10:].max() - l.iloc[-10:].min()) / last
    rng60 = (h.iloc[-60:].max() - l.iloc[-60:].min()) / last
    squeeze = float(np.clip(1 - rng10 / (rng60 * 0.5), 0, 1)) * 10

    flow_ratio = None
    flow = 0.0
    if {"foreign", "inst"} <= set(df.columns):
        val5 = (c * v).iloc[-5:].sum()
        flow_ratio = (df["foreign"].iloc[-5:].sum() + df["inst"].iloc[-5:].sum()) / val5
        both = df["foreign"].iloc[-5:].sum() > 0 and df["inst"].iloc[-5:].sum() > 0
        flow = float(np.clip(flow_ratio / 0.08, 0, 1)) * 15 + (5 if both else 0)
    max_pts = 100 if flow_ratio is not None else 80
    score = (trend + near + volume + squeeze + flow) / max_pts * 100

    ret5, ret20 = last / c.iloc[-6] - 1, last / c.iloc[-21] - 1
    ext = last / ma20.iloc[-1]
    overheated = ret5 >= OVERHEAT_RET5 or ext >= OVERHEAT_EXT
    if overheated:
        stage = "과열(추격금지)"
    elif score >= PICK_SCORE and ret20 < EARLY_RET20:
        stage = "선점 후보"
    elif score >= PICK_SCORE:
        stage = "상승 진행(눌림 대기)"
    else:
        stage = "관망"
    indiv_ratio = None
    if "individual" in df.columns:
        indiv_ratio = df["individual"].iloc[-5:].sum() / (c * v).iloc[-5:].sum()
    both_buy = flow_ratio is not None and df["foreign"].iloc[-5:].sum() > 0 and df["inst"].iloc[-5:].sum() > 0
    both_sell = flow_ratio is not None and df["foreign"].iloc[-5:].sum() < 0 and df["inst"].iloc[-5:].sum() < 0
    bias, bias_pts, bias_why = lean(
        above20=last > ma20.iloc[-1], aligned=ma20.iloc[-1] > ma60.iloc[-1], slope_up=ma20.iloc[-1] > ma20.iloc[-6],
        ret5=ret5, vol_ratio=vol_ratio, both_buy=both_buy, both_sell=both_sell, overheated=overheated)
    return {"bias": bias, "bias_pts": bias_pts, "bias_why": bias_why,
            "score": round(score, 1), "stage": stage, "indiv_ratio": indiv_ratio, "both_buy": bool(both_buy), "ret5": ret5, "ret20": ret20, "ext": ext,
            "vol_ratio": vol_ratio, "prox60": prox, "flow_ratio": flow_ratio,
            "above_ma20": bool(last > ma20.iloc[-1]), "value5": float((c * v).iloc[-5:].sum()),
            "value_prev": float((c * v).iloc[-25:-5].sum() / 4)}


def analyze_themes(market, themes):
    """테마 점수 = 상위3종목 평균 점수 60% + 20일선 위 종목 비율 25% + 거래대금 증가 15%. 대장주 = 테마 내 최고점."""
    rows, stocks = [], []
    for theme, members in themes.items():
        res = {}
        for code, name in members.items():
            if code in market:
                r = analyze_stock(market[code])
                if r:
                    res[code] = {"code": code, "name": name, "theme": theme, **r}
        if not res:
            continue
        stocks.extend(res.values())
        sc = sorted((r["score"] for r in res.values()), reverse=True)
        top3 = float(np.mean(sc[:3]))
        breadth = float(np.mean([r["above_ma20"] for r in res.values()]))
        v5 = sum(r["value5"] for r in res.values())
        vp = sum(r["value_prev"] for r in res.values())
        vgrow = v5 / vp - 1 if vp else 0
        tscore = top3 * 0.6 + breadth * 100 * 0.25 + float(np.clip(vgrow / 1.0, 0, 1)) * 100 * 0.15
        leader = max(res.values(), key=lambda r: r["score"])
        rows.append({"theme": theme, "score": round(tscore, 1), "breadth": breadth, "value_growth": vgrow,
                     "leader": leader["name"], "leader_score": leader["score"], "leader_stage": leader["stage"]})
    t = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    s = pd.DataFrame(stocks).sort_values("score", ascending=False).reset_index(drop=True)
    return t, s


def decide(row, regime, disclosure=None, kr=True):
    """대응 방향: 즉시 진입 / 분할 진입 / 관망(패스) + 사유. disclosure=dart.summarize 결과(없으면 미확인)."""
    why = []
    if row["stage"].startswith("과열"):
        return "관망(패스)", ["과열: 이미 급등, 추격 금지"]
    if disclosure and disclosure["negative"]:
        return "관망(패스)", ["악재성 공시: " + disclosure["negative"][0]]
    if row["stage"] == "관망":
        return "관망(패스)", ["점수 부족"]
    level = 2 if (row["score"] >= 75 and row["ret20"] < 0.12 and row["both_buy"]) else 1
    if not kr:
        level = 1  # 미국: 수급·공시 정보가 없어 분할 진입까지만
    why.append(f"점수 {row['score']}" + (", 외국인·기관 동반 순매수" if row["both_buy"] else ""))
    if row["stage"] != "선점 후보":
        level = min(level, 1)
        why.append("이미 상승 진행 중 → 눌림 때만")
    if not kr:
        why.append("미국: 수급·공시 미반영 → 분할 진입까지만")
    elif disclosure is None:
        level = min(level, 1)
        why.append("공시 미확인(--dart 로 확인 전엔 분할 진입까지만)")
    elif disclosure["positive"]:
        why.append("호재성 공시: " + disclosure["positive"][0])
    else:
        level = min(level, 1)
        why.append("최근 3개월 뚜렷한 호재 공시 없음")
    if regime["points"] <= 1:
        level -= 1
        why.append(f"시황 {regime['label']} → 한 단계 보수적")
    return {2: "즉시 진입", 1: "분할 진입", 0: "관망(패스)"}.get(max(level, 0), "관망(패스)"), why
