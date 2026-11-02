"""모의투자: 미리 정해 둔 규칙대로 가상 매매를 매일 기록해 실제 성과 데이터를 쌓는다(주문 없음, 실제 돈 없음).

규칙(운영 중 바꾸지 않는다 — 결과를 보고 규칙을 고치면 시험이 아니라 끼워 맞추기가 된다):
  - 신호: 훑기 스냅샷(scan/날짜.csv)의 '선점 후보'(화면 표시는 관찰 후보) 중 점수 상위 MAX_NEW개, 이미 보유·대기 중인 종목 제외, 동시 MAX_ACTIVE개까지
  - 진입: 신호 다음 거래일 시가 / 청산: 신호일로부터 HOLD 거래일째 종가(백테스트의 ro5 와 같은 정의)
  - 비용: 왕복 COST(수수료·거래세·슬리피지 단순화) 차감
  - 비교: 같은 기간 KOSPI(신호일 종가→청산일 종가). 훑은 종목 전체 평균과의 비교는 stock-backtest 의 '쌓은 데이터 시험'이 한다
"""
import os
import sys

import numpy as np
import pandas as pd

HOLD = 5
MAX_NEW = 3
MAX_ACTIVE = 9
COST = 0.003
COLS = ["code", "name", "signal_date", "signal_close", "score", "status", "entry_date", "entry_px", "exit_date", "exit_px", "ret_net", "kospi_ret"]
MIN_FOR_VERDICT = 60  # 이 건수 전에는 평균이 우연인지 가를 수 없다


def trim_partial(market, tz, close_hour, now=None):
    """장이 끝나기 전에 받은 '오늘' 봉은 아직 완성되지 않았으므로 버린다(신호·진입·청산이 장중 값으로 만들어지는 것을 막는다).

    tz 현지 시각이 close_hour 이전이면 오늘 날짜 이후의 봉을 모두 제외한 새 dict 를 돌려준다.
    """
    now = now if now is not None else pd.Timestamp.now(tz=tz)
    if now.hour >= close_hour:
        return market
    today = pd.Timestamp(now.date())
    out = {}
    for k, df in market.items():
        d = df[df.index < today]
        d.attrs = dict(df.attrs)
        out[k] = d
    return out


def load_ledger(path, name="ledger.csv"):
    f = os.path.join(path, "paper", name)
    if not os.path.exists(f):
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(f, dtype={"code": str})


def save_ledger(path, ledger, name="ledger.csv"):
    os.makedirs(os.path.join(path, "paper"), exist_ok=True)
    ledger[COLS].to_csv(os.path.join(path, "paper", name), index=False, encoding="utf-8-sig")


def step(ledger, scan, prices, bench="KOSPI"):
    """ledger 를 한 걸음 진행: 대기→진입, 보유→청산, 새 신호 추가. 가격은 prices[code] (open/close 열, 날짜 인덱스)와 비교 지수 prices[bench].

    scan 은 가장 최근 훑기 스냅샷(date, code, name, stage, score, close 열). 새로 만든 표와 이번에 일어난 일(events)을 돌려준다.
    """
    L = ledger.copy()
    for c in ("signal_date", "entry_date", "exit_date"):
        L[c] = L[c].astype("object")
    ev = {"entered": [], "closed": [], "new": [], "dropped": []}
    kospi = prices.get(bench)
    if kospi is not None and len(kospi):  # 완성된 마지막 봉보다 늦은 날짜로 만들어진 대기 신호(장중 값)는 취소하고 다시 만든다
        late = (L["status"] == "pending") & (pd.to_datetime(L["signal_date"]) > kospi.index[-1])
        ev["dropped"] = list(L.loc[late, "name"])
        L = L[~late].reset_index(drop=True)
    for i, r in L.iterrows():
        px = prices.get(r["code"])
        if px is None or r["status"] not in ("pending", "open"):
            continue
        after = px.index[px.index > pd.Timestamp(r["signal_date"])]
        if r["status"] == "pending" and len(after):
            o = float(px.loc[after[0], "open"])
            if not np.isfinite(o) or o <= 0:
                L.loc[i, "status"] = "skip"
                continue
            L.loc[i, ["status", "entry_date", "entry_px"]] = ["open", f"{after[0]:%Y-%m-%d}", o]
            ev["entered"].append(r["name"])
            r = L.loc[i]
        if r["status"] == "open" and len(after) >= HOLD:
            xd = after[HOLD - 1]
            xp = float(px.loc[xd, "close"])
            ret = xp / float(r["entry_px"]) - 1 - COST
            kr = np.nan
            if kospi is not None:
                k0 = kospi["close"][kospi.index <= pd.Timestamp(r["signal_date"])]
                if len(k0) and xd in kospi.index:
                    kr = float(kospi.loc[xd, "close"] / k0.iloc[-1] - 1)
            L.loc[i, ["status", "exit_date", "exit_px", "ret_net", "kospi_ret"]] = ["closed", f"{xd:%Y-%m-%d}", xp, ret, kr]
            ev["closed"].append((r["name"], ret, kr))
    if len(scan):
        d = scan["date"].max()
        have = set(zip(L["signal_date"], L["code"]))
        active = int(L["status"].isin(["pending", "open"]).sum())
        held = set(L.loc[L["status"].isin(["pending", "open"]), "code"])
        cand = scan[(scan["date"] == d) & (scan["stage"] == "선점 후보")].sort_values("score", ascending=False)
        n = int((L["signal_date"] == d).sum())  # 같은 날 다시 돌려도 그날 신호는 MAX_NEW개를 넘지 않는다
        for _, r in cand.iterrows():
            if n >= MAX_NEW or active >= MAX_ACTIVE:
                break
            if r["code"] in held or (d, r["code"]) in have:
                continue
            L.loc[len(L)] = [r["code"], r["name"], d, float(r["close"]), float(r["score"]), "pending", None, np.nan, None, np.nan, np.nan, np.nan]
            held.add(r["code"])
            active += 1
            n += 1
            ev["new"].append((r["name"], float(r["score"])))
    return L, ev


def stats(ledger):
    c = ledger[ledger["status"] == "closed"].copy()
    c["ex"] = c["ret_net"] - c["kospi_ret"]
    out = {"n": len(c), "open": int(ledger["status"].isin(["pending", "open"]).sum())}
    if len(c):
        out.update(mean=float(c["ret_net"].mean()), win=float((c["ret_net"] > 0).mean()))
        e = c["ex"].dropna()
        out["ex_mean"] = float(e.mean()) if len(e) else np.nan
        out["t"] = float(e.mean() / (e.std() / np.sqrt(len(e)))) if len(e) >= 10 and e.std() > 0 else None
    return out


def report(ledger, ev, market="한국", bench_name="KOSPI"):
    s = stats(ledger)
    lines = [f"[모의투자·{market}] 규칙: 관찰 후보 상위 3개를 다음 날 시가에 가상 매수, 5거래일째 종가에 가상 매도(왕복 비용 0.3% 차감). 실제 매매·주문 없음."]
    if ev["new"]:
        lines.append("오늘 새 신호(내일 시가에 가상 진입): " + ", ".join(f"{n}(점수 {sc:.0f})" for n, sc in ev["new"]))
    if ev["entered"]:
        lines.append("가상 진입: " + ", ".join(ev["entered"]))
    for n, r, k in ev["closed"]:
        lines.append(f"가상 청산: {n} {r:+.1%}" + ("" if pd.isna(k) else f" (같은 기간 {bench_name} {k:+.1%})"))
    if ev.get("dropped"):
        lines.append("장중 값으로 잘못 만든 신호 취소: " + ", ".join(ev["dropped"]))
    if not (ev["new"] or ev["entered"] or ev["closed"] or ev.get("dropped")):
        lines.append("오늘은 새 신호·진입·청산이 없습니다.")
    lines.append(f"누적: 청산 {s['n']}건 · 진행 중 {s['open']}건")
    if s["n"]:
        lines.append(f"  평균 {s['mean']:+.2%}(비용 반영) · 수익 낸 비율 {s['win']:.0%}" + ("" if pd.isna(s.get("ex_mean", np.nan)) else f" · {bench_name} 대비 평균 {s['ex_mean']:+.2%}p"))
        if s["n"] < MIN_FOR_VERDICT:
            lines.append(f"  ※ {s['n']}건은 아직 적어 평균이 우연일 수 있습니다. 최소 {MIN_FOR_VERDICT}건 이상 쌓인 뒤 판단하세요.")
        elif s.get("t") is not None:
            lines.append(f"  {bench_name} 대비 t값 {s['t']:+.1f} (절댓값 2 이상이어야 우연이 아니라고 볼 수 있음, 같은 날 신호끼리는 서로 비슷해 실제 근거는 더 약함)")
    return "\n".join(lines)


def run_us(path, log=print):
    """미국: 미국 관심 테마(us_themes)를 분석해 같은 규칙으로 가상 매매한다. 장부 paper/ledger_us.csv, 비교 지수 S&P500."""
    from stock.analyze import analyze_themes
    from stock.snapshot import build_snapshot
    from stock.us_themes import US_THEMES
    from stock.yahoo import load_yahoo
    market = load_yahoo([c for m in US_THEMES.values() for c in m], log=log)
    market = trim_partial(market, "America/New_York", 17)  # 미국 장 마감(동부 16:00) 이후에 만든 봉만 쓴다
    _, stocks = analyze_themes(market, US_THEMES)
    snap = build_snapshot(stocks.drop_duplicates("code"), market)
    last = snap[snap["date"] == snap["date"].max()]
    L = load_ledger(path, "ledger_us.csv")
    L2, ev = step(L, last, market, bench="SPX")
    save_ledger(path, L2, "ledger_us.csv")
    return report(L2, ev, "미국", "S&P500")


def run(path, log=print):
    from stock.naver import load_naver
    from stock.snapshot import load_scan_snapshots
    scan = load_scan_snapshots(path)
    if scan.empty:
        return "[모의투자] 훑기 스냅샷이 아직 없어 신호를 만들 수 없습니다."
    last = scan[scan["date"] == scan["date"].max()]
    L = load_ledger(path)
    need = sorted(set(L.loc[L["status"].isin(["pending", "open"]), "code"]) | set(last.loc[last["stage"] == "선점 후보", "code"]))
    prices = trim_partial(load_naver(need, days=40, with_flow=False, log=log), "Asia/Seoul", 16) if need else {}
    L2, ev = step(L, last, prices)
    save_ledger(path, L2)
    return report(L2, ev)


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "."
    us = "--us" in sys.argv
    out = run_us(p) if us else run(p)
    print(out)
    if "--telegram" in sys.argv:
        from trader import notify
        notify.send(("🧪 모의투자(미국)\n" if us else "🧪 모의투자\n") + out)
