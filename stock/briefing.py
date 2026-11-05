"""매일 브리핑: 국내(아침 8시 30분)와 미국(오후 5시)에 '오늘 눈여겨볼 곳'을 알아서 찾아 텔레그램으로 보낸다. 주문 없음.
python -m stock.briefing kr <stock-data 폴더> [--telegram]
python -m stock.briefing us <stock-data 폴더> [--telegram]

국내: 어제 장 마감 뒤 저장된 시장 전체 훑기(stock-data 의 scan/)에서 테마·관찰 후보·과열 종목을 뽑고, 후보와 관심 종목의 최근 공시(DART)를 붙인다.
미국: 기본 목록(US_THEMES)과 확장 목록(US_DISCOVERY)을 합쳐 어젯밤 마감 기준으로 테마·관찰 후보·전일 등락 상위를 뽑는다.
점수·단계는 시험에서 이후 수익률과의 연결이 입증되지 않았다 — '어디가 활발한가'를 보는 용도다."""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

SHOW = {"선점 후보": "관찰 후보"}
NOTE = "※ 참고용 요약이며 투자 권유가 아닙니다. 점수·단계가 이후 수익을 예측한다는 근거는 시험에서 확인되지 않았습니다. 사기 전에 금액과 손절 기준을 먼저 정하세요."
LIMIT = 3800  # 텔레그램 메시지 길이 여유


def _stage(s: str) -> str:
    return SHOW.get(s, s)


def _paper_lines(path: str) -> list[str]:
    from stock.paper import load_ledger
    out = []
    for name, label in (("ledger.csv", "한국"), ("ledger_us.csv", "미국")):
        L = load_ledger(path, name)
        act = L[L["status"].isin(["pending", "open"])] if len(L) else L
        done = L[L["status"] == "closed"] if len(L) else L
        if not len(act) and not len(done):
            continue
        line = f"· {label}: 진행 {len(act)}건"
        if len(act):
            line += " (" + ", ".join(f"{r['name']}{'' if r['status'] == 'open' else '·진입 전'}" for _, r in act.iterrows()) + ")"
        if len(done):
            line += f" · 청산 {len(done)}건 평균 {done['ret_net'].mean():+.2%}"
        out.append(line)
    return out


def _fit(text: str) -> str:
    return text if len(text) <= LIMIT else text[:LIMIT - 20] + "\n…(길어서 줄임)"


def kr_briefing(path: str, disclosures=None) -> str:
    """path = stock-data 폴더. disclosures(codes) → {code: [(날짜, 제목, 표시), ...]} 를 바꿔 끼울 수 있다(시험·DART 없음)."""
    from stock.snapshot import load_scan_snapshots
    snap = load_scan_snapshots(path)
    if snap.empty:
        return "[국내 브리핑] 저장된 훑기 데이터가 아직 없습니다."
    d = snap["date"].max()
    cur = snap[snap["date"] == d]
    lines = [f"📋 국내 브리핑 — 기준: {d} 장 마감 (훑기 {len(cur)}종목)"]
    tf = os.path.join(path, "scan", "themes", f"{d}.csv")
    if os.path.exists(tf):
        th = pd.read_csv(tf).head(5)
        lines.append("\n[돈이 몰린 테마 상위 5]")
        for i, r in th.iterrows():
            lines.append(f"{i + 1}. {r['theme']} · 점수 {r['score']:.0f} · 강세 {r['breadth']:.0%} · 거래대금 {r['value_growth']:+.0%} · 대장 {r['leader']}")
    up = float((cur["ret20"] > 0).mean())
    lines.append(f"\n[분위기] 훑은 종목 중 20일 수익률 플러스 {up:.0%}")
    cand = cur[cur["stage"] == "선점 후보"].sort_values("score", ascending=False).head(5)
    lines.append("\n[관찰 후보 — 점수는 높지만 아직 덜 오른 종목]" if len(cand) else "\n[관찰 후보] 없음")
    for _, r in cand.iterrows():
        flow = "" if pd.isna(r["flow_ratio"]) else f" · 수급 {r['flow_ratio']:+.1%}" + (" 외국인·기관 동반 순매수" if r["both_buy"] else "")
        lines.append(f"- {r['name']}({r['best_theme']}) 점수 {r['score']:.0f} · 20일 {r['ret20']:+.1%} · 거래량 x{r['vol_ratio']:.1f}{flow}")
    hot = cur[cur["stage"].str.startswith("과열")].sort_values("ret5", ascending=False).head(3)
    if len(hot):
        lines.append("\n[과열 — 급등 직후, 이후 방향은 불확실] " + ", ".join(f"{r['name']}(5일 {r['ret5']:+.0%})" for _, r in hot.iterrows()))
    codes = list(cand["code"])
    lk = None
    if os.path.isdir(os.path.join(path, "kr")):
        from stock.snapshot import load_snapshots
        ks = load_snapshots(path)
        if len(ks):
            lk = ks[ks["date"] == ks["date"].max()]
            from stock.themes import WATCHLIST
            w = lk[lk["code"].isin(WATCHLIST)]
            if len(w):
                lines.append("\n[관심 종목]")
                for _, r in w.iterrows():
                    lines.append(f"- {r['name']} {_stage(r['stage'])} · 점수 {r['score']:.0f} · 5일 {r['ret5']:+.1%} · 20일 {r['ret20']:+.1%}")
                codes += [c for c in w["code"] if c not in codes]
    if codes:
        lines.append("\n[최근 공시 (7일)]")
        try:
            disc = (disclosures or _dart_titles)(codes)
        except Exception as e:  # 공시 실패가 브리핑 전체를 막지 않게
            disc, err = {}, f"{type(e).__name__} {str(e)[:60]}"
        else:
            err = ""
        names = dict(zip(cur["code"], cur["name"]))
        if lk is not None:
            names.update(dict(zip(lk["code"], lk["name"])))
        shown = False
        for c in codes:
            items = disc.get(c) or []
            flag = [x for x in items if x[2]]
            if flag:
                shown = True
                lines.append(f"- {names.get(c, c)}: " + "; ".join(f"{x[0][5:]} {x[1]} {x[2]}" for x in flag[:3]))
            elif items:
                lines.append(f"- {names.get(c, c)}: {len(items)}건 (주의 키워드 없음)")
                shown = True
        if not shown:
            lines.append("- 확인된 공시 없음" + (f" ({err})" if err else ""))
    pl = _paper_lines(path)
    if pl:
        lines += ["\n[모의투자 현황 — 실제 매매 아님]"] + pl
    lines += ["", NOTE]
    return _fit("\n".join(lines))


def _dart_titles(codes, days=7):
    """종목코드 → [(접수일 YYYY-MM-DD, 제목, '＋호재성'|'－악재성'|'')]. 키가 없으면 빈 결과."""
    from stock import dart
    key = os.environ.get("DART_API_KEY")
    if not key:
        return {}
    cc = dart.corp_codes(key)
    out = {}
    for c in codes:
        if c not in cc:
            continue
        try:
            items = dart.recent_items(key, cc[c], days)
        except Exception:
            continue
        mark = {1: "＋호재성", -1: "－악재성", 0: ""}
        out[c] = [(f"{d[:4]}-{d[4:6]}-{d[6:]}", t, mark[dart.classify_title(t)]) for d, t in items]
    return out


def us_briefing(market, path: str | None = None, log=print) -> str:
    """market = load_yahoo 결과(SPX 포함). 기본 목록과 확장 목록을 합쳐 분석한다."""
    from stock.analyze import analyze_themes, market_regime
    from stock.us_themes import US_DISCOVERY, US_THEMES, US_WATCHLIST
    themes_map = {**US_THEMES, **US_DISCOVERY}
    t, s = analyze_themes(market, themes_map)
    reg = market_regime(market["SPX"])
    idx = market["SPX"]["close"]
    last = idx.index[-1]
    got = sum(len(m) for m in themes_map.values())
    have = len(s)
    lines = [f"📋 미국 브리핑 — 기준: {last:%Y-%m-%d} 미국장 마감 (분석 {have}/{got}종목)",
             f"[시황] {reg['label']} · S&P500 20일 {reg['ret20']:+.1%} · 전일 {idx.iloc[-1] / idx.iloc[-2] - 1:+.1%}"]
    lines.append("\n[돈이 몰린 테마 상위 5]")
    for i, r in t.head(5).iterrows():
        lines.append(f"{i + 1}. {r['theme']} · 점수 {r['score']:.0f} · 강세 {r['breadth']:.0%} · 거래대금 {r['value_growth']:+.0%} · 대장 {r['leader']}")
    cand = s[s["stage"] == "선점 후보"].head(5)
    lines.append("\n[관찰 후보 — 점수는 높지만 아직 덜 오른 종목]" if len(cand) else "\n[관찰 후보] 없음")
    for _, r in cand.iterrows():
        lines.append(f"- {r['name']}({r['theme']}) 점수 {r['score']:.0f} · 20일 {r['ret20']:+.1%} · 거래량 x{r['vol_ratio']:.1f} · 60일 고점 {r['prox60']:.0%}")
    hot = s[s["stage"].str.startswith("과열")].sort_values("ret5", ascending=False).head(3)
    if len(hot):
        lines.append("\n[과열 — 급등 직후, 이후 방향은 불확실] " + ", ".join(f"{r['name']}(5일 {r['ret5']:+.0%})" for _, r in hot.iterrows()))
    chg = {}
    names = {c: n for m in themes_map.values() for c, n in m.items()}
    for c in s["code"]:
        cl = market[c]["close"]
        if len(cl) > 1:
            chg[c] = float(cl.iloc[-1] / cl.iloc[-2] - 1)
    if chg:
        rank = sorted(chg.items(), key=lambda x: x[1], reverse=True)
        lines.append("\n[전일 등락 — 상승 상위] " + ", ".join(f"{names[c]} {v:+.1%}" for c, v in rank[:4]))
        lines.append("[전일 등락 — 하락 상위] " + ", ".join(f"{names[c]} {v:+.1%}" for c, v in rank[-4:][::-1]))
    w = s[s["code"].isin(US_WATCHLIST)]
    if len(w):
        lines.append("\n[관심 종목]")
        for _, r in w.iterrows():
            lines.append(f"- {r['name']} {_stage(r['stage'])} · 점수 {r['score']:.0f} · 5일 {r['ret5']:+.1%} · 20일 {r['ret20']:+.1%} · 60일 고점 {r['prox60']:.0%}")
    missing = [names[c] for c in names if c not in set(s["code"])]
    if missing:
        lines.append(f"\n※ 못 받은 종목 {len(missing)}개: " + ", ".join(missing[:8]))
    if path:
        pl = _paper_lines(path)
        if pl:
            lines += ["\n[모의투자 현황 — 실제 매매 아님]"] + pl
    lines += ["", NOTE]
    return _fit("\n".join(lines))


def run_us(path: str | None, log=print) -> str:
    from stock.paper import trim_partial
    from stock.us_themes import US_DISCOVERY, US_THEMES
    from stock.yahoo import load_yahoo
    codes = sorted({c for m in {**US_THEMES, **US_DISCOVERY}.values() for c in m})
    market = trim_partial(load_yahoo(codes, delay=0.5, log=lambda m: None), "America/New_York", 17)
    return us_briefing(market, path, log)


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    which = a[0] if a else "kr"
    path = a[1] if len(a) > 1 and not a[1].startswith("--") else "."
    text = run_us(path if os.path.isdir(path) else None) if which == "us" else kr_briefing(path)
    print(text)
    if "--telegram" in a:
        from trader import notify
        notify.send(text)


if __name__ == "__main__":
    main()
