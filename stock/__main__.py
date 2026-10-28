"""사용: python -m stock --demo | --pykrx | --csv 폴더   (결과 CSV: --out 폴더)"""
import argparse
import contextlib
import io
import os

from stock.analyze import analyze_themes, decide, market_regime
from stock.dart import fetch_summaries
from stock.naver import load_naver
from stock.data import load_csv_dir, load_pykrx, synthetic_market
from stock.themes import THEMES, WATCHLIST
from trader import notify

DISCLAIMER = "※ '흐름'은 현재 지표가 기운 방향일 뿐 내일의 예측이 아니며 검증되지 않았습니다. 참고용 분석이며 투자 권유가 아닙니다. 손절 기준을 먼저 정하고, 과열 종목은 추격하지 마세요."


def main(argv=None):
    ap = argparse.ArgumentParser(description="한국 주식 종목 분석기")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--demo", action="store_true", help="합성 데이터로 시연")
    g.add_argument("--pykrx", action="store_true", help="pykrx 실데이터(pip install pykrx)")
    g.add_argument("--naver", action="store_true", help="네이버 금융 실데이터(해외 서버에서도 가능)")
    g.add_argument("--csv", metavar="DIR", help="<코드>.csv 폴더")
    ap.add_argument("--top", type=int, default=5, help="종목 표시 개수")
    ap.add_argument("--dart", action="store_true", help="DART 공시 확인(환경변수 DART_API_KEY 필요)")
    ap.add_argument("--telegram", action="store_true", help="결과를 텔레그램으로 전송(TELEGRAM_BOT_TOKEN/CHAT_ID)")
    ap.add_argument("--out", help="결과 CSV 저장 폴더")
    a = ap.parse_args(argv)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            run(a)
    except Exception as e:  # 실패도 알려서 조용히 멈추지 않게
        print(buf.getvalue())
        if a.telegram:
            notify.send(f"❌ 종목 분석기 오류: {type(e).__name__}: {str(e)[:200]}")
        raise
    print(buf.getvalue())
    if a.telegram:
        notify.send("📈 오늘의 종목 분석\n" + buf.getvalue())


def run(a):

    if a.demo:
        market = synthetic_market()
    elif a.naver:
        market = load_naver([c for m in THEMES.values() for c in m])
    elif a.pykrx:
        market = load_pykrx([c for m in THEMES.values() for c in m])
    else:
        market = load_csv_dir(a.csv)

    reg = market_regime(market["KOSPI"])
    themes, stocks = analyze_themes(market, THEMES)

    print(f"[시황] {reg['label']}  (KOSPI 20일 {reg['ret20']:+.1%})")
    if reg["points"] <= 1:
        print("  → 지수가 약하면 신규 진입 비중을 줄이세요.")
    n_flow = int(stocks["flow_ratio"].notna().sum())
    print(f"[수급 반영] {n_flow}/{len(stocks)}종목" + ("" if n_flow else "  ※ 수급 없이 가격·거래량만으로 분석됨"))
    print("\n[테마 순위]")
    for i, r in themes.iterrows():
        print(f"{i+1}. {r['theme']:<10} 점수 {r['score']:5.1f} | 강세종목비율 {r['breadth']:.0%} | "
              f"거래대금 {r['value_growth']:+.0%} | 대장 {r['leader']} ({r['leader_stage']})")
    print("\n[선점 후보] 점수 높고 아직 안 오른 종목")
    picks = stocks[stocks["stage"] == "선점 후보"].head(a.top)
    disc = fetch_summaries(list(picks["code"])) if a.dart else {}
    if a.dart and not disc:
        print("  (DART 조회 실패 또는 DART_API_KEY 없음 → 공시 미확인)")
    if picks.empty:
        print("  없음")
    for _, r in picks.iterrows():
        fl = "수급 n/a" if r["flow_ratio"] is None else f"수급 {r['flow_ratio']:+.1%}"
        print(f"  {r['name']}({r['theme']}) 점수 {r['score']} | 20일 {r['ret20']:+.1%} | "
              f"거래량 x{r['vol_ratio']:.1f} | 60일고점 {r['prox60']:.0%} | {fl}"
              + ("" if r["indiv_ratio"] is None else f" | 개인 {r['indiv_ratio']:+.1%}"))
        act, why = decide(r, reg, disc.get(r["code"]))
        print(f"    → 대응: {act}  ({'; '.join(why)})")
        print(f"    → 흐름: {r['bias']} ({r['bias_pts']:+d}점: {', '.join(r['bias_why'])})")
    hot = stocks[stocks["stage"].str.startswith("과열")]
    if len(hot):
        print("\n[과열 — 추격 금지] " + ", ".join(f"{r['name']}(5일 {r['ret5']:+.0%})" for _, r in hot.iterrows()))
    watch = stocks[stocks["code"].isin(WATCHLIST)]
    if len(watch):
        print("\n[관심 종목 현황]")
        wdisc = fetch_summaries(list(watch["code"])) if a.dart else {}
        for _, r in watch.iterrows():
            fl = "수급 n/a" if r["flow_ratio"] is None else f"수급 {r['flow_ratio']:+.1%}"
            ma = "20일선 위" if r["above_ma20"] else "20일선 아래"
            print(f"  {r['name']}({r['theme']}) {r['stage']} · 점수 {r['score']} | 5일 {r['ret5']:+.1%} · 20일 {r['ret20']:+.1%} | "
                  f"{ma} | 거래량 x{r['vol_ratio']:.1f} | 60일고점 {r['prox60']:.0%} | {fl}")
            act, why = decide(r, reg, wdisc.get(r["code"]))
            print(f"    → 대응: {act}  ({'; '.join(why)})")
            print(f"    → 흐름: {r['bias']} ({r['bias_pts']:+d}점: {', '.join(r['bias_why'])})")
    print("\n" + DISCLAIMER)

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        themes.to_csv(os.path.join(a.out, "themes.csv"), index=False, encoding="utf-8-sig")
        stocks.to_csv(os.path.join(a.out, "stocks.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
