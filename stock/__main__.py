"""사용: python -m stock --demo | --pykrx | --csv 폴더   (결과 CSV: --out 폴더)"""
import argparse
import contextlib
import io
import os

import pandas as pd

from stock.analyze import analyze_themes, decide, market_regime
from stock.dart import fetch_summaries
from stock.naver import load_naver
from stock.data import load_csv_dir, load_pykrx, synthetic_market
from stock.themes import THEMES, WATCHLIST
from stock.us_themes import US_INDEX, US_THEMES, US_WATCHLIST
from stock.yahoo import load_yahoo
from trader import notify

DISCLAIMER = ("※ 점수·흐름·관찰 우선순위는 현재 지표를 요약한 것일 뿐 내일의 예측이 아닙니다. 과거 3년 시험에서 이 점수가 기준선보다 "
              "나은 성과를 냈다는 근거는 확인되지 않았습니다(매주 토요일 '성과 확인' 참고). 참고용이며 투자 권유가 아닙니다. "
              "매수 전 손절 기준을 먼저 정하세요.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="한국 주식 종목 분석기")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--demo", action="store_true", help="합성 데이터로 시연")
    g.add_argument("--pykrx", action="store_true", help="pykrx 실데이터(pip install pykrx)")
    g.add_argument("--naver", action="store_true", help="네이버 금융 실데이터(해외 서버에서도 가능)")
    g.add_argument("--yahoo", action="store_true", help="미국 주식 실데이터(야후 파이낸스)")
    g.add_argument("--csv", metavar="DIR", help="<코드>.csv 폴더")
    ap.add_argument("--backtest", action="store_true", help="성과 확인: 과거 신호 이후 5·10일 수익률 평가(3년치 데이터)")
    ap.add_argument("--evaluate", metavar="DIR", help="쌓아 둔 한국 수급 스냅샷(DIR/kr/*.csv)으로 수급 버전 시험")
    ap.add_argument("--scan", action="store_true", help="한국 시장 훑기: 네이버 테마에서 움직이는 테마·종목을 자동 수집(--naver 불필요)")
    ap.add_argument("--us", action="store_true", help="미국 테마로 분석(--demo 와 함께 시험용)")
    ap.add_argument("--top", type=int, default=5, help="종목 표시 개수")
    ap.add_argument("--dart", action="store_true", help="DART 공시 확인(환경변수 DART_API_KEY 필요)")
    ap.add_argument("--telegram", action="store_true", help="결과를 텔레그램으로 전송(TELEGRAM_BOT_TOKEN/CHAT_ID)")
    ap.add_argument("--out", help="결과 CSV 저장 폴더")
    a = ap.parse_args(argv)
    if a.scan:
        if a.demo or a.pykrx or a.yahoo or a.csv or a.us:
            ap.error("--scan 은 한국 네이버 데이터만 지원합니다(--demo/--pykrx/--yahoo/--csv/--us 와 함께 쓸 수 없음)")
        a.naver = True
    elif not (a.demo or a.pykrx or a.naver or a.yahoo or a.csv):
        ap.error("데이터 출처를 고르세요: --demo | --pykrx | --naver | --yahoo | --csv DIR | --scan")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            run(a)
    except Exception as e:  # 실패도 알려서 조용히 멈추지 않게
        print(buf.getvalue())
        if a.telegram:
            notify.send(f"❌ {'미국 ' if (a.yahoo or a.us) else ''}종목 분석기 오류: {type(e).__name__}: {str(e)[:200]}")
        raise
    print(buf.getvalue())
    if a.telegram:
        head = "🇺🇸 미국" if (a.yahoo or a.us) else "🇰🇷 한국"
        notify.send(("🔎 한국 시장 훑기\n" if a.scan else f"📊 {head} 성과 확인\n" if a.backtest else f"📊 {head} 쌓은 데이터 시험(수급·훑기)\n" if a.evaluate else ("🇺🇸 오늘의 미국 종목 분석\n" if (a.yahoo or a.us) else "📈 오늘의 종목 분석\n")) + buf.getvalue())


def run(a):
    us = a.yahoo or a.us
    themes_map, watchlist, idx_key, idx_name = (US_THEMES, US_WATCHLIST, US_INDEX, "S&P500") if us else (THEMES, WATCHLIST, "KOSPI", "KOSPI")
    scan_stats = None
    if a.scan:
        from stock.scan import build_scan
        themes_map, scan_stats = build_scan(log=lambda m: None)
        codes = sorted({c for m in themes_map.values() for c in m})
        market = load_naver(codes, log=lambda m: None)  # 개별 실패는 개수만 집계(텔레그램 길이 제한)
        scan_stats["failed"] = len(codes) - (len(market) - 1)
    elif a.yahoo:
        market = load_yahoo([c for m in US_THEMES.values() for c in m], long=a.backtest)
    elif a.demo:
        n = 800 if (a.backtest or a.evaluate) else 260
        market = synthetic_market(n=n, themes=US_THEMES, index=US_INDEX, flow=False) if us else synthetic_market(n=n)
    elif a.naver:
        codes = [c for m in THEMES.values() for c in m]
        if a.evaluate:  # 쌓아 둔 훑기 스냅샷에 나온 종목 가격도 필요하다
            from stock.snapshot import scan_codes
            codes = sorted(set(codes) | set(scan_codes(a.evaluate)))
        market = load_naver(codes, days=800 if (a.backtest or a.evaluate) else 250, with_flow=not (a.backtest or a.evaluate))
    elif a.pykrx:
        market = load_pykrx([c for m in THEMES.values() for c in m])
    else:
        market = load_csv_dir(a.csv)

    if a.evaluate:
        from stock.snapshot import evaluate, evaluate_scan, load_scan_snapshots, load_snapshots
        print(evaluate(load_snapshots(a.evaluate), market))
        scan_snap = load_scan_snapshots(a.evaluate)
        if not scan_snap.empty:
            print("\n" + evaluate_scan(scan_snap, market))
        return
    if a.backtest:
        from stock.backtest import collect, format_report
        print(format_report(collect(market, themes_map)))
        return
    if not us:  # 수급은 직전 거래일까지로 통일(저장·시험과 같은 정의). 네이버가 당일 수급을 종목별로 순차 공개해 날짜가 섞이기 때문
        from stock.snapshot import lag_flow
        market = lag_flow(market)
    if market[idx_key].attrs.get("proxy"):
        idx_name = "받은 종목 평균(S&P500 대용)"
    reg = market_regime(market[idx_key])
    themes, stocks = analyze_themes(market, themes_map)
    stocks_all = stocks
    if scan_stats:  # 같은 종목이 여러 테마에 있으면 점수가 높은 한 줄만
        stocks = stocks.sort_values("score", ascending=False).drop_duplicates("code").reset_index(drop=True)

    print(f"[시황] {reg['label']}  ({idx_name} 20일 {reg['ret20']:+.1%})")
    if reg["points"] <= 1:
        print("  → 지수가 약하면 신규 진입 비중을 줄이세요.")
    if scan_stats:
        if scan_stats.get("source") == "ranking":
            tried = ", ".join(f"{k}→{v}" for k, v in scan_stats.get("member_errors", {}).items()) or "없음"
            print(f"[훑기] 테마 구성 종목 조회 실패(시도한 주소→상태: {tried}) → 거래대금·상승률 상위 종목 {scan_stats['stocks']}개로 대신 훑음 "
                  f"(테마 순위는 의미 없음, 데이터 수집 실패 {scan_stats['failed']}개)")
        else:
            print(f"[훑기] 네이버 테마 {scan_stats['themes_total']}개 중 오늘 오르는 {scan_stats['themes_picked']}개를 골라 "
                  f"종목 {scan_stats['stocks']}개 분석 (테마 {scan_stats['themes_used']}개 사용, 데이터 수집 실패 {scan_stats['failed']}개)")
        print("      ※ 이미 움직인 테마를 고르는 방식이라 늦을 수 있고, 같은 종목이 여러 테마에 있으면 한 번만 표시합니다.")
    expected = [c for m in themes_map.values() for c in m]
    missing = [c for c in expected if c not in set(stocks["code"])]
    if missing and not scan_stats:
        print(f"[수집] {len(expected) - len(missing)}/{len(expected)}종목 — 누락: {', '.join(missing)} (누락 종목이 있는 테마는 점수가 부정확)")
    n_flow = int(stocks["flow_ratio"].notna().sum())
    if us:
        print("[수급] 미국은 투자자별 수급 데이터가 없어 가격·거래량·차트만으로 분석")
    else:
        print(f"[수급 반영] {n_flow}/{len(stocks)}종목" + (" (가격은 오늘 종가, 수급은 직전 거래일까지)" if n_flow else "  ※ 수급 없이 가격·거래량만으로 분석됨"))
    print("\n[테마 순위]" + (" (상위 10개)" if scan_stats else ""))
    for i, r in (themes.head(10) if scan_stats else themes).iterrows():
        print(f"{i+1}. {r['theme']:<10} 점수 {r['score']:5.1f} | 강세종목비율 {r['breadth']:.0%} | "
              f"거래대금 {r['value_growth']:+.0%} | 대장 {r['leader']} ({r['leader_stage']})"
              + (f" | ⚠️종목 {r['n']}/{r['total']}" if r["n"] < r["total"] else ""))
    print("\n[선점 후보] 점수 높고 아직 안 오른 종목")
    picks = stocks[stocks["stage"] == "선점 후보"].head(a.top)
    disc = fetch_summaries(list(picks["code"])) if a.dart and not us else {}
    if a.dart and not us and not disc:
        print("  (DART 조회 실패 또는 DART_API_KEY 없음 → 공시 미확인)")
    if picks.empty:
        print("  없음")
    for _, r in picks.iterrows():
        fl = "수급 n/a" if pd.isna(r["flow_ratio"]) else f"수급 {r['flow_ratio']:+.1%}"
        print(f"  {r['name']}({r['theme']}) 점수 {r['score']} | 20일 {r['ret20']:+.1%} | "
              f"거래량 x{r['vol_ratio']:.1f} | 60일고점 {r['prox60']:.0%} | {fl}"
              + ("" if pd.isna(r["indiv_ratio"]) else f" | 개인 {r['indiv_ratio']:+.1%}"))
        act, why = decide(r, reg, disc.get(r["code"]), kr=not us)
        print(f"    → 관찰 우선순위: {act}  ({'; '.join(why)})")
        print(f"    → 흐름: {r['bias']} ({r['bias_pts']:+d}점: {', '.join(r['bias_why'])})")
    hot = stocks[stocks["stage"].str.startswith("과열")]
    if len(hot):
        print("\n[과열 — 급등 직후(이후 방향은 불확실, 시험에서 하락 근거 없음)] " + ", ".join(f"{r['name']}(5일 {r['ret5']:+.0%})" for _, r in hot.iterrows()))
    watch = stocks[stocks["code"].isin(watchlist)]
    if len(watch):
        print("\n[관심 종목 현황]")
        wdisc = fetch_summaries(list(watch["code"])) if a.dart and not us else {}
        for _, r in watch.iterrows():
            fl = "수급 n/a" if pd.isna(r["flow_ratio"]) else f"수급 {r['flow_ratio']:+.1%}"
            ma = "20일선 위" if r["above_ma20"] else "20일선 아래"
            print(f"  {r['name']}({r['theme']}) {r['stage']} · 점수 {r['score']} | 5일 {r['ret5']:+.1%} · 20일 {r['ret20']:+.1%} | "
                  f"{ma} | 거래량 x{r['vol_ratio']:.1f} | 60일고점 {r['prox60']:.0%} | {fl}")
            act, why = decide(r, reg, wdisc.get(r["code"]), kr=not us)
            print(f"    → 관찰 우선순위: {act}  ({'; '.join(why)})")
            print(f"    → 흐름: {r['bias']} ({r['bias_pts']:+d}점: {', '.join(r['bias_why'])})")
    print("\n" + DISCLAIMER)

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        themes.to_csv(os.path.join(a.out, "themes.csv"), index=False, encoding="utf-8-sig")
        stocks.to_csv(os.path.join(a.out, "stocks.csv"), index=False, encoding="utf-8-sig")
        if scan_stats:  # 훑기 결과는 소속 테마와 함께 매일 저장해 둔다(워크플로가 stock-data 브랜치의 scan/ 에 보관)
            from stock.snapshot import build_scan_snapshot
            build_scan_snapshot(stocks_all, themes, market).to_csv(os.path.join(a.out, "scan_snapshot.csv"), index=False, encoding="utf-8-sig")
            themes.to_csv(os.path.join(a.out, "scan_themes.csv"), index=False, encoding="utf-8-sig")
        elif not us:  # 한국 수급은 과거 이력을 구할 수 없어 매일 저장해 둔다(워크플로가 stock-data 브랜치의 kr/ 에 보관)
            from stock.snapshot import build_snapshot
            build_snapshot(stocks, market).to_csv(os.path.join(a.out, "snapshot.csv"), index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
