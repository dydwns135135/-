"""사용: python -m mt5bot check | run | test-order  (옵션은 --help)"""
import argparse
import time
from datetime import date

from . import broker
from .indicators import atr, ema, rsi
from .strategy import daily_loss_hit, entry_signal, should_exit, trend_ok


def cmd_check(args):
    broker.connect(args.allow_live)
    for k, v in broker.account_summary().items():
        print(f"{k}: {v}")
    times, highs, lows, closes = broker.get_rates(args.symbol, 600)
    r, e, a = rsi(closes, args.period), ema(closes, args.ema_period), atr(highs, lows, closes, args.atr_period)
    print(f"{args.symbol} M5 최근 확정봉 {times[-1]} 종가 {closes[-1]} RSI({args.period}) {r[-1]:.1f} "
          f"EMA({args.ema_period}) {e[-1]:.2f} ATR({args.atr_period}) {a[-1]:.2f}")
    broker.shutdown()


def cmd_test_order(args):
    """주문 경로 검증: 최소 랏으로 SL/TP 포함 매수 1건 체결 후 즉시 청산. 데모 전용."""
    broker.connect(False)  # 실계좌면 여기서 종료
    try:
        lots = args.lots or broker.min_volume(args.symbol)
        res = broker.open_order(args.symbol, "buy", lots, args.sl or 5.0, args.tp or 10.0)
        print(f"진입 성공: {res}", flush=True)
        time.sleep(2)
        pos = broker.positions(args.symbol)
        if not pos:
            print("경고: 포지션이 조회되지 않습니다(이미 SL/TP 체결?)")
            return
        p = pos[0]
        print(f"보유 확인: ticket {p.ticket} 볼륨 {p.volume} SL {p.sl} TP {p.tp} 손익 {p.profit}", flush=True)
        r = broker.close_position(p)
        print(f"청산 성공: 체결가 {r.price}", flush=True)
    finally:
        broker.shutdown()


def cmd_run(args):
    broker.connect(args.allow_live)
    sltp = (f"SL ${args.sl} TP ${args.tp} 고정" if args.sl and args.tp
            else f"SL ATR×{args.sl_atr} TP ATR×{args.tp_atr}")
    print(f"시작: {args.symbol} M5, RSI({args.period}) {args.oversold}/{args.overbought}, "
          f"추세필터 {'EMA'+str(args.ema_period) if not args.no_trend_filter else '끔'}, {sltp}, "
          f"일일손실한도 {args.max_daily_loss_pct}%", flush=True)
    last_bar = None
    day, day_equity = date.today(), broker.equity()
    try:
        while True:
            if date.today() != day:
                day, day_equity = date.today(), broker.equity()
            times, highs, lows, closes = broker.get_rates(args.symbol, 600)
            if times[-1] != last_bar:  # 새 봉이 확정됐을 때만 판단
                last_bar = times[-1]
                r = rsi(closes, args.period)
                pos = broker.positions(args.symbol)
                if pos:
                    p = pos[0]
                    side = "buy" if p.type == 0 else "sell"
                    if should_exit(side, r[-1]):
                        broker.close_position(p)
                        print(f"{last_bar} 청산({side}) RSI {r[-1]:.1f}", flush=True)
                else:
                    sig = entry_signal(r[-2], r[-1], args.oversold, args.overbought)
                    if sig and daily_loss_hit(day_equity, broker.equity(), args.max_daily_loss_pct):
                        print(f"{last_bar} 신호 {sig} 무시: 일일 손실 한도 도달(내일까지 정지)", flush=True)
                    elif sig:
                        e = ema(closes, args.ema_period)[-1]
                        if not args.no_trend_filter and not trend_ok(sig, closes[-1], e):
                            print(f"{last_bar} 신호 {sig} 무시: 추세 불일치(종가 {closes[-1]} EMA {e})", flush=True)
                        else:
                            a = atr(highs, lows, closes, args.atr_period)[-1]
                            sl = args.sl or args.sl_atr * a
                            tp = args.tp or args.tp_atr * a
                            vol = args.lots or broker.calc_volume(args.symbol, sl, args.risk)
                            res = broker.open_order(args.symbol, sig, vol, sl, tp)
                            print(f"{last_bar} 진입 {sig} RSI {r[-1]:.1f} ATR {a:.2f} {res}", flush=True)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("중지")
    finally:
        broker.shutdown()


def main():
    ap = argparse.ArgumentParser(prog="mt5bot")
    ap.add_argument("command", choices=["check", "run", "test-order"])
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--period", type=int, default=14, help="RSI 기간")
    ap.add_argument("--oversold", type=float, default=30)
    ap.add_argument("--overbought", type=float, default=70)
    ap.add_argument("--ema-period", type=int, default=200, help="추세 필터 EMA 기간")
    ap.add_argument("--no-trend-filter", action="store_true", help="추세 필터 끄기")
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--sl-atr", type=float, default=1.5, help="손절 = ATR × 이 값")
    ap.add_argument("--tp-atr", type=float, default=3.0, help="익절 = ATR × 이 값")
    ap.add_argument("--sl", type=float, default=0, help="고정 손절 거리(가격 단위). 0이면 ATR 사용")
    ap.add_argument("--tp", type=float, default=0, help="고정 익절 거리(가격 단위). 0이면 ATR 사용")
    ap.add_argument("--max-daily-loss-pct", type=float, default=2.0,
                    help="하루 시작 자산 대비 손실 %% 도달 시 그날 신규 진입 중지(0=끔, 봇 재시작 시 기준 초기화)")
    ap.add_argument("--risk", type=float, default=10.0, help="거래당 최대 손실 금액(계좌 통화)")
    ap.add_argument("--lots", type=float, default=0, help="랏 고정(0이면 --risk 로 계산)")
    ap.add_argument("--poll", type=int, default=5, help="폴링 간격(초)")
    ap.add_argument("--allow-live", action="store_true", help="실계좌 허용(기본 거부)")
    args = ap.parse_args()
    if bool(args.sl) != bool(args.tp):
        ap.error("--sl 과 --tp 는 함께 지정하세요(또는 둘 다 생략하고 ATR 사용)")
    {"check": cmd_check, "run": cmd_run, "test-order": cmd_test_order}[args.command](args)


if __name__ == "__main__":
    main()
