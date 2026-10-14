"""사용: python -m mt5bot check | run  (옵션은 --help)"""
import argparse
import time

from . import broker
from .indicators import rsi
from .strategy import entry_signal, should_exit


def cmd_check(args):
    info = broker.connect(args.allow_live)
    for k, v in broker.account_summary().items():
        print(f"{k}: {v}")
    times, closes = broker.get_closes(args.symbol, 100)
    r = rsi(closes, args.period)
    print(f"{args.symbol} M5 최근 확정봉 {times[-1]} 종가 {closes[-1]} RSI({args.period}) {r[-1]:.1f}")
    broker.shutdown()


def cmd_test_order(args):
    """주문 경로 검증: 최소 랏으로 SL/TP 포함 매수 1건 체결 후 즉시 청산. 데모 전용."""
    info = broker.connect(False)  # 실계좌면 여기서 종료
    try:
        lots = args.lots or broker.min_volume(args.symbol)
        res = broker.open_order(args.symbol, "buy", lots, args.sl, args.tp)
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
    print(f"시작: {args.symbol} M5, RSI({args.period}) {args.oversold}/{args.overbought}, "
          f"SL ${args.sl} TP ${args.tp}, 거래당 위험 {args.risk}", flush=True)
    last_bar = None
    try:
        while True:
            times, closes = broker.get_closes(args.symbol, 300)
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
                    if sig:
                        vol = args.lots or broker.calc_volume(args.symbol, args.sl, args.risk)
                        res = broker.open_order(args.symbol, sig, vol, args.sl, args.tp)
                        print(f"{last_bar} 진입 {sig} RSI {r[-1]:.1f} {res}", flush=True)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("중지")
    finally:
        broker.shutdown()


def main():
    ap = argparse.ArgumentParser(prog="mt5bot")
    ap.add_argument("command", choices=["check", "run", "test-order"])
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--period", type=int, default=14)
    ap.add_argument("--oversold", type=float, default=30)
    ap.add_argument("--overbought", type=float, default=70)
    ap.add_argument("--sl", type=float, default=5.0, help="고정 손절 거리(가격 단위, 금은 달러)")
    ap.add_argument("--tp", type=float, default=10.0, help="고정 익절 거리(가격 단위)")
    ap.add_argument("--risk", type=float, default=10.0, help="거래당 최대 손실 금액(계좌 통화)")
    ap.add_argument("--lots", type=float, default=0, help="랏 고정(0이면 --risk 로 계산)")
    ap.add_argument("--poll", type=int, default=5, help="폴링 간격(초)")
    ap.add_argument("--allow-live", action="store_true", help="실계좌 허용(기본 거부)")
    args = ap.parse_args()
    {"check": cmd_check, "run": cmd_run, "test-order": cmd_test_order}[args.command](args)


if __name__ == "__main__":
    main()
