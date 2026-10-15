"""사용: python -m mt5bot check | run | test-order | alert | telegram-test | backtest  (옵션은 --help)"""
import argparse
import time
from datetime import date

from . import backtest, broker, notify
from .indicators import atr, ema, rsi
from .strategy import daily_loss_hit, entry_signal, should_exit, sl_tp_prices, trend_ok, virtual_exit


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


def cmd_backtest(args):
    broker.connect(True)  # 조회만
    times, highs, lows, closes = broker.get_history(args.symbol, args.bars)
    broker.shutdown()
    days = (times[-1] - times[0]).days
    print(f"데이터: {args.symbol} M5 {len(closes)}봉 ({times[0]:%Y-%m-%d} ~ {times[-1]:%Y-%m-%d}, 약 {days}일), "
          f"거래당 비용 {args.spread} (가격 단위)", flush=True)
    rows = backtest.run_grid(highs, lows, closes, args.spread)
    # 검증 구간(뒤 30%) 총손익 기준 정렬: 과거에 끼워 맞춘 설정이 위로 오는 걸 줄이기 위함
    rows = [r for r in rows if r["in"]["n"] >= 10 and r["out"]["n"] >= 5]
    rows.sort(key=lambda r: r["out"]["total"], reverse=True)
    print("\n상위 10개 (정렬: 검증구간 총손익). 손익 단위는 가격 단위=달러(0.01랏 약 1온스당)")
    print(f"{'전략':<22}{'SL':>4}{'TP':>5} | {'전체: 거래 승률 총손익 PF 최대낙폭':<34} | {'학습 총손익':>9} {'검증 총손익':>9}")
    for r in rows[:10]:
        a = r["all"]
        print(f"{r['name']:<22}{r['sl']:>4}{r['tp']:>5} | {a['n']:>5} {a['win']:>5.1f}% {a['total']:>9.1f} {a['pf']:>4.2f} {a['maxdd']:>8.1f} | "
              f"{r['in']['total']:>9.1f} {r['out']['total']:>9.1f}")
    pos_both = [r for r in rows if r["in"]["total"] > 0 and r["out"]["total"] > 0]
    print(f"\n학습·검증 구간 모두 흑자인 조합: {len(pos_both)}개 / 전체 {len(rows)}개")
    if not pos_both:
        print("-> 비용을 빼면 두 구간 모두에서 이기는 설정이 없습니다. 이 전략들은 실거래에 쓰지 않는 게 맞습니다.")


def cmd_telegram_test(args):
    res, ids = notify.find_chat_ids()
    if not res.get("ok"):
        print(f"토큰 확인 실패: {res.get('description')}")
        return
    for cid, kind in ids.items():
        print(f"chat id 후보: {cid} ({kind})  <- private 인 숫자가 내 번호")
    if not ids:
        print("봇이 받은 메시지가 없습니다. 텔레그램에서 내 봇에게 아무 메시지나 보낸 뒤 다시 실행하세요.")
    r = notify.send("알림 테스트: MT5 봇 알림이 연결되었습니다.")
    print("테스트 알림 전송 성공(텔레그램 확인)" if r.get("ok") else f"테스트 알림 실패: {r.get('description')}")


_warned = [False]


def _tell(msg):
    """콘솔에 출력하고, 텔레그램이 설정되어 있으면 알림도 보낸다. 실패해도 매매는 계속(실패 문구는 한 번만 표시)."""
    print(msg, flush=True)
    r = notify.send(msg)
    if not r.get("ok") and not _warned[0]:
        _warned[0] = True
        print(f"  (텔레그램 전송 안 됨: {r.get('description')} - 이후 실패는 표시하지 않음)", flush=True)


def _heartbeat(args, times, closes, r, e, extra=""):
    trend = "상승(롱만)" if closes[-1] > e else "하락(숏만)"
    if args.no_trend_filter:
        trend = "필터 끔"
    return (f"[정상 작동 중] {args.symbol} {times[-1]:%m-%d %H:%M} 종가 {closes[-1]} "
            f"RSI {r[-1]:.1f} EMA{args.ema_period} {e:.2f} ({trend}){extra}")


def cmd_alert(args):
    """주문은 내지 않고, 진입/청산 타이밍을 텔레그램으로 알린다(폰 MT5 에서 수동 매매용)."""
    broker.connect(True)  # 조회만 하므로 데모/실계좌 모두 허용
    _tell(f"알림 모드 시작: {args.symbol} M5 RSI({args.period}) {args.oversold}/{args.overbought}, "
          f"추세필터 {'EMA'+str(args.ema_period) if not args.no_trend_filter else '끔'}. 주문은 내지 않습니다.")
    digits = 2
    last_bar, virt, errors = None, None, 0
    last_hb = time.monotonic()
    try:
        while True:
            try:
                times, highs, lows, closes = broker.get_rates(args.symbol, 600)
                r = rsi(closes, args.period)
                if args.heartbeat_hours > 0 and time.monotonic() - last_hb >= args.heartbeat_hours * 3600:
                    last_hb = time.monotonic()
                    _tell(_heartbeat(args, times, closes, r, ema(closes, args.ema_period)[-1]))
                if times[-1] != last_bar:
                    last_bar = times[-1]
                    if virt:
                        out = virtual_exit(virt["side"], virt["sl"], virt["tp"], highs[-1], lows[-1], r[-1])
                        if out:
                            reason, px = out
                            _tell(f"[청산 알림] {args.symbol} {virt['side'].upper()} 정리 고려\n사유: {reason}"
                                  + (f"\n기준가: {px}" if px else f"\n현재가(종가): {closes[-1]}"))
                            virt = None
                    else:
                        sig = entry_signal(r[-2], r[-1], args.oversold, args.overbought)
                        if sig:
                            e = ema(closes, args.ema_period)[-1]
                            if not args.no_trend_filter and not trend_ok(sig, closes[-1], e):
                                print(f"{last_bar} 신호 {sig} 무시: 추세 불일치(종가 {closes[-1]} EMA {e:.2f})", flush=True)
                            else:
                                a = atr(highs, lows, closes, args.atr_period)[-1]
                                sl_d = args.sl or args.sl_atr * a
                                tp_d = args.tp or args.tp_atr * a
                                sl, tp = sl_tp_prices(sig, closes[-1], sl_d, tp_d, digits)
                                virt = {"side": sig, "sl": sl, "tp": tp}
                                _tell(f"[진입 알림] {args.symbol} {'매수(BUY)' if sig == 'buy' else '매도(SELL)'}\n"
                                      f"기준가(종가): {closes[-1]}  RSI {r[-1]:.1f}\n"
                                      f"손절(SL): {sl}  익절(TP): {tp}  (ATR {a:.2f})\n"
                                      f"폰 MT5 에서 주문 시 SL/TP 를 꼭 같이 입력하세요. 시각 {last_bar}")
                errors = 0
            except Exception as ex:  # 일시적 오류(연결 끊김 등)로 봇이 죽지 않게: 알리고 재시도
                errors += 1
                print(f"오류 {type(ex).__name__}: {ex}", flush=True)
                if errors == 1 or errors % 60 == 0:
                    _tell(f"[오류] 알림 봇에 문제가 생겼습니다(재시도 중, 연속 {errors}회): {type(ex).__name__}: {ex}")
                time.sleep(30)
                continue
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("중지")
    finally:
        broker.shutdown()


def cmd_run(args):
    broker.connect(args.allow_live)
    sltp = (f"SL ${args.sl} TP ${args.tp} 고정" if args.sl and args.tp
            else f"SL ATR×{args.sl_atr} TP ATR×{args.tp_atr}")
    _tell(f"자동매매 시작: {args.symbol} M5, RSI({args.period}) {args.oversold}/{args.overbought}, "
          f"추세필터 {'EMA'+str(args.ema_period) if not args.no_trend_filter else '끔'}, {sltp}, "
          f"{args.lots}랏, 일일손실한도 {args.max_daily_loss_pct}%")
    last_bar, errors, had_pos, we_closed = None, 0, False, False
    day, day_equity = date.today(), broker.equity()
    last_hb = time.monotonic()
    try:
        while True:
            try:
                if date.today() != day:
                    day, day_equity = date.today(), broker.equity()
                times, highs, lows, closes = broker.get_rates(args.symbol, 600)
                r = rsi(closes, args.period)
                pos = broker.positions(args.symbol)
                if had_pos and not pos and not we_closed:  # 우리가 청산하지 않았는데 사라짐 = SL/TP 체결
                    _tell(f"[포지션 종료] {args.symbol} SL/TP 에 걸려 정리된 것으로 보입니다. 자산 {broker.equity():,.0f}")
                had_pos, we_closed = bool(pos), False
                if args.heartbeat_hours > 0 and time.monotonic() - last_hb >= args.heartbeat_hours * 3600:
                    last_hb = time.monotonic()
                    _tell(_heartbeat(args, times, closes, r, ema(closes, args.ema_period)[-1],
                                     f" 보유 {len(pos)}건 자산 {broker.equity():,.0f}"))
                if times[-1] != last_bar:  # 새 봉이 확정됐을 때만 판단
                    last_bar = times[-1]
                    if pos:
                        p = pos[0]
                        side = "buy" if p.type == 0 else "sell"
                        if should_exit(side, r[-1]):
                            broker.close_position(p)
                            we_closed = True
                            _tell(f"[청산] {args.symbol} {side} RSI {r[-1]:.1f} 중립선 되돌파로 청산 (시각 {last_bar})")
                    else:
                        sig = entry_signal(r[-2], r[-1], args.oversold, args.overbought)
                        if sig and daily_loss_hit(day_equity, broker.equity(), args.max_daily_loss_pct):
                            print(f"{last_bar} 신호 {sig} 무시: 일일 손실 한도 도달(내일까지 정지)", flush=True)
                        elif sig:
                            e = ema(closes, args.ema_period)[-1]
                            if not args.no_trend_filter and not trend_ok(sig, closes[-1], e):
                                print(f"{last_bar} 신호 {sig} 무시: 추세 불일치(종가 {closes[-1]} EMA {e:.2f})", flush=True)
                            else:
                                a = atr(highs, lows, closes, args.atr_period)[-1]
                                sl = args.sl or args.sl_atr * a
                                tp = args.tp or args.tp_atr * a
                                vol = args.lots or broker.calc_volume(args.symbol, sl, args.risk)
                                res = broker.open_order(args.symbol, sig, vol, sl, tp)
                                had_pos = True
                                _tell(f"[진입] {args.symbol} {'매수' if sig == 'buy' else '매도'} {vol}랏 RSI {r[-1]:.1f}\n"
                                      f"체결 {res['filled']} (요청 {res['requested']}, 슬리피지 {res['slippage_points']:.0f}포인트)\n"
                                      f"SL {res['sl']}  TP {res['tp']}  (ATR {a:.2f})  시각 {last_bar}")
                errors = 0
            except Exception as ex:  # 일시적 오류(연결 끊김 등)로 봇이 죽지 않게: 알리고 재시도
                errors += 1
                print(f"오류 {type(ex).__name__}: {ex}", flush=True)
                if errors == 1 or errors % 60 == 0:
                    _tell(f"[오류] 자동매매 봇에 문제가 생겼습니다(재시도 중, 연속 {errors}회): {type(ex).__name__}: {ex}")
                time.sleep(30)
                continue
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("중지")
    finally:
        broker.shutdown()


def main():
    ap = argparse.ArgumentParser(prog="mt5bot")
    ap.add_argument("command", choices=["check", "run", "test-order", "alert", "telegram-test", "backtest"])
    ap.add_argument("--symbol", default="GOLD", help="심볼(XM 은 GOLD)")
    ap.add_argument("--period", type=int, default=14, help="RSI 기간")
    ap.add_argument("--oversold", type=float, default=30)
    ap.add_argument("--overbought", type=float, default=70)
    ap.add_argument("--ema-period", type=int, default=200, help="추세 필터 EMA 기간")
    ap.add_argument("--no-trend-filter", action="store_true", help="추세 필터 끄기")
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--sl-atr", type=float, default=2.0, help="손절 = ATR × 이 값 (백테스트 최적 2.0)")
    ap.add_argument("--tp-atr", type=float, default=3.0, help="익절 = ATR × 이 값 (백테스트 최적 3.0)")
    ap.add_argument("--sl", type=float, default=0, help="고정 손절 거리(가격 단위). 0이면 ATR 사용")
    ap.add_argument("--tp", type=float, default=0, help="고정 익절 거리(가격 단위). 0이면 ATR 사용")
    ap.add_argument("--max-daily-loss-pct", type=float, default=2.0,
                    help="하루 시작 자산 대비 손실 %% 도달 시 그날 신규 진입 중지(0=끔, 봇 재시작 시 기준 초기화)")
    ap.add_argument("--risk", type=float, default=10.0, help="거래당 최대 손실 금액(계좌 통화)")
    ap.add_argument("--lots", type=float, default=0.01, help="랏 고정(0이면 --risk 로 계산)")
    ap.add_argument("--bars", type=int, default=30000, help="백테스트에 쓸 5분봉 개수")
    ap.add_argument("--spread", type=float, default=0.3, help="백테스트 거래당 비용(가격 단위, 스프레드+슬리피지)")
    ap.add_argument("--heartbeat-hours", type=float, default=6,
                    help="이 시간마다 텔레그램으로 정상 작동 메시지 전송(0=끔). 이 메시지가 끊기면 봇이 멈춘 것")
    ap.add_argument("--poll", type=int, default=5, help="폴링 간격(초)")
    ap.add_argument("--allow-live", action="store_true", help="실계좌 허용(기본 거부)")
    args = ap.parse_args()
    if bool(args.sl) != bool(args.tp):
        ap.error("--sl 과 --tp 는 함께 지정하세요(또는 둘 다 생략하고 ATR 사용)")
    {"check": cmd_check, "run": cmd_run, "test-order": cmd_test_order,
     "alert": cmd_alert, "telegram-test": cmd_telegram_test,
     "backtest": cmd_backtest}[args.command](args)


if __name__ == "__main__":
    main()
