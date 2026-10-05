"""자동매매 실행기. 한 번 실행하면 전 종목을 점검하고 종료(크론/스케줄러로 장 마감 후 일 1회 실행 권장).

기본은 모의 모드. 실거래는 --live 와 환경변수 TRADER_CONFIRM_LIVE=yes 를 모두 요구한다.
"""
from __future__ import annotations

import argparse
import logging
import os

from . import notify
from .broker import Broker
from .risk import RiskLimits, RiskManager
from .strategy import StrategyParams, latest_signal

log = logging.getLogger("trader")


def run_once(broker: Broker, symbols: list[str], risk: RiskManager, params: StrategyParams = StrategyParams()) -> list[str]:
    notes: list[str] = []
    positions = broker.positions()
    prices = {s: broker.price(s) for s in set(symbols) | set(positions)}
    held_value = sum(p.qty * prices[s] for s, p in positions.items())
    cash = broker.cash()
    equity = cash + held_value
    log.info("자산 %.2f (현금 %.2f, 보유 %.2f)", equity, cash, held_value)

    for sym in symbols:
        df = broker.candles(sym, 200)
        sig, px, pos = latest_signal(df, params), prices[sym], positions.get(sym)

        if pos and (sig == "sell" or risk.stop_hit(pos.avg_cost, px)):
            log.info("%s 청산 신호(신호=%s)", sym, sig)
            notes.append(f"🔴 매도 {sym} x{pos.qty} @ {px:.2f} (신호={sig})")
            broker.sell(sym, pos.qty, px)
            cash += pos.qty * px
            held_value -= pos.qty * px
            positions.pop(sym)
        elif not pos and sig == "buy":
            qty = risk.buy_qty(px, equity, cash, held_value, len(positions))
            if qty > 0:
                broker.buy(sym, qty, px)
                notes.append(f"🟢 매수 {sym} x{qty} @ {px:.2f}")
                cash -= qty * px
                held_value += qty * px
                positions[sym] = None  # 보유 종목 수 집계용
            else:
                log.info("%s 매수 신호지만 리스크 한도로 보류", sym)
                notes.append(f"⚪ {sym} 매수 신호, 리스크 한도로 보류")
        else:
            log.info("%s %s (보유 %s)", sym, sig, "O" if pos else "X")
    return notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("symbols", nargs="+", help="예: AAPL MSFT NVDA")
    ap.add_argument("--live", action="store_true", help="실거래(기본: 모의)")
    ap.add_argument("--state", default="state.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from .toss_client import TossBroker, TossClient

    real = TossBroker(TossClient.from_env())
    if a.live:
        if os.environ.get("TRADER_CONFIRM_LIVE") != "yes":
            raise SystemExit("실거래는 TRADER_CONFIRM_LIVE=yes 환경변수가 필요합니다.")
        broker: Broker = real
        log.warning("*** 실거래 모드 ***")
    else:
        from .broker import PaperBroker

        broker = PaperBroker(real)
        log.info("모의 모드 (주문 전송 안 함)")
    mode = "실거래" if a.live else "모의"
    try:
        notes = run_once(broker, a.symbols, RiskManager(RiskLimits(), a.state))
    except Exception as e:  # 장애도 휴대폰으로 알린다
        notify.send(f"❌ 자동매매 오류 [{mode}]: {type(e).__name__}: {e}")
        raise
    notify.send(f"📈 자동매매 [{mode}] {', '.join(a.symbols)}\n" + ("\n".join(notes) or "거래 없음"))


if __name__ == "__main__":
    main()
