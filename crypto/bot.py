"""Bitget 선물 봇: 마감된 최근 봉 기준으로 한 번 점검하고 종료(4시간마다 실행 권장).
기본은 데모 모드. 실거래는 --live + 환경변수 CRYPTO_CONFIRM_LIVE=yes 가 모두 필요."""
from __future__ import annotations

import argparse
import logging
import os

from trader import notify
from trader.strategy import StrategyParams

from .backtest import FuturesConfig
from .strategy import latest_action

log = logging.getLogger("crypto")


def run_once(broker, timeframe: str, cfg: FuturesConfig, params: StrategyParams = StrategyParams()) -> list[str]:
    act = latest_action(broker.closed_candles(timeframe), params)
    pos, px = broker.position(), broker.price()
    notes: list[str] = []

    if pos:
        exit_sig = act["long_exit"] if pos["side"] == "long" else act["short_exit"]
        if exit_sig:
            broker.close(pos["side"], pos["contracts"])
            notes.append(f"🔴 {pos['side']} 청산 {pos['contracts']} @ ~{px:.2f}")
        else:
            notes.append(f"보유 유지: {pos['side']} {pos['contracts']} (진입 {pos['entry']:.2f}, 현재 {px:.2f})")
        return notes

    side = "buy" if act["long_entry"] and not act["long_exit"] else (
        "sell" if cfg.allow_short and act["short_entry"] and not act["short_exit"] else None)
    if not side:
        return ["신호 없음, 포지션 없음"]
    margin = broker.balance() * cfg.margin_pct
    amount = margin * cfg.leverage / px
    stop = px * (1 - cfg.stop_loss) if side == "buy" else px * (1 + cfg.stop_loss)
    broker.open(side, amount, stop)
    notes.append(f"🟢 {'롱' if side == 'buy' else '숏'} 진입 {amount:.5f} @ ~{px:.2f}, 손절 {stop:.2f}, x{cfg.leverage:g}")
    return notes


def test_open(broker, cfg: FuturesConfig) -> list[str]:
    """데모 전용: 최소 수량 롱 1건 + 손절 주문 → 붙었는지 진단 출력."""
    if broker.position():
        return ["이미 포지션이 있어 테스트 주문을 건너뜀 (--test-close 로 먼저 정리)"]
    px, amt = broker.price(), broker.test_amount()
    stop = px * (1 - cfg.stop_loss)
    broker.open("buy", amt, stop)
    info = broker.stop_info()
    mode = {None: "?", False: "단방향", True: "헤지"}[getattr(broker, "hedged", None)]
    return [f"🧪 테스트 롱 {amt} @ ~{px:.2f}, 손절 {stop:.2f}, 계좌 포지션모드={mode}", f"손절 진단: {info}",
            "Bitget 앱(데모)에서 포지션과 손절(TP/SL)이 보이는지 확인 → 확인 후 test-close 실행"]


def test_close(broker) -> list[str]:
    pos = broker.position()
    if not pos:
        return ["정리할 포지션 없음"]
    broker.close(pos["side"], pos["contracts"])
    return [f"🧪 테스트 포지션 청산: {pos['side']} {pos['contracts']}"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--leverage", type=int, default=2)
    ap.add_argument("--margin-pct", type=float, default=0.25)
    ap.add_argument("--stop-loss", type=float, default=0.03)
    ap.add_argument("--no-short", action="store_true")
    ap.add_argument("--live", action="store_true", help="실거래(기본: 데모)")
    ap.add_argument("--test-open", action="store_true", help="[데모 전용] 최소 수량 테스트 주문 + 손절 확인")
    ap.add_argument("--test-close", action="store_true", help="[데모 전용] 테스트 포지션 청산")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.live and os.environ.get("CRYPTO_CONFIRM_LIVE") != "yes":
        raise SystemExit("실거래는 CRYPTO_CONFIRM_LIVE=yes 환경변수가 필요합니다.")
    if (a.test_open or a.test_close) and a.live:
        raise SystemExit("테스트 주문은 데모 모드에서만 가능합니다 (--live 와 함께 쓸 수 없음).")
    cfg = FuturesConfig(leverage=a.leverage, margin_pct=a.margin_pct, stop_loss=a.stop_loss, allow_short=not a.no_short)
    mode = "실거래" if a.live else "데모"
    from .exchange import BitgetFutures
    try:
        broker = BitgetFutures.from_env(a.symbol, a.leverage, demo=not a.live)
        if a.test_open:
            notes = test_open(broker, cfg)
        elif a.test_close:
            notes = test_close(broker)
        else:
            notes = run_once(broker, a.timeframe, cfg)
    except Exception as e:
        notify.send(f"❌ 선물봇 오류 [{mode}]: {type(e).__name__}: {e}")
        raise
    notify.send(f"🪙 선물봇 [{mode}] {a.symbol} {a.timeframe}\n" + "\n".join(notes))
    log.info("\n".join(notes))


if __name__ == "__main__":
    main()
