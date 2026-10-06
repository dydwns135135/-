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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--leverage", type=int, default=2)
    ap.add_argument("--margin-pct", type=float, default=0.25)
    ap.add_argument("--stop-loss", type=float, default=0.03)
    ap.add_argument("--no-short", action="store_true")
    ap.add_argument("--live", action="store_true", help="실거래(기본: 데모)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.live and os.environ.get("CRYPTO_CONFIRM_LIVE") != "yes":
        raise SystemExit("실거래는 CRYPTO_CONFIRM_LIVE=yes 환경변수가 필요합니다.")
    cfg = FuturesConfig(leverage=a.leverage, margin_pct=a.margin_pct, stop_loss=a.stop_loss, allow_short=not a.no_short)
    mode = "실거래" if a.live else "데모"
    from .exchange import BitgetFutures
    try:
        broker = BitgetFutures.from_env(a.symbol, a.leverage, demo=not a.live)
        notes = run_once(broker, a.timeframe, cfg)
    except Exception as e:
        notify.send(f"❌ 선물봇 오류 [{mode}]: {type(e).__name__}: {e}")
        raise
    notify.send(f"🪙 선물봇 [{mode}] {a.symbol} {a.timeframe}\n" + "\n".join(notes))
    log.info("\n".join(notes))


if __name__ == "__main__":
    main()
