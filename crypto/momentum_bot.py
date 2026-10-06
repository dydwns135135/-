"""일봉 모멘텀(롱만) 봇: 매일 한 번, 마감된 일봉 기준으로 판단하고 종료.

규칙(백테스트로 검증된 것과 동일): 종가가 lookback 일 전 종가보다 높으면 롱 보유, 아니면 현금.
레버리지 1배·롱만. 숏은 쓰지 않는다(백테스트에서 도움이 안 됨).
기본은 데모. 실거래는 --live + MOMENTUM_CONFIRM_LIVE=yes 가 모두 있어야 한다(워크플로는 데모 전용)."""
from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from trader import notify

log = logging.getLogger("momentum")


def momentum_positive(closes: pd.Series, lookback: int) -> bool | None:
    """마지막 마감 종가가 lookback 일 전 종가보다 높은가. 데이터가 부족하면 None."""
    if len(closes) <= lookback:
        return None
    return bool(closes.iloc[-1] > closes.iloc[-1 - lookback])


def decide(want_long: bool, has_position: bool) -> str:
    if want_long and not has_position:
        return "open"
    if not want_long and has_position:
        return "close"
    return "hold"


def run_once(broker, lookback: int = 90, alloc: float = 0.5, stop_loss: float | None = None) -> list[str]:
    candles = broker.closed_candles("1d", limit=lookback + 40)
    want = momentum_positive(candles["close"], lookback)
    if want is None:
        return [f"일봉이 부족해 판단 보류 ({len(candles)}개, 필요 {lookback + 1}개 이상)"]
    last = candles["close"].iloc[-1]
    ref = candles["close"].iloc[-1 - lookback]
    pos = broker.position()
    act = decide(want, pos is not None)
    head = f"{lookback}일 모멘텀 {'양(+)' if want else '음(-)'}: 종가 {last:,.0f} vs {lookback}일 전 {ref:,.0f} ({last / ref - 1:+.1%})"

    if act == "open":
        px = broker.price()
        amount = broker.balance() * alloc / px  # 레버리지 1배 기준 비중
        stop = px * (1 - stop_loss) if stop_loss else None
        broker.open("buy", amount, stop)
        return [head, f"🟢 롱 진입 {amount:.5f} BTC @ ~{px:,.0f} (잔고의 {alloc:.0%})" + (f", 손절 {stop:,.0f}" if stop else "")]
    if act == "close":
        broker.close(pos["side"], pos["contracts"])
        return [head, f"🔴 롱 청산 {pos['contracts']} BTC (진입 {pos['entry']:,.0f})"]
    return [head, "보유 유지" if pos else "현금 유지(신호 없음)"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--alloc", type=float, default=0.5, help="잔고 중 투입 비율(레버리지 1배 기준)")
    ap.add_argument("--stop-loss", type=float, default=None, help="선택: 진입가 대비 손절 비율(백테스트에는 없음)")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.live and os.environ.get("MOMENTUM_CONFIRM_LIVE") != "yes":
        raise SystemExit("실거래는 MOMENTUM_CONFIRM_LIVE=yes 환경변수가 필요합니다.")
    mode = "실거래" if a.live else "데모"
    from .exchange import BitgetFutures
    try:
        broker = BitgetFutures.from_env(a.symbol, 1, demo=not a.live)  # 레버리지 1배 고정
        notes = run_once(broker, a.lookback, a.alloc, a.stop_loss)
    except Exception as e:
        notify.send(f"❌ 모멘텀봇 오류 [{mode}]: {type(e).__name__}: {str(e)[:300]}")
        raise
    text = "\n".join(notes)
    log.info(text)
    notify.send(f"🪙 모멘텀봇 [{mode}] {a.symbol}\n{text}")


if __name__ == "__main__":
    main()
