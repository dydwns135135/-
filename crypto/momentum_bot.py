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


ENSEMBLE_LOOKBACKS = (30, 45, 60, 90, 120, 150, 180, 250)


def ensemble_weight(closes: pd.Series, lookbacks=ENSEMBLE_LOOKBACKS) -> tuple[float, int] | None:
    """(양(+)인 기간의 비율, 양(+)인 기간 수). 데이터가 가장 긴 기간보다 짧으면 None."""
    if len(closes) <= max(lookbacks):
        return None
    pos = sum(bool(closes.iloc[-1] > closes.iloc[-1 - lb]) for lb in lookbacks)
    return pos / len(lookbacks), pos


def position_budget(equity: float, alloc: float, max_notional: float | None = None) -> float:
    """비중 100% 일 때의 포지션 금액(USDT). 잔고 × alloc 이되 max_notional 이 있으면 그 이하로 제한한다."""
    budget = equity * alloc
    return min(budget, max_notional) if max_notional else budget


def plan_rebalance(target_w: float, alloc: float, equity: float, price: float, current: float,
                   step: float = 0.10, min_notional: float = 6.0, max_notional: float | None = None) -> tuple[str, float]:
    """목표 비중(0~1)에 맞추기 위한 ('buy'|'sell'|'hold', 계약 수). 잔고 대비 step(기본 10%p) 미만의
    미세 조정과 최소 주문 금액 미만은 하지 않는다(수수료 절감). 목표가 0이면 전량 청산."""
    full = position_budget(equity, alloc, max_notional) / price  # 비중 100% 일 때의 수량
    delta = full * target_w - current
    if target_w <= 0:
        return ("sell", current) if current * price >= min_notional else ("hold", 0.0)
    if abs(delta) * price < min_notional or abs(delta) < step * full:
        return "hold", 0.0
    return ("buy", delta) if delta > 0 else ("sell", -delta)


def kill_switch(broker, equity: float, min_equity: float | None) -> list[str] | None:
    """잔고가 min_equity 아래면 전량 청산하고 알림 문구를 돌려준다(이후 신규 진입 없음). 해당 없으면 None."""
    if not min_equity or equity >= min_equity:
        return None
    pos = broker.position()
    msg = [f"🛑 차단 장치: 잔고 {equity:,.2f} < 최소 잔고 {min_equity:,.2f} → 신규 진입 중단"]
    if pos:
        broker.close(pos["side"], pos["contracts"])
        msg.append(f"🔴 보유 {pos['contracts']} BTC 전량 청산")
    else:
        msg.append("보유 포지션 없음")
    msg.append("자동 재개되지 않습니다. 원인을 확인한 뒤 MOMENTUM_MIN_EQUITY 를 낮추거나 비워서 재개하세요.")
    return msg


def mix_weight(candles: pd.DataFrame) -> tuple[float, int, bool, float] | None:
    """섞기(교집합): 종가가 일목 구름 위일 때만 앙상블 비중, 아래면 0.
    (목표 비중, 양(+) 기간 수, 구름 위 여부, 다음 일봉의 구름 상단 = 다음 판단의 기준 가격)"""
    from .upbit_indicators import ichimoku_cloud, next_cloud_top
    res = ensemble_weight(candles["close"])
    if res is None:
        return None
    w, npos = res
    above = bool(ichimoku_cloud(candles).iloc[-1] > 0)
    return (w if above else 0.0), npos, above, next_cloud_top(candles)


def run_once_ensemble(broker, alloc: float = 0.5, stop_loss: float | None = None,
                      max_notional: float | None = None, min_equity: float | None = None,
                      mix: bool = False) -> list[str]:
    """mix=True 면 섞기(교집합: 앙상블 × 일목 구름 위 여부), 백테스트와 같이 25%p 이상 차이일 때만 조정."""
    candles = broker.closed_candles("1d", limit=max(ENSEMBLE_LOOKBACKS) + 40, deep=True)
    if mix:
        res = mix_weight(candles)
        if res is None:
            return [f"일봉이 부족해 판단 보류 ({len(candles)}개, 필요 {max(ENSEMBLE_LOOKBACKS) + 1}개 이상)"]
        w, npos, above, next_top = res
    else:
        res = ensemble_weight(candles["close"])
        if res is None:
            return [f"일봉이 부족해 판단 보류 ({len(candles)}개, 필요 {max(ENSEMBLE_LOOKBACKS) + 1}개 이상)"]
        w, npos = res
    pos = broker.position()
    current = float(pos["contracts"]) if pos else 0.0
    equity, px = broker.balance(), broker.price()
    killed = kill_switch(broker, equity, min_equity)
    if killed:
        return killed
    act, qty = plan_rebalance(w, alloc, equity, px, current, step=0.25 if mix else 0.10, max_notional=max_notional)
    head = (f"앙상블 모멘텀: {len(ENSEMBLE_LOOKBACKS)}개 기간 중 {npos}개 양(+) → 목표 비중 {w:.0%} "
            f"(잔고의 {w * alloc:.0%}), 현재 {current:.5f} BTC")
    if mix:
        head = ("섞기(앙상블×일목): " + head + f" · 일목 구름 {'위 → 앙상블 비중 사용' if above else '아래 → 목표 0%'}\n"
                + (f"📍 기준 가격: 다음 일봉(한국 09시 마감) 종가가 구름 상단 {next_top:,.0f} 아래로 마감하면 정리"
                   if above else
                   f"📍 재진입 기준: 다음 일봉(한국 09시 마감) 종가가 구름 상단 {next_top:,.0f} 위로 마감하면 매수"))
    if act == "buy":
        stop = px * (1 - stop_loss) if stop_loss else None
        broker.open("buy", qty, stop)
        return [head, f"🟢 {qty:.5f} BTC 추가 매수 @ ~{px:,.0f}"]
    if act == "sell":
        broker.close("long", qty)
        kind = "전량 청산" if qty >= current - 1e-12 else "일부 청산"
        return [head, f"🔴 {qty:.5f} BTC {kind} @ ~{px:,.0f}"]
    return [head, "조정 없음(차이가 작거나 최소 주문 미만)"]


def run_once(broker, lookback: int = 90, alloc: float = 0.5, stop_loss: float | None = None,
             max_notional: float | None = None, min_equity: float | None = None) -> list[str]:
    candles = broker.closed_candles("1d", limit=lookback + 40, deep=True)
    want = momentum_positive(candles["close"], lookback)
    if want is None:
        return [f"일봉이 부족해 판단 보류 ({len(candles)}개, 필요 {lookback + 1}개 이상)"]
    last = candles["close"].iloc[-1]
    ref = candles["close"].iloc[-1 - lookback]
    pos = broker.position()
    killed = kill_switch(broker, broker.balance(), min_equity)
    if killed:
        return killed
    act = decide(want, pos is not None)
    head = f"{lookback}일 모멘텀 {'양(+)' if want else '음(-)'}: 종가 {last:,.0f} vs {lookback}일 전 {ref:,.0f} ({last / ref - 1:+.1%})"

    if act == "open":
        px = broker.price()
        amount = position_budget(broker.balance(), alloc, max_notional) / px  # 레버리지 1배 기준 비중(상한 적용)
        stop = px * (1 - stop_loss) if stop_loss else None
        broker.open("buy", amount, stop)
        return [head, f"🟢 롱 진입 {amount:.5f} BTC @ ~{px:,.0f} (잔고의 {alloc:.0%})" + (f", 손절 {stop:,.0f}" if stop else "")]
    if act == "close":
        broker.close(pos["side"], pos["contracts"])
        return [head, f"🔴 롱 청산 {pos['contracts']} BTC (진입 {pos['entry']:,.0f})"]
    return [head, "보유 유지" if pos else "현금 유지(신호 없음)"]


class CheckBroker:
    """점검 모드: 실제 브로커의 조회 기능은 그대로 쓰고 주문(open/close)만 가로채 기록한다."""
    def __init__(self, broker):
        self._b, self.blocked = broker, []

    def __getattr__(self, name):
        return getattr(self._b, name)

    def open(self, side, amount, stop=None):
        self.blocked.append(f"(주문 안 함) 진입 {side} {amount:.5f}")

    def close(self, side, contracts):
        self.blocked.append(f"(주문 안 함) 청산 {side} {contracts}")


def check_notes(broker, notes: list[str]) -> list[str]:
    """점검 결과 문구: 잔고·포지션·계획된 주문(실행 안 함)·진단."""
    out = ["🔎 점검 모드: 주문하지 않고 연결·잔고·계획만 확인"] + notes
    out += broker.blocked or ["(계획된 주문 없음)"]
    try:
        out.append(f"잔고 {broker.balance():,.2f} USDT · 포지션 {broker.position() or '없음'}")
    except Exception as e:  # 조회 실패도 그대로 보여 준다
        out.append(f"잔고/포지션 조회 실패: {type(e).__name__}: {str(e)[:150]}")
    if hasattr(broker, "diagnose"):
        out.append(f"진단: {str(broker.diagnose())[:300]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--alloc", type=float, default=0.5, help="잔고 중 투입 비율(레버리지 1배 기준)")
    ap.add_argument("--stop-loss", type=float, default=None, help="선택: 진입가 대비 손절 비율(백테스트에는 없음)")
    ap.add_argument("--mode", choices=["single", "ensemble", "mix"], default="single",
                    help="single: lookback 하나 / ensemble: 30~250일 8개 기간의 양(+) 비율만큼 보유 / "
                         "mix: 앙상블 × 일목 구름 위 여부(교집합, 25%%p 조정)")
    ap.add_argument("--max-notional", type=float, default=None, help="포지션 금액 상한(USDT). 잔고가 커도 이 금액을 넘기지 않음")
    ap.add_argument("--min-equity", type=float, default=None, help="최소 잔고(USDT). 이 아래면 전량 청산하고 신규 진입 중단")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--check", action="store_true", help="주문하지 않고 연결·잔고·계획만 확인")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.live and os.environ.get("MOMENTUM_CONFIRM_LIVE") != "yes":
        raise SystemExit("실거래는 MOMENTUM_CONFIRM_LIVE=yes 환경변수가 필요합니다.")
    mode = "실거래" if a.live else "데모"
    from .exchange import BitgetFutures
    try:
        broker = BitgetFutures.from_env(a.symbol, 1, demo=not a.live)  # 레버리지 1배 고정
        if a.check:
            broker = CheckBroker(broker)
        notes = (run_once_ensemble(broker, a.alloc, a.stop_loss, a.max_notional, a.min_equity, mix=a.mode == "mix")
                 if a.mode in ("ensemble", "mix")
                 else run_once(broker, a.lookback, a.alloc, a.stop_loss, a.max_notional, a.min_equity))
    except Exception as e:
        notify.send(f"❌ 모멘텀봇 오류 [{mode}]: {type(e).__name__}: {str(e)[:300]}")
        raise
    if a.check:
        notes = check_notes(broker, notes)
    text = "\n".join(notes)
    log.info(text)
    notify.send(f"🪙 모멘텀봇 [{mode}/{a.mode}] {a.symbol}\n{text}")


if __name__ == "__main__":
    main()
