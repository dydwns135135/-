"""데모 전용 단타 봇: Bitget 데모 계좌에서 ETH 1시간봉 돌파 신호(롱·숏)로 자동매매한다. 실거래 옵션 없음.
python -m crypto.scalp_bot [--budget 1000]   (매시간 실행)

신호·손절·목표는 crypto.futures_signal 과 같다(백테스트: 비용 후 신호당 평균 -0.09R → 손실 가능성이 큰 규칙).
- 롱: 직전 24봉 고점 돌파 + 거래량 2배 + EMA100 위 / 숏: 반대
- 손절 1.5×ATR(=1R, 거래소에 손절 주문으로 걸어 둠), 목표1 +1.5R 에 절반 정리, 목표2 +3R 에 나머지 정리, 48봉 지나면 정리
- 포지션이 있으면 새로 진입하지 않는다. 하루(한국 시간 기준) 최대 3번 진입.
- 크기: 손절 시 손실이 budget 의 1%가 되도록(명목금액 = budget×1%/손절폭), 최대 budget×3배. 레버리지는 그에 맞춰 1~3배.
BTC 섞기 봇(crypto-momentum)과 포지션이 꼬이지 않게 ETH 만 거래한다.
목표 정리는 봇이 매시간 마감봉을 보고 시장가로 하므로, 백테스트(정확히 목표가 체결)보다 체결가가 불리할 수 있다."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from . import futures_signal as fs

SYMBOL = "ETH/USDT:USDT"
MAX_PER_DAY = 3


def kst_day(ts: pd.Timestamp) -> str:
    return (ts + pd.Timedelta(hours=9)).strftime("%Y-%m-%d")


def load_state(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def save_state(path: str, state: dict) -> None:
    Path(path).write_text(json.dumps(state, ensure_ascii=False, indent=1))


def size_for(budget: float, stop_pct: float) -> tuple[float, int]:
    """(명목금액 USDT, 레버리지). 손절 시 손실 = budget×1%, 명목금액은 budget×3배 이하."""
    notional = min(budget * fs.RISK / stop_pct, budget * fs.LEV_CAP)
    return notional, max(1, min(int(fs.LEV_CAP), math.ceil(notional / budget)))


def tp1_reached(side: int, tp1: float, bars: pd.DataFrame) -> bool:
    return bool((bars["high"] >= tp1).any()) if side > 0 else bool((bars["low"] <= tp1).any())


def _record(state: dict, o: dict, status: str, r: float) -> str:
    net = fs.net_r(r, o["r"] / o["entry"])
    state.setdefault("done", []).append({"side": o["side"], "time": o["bar_time"], "status": status, "net_r": net})
    state["open"] = None
    return f"📒 {'롱' if o['side'] > 0 else '숏'} 종료: {status} ({net:+.2f}R, 비용 반영 추정)\n{fs.tally(state['done'])}"


def run_once(broker, bars: pd.DataFrame, state: dict, now: pd.Timestamp, budget: float) -> list[str]:
    """bars: 마감된 1시간봉(open/high/low/close/volume). 진행 중 포지션을 관리하고, 조건이 되면 새로 진입한다."""
    msgs: list[str] = []
    today = kst_day(now)
    if state.get("day") != today:
        state["day"], state["entries"] = today, 0
    d = fs.prepare(bars)
    pos = broker.position()
    o = state.get("open")

    # 1) 진행 중인 포지션 관리
    if o:
        if pos is None:   # 거래소 손절 주문(또는 수동)으로 이미 정리됨
            msgs.append(_record(state, o, "손절(거래소 주문)" if not o.get("tp1_done") else "목표1 후 손절(거래소 주문)",
                                (0.75 - 0.5) if o.get("tp1_done") else -1.0))
        else:
            after = d[d.index > pd.Timestamp(o["bar_time"])]
            res = fs.resolve(o["side"], o["entry"], o["r"], after)
            if res["status"] != "진행중":
                broker.close(pos["side"], pos["contracts"])
                msgs.append(_record(state, o, res["status"], res["r"]))
            elif not o.get("tp1_done") and tp1_reached(o["side"], o["tp1"], after):
                half = float(broker.ex.amount_to_precision(broker.symbol, pos["contracts"] / 2)) if hasattr(broker, "ex") else pos["contracts"] / 2
                if half > 0:
                    broker.close(pos["side"], half)
                    msgs.append(f"🟡 목표1 도달 → 절반({half}) 정리, 나머지는 목표2 {fs.fmt_price(o['tp2'])} 또는 손절까지 보유")
                o["tp1_done"] = True
        pos = broker.position()

    # 2) 새 진입
    if state.get("open") or pos is not None:
        if pos is not None and not state.get("open"):
            msgs.append(f"⚠️ 봇이 모르는 {SYMBOL} 포지션이 있어 새로 진입하지 않음(앱에서 직접 연 포지션?)")
        return msgs
    last = d.index[-1]
    if now - (last + pd.Timedelta(hours=1)) > pd.Timedelta(hours=3):
        return msgs + ["⚠️ 시세가 오래되어 판단 보류"]
    sig = int(fs.signals(d).iloc[-1])
    if sig == 0 or state.get("last_signal_bar") == last.isoformat():
        return msgs
    if state.get("entries", 0) >= MAX_PER_DAY:
        return msgs + [f"⏸ 오늘 진입 {MAX_PER_DAY}번을 다 써서 이번 신호({'롱' if sig > 0 else '숏'})는 건너뜀"]
    px = float(broker.price())
    p = fs.plan(px, float(d["atr"].iloc[-1]), sig)
    notional, lev = size_for(budget, p["stop_pct"])
    qty = notional / px
    broker.leverage = lev
    broker.open("buy" if sig > 0 else "sell", qty, p["stop"])
    state["open"] = {"side": sig, "bar_time": last.isoformat(), "entry": px, "r": p["r"], "stop": p["stop"],
                     "tp1": p["tp1"], "tp2": p["tp2"], "tp1_done": False}
    state["entries"] = state.get("entries", 0) + 1
    state["last_signal_bar"] = last.isoformat()
    vol_x = float(d["volume"].iloc[-1] / d["vavg"].iloc[-1])
    msgs.append(f"{'📈 롱' if sig > 0 else '📉 숏'} 진입 {qty:.4f} ETH @ ~{fs.fmt_price(px)} (레버리지 {lev}배, 거래량 {vol_x:.1f}배)\n"
                f"손절 {fs.fmt_price(p['stop'])} (거래소 주문) · 목표1 {fs.fmt_price(p['tp1'])} · 목표2 {fs.fmt_price(p['tp2'])}\n"
                f"손절 시 손실 ≈ {budget * fs.RISK:,.0f} USDT · 오늘 진입 {state['entries']}/{MAX_PER_DAY}")
    return msgs


def main() -> None:
    import ccxt
    from trader import notify
    from .backtest_report import fetch_history
    from .exchange import BitgetFutures
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=1000.0, help="이 봇이 쓰는 기준 금액(USDT). 손절 1번 = 이 금액의 1%")
    ap.add_argument("--state", default="scalp_state.json")
    a = ap.parse_args()
    state = load_state(a.state)
    try:
        public = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        bars = fetch_history(public, SYMBOL, timeframe="1h", years=0.05, keep_volume=True)
        broker = BitgetFutures.from_env(SYMBOL, 1, demo=True)   # 데모 전용
        msgs = run_once(broker, bars, state, pd.Timestamp.now(tz="UTC"), a.budget)
    except Exception as e:
        notify.send(f"❌ 단타봇 오류 [데모/ETH]: {type(e).__name__}: {str(e)[:300]}")
        raise
    finally:
        save_state(a.state, state)
    for m in msgs:
        notify.send(f"⚡ 단타봇 [데모/ETH 1시간봉]\n{m}")
    print("\n".join(msgs) or "변화 없음")


if __name__ == "__main__":
    main()
