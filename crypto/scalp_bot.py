"""데모 전용 단타 봇: Bitget 데모 계좌에서 자동매매한다. 실거래 옵션 없음.
--strategy rsi_ema: ETH·SOL·XRP·DOGE·BNB 4시간봉 RSI반전+EMA200(사용자 제공 Pine). 코인당 1개, 동시 최대 3개, 하루 3번 진입.
--strategy breakout(기본): ETH 1시간봉 돌파 신호(롱·숏), 아래 설명.
python -m crypto.scalp_bot [--budget 1000]   (매시간 실행)

신호·손절·목표는 crypto.futures_signal 과 같다(백테스트: 비용 후 신호당 평균 -0.09R → 손실 가능성이 큰 규칙).
- 롱: 직전 24봉 고점 돌파 + 거래량 2배 + EMA100 위 / 숏: 반대
- 손절 1.5×ATR(=1R, 거래소에 손절 주문으로 걸어 둠), 목표1 +1.5R 에 절반 정리, 목표2 +3R 에 나머지 정리, 48봉 지나면 정리
- 포지션이 있으면 새로 진입하지 않는다. 하루(한국 시간 기준) 최대 3번 진입.
- 크기: 손절 시 손실이 budget×risk(기본 1%)가 되도록(명목금액 = budget×risk/손절폭), 최대 budget×leverage.
  레버리지(기본 3, 최대 20)는 거래소 설정값이며, 청산가가 손절가보다 가까워지면 자동으로 낮춘다.
  레버리지를 올려도 risk 가 같으면 포지션 크기·손익은 거의 같고 증거금만 줄어든다(크기를 키우려면 risk 를 올려야 함).
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


MAX_LEVERAGE, MAX_RISK = 20, 0.05


def size_for(budget: float, stop_pct: float, risk: float = fs.RISK, leverage: int = int(fs.LEV_CAP)) -> tuple[float, int]:
    """(명목금액 USDT, 거래소에 설정할 레버리지).
    - 명목금액 = budget×risk/손절폭 (손절 시 손실 = budget×risk), 단 budget×leverage 이하.
    - 레버리지는 요청값을 쓰되, 청산가(≈ 진입가에서 90%/레버리지)가 손절가보다 가까워지면 손절이 먼저 닿도록 낮춘다."""
    notional = min(budget * risk / stop_pct, budget * leverage)
    safe = max(1, math.floor(0.9 / (stop_pct * 1.5)))         # 청산 거리 ≥ 손절폭×1.5
    return notional, max(1, min(leverage, safe))


def tp1_reached(side: int, tp1: float, bars: pd.DataFrame) -> bool:
    return bool((bars["high"] >= tp1).any()) if side > 0 else bool((bars["low"] <= tp1).any())


def _record(state: dict, o: dict, status: str, r: float) -> str:
    net = fs.net_r(r, o["r"] / o["entry"])
    state.setdefault("done", []).append({"side": o["side"], "time": o["bar_time"], "status": status, "net_r": net})
    state["open"] = None
    return f"📒 {'롱' if o['side'] > 0 else '숏'} 종료: {status} ({net:+.2f}R, 비용 반영 추정)\n{fs.tally(state['done'])}"


def run_once(broker, bars: pd.DataFrame, state: dict, now: pd.Timestamp, budget: float,
             risk: float = fs.RISK, leverage: int = int(fs.LEV_CAP)) -> list[str]:
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
    notional, lev = size_for(budget, p["stop_pct"], risk, leverage)
    qty = notional / px
    broker.leverage = lev
    broker.open("buy" if sig > 0 else "sell", qty, p["stop"])
    state["open"] = {"side": sig, "bar_time": last.isoformat(), "entry": px, "r": p["r"], "stop": p["stop"],
                     "tp1": p["tp1"], "tp2": p["tp2"], "tp1_done": False}
    state["entries"] = state.get("entries", 0) + 1
    state["last_signal_bar"] = last.isoformat()
    vol_x = float(d["volume"].iloc[-1] / d["vavg"].iloc[-1])
    lev_note = "" if lev == leverage else f" · 요청 {leverage}배는 청산가가 손절가보다 가까워 {lev}배로 낮춤"
    msgs.append(f"{'📈 롱' if sig > 0 else '📉 숏'} 진입 {qty:.4f} ETH @ ~{fs.fmt_price(px)} "
                f"(명목 {notional:,.0f} USDT, 레버리지 {lev}배{lev_note}, 거래량 {vol_x:.1f}배)\n"
                f"손절 {fs.fmt_price(p['stop'])} (거래소 주문) · 목표1 {fs.fmt_price(p['tp1'])} · 목표2 {fs.fmt_price(p['tp2'])}\n"
                f"손절 시 손실 ≈ {notional * p['stop_pct']:,.0f} USDT (기준금액의 {notional * p['stop_pct'] / budget:.1%}) · 오늘 진입 {state['entries']}/{MAX_PER_DAY}")
    return msgs


RSI_SYMBOLS = ("ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT", "BNB/USDT:USDT")  # BTC 는 섞기 봇 몫
MAX_OPEN = 3      # 동시에 들고 있을 수 있는 코인 수(코인당 포지션 1개)


def run_once_rsi(broker, bars: pd.DataFrame, state: dict, now: pd.Timestamp, budget: float,
                 risk: float = fs.RISK, leverage: int = int(fs.LEV_CAP), tf_hours: int = 4,
                 symbol: str = SYMBOL, can_enter: bool = True) -> list[str]:
    """RSI반전+EMA200 전략(사용자 제공 Pine). bars: 마감된 N시간봉. 손절은 거래소 주문, 익절·RSI50 청산은 봇이 시장가로."""
    from . import rsi_ema
    msgs: list[str] = []
    today = kst_day(now)
    if state.get("day") != today:
        state["day"], state["entries"] = today, 0
    d = rsi_ema.indicators(bars)
    pos = broker.position()
    o = state.get("open")
    if o:
        side_kr = "롱" if o["side"] > 0 else "숏"
        if pos is None:      # 거래소 손절 주문으로 정리됨
            msgs.append(_record(state, o, "손절(거래소 주문)", -1.0))
        else:
            hit = rsi_ema.exit_check(o["side"], o["stop"], o["tp"], d[d.index > pd.Timestamp(o["bar_time"])])
            if hit:
                why, ref = hit
                broker.close(pos["side"], pos["contracts"])
                msgs.append(f"🔚 {side_kr} 정리: {why}")
                msgs.append(_record(state, o, why, o["side"] * (ref - o["entry"]) / o["r"]))
        pos = broker.position()
    if state.get("open") or pos is not None:
        if pos is not None and not state.get("open"):
            msgs.append(f"⚠️ 봇이 모르는 {symbol} 포지션이 있어 새로 진입하지 않음(앱에서 직접 연 포지션?)")
        return msgs
    last = d.index[-1]
    if now - (last + pd.Timedelta(hours=tf_hours)) > pd.Timedelta(hours=3):
        return msgs + ["⚠️ 시세가 오래되어 판단 보류"]
    sig = int(rsi_ema.signals(d).iloc[-1])
    if sig == 0 or state.get("last_signal_bar") == last.isoformat():
        return msgs
    if state.get("entries", 0) >= MAX_PER_DAY:
        return msgs + [f"⏸ 오늘 진입 {MAX_PER_DAY}번을 다 써서 이번 신호({'롱' if sig > 0 else '숏'})는 건너뜀"]
    if not can_enter:
        state["last_signal_bar"] = last.isoformat()
        return msgs + [f"⏸ 이미 {MAX_OPEN}개 코인을 보유 중이라 이번 신호({'롱' if sig > 0 else '숏'})는 건너뜀"]
    c, atr = float(d["close"].iloc[-1]), float(d["atr"].iloc[-1])
    stop, tp = c - sig * rsi_ema.SL_MULT * atr, c + sig * rsi_ema.TP_MULT * atr
    px = float(broker.price())
    stop_pct = abs(px - stop) / px
    if (sig > 0 and px <= stop) or (sig < 0 and px >= stop):
        return msgs + ["⚠️ 신호 이후 가격이 이미 손절가를 넘어 진입하지 않음"]
    notional, lev = size_for(budget, stop_pct, risk, leverage)
    qty = notional / px
    broker.leverage = lev
    broker.open("buy" if sig > 0 else "sell", qty, stop)
    state["open"] = {"side": sig, "bar_time": last.isoformat(), "entry": px, "r": abs(px - stop), "stop": stop, "tp": tp}
    state["entries"] = state.get("entries", 0) + 1
    state["last_signal_bar"] = last.isoformat()
    lev_note = "" if lev == leverage else f" · 요청 {leverage}배는 청산가가 손절가보다 가까워 {lev}배로 낮춤"
    msgs.append(f"{'📈 롱' if sig > 0 else '📉 숏'} 진입 {qty:.4f} {symbol.split('/')[0]} @ ~{fs.fmt_price(px)} "
                f"(RSI {d['rsi'].iloc[-1]:.0f}, EMA200 {fs.fmt_price(float(d['ema'].iloc[-1]))}, 명목 {notional:,.0f} USDT, 레버리지 {lev}배{lev_note})\n"
                f"손절 {fs.fmt_price(stop)} (거래소 주문) · 익절 {fs.fmt_price(tp)} · RSI 50 도달 시 정리\n"
                f"손절 시 손실 ≈ {notional * stop_pct:,.0f} USDT (기준금액의 {notional * stop_pct / budget:.1%}) · 오늘 진입 {state['entries']}/{MAX_PER_DAY}")
    return msgs


def run_rsi_multi(brokers: dict, bars: dict, state: dict, now: pd.Timestamp, budget: float,
                  risk: float = fs.RISK, leverage: int = int(fs.LEV_CAP), tf_hours: int = 4) -> list[str]:
    """여러 코인에 RSI반전 전략. 코인당 포지션 1개, 동시 보유 최대 MAX_OPEN 개, 진입은 전체 합쳐 하루 MAX_PER_DAY 번.
    상태: state['rsi_multi'] = {day, entries, done(공통 성적표), coins: {심볼: {open, last_signal_bar}}}."""
    rs = state.setdefault("rsi_multi", {"coins": {}, "done": []})
    today = kst_day(now)
    if rs.get("day") != today:
        rs["day"], rs["entries"] = today, 0
    msgs: list[str] = []
    for sym in brokers:
        if sym not in bars or bars[sym] is None or len(bars[sym]) < 210:
            continue
        cs = rs["coins"].setdefault(sym, {})
        n_open = sum(1 for k, v in rs["coins"].items() if k != sym and v.get("open"))
        cs.update(day=rs["day"], entries=rs.get("entries", 0), done=rs["done"])
        try:
            out = run_once_rsi(brokers[sym], bars[sym], cs, now, budget, risk, leverage, tf_hours,
                               symbol=sym, can_enter=n_open < MAX_OPEN)
        except Exception as e:      # 한 코인 오류가 나머지 코인 관리를 막지 않게
            out = [f"⚠️ 오류로 이번 회차 건너뜀: {type(e).__name__}: {str(e)[:150]}"]
        rs["entries"] = cs.pop("entries")
        cs.pop("day", None)
        cs.pop("done", None)
        coin = sym.split("/")[0]
        msgs += [f"[{coin}] {m}" for m in out]
    return msgs


def main() -> None:
    import ccxt
    from trader import notify
    from .backtest_report import fetch_history
    from .exchange import BitgetFutures
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=1000.0, help="이 봇이 쓰는 기준 금액(USDT). 손절 1번 = 이 금액의 1%")
    ap.add_argument("--state", default="scalp_state.json")
    ap.add_argument("--leverage", type=int, default=int(fs.LEV_CAP), help=f"거래소 레버리지(1~{MAX_LEVERAGE})")
    ap.add_argument("--risk", type=float, default=fs.RISK, help=f"손절 1번에 잃을 기준금액 비율(0~{MAX_RISK})")
    ap.add_argument("--symbols", default=",".join(x.split("/")[0] for x in RSI_SYMBOLS),
                    help="rsi_ema 전략에서 볼 코인(쉼표 구분, BTC 제외 권장)")
    ap.add_argument("--strategy", choices=("breakout", "rsi_ema"), default="breakout",
                    help="breakout: 1시간봉 돌파(기존) / rsi_ema: 4시간봉 RSI반전+EMA200(사용자 제공 Pine)")
    a = ap.parse_args()
    if not 1 <= a.leverage <= MAX_LEVERAGE:
        raise SystemExit(f"--leverage 는 1~{MAX_LEVERAGE}")
    if not 0 < a.risk <= MAX_RISK:
        raise SystemExit(f"--risk 는 0 초과 {MAX_RISK} 이하")
    state = load_state(a.state)
    try:
        public = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        now = pd.Timestamp.now(tz="UTC")
        if a.strategy == "rsi_ema":
            syms = [f"{c.strip().upper()}/USDT:USDT" for c in a.symbols.split(",") if c.strip()]
            brokers, bars = {}, {}
            for sym in syms:
                try:
                    brokers[sym] = BitgetFutures.from_env(sym, 1, demo=True)   # 데모 전용
                    bars[sym] = fetch_history(public, sym, timeframe="4h", years=0.2)
                except Exception as e:
                    notify.send(f"⚠️ 단타봇 {sym.split('/')[0]} 준비 실패(이번 회차 제외): {type(e).__name__}: {str(e)[:150]}")
                    brokers.pop(sym, None)
            msgs = run_rsi_multi(brokers, bars, state, now, a.budget, a.risk, a.leverage, tf_hours=4)
        else:
            broker = BitgetFutures.from_env(SYMBOL, 1, demo=True)   # 데모 전용
            bars = fetch_history(public, SYMBOL, timeframe="1h", years=0.05, keep_volume=True)
            msgs = run_once(broker, bars, state, now, a.budget, a.risk, a.leverage)
    except Exception as e:
        notify.send(f"❌ 단타봇 오류 [데모/{a.strategy}]: {type(e).__name__}: {str(e)[:300]}")
        raise
    finally:
        save_state(a.state, state)
    for m in msgs:
        notify.send(f"⚡ 단타봇 [데모/{'4시간봉 RSI반전 ' + a.symbols if a.strategy == 'rsi_ema' else 'ETH 1시간봉 돌파'}]\n{m}")
    print("\n".join(msgs) or "변화 없음")


if __name__ == "__main__":
    main()
