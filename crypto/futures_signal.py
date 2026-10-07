"""선물 단타 진입 신호(롱·숏)를 찾아 텔레그램으로 알려 주는 알림 전용 도구. 주문은 하지 않는다.

신호 규칙(미리 정해 고정, 결과를 보고 바꾸지 않음) — Bitget USDT 무기한 BTC·ETH·SOL·XRP, 1시간봉:
- 롱: 종가가 직전 24봉 최고가를 넘고, 거래량이 직전 24봉 평균의 2배 이상이며, 종가가 EMA(100) 위
- 숏: 반대(직전 24봉 최저가 이탈 + 거래량 2배 + 종가가 EMA(100) 아래)
- 손절: 진입가 ∓ 1.5×ATR(14)  (= 1R)   목표1: +1.5R(절반 정리)   목표2: +3R(나머지)   48봉 지나면 정리
- 권장 레버리지: 손절 시 손실이 투입금의 1%가 되도록 계산(=1%/손절폭), 최대 3배로 제한
백테스트:  python -m crypto.futures_signal --backtest [--years 3]
알림:     python -m crypto.futures_signal --alert   (신호·결과를 signals_state.json 에 기록, 성적표 포함)
손절과 목표가 같은 봉에서 동시에 닿으면 손절이 먼저라고 보수적으로 계산한다. 펀딩비는 반영하지 않았다(보유 몇 시간~이틀)."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT")
N_BREAK, VOL_MULT, EMA_N, ATR_N = 24, 2.0, 100, 14
STOP_ATR, TP1_R, TP2_R, TIMEOUT = 1.5, 1.5, 3.0, 48
RISK, LEV_CAP = 0.01, 3.0
RT_COST = 0.0022          # 왕복 비용: 수수료 0.06%×2 + 슬리피지 0.05%×2
MIN_BARS = 150


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    prev_close = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev_close).abs(), (d["low"] - prev_close).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(ATR_N).mean()
    d["ema"] = d["close"].ewm(span=EMA_N, adjust=False).mean()
    d["hh"] = d["high"].rolling(N_BREAK).max().shift(1)
    d["ll"] = d["low"].rolling(N_BREAK).min().shift(1)
    d["vavg"] = d["volume"].rolling(N_BREAK).mean().shift(1)
    return d


def signals(d: pd.DataFrame) -> pd.Series:
    """봉 마감 시점의 신호: +1 롱, -1 숏, 0 없음(준비 구간은 0)."""
    vol_ok = d["volume"] >= VOL_MULT * d["vavg"]
    long_ = (d["close"] > d["hh"]) & vol_ok & (d["close"] > d["ema"])
    short = (d["close"] < d["ll"]) & vol_ok & (d["close"] < d["ema"])
    out = pd.Series(0, index=d.index)
    out[long_.fillna(False)] = 1
    out[short.fillna(False)] = -1
    out[d["atr"].isna() | (d.index < d.index[min(MIN_BARS, len(d) - 1)])] = 0
    return out


def plan(entry: float, atr: float, side: int) -> dict:
    r = STOP_ATR * atr
    stop_pct = r / entry
    lev = min(LEV_CAP, RISK / stop_pct)
    lev = math.floor(lev * 10) / 10
    return {"side": side, "entry": entry, "r": r, "stop": entry - side * r,
            "tp1": entry + side * TP1_R * r, "tp2": entry + side * TP2_R * r,
            "stop_pct": stop_pct, "leverage": lev, "liq": entry * (1 - side * 0.9 / lev) if lev > 0 else None}


def resolve(side: int, entry: float, r: float, bars: pd.DataFrame) -> dict:
    """진입 이후 봉들로 결과 판정. R(손절폭 배수) 기준, 비용 반영 전. status: 진행중 / 손절 / 목표1후손절 / 목표2 / 시간초과."""
    stop, tp1, tp2 = entry - side * r, entry + side * TP1_R * r, entry + side * TP2_R * r
    remaining, realized, tp1_done = 1.0, 0.0, False
    for j, (hi, lo, cl) in enumerate(zip(bars["high"], bars["low"], bars["close"])):
        if (lo <= stop) if side > 0 else (hi >= stop):
            return {"status": "목표1후손절" if tp1_done else "손절", "r": realized - remaining, "held": j + 1}
        if not tp1_done and ((hi >= tp1) if side > 0 else (lo <= tp1)):
            realized, remaining, tp1_done = realized + 0.5 * TP1_R, 0.5, True
        if tp1_done and ((hi >= tp2) if side > 0 else (lo <= tp2)):
            return {"status": "목표2", "r": realized + remaining * TP2_R, "held": j + 1}
        if j + 1 >= TIMEOUT:
            return {"status": "시간초과", "r": realized + remaining * side * (cl - entry) / r, "held": j + 1}
    return {"status": "진행중", "r": None, "held": len(bars)}


def net_r(res_r: float, stop_pct: float) -> float:
    return res_r - RT_COST / stop_pct


def backtest_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    d = prepare(df)
    sig = signals(d)
    opens, atrs = d["open"].to_numpy(), d["atr"].to_numpy()
    rows, free_at = [], 0
    for i in np.flatnonzero(sig.to_numpy() != 0):
        if i < free_at or i + 1 >= len(d):
            continue
        side = int(sig.iloc[i])
        entry, r = opens[i + 1], STOP_ATR * atrs[i]
        res = resolve(side, entry, r, d.iloc[i + 1: i + 1 + TIMEOUT])
        if res["status"] == "진행중":
            continue
        free_at = i + 1 + res["held"]
        rows.append({"symbol": symbol, "time": d.index[i], "exit": d.index[min(i + res["held"], len(d) - 1)], "side": side,
                     "status": res["status"], "gross_r": res["r"], "net_r": net_r(res["r"], r / entry), "held": res["held"]})
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame, days: float) -> dict:
    if trades.empty:
        return {"신호수": 0, "코인당 하루신호": 0.0, "승률": float("nan"), "평균R(비용후)": float("nan"), "평균R(비용전)": float("nan"),
                "합계R": 0.0, "최대낙폭": 0.0}
    t = trades.sort_values("exit")
    eq = (1 + RISK * t["net_r"]).cumprod()
    return {"신호수": len(t), "코인당 하루신호": len(t) / max(days, 1), "승률": float((t["net_r"] > 0).mean()),
            "평균R(비용후)": float(t["net_r"].mean()), "평균R(비용전)": float(t["gross_r"].mean()),
            "합계R": float(t["net_r"].sum()), "최대낙폭": float((eq / eq.cummax() - 1).min())}


def fmt_summary(rows: dict[str, dict]) -> str:
    t = pd.DataFrame(rows).T
    t["신호수"] = t["신호수"].astype(int)
    t["코인당 하루신호"] = t["코인당 하루신호"].map(lambda x: f"{x:.2f}")
    t["승률"] = t["승률"].map(lambda x: "-" if x != x else f"{x * 100:.0f}%")
    for c in ("평균R(비용후)", "평균R(비용전)", "합계R"):
        t[c] = t[c].map(lambda x: "-" if x != x else f"{x:+.2f}")
    t["최대낙폭"] = t["최대낙폭"].map(lambda x: f"{x * 100:.0f}%")
    return t.to_string()


# ---------------------------------------------------------------- 알림 ----------------------------------------------------------------
def load_state(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {"open": [], "done": []}


def save_state(path: str, state: dict) -> None:
    Path(path).write_text(json.dumps(state, ensure_ascii=False, indent=1))


def tally(done: list[dict]) -> str:
    if not done:
        return "기록된 신호 없음"
    r = [x["net_r"] for x in done]
    wins = sum(1 for x in r if x > 0)
    return f"누적 {len(r)}건 · 승률 {wins / len(r) * 100:.0f}% · 평균 {sum(r) / len(r):+.2f}R · 합계 {sum(r):+.2f}R (비용 반영)"


def fmt_price(x: float) -> str:
    return f"{x:,.2f}" if x >= 100 else f"{x:,.4f}"


def signal_message(symbol: str, p: dict, vol_x: float, done: list[dict]) -> str:
    coin = symbol.split("/")[0]
    arrow, kind = ("📈", "롱(매수)") if p["side"] > 0 else ("📉", "숏(매도)")
    cond = "직전 24봉 고점 돌파" if p["side"] > 0 else "직전 24봉 저점 이탈"
    return (f"🔔 선물 단타 신호 [알림 전용 · 주문 안 함]\n{arrow} {coin} {kind} (1시간봉: {cond}, 거래량 {vol_x:.1f}배)\n"
            f"진입 참고가 {fmt_price(p['entry'])} (다음 봉 시가 근처)\n"
            f"손절 {fmt_price(p['stop'])} ({p['stop_pct'] * 100:.2f}% 반대 방향)\n"
            f"목표1 {fmt_price(p['tp1'])} (+1.5R, 절반 정리) / 목표2 {fmt_price(p['tp2'])} (+3R)\n"
            f"권장 레버리지 최대 {p['leverage']:.1f}배 → 손절 시 손실 ≈ 투입금의 {RISK * 100:.0f}%\n"
            f"청산가는 약 {fmt_price(p['liq'])} (참고, 수수료·유지증거금 제외)\n"
            f"48시간 안에 목표/손절이 안 나오면 정리하세요. 교차마진·고배율 금지.\n"
            f"📒 {tally(done)}\n※ 백테스트로 수익이 확인된 규칙이 아닙니다. 잃을 수 있는 금액만 쓰세요.")


def result_message(x: dict, done: list[dict]) -> str:
    coin = x["symbol"].split("/")[0]
    side = "롱" if x["side"] > 0 else "숏"
    return f"📒 신호 결과: {coin} {side} → {x['status']} ({x['net_r']:+.2f}R, 비용 반영)\n{tally(done)}"


def run_alert(fetch, send, state_path: str, now: pd.Timestamp | None = None) -> dict:
    """fetch(symbol)->마감봉 DataFrame(open/high/low/close/volume). 새 신호는 알리고, 진행 중 신호는 결과를 판정한다."""
    now = now or pd.Timestamp.now(tz="UTC")
    state = load_state(state_path)
    for symbol in SYMBOLS:
        df = fetch(symbol)
        if df is None or len(df) < MIN_BARS + 10:
            continue
        d = prepare(df)
        # 1) 진행 중 신호 판정
        for x in [o for o in state["open"] if o["symbol"] == symbol]:
            bars = d[d.index > pd.Timestamp(x["bar_time"])]
            if len(bars) == 0:
                continue
            res = resolve(x["side"], x["entry"], x["r"], bars)
            if res["status"] != "진행중":
                x.update(status=res["status"], net_r=net_r(res["r"], x["r"] / x["entry"]))
                state["open"].remove(x)
                state["done"].append(x)
                send(result_message(x, state["done"]))
        # 2) 새 신호
        if any(o["symbol"] == symbol for o in state["open"]):
            continue
        last = d.index[-1]
        if now - (last + pd.Timedelta(hours=1)) > pd.Timedelta(hours=3):   # 시세가 오래됨
            continue
        s = int(signals(d).iloc[-1])
        if s == 0:
            continue
        if any(o["symbol"] == symbol and o["bar_time"] == last.isoformat() for o in state["done"]):
            continue
        entry, atr = float(d["close"].iloc[-1]), float(d["atr"].iloc[-1])
        p = plan(entry, atr, s)
        vol_x = float(d["volume"].iloc[-1] / d["vavg"].iloc[-1])
        state["open"].append({"symbol": symbol, "side": s, "bar_time": last.isoformat(), "entry": entry, "r": p["r"]})
        send(signal_message(symbol, p, vol_x, state["done"]))
    save_state(state_path, state)
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--alert", action="store_true")
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--state", default="signals_state.json")
    a = ap.parse_args()
    if a.backtest:
        from .backtest_report import load_data
        all_tr, days, lines = [], 0.0, []
        for sym in SYMBOLS:
            ex, df = load_data(a.years, symbol=sym, min_bars=3000, timeframe="1h", fallback_years=(3, 2, 1), keep_volume=True)
            tr = backtest_symbol(df, sym)
            all_tr.append(tr)
            days = max(days, (df.index[-1] - df.index[0]).days)
            lines.append(f"{sym.split('/')[0]}: {len(df)}봉 {df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d} ({ex})")
        tr = pd.concat(all_tr, ignore_index=True)
        mid = tr["time"].min() + (tr["time"].max() - tr["time"].min()) / 2
        sets = {"전체": tr, "앞 절반": tr[tr["time"] < mid], "뒤 절반": tr[tr["time"] >= mid],
                "롱만": tr[tr["side"] > 0], "숏만": tr[tr["side"] < 0]}
        out = ["데이터: " + " | ".join(lines)]
        out += ["", "=== 전체 합산 (4개 코인) ===", fmt_summary({k: summarize(v, days * (0.5 if k in ("앞 절반", "뒤 절반") else 1) * len(SYMBOLS)) for k, v in sets.items()}),
                "", "=== 코인별 ===", fmt_summary({s.split('/')[0]: summarize(tr[tr["symbol"] == s], days) for s in SYMBOLS}),
                "", "=== 결과 종류 ===", tr["status"].value_counts().to_string(),
                "", f"비용: 왕복 {RT_COST * 100:.2f}%(손절폭 대비 R로 환산), 신호당 위험 {RISK * 100:.0f}%로 복리 가정(여러 코인 동시 보유 위험은 합산 안 함).",
                "'평균R(비용후)'이 0보다 커야 수익 가능성이 있는 규칙입니다. 과거 결과이며 수익을 보장하지 않습니다."]
        text = "\n".join(out)
        print(text)
        os.makedirs("data", exist_ok=True)
        Path("data/futures_signal_backtest.txt").write_text(text)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as f:
                f.write("```\n" + text + "\n```\n")
        return
    if a.alert:
        import ccxt
        from .backtest_report import fetch_history
        from trader.notify import send
        ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        try:
            run_alert(lambda s: fetch_history(ex, s, timeframe="1h", years=0.05, keep_volume=True), send, a.state)
        except Exception as e:
            send(f"⚠️ 선물 단타 알림 오류: {type(e).__name__}: {str(e)[:150]}")
            raise
        return
    ap.print_help()


if __name__ == "__main__":
    main()
