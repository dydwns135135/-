"""파인스크립트 3종(샹들리에·EMA+거래량·볼린저+RSI)의 신호를 CCXT 로 현물 자동매매하는 봇. 바이낸스(USDT)·업비트(KRW) 지원.
python -m crypto.spot_bot --exchange upbit --symbol BTC/KRW --timeframe 1d --strategy chandelier            # 모의(주문 안 함, 기본)
python -m crypto.spot_bot --exchange binance --symbol BTC/USDT --timeframe 4h --strategy chandelier --live --budget 100   # 실거래

- 신호는 백테스트와 같은 함수(crypto.chandelier / ema_volume / bb_rsi)를 그대로 쓴다. 마감된 봉만 사용하고, 봉마다 한 번만 판단한다.
- 현물·롱만·1배·시장가 전액 매수/전량 매도. 숏·레버리지·선물은 지원하지 않는다(의도적).
- 기본은 모의 모드: 거래소 키 없이 공개 시세만 읽고, 가상 포지션을 상태 파일에 기록하며 텔레그램으로 알린다.
- 실거래(--live)는 --budget(quote 통화 기준 금액 상한)이 필수이고 BINANCE_API_KEY/SECRET 또는 UPBIT_API_KEY/SECRET 환경변수가 있어야 한다.
  키는 '출금 권한 없음'으로 만들고, 업비트는 키마다 허용 IP 등록이 필수라 고정 IP 서버에서만 동작한다(GitHub Actions 불가).
- 상태가 필요한 신호(볼린저+RSI 의 진입 시점)는 상태 파일에 기록한다. 샹들리에·EMA 신호는 최근 200봉으로 방향을 다시 계산한다.
수익을 보장하지 않는다. 백테스트로 확인되지 않은 규칙으로 실거래하지 말 것."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from . import bb_rsi, chandelier, ema_volume

STRATEGIES = ("chandelier", "ema_volume", "bb_rsi")
BARS = 200
FEE_BUFFER = 0.995          # 시장가 매수 시 수수료 여유


def to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    return df[~df.index.duplicated()].sort_index()


def closed_bars(ex, symbol: str, timeframe: str) -> pd.DataFrame:
    df = to_frame(ex.fetch_ohlcv(symbol, timeframe, limit=BARS))
    return df.iloc[:-1]                      # 마지막은 진행 중인 봉


def decide(strategy: str, df: pd.DataFrame, entry_bar: str | None) -> tuple[bool, str, str | None]:
    """(보유해야 하는가, 이유, 새 entry_bar). entry_bar 는 볼린저+RSI 의 진입 봉(상태)."""
    if strategy == "chandelier":
        d = float(chandelier.direction(df).iloc[-1])
        return d > 0, f"샹들리에 방향 {'롱' if d > 0 else '숏/대기'}", None
    if strategy == "ema_volume":
        b, s = ema_volume.buy_sell(df)
        pos = float(ema_volume.positions(b, s, False).iloc[-1])
        return pos > 0, f"EMA20/50 교차 {'매수 상태' if pos > 0 else '청산 상태'}", None
    if strategy == "bb_rsi":
        last = df.index[-1]
        if entry_bar is None:
            if bool(bb_rsi.buy_condition(df).iloc[-1]):
                return True, "볼린저 하단 이탈 + RSI<30 진입", last.isoformat()
            return False, "진입 조건 없음", None
        since = int((df.index > pd.Timestamp(entry_bar)).sum())
        mid = float(df["close"].rolling(bb_rsi.BB_LEN).mean().iloc[-1])
        if df["close"].iloc[-1] > mid or since >= bb_rsi.TIMEOUT:
            return False, "중심선 회복" if df["close"].iloc[-1] > mid else f"{bb_rsi.TIMEOUT}봉 경과 청산", None
        return True, f"보유 유지({since}봉째)", entry_bar
    raise ValueError(strategy)


def load_state(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def save_state(path: str, state: dict) -> None:
    Path(path).write_text(json.dumps(state, ensure_ascii=False, indent=1))


def min_cost(ex, symbol: str) -> float:
    try:
        return float(((ex.markets[symbol].get("limits") or {}).get("cost") or {}).get("min") or 0)
    except (KeyError, TypeError, ValueError):
        return 0.0


def run_once(ex, args, send, state_path: str) -> dict:
    """한 번 실행: 마감 봉 기준으로 판단하고 필요하면 주문(실거래) 또는 가상 기록(모의)을 한다."""
    state = load_state(state_path)
    df = closed_bars(ex, args.symbol, args.timeframe)
    if len(df) < 120:
        send(f"⚠️ 현물봇: 봉이 부족해 판단 보류 ({len(df)}개)")
        return state
    last_bar = df.index[-1].isoformat()
    if state.get("last_bar") == last_bar:
        return state                                         # 이미 처리한 봉
    want, reason, new_entry = decide(args.strategy, df, state.get("entry_bar"))
    base, quote = args.symbol.split("/")
    mode = "실거래" if args.live else "모의"
    price = float(df["close"].iloc[-1])
    if args.live:
        price = float(ex.fetch_ticker(args.symbol)["last"])
        bal = ex.fetch_balance()
        base_free = float((bal.get(base) or {}).get("free") or 0)
        quote_free = float((bal.get(quote) or {}).get("free") or 0)
        holding = base_free * price >= max(min_cost(ex, args.symbol), 1e-12)
    else:
        holding = bool(state.get("paper_long", False))
    if args.strategy == "bb_rsi" and want and not holding and state.get("entry_bar"):
        want, reason, new_entry = False, "진입 기록은 있으나 실제 포지션 없음 → 상태 초기화", None
    tag = f"🤖 현물봇 [{mode}] {ex.id.upper()} {args.symbol} {args.timeframe} {args.strategy}"
    if want and not holding:
        cost = min(args.budget * args.alloc if args.budget else float("inf"), quote_free if args.live else float("inf"))
        if args.live and cost < max(min_cost(ex, args.symbol), 1e-12):
            send(f"{tag}\n매수 신호지만 주문 가능 금액이 최소 주문액 미만이라 건너뜀 ({cost:,.2f} {quote})")
        else:
            if args.live:
                amount = float(ex.amount_to_precision(args.symbol, cost * FEE_BUFFER / price))
                ex.create_order(args.symbol, "market", "buy", amount, price)   # 업비트는 price 로 주문총액 계산
                send(f"{tag}\n✅ 매수 {amount} {base} (≈{amount * price:,.0f} {quote}) · {reason}")
            else:
                state["paper_long"], state["paper_entry"] = True, price
                send(f"{tag}\n(모의) 매수 신호 @ {price:,.2f} · {reason}")
            state["entry_bar"] = new_entry
    elif not want and holding:
        if args.live:
            amount = float(ex.amount_to_precision(args.symbol, base_free))
            ex.create_order(args.symbol, "market", "sell", amount)
            send(f"{tag}\n✅ 전량 매도 {amount} {base} · {reason}")
        else:
            entry = state.get("paper_entry") or price
            state["paper_long"] = False
            send(f"{tag}\n(모의) 매도 신호 @ {price:,.2f} ({(price / entry - 1) * 100:+.1f}%) · {reason}")
        state["entry_bar"] = None
    else:
        state["entry_bar"] = new_entry if (want and holding) else None
    state["last_bar"] = last_bar
    save_state(state_path, state)
    return state


def build_exchange(args):
    import ccxt
    cfg = {"enableRateLimit": True}
    if args.live:
        key, secret = (os.environ.get(f"{args.exchange.upper()}_API_KEY"), os.environ.get(f"{args.exchange.upper()}_API_SECRET"))
        if not key or not secret:
            raise SystemExit(f"실거래에는 {args.exchange.upper()}_API_KEY / {args.exchange.upper()}_API_SECRET 환경변수가 필요합니다.")
        cfg.update(apiKey=key, secret=secret)
    ex = getattr(ccxt, args.exchange)(cfg)
    ex.load_markets()
    return ex


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="파인스크립트 신호 현물 자동매매 봇(기본: 모의)")
    ap.add_argument("--exchange", choices=("binance", "upbit"), required=True)
    ap.add_argument("--symbol", required=True, help="예: BTC/USDT (바이낸스), BTC/KRW (업비트)")
    ap.add_argument("--timeframe", default="4h", choices=("1h", "4h", "1d"))
    ap.add_argument("--strategy", choices=STRATEGIES, required=True)
    ap.add_argument("--alloc", type=float, default=1.0, help="예산 중 사용할 비율(0~1)")
    ap.add_argument("--budget", type=float, default=0.0, help="사용할 최대 금액(quote 통화 기준). 실거래 필수")
    ap.add_argument("--live", action="store_true", help="실제 주문(기본은 모의)")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    if not 0 < a.alloc <= 1:
        raise SystemExit("--alloc 은 0 초과 1 이하")
    if a.live and a.budget <= 0:
        raise SystemExit("실거래에는 --budget(최대 사용 금액)이 필수입니다.")
    if a.exchange == "upbit" and not a.symbol.endswith("/KRW"):
        raise SystemExit("업비트는 */KRW 마켓만 지원합니다.")
    if a.exchange == "binance" and not a.symbol.endswith("/USDT"):
        raise SystemExit("바이낸스는 */USDT 마켓만 지원합니다.")
    a.state = a.state or f"spot_bot_{a.exchange}_{a.symbol.replace('/', '')}_{a.strategy}_{a.timeframe}.json"
    return a


def main(argv=None) -> None:
    from trader.notify import send
    args = parse_args(argv)
    ex = build_exchange(args)
    try:
        run_once(ex, args, send, args.state)
    except Exception as e:
        send(f"⚠️ 현물봇 오류: {type(e).__name__}: {str(e)[:150]}")
        raise


if __name__ == "__main__":
    main()
