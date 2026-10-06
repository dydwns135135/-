"""Bitget USDT 무기한 선물 브로커 (ccxt).

!! 미검증 !!  이 개발 환경에서 Bitget에 접속할 수 없어 실제 호출로 확인하지 못했다.
특히 (1) 데모(샌드박스) 모드의 심볼/키, (2) 주문에 손절 붙이는 params 형식, (3) 포지션 응답 필드는
ccxt·Bitget 버전에 따라 다를 수 있으니 데모 모드에서 먼저 검증할 것(`TODO(verify)`).
"""
from __future__ import annotations

import os

import ccxt
import pandas as pd


def configure_demo(ex: ccxt.Exchange) -> None:
    """데모(모의) 거래 설정: 일반 종목(BTC/USDT:USDT)·USDT 증거금에 PAPTRADING 헤더를 붙인다.
    - 데모 계정은 통합 계정(UTA)이다: 키 권한 화면이 'Unified account' 이고, 데모 키로 UTA 설정 조회가
      성공했다(run #1~#5 에서 ccxt 가 UTA 로 자동 판별). 따라서 UTA API 를 쓴다.
      (일반 계정 API 로 강제하면 40014 권한 오류 — run #9, #10)
    - 헤더는 모든 요청에 강제한다."""
    ex.options["uta"] = True
    ex.set_sandbox_mode(True)
    ex.headers = {**(ex.headers or {}), "PAPTRADING": "1"}


class BitgetFutures:
    def __init__(self, ex: ccxt.Exchange, symbol: str, leverage: int,
                 data_symbol: str | None = None, demo: bool = False):
        """symbol: 주문 종목, data_symbol: 시세/캔들용 종목(기본 symbol 과 동일)."""
        self.ex, self.symbol, self.leverage = ex, symbol, leverage
        self.data_symbol, self.demo = data_symbol or symbol, demo
        self.setup_errors: list[str] = []  # 격리마진/레버리지 설정 실패 기록(진단용)
        self.hedged: bool | None = None  # 계좌 포지션 모드: 주문이 성공한 방식으로 학습(None=아직 모름)
        ex.load_markets()

    @classmethod
    def from_env(cls, symbol: str, leverage: int, demo: bool) -> "BitgetFutures":
        ex = ccxt.bitget({
            "apiKey": os.environ["BITGET_API_KEY"],
            "secret": os.environ["BITGET_API_SECRET"],
            "password": os.environ["BITGET_API_PASSPHRASE"],
            "options": {"defaultType": "swap"},
            "enableRateLimit": True,
        })
        if demo:
            # 데모: 일반 종목(BTC/USDT:USDT) + USDT 증거금 + PAPTRADING 헤더 + 일반 선물(classic) API (UTA 자동감지 끔).
            configure_demo(ex)
            return cls(ex, symbol, leverage, demo=True)
        return cls(ex, symbol, leverage)

    def closed_candles(self, timeframe: str, limit: int = 300, deep: bool = False) -> pd.DataFrame:
        """마감된 봉만. deep=True 는 기간(since)을 지정해 과거 봉을 이어 받는다 — Bitget 의 기본 캔들 조회는
        일봉을 최근 ~90개까지만 주므로(run: 89개), 90일 이상 필요한 규칙은 deep 으로 받는다."""
        dur_ms = int(self.ex.parse_timeframe(timeframe) * 1000)
        if deep:
            since, rows = self._now_ms() - (limit + 2) * dur_ms, []
            for _ in range(8):
                batch = self.ex.fetch_ohlcv(self.data_symbol, timeframe, since=since, limit=min(limit + 2, 200))
                new = [r for r in batch if not rows or r[0] > rows[-1][0]]
                if not new:
                    break
                rows += new
                since = rows[-1][0] + 1
                if rows[-1][0] + dur_ms >= self._now_ms():  # 현재 봉까지 도달
                    break
        else:
            rows = self.ex.fetch_ohlcv(self.data_symbol, timeframe, limit=limit)
        closed = [r for r in rows if r[0] + dur_ms <= self._now_ms()]  # 시각이 지나 마감된 봉만
        df = pd.DataFrame(closed, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        return df[["open", "high", "low", "close"]]

    @staticmethod
    def _now_ms() -> int:
        import time
        return int(time.time() * 1000)

    def price(self) -> float:
        return float(self.ex.fetch_ticker(self.data_symbol)["last"])

    def _bal_params(self) -> tuple[dict, str]:
        return {}, "USDT"  # 데모도 일반 종목·USDT 증거금(Bitget 데모 화면: BTCUSDT, USDT 잔고)

    def balance(self) -> float:
        params, coin = self._bal_params()
        return float(self.ex.fetch_balance(params)[coin]["total"])

    def position(self) -> dict | None:
        """{'side': 'long'|'short', 'contracts': float, 'entry': float} 또는 None."""
        for p in self.ex.fetch_positions([self.symbol]):
            if float(p.get("contracts") or 0) > 0:
                return {"side": p["side"], "contracts": float(p["contracts"]), "entry": float(p["entryPrice"])}
        return None

    def open(self, side: str, amount: float, stop_price: float | None = None) -> None:
        """side: 'buy'(롱) | 'sell'(숏). 격리마진 시도 + (지정하면) 거래소 측 손절 주문."""
        for fn, arg in ((self.ex.set_margin_mode, "isolated"), (self.ex.set_leverage, self.leverage)):
            try:
                fn(arg, self.symbol)
            except ccxt.BaseError as e:  # 이미 같은 값이면 오류가 날 수 있음 → 기록만 하고 진행
                self.setup_errors.append(f"{fn.__name__}: {str(e)[:160]}")
                if not self.demo and "40014" in str(e):  # 실거래에서 권한 부족이면 격리/레버리지 미설정 상태로 주문하지 않는다
                    raise
        amt = float(self.ex.amount_to_precision(self.symbol, amount))
        self._order(side, amt, {"stopLoss": {"triggerPrice": stop_price}} if stop_price else {})

    def _order(self, side: str, amount: float, params: dict):
        """단방향/헤지 모드를 모르면 단방향으로 먼저 시도하고, Bitget 오류 25236
        (Incorrect position open type)이면 헤지 방식으로 재시도한다. 성공한 방식은 기억한다."""
        modes = [self.hedged] if self.hedged is not None else [False, True]
        err: Exception | None = None
        for hedged in modes:
            try:
                order = self.ex.create_order(self.symbol, "market", side, amount,
                                             params={**params, "hedged": hedged})
                self.hedged = hedged
                return order
            except ccxt.ExchangeError as e:
                if "25236" not in str(e):
                    raise
                err = e
        raise err  # type: ignore[misc]

    def test_amount(self, min_notional: float = 7.0) -> float:
        """거래소 최소 수량 단위로, 명목가치가 min_notional USDT 이상 되는 가장 작은 수량."""
        import math
        step = float(self.ex.market(self.symbol)["limits"]["amount"]["min"] or 0.001)
        return round(math.ceil(min_notional / (self.price() * step)) * step, 8)

    def stop_info(self) -> dict:
        """손절 주문이 붙었는지 확인하기 위한 진단 정보(거래소 응답 그대로 일부)."""
        out: dict = {}
        for name, params in (("trigger_orders", {"trigger": True}), ("stop_orders", {"stop": True})):
            try:
                out[name] = [
                    {k: o.get(k) for k in ("id", "type", "side", "amount", "triggerPrice", "stopLossPrice", "reduceOnly")}
                    for o in self.ex.fetch_open_orders(self.symbol, params=params)
                ]
            except ccxt.BaseError as e:
                out[name] = f"조회 실패: {type(e).__name__}"
        for p in self.ex.fetch_positions([self.symbol]):
            if float(p.get("contracts") or 0) > 0:
                out["position_stop"] = {k: v for k, v in (p.get("info") or {}).items() if "stop" in k.lower() or "sl" == k.lower()[:2]}
        return out

    def diagnose(self) -> dict:
        """주문 실패 시 원인 파악용: 잔고, 데모 종목 존재 여부, 설정 오류."""
        out: dict = {"symbol": self.symbol, "leverage": self.leverage, "hedged": self.hedged}
        try:
            out["settle"] = self.ex.market(self.symbol).get("settle")
            out["market_id"] = self.ex.market(self.symbol).get("id")
            out["data_symbol"], out["demo"] = self.data_symbol, self.demo
        except ccxt.BaseError as e:
            out["market"] = f"조회 실패: {type(e).__name__}"
        try:
            total = self.ex.fetch_balance(self._bal_params()[0]).get("total") or {}
            out["balances"] = {k: v for k, v in total.items() if v}
        except ccxt.BaseError as e:
            out["balances"] = f"조회 실패: {type(e).__name__}: {str(e)[:120]}"
        out["setup_errors"] = self.setup_errors
        return out

    def close(self, side: str, contracts: float) -> None:
        opp = "sell" if side == "long" else "buy"
        amt = float(self.ex.amount_to_precision(self.symbol, contracts))
        self._order(opp, amt, {"reduceOnly": True})

