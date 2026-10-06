"""Bitget USDT 무기한 선물 브로커 (ccxt).

!! 미검증 !!  이 개발 환경에서 Bitget에 접속할 수 없어 실제 호출로 확인하지 못했다.
특히 (1) 데모(샌드박스) 모드의 심볼/키, (2) 주문에 손절 붙이는 params 형식, (3) 포지션 응답 필드는
ccxt·Bitget 버전에 따라 다를 수 있으니 데모 모드에서 먼저 검증할 것(`TODO(verify)`).
"""
from __future__ import annotations

import os

import ccxt
import pandas as pd


class BitgetFutures:
    def __init__(self, ex: ccxt.Exchange, symbol: str, leverage: int):
        self.ex, self.symbol, self.leverage = ex, symbol, leverage
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
            ex.set_sandbox_mode(True)  # TODO(verify): 데모 전용 API 키 필요, 심볼은 SBTC/SUSDT:SUSDT 형태일 수 있음
        return cls(ex, symbol, leverage)

    def closed_candles(self, timeframe: str, limit: int = 300) -> pd.DataFrame:
        rows = self.ex.fetch_ohlcv(self.symbol, timeframe, limit=limit)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        return df.iloc[:-1][["open", "high", "low", "close"]]  # 마지막은 진행 중인 봉이라 제외

    def price(self) -> float:
        return float(self.ex.fetch_ticker(self.symbol)["last"])

    def balance(self) -> float:
        return float(self.ex.fetch_balance()["USDT"]["total"])

    def position(self) -> dict | None:
        """{'side': 'long'|'short', 'contracts': float, 'entry': float} 또는 None."""
        for p in self.ex.fetch_positions([self.symbol]):
            if float(p.get("contracts") or 0) > 0:
                return {"side": p["side"], "contracts": float(p["contracts"]), "entry": float(p["entryPrice"])}
        return None

    def open(self, side: str, amount: float, stop_price: float) -> None:
        """side: 'buy'(롱) | 'sell'(숏). 격리마진 + 거래소 측 손절 주문."""
        for fn, arg in ((self.ex.set_margin_mode, "isolated"), (self.ex.set_leverage, self.leverage)):
            try:
                fn(arg, self.symbol)
            except ccxt.BaseError:
                pass  # 이미 같은 값이면 오류가 날 수 있음
        amt = float(self.ex.amount_to_precision(self.symbol, amount))
        self.ex.create_order(self.symbol, "market", side, amt,
                             params={"stopLoss": {"triggerPrice": stop_price}})  # TODO(verify)

    def close(self, side: str, contracts: float) -> None:
        opp = "sell" if side == "long" else "buy"
        self.ex.create_order(self.symbol, "market", opp, contracts, params={"reduceOnly": True})
