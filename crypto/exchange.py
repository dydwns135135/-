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
    """데모(모의) 거래 설정. ccxt 는 S 접두 상품(SUSDT-FUTURES) 요청에는 PAPTRADING 헤더를 빼지만
    Bitget 은 데모 키 요청에 항상 이 헤더를 요구한다(없으면 40099 exchange environment is incorrect)."""
    ex.options["uta"] = False  # 일반 선물 API 사용(UTA 자동감지 끔)
    ex.set_sandbox_mode(True)
    ex.headers = {**(ex.headers or {}), "PAPTRADING": "1"}


class BitgetFutures:
    def __init__(self, ex: ccxt.Exchange, symbol: str, leverage: int,
                 data_symbol: str | None = None, demo: bool = False):
        """symbol: 주문 종목(데모면 SBTC/SUSDT:SUSDT), data_symbol: 시세/캔들용 실제 종목."""
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
            # 데모는 일반 선물(classic) API + 데모 종목(SBTC/SUSDT) + SUSDT 증거금으로 동작(UTA 자동감지 끔).
            configure_demo(ex)
            return BitgetDemoFutures(ex, symbol, leverage)
        return cls(ex, symbol, leverage)

    def closed_candles(self, timeframe: str, limit: int = 300) -> pd.DataFrame:
        rows = self.ex.fetch_ohlcv(self.data_symbol, timeframe, limit=limit)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        return df.iloc[:-1][["open", "high", "low", "close"]]  # 마지막은 진행 중인 봉이라 제외

    def price(self) -> float:
        return float(self.ex.fetch_ticker(self.data_symbol)["last"])

    def _bal_params(self) -> tuple[dict, str]:
        return ({"productType": "SUSDT-FUTURES"}, "SUSDT") if self.demo else ({}, "USDT")

    def balance(self) -> float:
        params, coin = self._bal_params()
        return float(self.ex.fetch_balance(params)[coin]["total"])

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
            except ccxt.BaseError as e:  # 이미 같은 값이면 오류가 날 수 있음 → 기록만 하고 진행
                self.setup_errors.append(f"{fn.__name__}: {str(e)[:160]}")
                if not self.demo and "40014" in str(e):  # 실거래에서 권한 부족이면 격리/레버리지 미설정 상태로 주문하지 않는다
                    raise
        amt = float(self.ex.amount_to_precision(self.symbol, amount))
        self._order(side, amt, {"stopLoss": {"triggerPrice": stop_price}})  # TODO(verify)

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
            out["demo_symbols"] = sorted(k for k, v in self.ex.markets.items()
                                          if str(v.get("id", "")).startswith("SBTC"))[:5]
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
        self._order(opp, contracts, {"reduceOnly": True})


DEMO_PT, DEMO_COIN = "SUSDT-FUTURES", "SUSDT"


class BitgetDemoFutures:
    """Bitget 데모(모의) 선물. ccxt 종목 목록에는 데모 종목(SBTCSUSDT)이 등록되지 않아(run #8 로그)
    통합 심볼 대신 Bitget v2 mix REST 를 ccxt 의 직접 호출 메서드로 부른다.
    시세/캔들은 실제 BTC/USDT:USDT 를 ccxt 로 가져온다. 주문 종목 ID = S + 기초자산 + SUSDT.

    !! 미검증 !!  요청 파라미터·응답 필드는 Bitget v2 문서 기준 추정이며 데모에서 확인 중이다."""

    def __init__(self, ex: ccxt.Exchange, data_symbol: str, leverage: int):
        self.ex, self.data_symbol, self.leverage = ex, data_symbol, leverage
        self.hedged: bool | None = None
        self.setup_errors: list[str] = []
        self.demo = True
        ex.load_markets()
        self.market_id = "S" + ex.market(data_symbol)["baseId"] + DEMO_COIN  # 예: SBTCSUSDT
        self._base = {"symbol": self.market_id, "productType": DEMO_PT, "marginCoin": DEMO_COIN}

    # --- 시세 (실제 시장, ccxt 통합 API) ---
    def closed_candles(self, timeframe: str, limit: int = 300) -> pd.DataFrame:
        rows = self.ex.fetch_ohlcv(self.data_symbol, timeframe, limit=limit)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        return df.iloc[:-1][["open", "high", "low", "close"]]

    def price(self) -> float:
        return float(self.ex.fetch_ticker(self.data_symbol)["last"])

    # --- 계좌/포지션 (데모, 직접 호출) ---
    def balance(self) -> float:
        r = self.ex.privateMixGetV2MixAccountAccounts({"productType": DEMO_PT})
        for a in r.get("data") or []:
            if a.get("marginCoin") == DEMO_COIN:
                return float(a.get("accountEquity") or a.get("available") or 0)
        return 0.0

    def position(self) -> dict | None:
        r = self.ex.privateMixGetV2MixPositionAllPosition({"productType": DEMO_PT, "marginCoin": DEMO_COIN})
        for p in r.get("data") or []:
            if p.get("symbol") == self.market_id and float(p.get("total") or 0) > 0:
                return {"side": p["holdSide"], "contracts": float(p["total"]), "entry": float(p.get("openPriceAvg") or 0)}
        return None

    def test_amount(self, min_notional: float = 7.0) -> float:
        import math
        m = self.ex.market(self.data_symbol)
        step = float(m["limits"]["amount"]["min"] or 0.001)
        return round(math.ceil(min_notional / (self.price() * step)) * step, 8)

    def _size(self, amount: float) -> str:
        return str(self.ex.amount_to_precision(self.data_symbol, amount))

    def _setup(self) -> None:
        calls = (
            ("set_position_mode", self.ex.privateMixPostV2MixAccountSetPositionMode,
             {"productType": DEMO_PT, "posMode": "one_way_mode"}),
            ("set_margin_mode", self.ex.privateMixPostV2MixAccountSetMarginMode,
             {**self._base, "marginMode": "isolated"}),
            ("set_leverage", self.ex.privateMixPostV2MixAccountSetLeverage,
             {**self._base, "leverage": str(self.leverage)}),
        )
        for name, fn, params in calls:
            try:
                fn(params)
            except ccxt.BaseError as e:  # 이미 같은 값/포지션 보유 중이면 거부될 수 있음 → 기록만
                self.setup_errors.append(f"{name}: {str(e)[:160]}")

    def _place(self, side: str, size: str, extra: dict) -> dict:
        """단방향으로 먼저 시도, 포지션 모드 불일치 오류(40774/25236)면 헤지 방식(tradeSide)으로 재시도."""
        body = {**self._base, "marginMode": "isolated", "side": side, "orderType": "market", "size": size, **extra}
        modes = [self.hedged] if self.hedged is not None else [False, True]
        err: Exception | None = None
        for hedged in modes:
            b = dict(body)
            if hedged:
                b["tradeSide"] = "close" if b.pop("reduceOnly", None) else "open"
                if b["tradeSide"] == "close":  # 헤지 청산: side 는 청산할 포지션의 방향
                    b["side"] = "buy" if side == "sell" else "sell"
            try:
                r = self.ex.privateMixPostV2MixOrderPlaceOrder(b)
                self.hedged = hedged
                return r
            except ccxt.ExchangeError as e:
                if not any(c in str(e) for c in ("40774", "25236")):
                    raise
                err = e
        raise err  # type: ignore[misc]

    def open(self, side: str, amount: float, stop_price: float) -> None:
        self._setup()
        self._place(side, self._size(amount), {"presetStopLossPrice": str(self.ex.price_to_precision(self.data_symbol, stop_price))})

    def close(self, side: str, contracts: float) -> None:
        opp = "sell" if side == "long" else "buy"
        self._place(opp, self._size(contracts), {"reduceOnly": "YES"})

    # --- 진단 ---
    def stop_info(self) -> dict:
        out: dict = {}
        try:
            r = self.ex.privateMixGetV2MixOrderOrdersPlanPending(
                {"productType": DEMO_PT, "planType": "profit_loss", "symbol": self.market_id})
            d = r.get("data") or {}
            rows = d.get("entrustedList") if isinstance(d, dict) else d
            out["plan_orders"] = [{k: o.get(k) for k in ("orderId", "planType", "side", "size", "triggerPrice", "triggerType")}
                                  for o in (rows or [])]
        except ccxt.BaseError as e:
            out["plan_orders"] = f"조회 실패: {type(e).__name__}: {str(e)[:120]}"
        return out

    def diagnose(self) -> dict:
        out: dict = {"market_id": self.market_id, "hedged": self.hedged, "setup_errors": self.setup_errors}
        try:
            c = self.ex.publicMixGetV2MixMarketContracts({"productType": DEMO_PT, "symbol": self.market_id})
            out["contract"] = (c.get("data") or [{}])[0].get("symbol") if c.get("data") else "없음(빈 목록)"
        except ccxt.BaseError as e:
            out["contract"] = f"조회 실패: {type(e).__name__}: {str(e)[:120]}"
        try:
            out["balance"] = self.balance()
        except ccxt.BaseError as e:
            out["balance"] = f"조회 실패: {type(e).__name__}: {str(e)[:120]}"
        return out
