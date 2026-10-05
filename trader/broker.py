"""브로커 추상화: 모의(PaperBroker) / 토스증권(TossBroker)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

log = logging.getLogger("trader")


@dataclass
class Position:
    symbol: str
    qty: int
    avg_cost: float


class Broker(Protocol):
    def candles(self, symbol: str, count: int) -> pd.DataFrame: ...
    def price(self, symbol: str) -> float: ...
    def cash(self) -> float: ...
    def positions(self) -> dict[str, Position]: ...
    def buy(self, symbol: str, qty: int, price: float) -> None: ...
    def sell(self, symbol: str, qty: int, price: float) -> None: ...


class PaperBroker:
    """실제 주문 없이 로그만 남기고 가상 체결한다. 시세는 `data_source`(Broker)에서 가져온다."""

    def __init__(self, data_source: Broker, cash: float = 10_000.0):
        self.src, self._cash, self._pos = data_source, cash, {}

    def candles(self, symbol, count):
        return self.src.candles(symbol, count)

    def price(self, symbol):
        return self.src.price(symbol)

    def cash(self):
        return self._cash

    def positions(self):
        return dict(self._pos)

    def buy(self, symbol, qty, price):
        log.info("[PAPER] 매수 %s x%d @ %.2f", symbol, qty, price)
        p = self._pos.get(symbol, Position(symbol, 0, 0.0))
        total = p.qty + qty
        self._pos[symbol] = Position(symbol, total, (p.avg_cost * p.qty + price * qty) / total)
        self._cash -= qty * price

    def sell(self, symbol, qty, price):
        log.info("[PAPER] 매도 %s x%d @ %.2f", symbol, qty, price)
        p = self._pos[symbol]
        left = p.qty - qty
        if left <= 0:
            del self._pos[symbol]
        else:
            self._pos[symbol] = Position(symbol, left, p.avg_cost)
        self._cash += qty * price
