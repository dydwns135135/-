"""토스증권 OpenAPI 클라이언트.

!! 주의 !!  이 파일의 엔드포인트 경로/필드명은 공개 자료(검색 결과)만 보고 작성한 *가정*이며,
개발 환경에서 공식 문서에 접근하지 못해 검증하지 못했다. 확인된 사실은 다음뿐이다:
  - 베이스 URL https://openapi.tossinvest.com, OAuth 2.0 Client Credentials 인증
  - 계좌·주문 계열은 `X-Tossinvest-Account` 헤더 필요
  - 조건주문: POST /api/v1/conditional-orders
실거래 전에 아래 `ENDPOINTS` 와 파싱 로직을 공식 문서와 대조해 고칠 것. 경로를 한 곳에 모아 두었다.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

from .broker import Position

BASE_URL = "https://openapi.tossinvest.com"

# TODO(verify): 공식 문서와 대조 필요
ENDPOINTS = {
    "token": "/oauth2/token",
    "candles": "/api/v1/quotes/{symbol}/candles",
    "quote": "/api/v1/quotes/{symbol}",
    "balance": "/api/v1/account/balance",
    "positions": "/api/v1/account/positions",
    "order": "/api/v1/orders",
}


class TossClient:
    def __init__(self, client_id: str, client_secret: str, account: str, base_url: str = BASE_URL):
        self.id, self.secret, self.account, self.base = client_id, client_secret, account, base_url
        self._token, self._exp = None, 0.0
        self.s = requests.Session()

    @classmethod
    def from_env(cls) -> "TossClient":
        return cls(os.environ["TOSS_CLIENT_ID"], os.environ["TOSS_CLIENT_SECRET"], os.environ["TOSS_ACCOUNT"])

    def _auth(self) -> dict:
        if not self._token or time.time() > self._exp - 30:
            r = self.s.post(
                self.base + ENDPOINTS["token"],
                data={"grant_type": "client_credentials", "client_id": self.id, "client_secret": self.secret},
                timeout=10,
            )
            r.raise_for_status()
            j = r.json()
            self._token, self._exp = j["access_token"], time.time() + int(j.get("expires_in", 3600))
        return {"Authorization": f"Bearer {self._token}", "X-Tossinvest-Account": self.account}

    def _req(self, method: str, key: str, **kw):
        path = ENDPOINTS[key].format(**kw.pop("fmt", {}))
        r = self.s.request(method, self.base + path, headers=self._auth(), timeout=10, **kw)
        r.raise_for_status()
        return r.json()


class TossBroker:
    """실거래 브로커. 응답 필드명은 가정이므로 검증 후 수정할 것."""

    def __init__(self, client: TossClient):
        self.c = client

    def candles(self, symbol, count):
        j = self.c._req("GET", "candles", fmt={"symbol": symbol}, params={"interval": "1d", "count": count})
        df = pd.DataFrame(j["candles"])  # TODO(verify)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()[["open", "high", "low", "close"]].astype(float)

    def price(self, symbol):
        return float(self.c._req("GET", "quote", fmt={"symbol": symbol})["price"])  # TODO(verify)

    def cash(self):
        return float(self.c._req("GET", "balance")["usdCash"])  # TODO(verify)

    def positions(self):
        j = self.c._req("GET", "positions")
        return {
            p["symbol"]: Position(p["symbol"], int(p["quantity"]), float(p["avgPrice"]))
            for p in j["positions"]  # TODO(verify)
        }

    def _order(self, side: str, symbol: str, qty: int, price: float):
        body = {"symbol": symbol, "side": side, "type": "LIMIT", "quantity": qty, "price": price}
        return self.c._req("POST", "order", json=body)  # TODO(verify)

    def buy(self, symbol, qty, price):
        self._order("BUY", symbol, qty, price)

    def sell(self, symbol, qty, price):
        self._order("SELL", symbol, qty, price)
