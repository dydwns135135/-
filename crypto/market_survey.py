"""Bitget USDT 무기한 선물 상품 조사(조회 전용): 어떤 상품이 있는지, 코인이 아닌 후보(지수·주식·금·원유 등)의
과거 데이터가 얼마나 되는지, 거래가 활발한지를 확인한다.
python -m crypto.market_survey   (공개 시세만 사용, API 키·주문 없음)"""
from __future__ import annotations

import argparse
import os
import re

import ccxt
import pandas as pd

from .backtest_report import fetch_history

# 코인이 아닌 후보를 찾기 위한 키워드(대소문자 무시). 이름 일부가 포함되면 후보로 표시하고 사람이 판단한다.
KEYWORDS = (
    "NAS", "NDX", "QQQ", "US100", "SP500", "SPX", "US500", "SPY", "DOW", "DJI", "RUSS", "KOSPI", "NIKKEI", "DAX",
    "XAU", "GOLD", "XAG", "SILVER", "OIL", "WTI", "BRENT", "COPPER", "EUR", "JPY", "GBP", "DXY",
    "SAMSUNG", "SKHY", "SNDK", "TSLA", "AAPL", "NVDA", "MSFT", "AMZN", "GOOG", "META", "MSTR", "COIN", "HOOD", "CRCL",
)
PROBE_YEARS = (8, 5, 3, 2, 1, 0.5)
MAX_PROBES = 60


def usdt_perps(markets: dict) -> list[dict]:
    return [m for m in markets.values() if m.get("swap") and m.get("linear") and m.get("quote") == "USDT" and m.get("active", True) is not False]


def candidates(perps: list[dict]) -> list[dict]:
    pat = re.compile("|".join(KEYWORDS), re.I)
    return [m for m in perps if pat.search(m["base"])]


def probe(ex, symbol: str) -> dict:
    """일봉을 받아 기간·거래대금을 요약한다. 상장 전 시점 요청이 빈 결과일 수 있어 기간을 줄여가며 시도."""
    for y in PROBE_YEARS:
        try:
            df = fetch_history(ex, symbol, timeframe="1d", years=y)
        except Exception:  # 한 상품 실패가 전체를 막지 않게
            continue
        if len(df) > 0:
            return {"bars": len(df), "first": f"{df.index[0]:%Y-%m-%d}", "last": f"{df.index[-1]:%Y-%m-%d}", "req_years": y}
    return {"bars": 0, "first": "-", "last": "-", "req_years": None}


def with_liquidity(ex, symbol: str, base_info: dict) -> dict:
    """최근 30일 평균 거래대금(USDT, 종가×거래량 근사)."""
    try:
        rows = ex.fetch_ohlcv(symbol, "1d", limit=40)
        df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"]).iloc[:-1].tail(30)
        base_info["avg_turnover_usdt"] = float((df["c"] * df["v"]).mean()) if len(df) else 0.0
    except Exception:
        base_info["avg_turnover_usdt"] = float("nan")
    return base_info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    perps = usdt_perps(ex.markets)
    bases = sorted({m["base"] for m in perps})
    print(f"Bitget USDT 무기한 선물 {len(perps)}개 상품")
    for i in range(0, len(bases), 12):
        print("  " + ", ".join(bases[i:i + 12]))

    cands = candidates(perps)
    print(f"\n코인이 아닐 가능성이 있는 후보 {len(cands)}개 (이름 키워드 일치, 사람이 판단 필요)")
    if cands:
        print("예시 상품 정보 필드:", {k: v for k, v in (cands[0].get("info") or {}).items() if k in ("symbol", "baseCoin", "symbolType", "launchTime", "maxLever", "minTradeUSDT")})
    rows = []
    for m in sorted(cands, key=lambda x: x["base"])[:MAX_PROBES]:
        r = with_liquidity(ex, m["symbol"], {"종목": m["base"], **probe(ex, m["symbol"])})
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["봉수"] = df["bars"]
        df["첫 거래일"] = df["first"]
        df["일평균 거래대금(만USDT)"] = (df["avg_turnover_usdt"] / 1e4).round(1)
        view = df[["종목", "봉수", "첫 거래일", "일평균 거래대금(만USDT)"]].sort_values("봉수", ascending=False)
        os.makedirs(a.out, exist_ok=True)
        view.to_csv(os.path.join(a.out, "market_survey.csv"), index=False)
        text = view.to_string(index=False)
        print("\n" + text)
        long_ok = view[view["봉수"] >= 1000]
        print(f"\n→ 일봉 1,000개(약 2.7년) 이상: {len(long_ok)}개: {', '.join(long_ok['종목']) or '없음'}")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                f.write("### Bitget 비(非)코인 후보 상품\n\n```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
