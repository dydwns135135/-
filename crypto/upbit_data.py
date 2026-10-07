"""업비트 공개 시세(일봉, 인증 불필요) 내려받기. 일봉은 한국 시간 09:00(UTC 00:00) 기준으로 마감된다."""
from __future__ import annotations

import time

import pandas as pd
import requests

URL = "https://api.upbit.com/v1/candles/days"
MINUTES_URL = "https://api.upbit.com/v1/candles/minutes/{unit}"


def parse_upbit(rows: list[dict]) -> pd.DataFrame:
    """업비트 캔들 JSON(최신순) → open/high/low/close (UTC 시각 오름차순)."""
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "open": df["opening_price"].astype(float), "high": df["high_price"].astype(float),
        "low": df["low_price"].astype(float), "close": df["trade_price"].astype(float),
    })
    out.index = pd.to_datetime(df["candle_date_time_utc"], utc=True)
    return out[~out.index.duplicated()].sort_index()


def fetch_upbit_candles(url: str, market: str, total: int, get=requests.get, sleep=time.sleep) -> pd.DataFrame:
    """최신 → 과거로 200개씩 이어 받는다(`to` 는 이 시각보다 이전만 반환). 요청 제한을 지키려고 호출 사이에 쉰다."""
    rows: list[dict] = []
    to = None
    for _ in range(total // 200 + 2):
        params = {"market": market, "count": 200}
        if to:
            params["to"] = to
        r = get(url, params=params, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"}, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"Upbit HTTP {r.status_code}: {r.text[:80]!r}")
        batch = r.json()
        if not batch:
            break
        rows += batch
        if len(batch) < 200 or len(rows) >= total:
            break
        to = batch[-1]["candle_date_time_utc"] + "Z"   # 가장 오래된 봉 이전
        sleep(0.15)
    if not rows:
        raise RuntimeError("Upbit 응답이 비어 있음")
    return parse_upbit(rows)


def fetch_upbit_daily(market: str = "KRW-BTC", total: int = 4000, get=requests.get, sleep=time.sleep) -> pd.DataFrame:
    return fetch_upbit_candles(URL, market, total, get, sleep)


def fetch_upbit_minutes(unit: int, market: str = "KRW-BTC", total: int = 20000, get=requests.get, sleep=time.sleep) -> pd.DataFrame:
    """분봉(unit=1,3,5,10,15,30,60,240)."""
    return fetch_upbit_candles(MINUTES_URL.format(unit=unit), market, total, get, sleep)


def closed_only(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """진행 중인 봉(시작+1일이 아직 안 지난 봉)을 제외한다."""
    now = now or pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(days=1) <= now]
