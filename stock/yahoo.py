"""야후 파이낸스 일봉 로더(미국). 수급(외국인·기관 순매수)은 제공되지 않아 가격·거래량만 쓴다."""
import time

import pandas as pd
import requests

HEAD = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"


def parse_chart(data):
    """야후 chart API JSON → DataFrame(open, high, low, close, volume). 거래 없는 날(NaN)은 제외."""
    res = data["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    off = res.get("meta", {}).get("gmtoffset", 0)
    idx = pd.to_datetime(pd.Series(res["timestamp"]) + off, unit="s").dt.normalize()
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")})
    df.index = pd.DatetimeIndex(idx)
    df = df.dropna()
    if df.empty:
        raise ValueError("야후 응답에 데이터 없음")
    return df.astype(float).sort_index()


def fetch(sym, days="1y"):
    r = requests.get(URL.format(sym=sym), params={"range": days, "interval": "1d"}, headers=HEAD, timeout=20)
    r.raise_for_status()
    return parse_chart(r.json())


def load_yahoo(tickers, delay=0.3, log=print):
    out = {"SPX": fetch("^GSPC")[["close", "volume"]]}
    for t in tickers:
        try:
            out[t] = fetch(t)
        except Exception as e:
            log(f"일봉 실패 {t}: {type(e).__name__}: {str(e)[:100]}")
        time.sleep(delay)
    if len(out) < 3:
        raise RuntimeError("야후에서 종목 데이터를 받지 못함")
    return out
