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


def parse_stooq(text):
    """Stooq 일봉 CSV(Date,Open,High,Low,Close,Volume) → DataFrame."""
    from io import StringIO

    if not text.lstrip().startswith("Date"):
        raise ValueError("Stooq 응답이 CSV가 아님: " + text.strip()[:60])
    df = pd.read_csv(StringIO(text), index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna().sort_index()
    if df.empty:
        raise ValueError("Stooq 응답에 데이터 없음")
    return df.astype(float).tail(260)


def _stooq_sym(sym):
    return "^spx" if sym == "^GSPC" else sym.lower().replace("-", ".") + ".us"


def fetch_yahoo(sym, days="1y", tries=3, sleep=time.sleep):
    """야후: query1/query2 번갈아 시도, 429·5xx면 점점 길게 쉬고 재시도."""
    last = None
    for i in range(tries):
        host = "query1" if i % 2 == 0 else "query2"
        try:
            r = requests.get(URL.format(sym=sym).replace("query1", host), params={"range": days, "interval": "1d"},
                             headers=HEAD, timeout=20)
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                sleep(3 * (i + 1))
                continue
            r.raise_for_status()
            return parse_chart(r.json())
        except (requests.RequestException, ValueError, KeyError) as e:
            last = f"{type(e).__name__}"
            sleep(3 * (i + 1))
    raise RuntimeError(f"야후 실패({last})")


def fetch_stooq(sym):
    r = requests.get("https://stooq.com/q/d/l/", params={"s": _stooq_sym(sym), "i": "d"}, headers=HEAD, timeout=20)
    r.raise_for_status()
    return parse_stooq(r.text)


def fetch(sym, days="1y", log=print):
    """야후 → 실패 시 Stooq. 둘 다 실패하면 두 오류를 함께 올린다."""
    try:
        return fetch_yahoo(sym, days)
    except Exception as e1:
        try:
            df = fetch_stooq(sym)
            log(f"{sym}: 야후 실패({e1}) → Stooq 사용")
            return df
        except Exception as e2:
            raise RuntimeError(f"야후: {e1} / Stooq: {type(e2).__name__}: {str(e2)[:60]}")


def load_yahoo(tickers, delay=0.8, log=print):
    try:
        idx = fetch("^GSPC", log=log)
    except Exception:
        idx = fetch("SPY", log=log)  # 지수가 안 되면 S&P500 ETF로 대체
        log("S&P500 지수 대신 SPY(ETF) 사용")
    out = {"SPX": idx[["close", "volume"]]}
    for t in tickers:
        try:
            out[t] = fetch(t, log=log)
        except Exception as e:
            log(f"일봉 실패 {t}: {str(e)[:150]}")
        time.sleep(delay)
    if len(out) < 3:
        raise RuntimeError("미국 종목 데이터를 받지 못함")
    return out
