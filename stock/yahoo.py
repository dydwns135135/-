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
    r = requests.get("https://stooq.com/q/d/l/", params={"s": _stooq_sym(sym), "i": "d"}, headers=HEAD, timeout=8)
    r.raise_for_status()
    return parse_stooq(r.text)


def parse_naver_us(rows):
    """네이버 해외 일봉 JSON(list) → DataFrame. 필드: localDate, openPrice, highPrice, lowPrice, closePrice, accumulatedTradingVolume."""
    recs = {}
    for d in rows:
        try:
            recs[pd.Timestamp(str(d["localDate"]))] = [float(d[k]) for k in
                ("openPrice", "highPrice", "lowPrice", "closePrice", "accumulatedTradingVolume")]
        except (KeyError, ValueError, TypeError):
            continue
    if not recs:
        raise ValueError("네이버 해외 응답에 데이터 없음")
    return pd.DataFrame(recs, index=["open", "high", "low", "close", "volume"]).T.sort_index().tail(260)


def fetch_naver_us(sym):
    """네이버 해외주식 일봉. 거래소 접미사(.O 나스닥, .N 뉴욕, .K 아카)를 차례로 시도한다."""
    end = pd.Timestamp.today().normalize()
    p = {"startDateTime": (end - pd.Timedelta(days=420)).strftime("%Y%m%d0000"), "endDateTime": end.strftime("%Y%m%d2359")}
    errs = []
    for suf in ((".O", ".N", ".K") if "." not in sym else ("",)):
        try:
            r = requests.get(f"https://api.stock.naver.com/chart/foreign/item/{sym}{suf}/day", params=p,
                             headers=HEAD, timeout=15)
            r.raise_for_status()
            return parse_naver_us(r.json())
        except Exception as e:
            errs.append(f"{suf or '-'}:{type(e).__name__}")
    raise RuntimeError("네이버 " + ",".join(errs))


def fetch(sym, days="1y", log=print):
    """네이버(GitHub 서버에서 접속되는 것으로 확인된 곳) → 야후 → Stooq 순서. 모두 실패하면 오류를 모아 올린다."""
    errs = []
    if sym != "^GSPC":  # 네이버는 지수 주소를 몰라 지수는 SPY로 대체하게 둔다
        try:
            return fetch_naver_us(sym)
        except Exception as e:
            errs.append(str(e)[:80])
    try:
        df = fetch_yahoo(sym, days, tries=2)
        if errs:
            log(f"{sym}: 네이버 실패 → 야후 사용")
        return df
    except Exception as e:
        errs.append(str(e)[:60])
    try:
        df = fetch_stooq(sym)
        log(f"{sym}: Stooq 사용")
        return df
    except Exception as e:
        errs.append(f"Stooq {type(e).__name__}")
    raise RuntimeError(" / ".join(errs))


def load_yahoo(tickers, delay=0.8, log=print):
    try:
        idx = fetch("SPY", log=log)  # S&P500 ETF를 시황 지수로 사용(네이버에서 받을 수 있음)
    except Exception:
        idx = fetch("^GSPC", log=log)
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
