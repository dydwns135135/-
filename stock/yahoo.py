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


def _find_rows(j):
    """JSON 어디에 있든 localDate 가 든 dict 목록을 찾아 돌려준다(응답 구조가 달라도 견디기 위함)."""
    if isinstance(j, list):
        if any(isinstance(d, dict) and "localDate" in d for d in j):
            return j
        for x in j:
            r = _find_rows(x)
            if r:
                return r
    elif isinstance(j, dict):
        for v in j.values():
            r = _find_rows(v)
            if r:
                return r
    return None


def parse_naver_us(j):
    """네이버 해외 일봉 JSON → DataFrame. 필드: localDate, openPrice, highPrice, lowPrice, closePrice, accumulatedTradingVolume."""
    recs = {}
    for d in _find_rows(j) or []:
        try:
            recs[pd.Timestamp(str(d["localDate"]))] = [float(d[k]) for k in
                ("openPrice", "highPrice", "lowPrice", "closePrice", "accumulatedTradingVolume")]
        except (KeyError, ValueError, TypeError):
            continue
    if not recs:
        raise ValueError("네이버 해외 응답에 일봉 없음")
    return pd.DataFrame(recs, index=["open", "high", "low", "close", "volume"]).T.sort_index().tail(260)


def _naver_urls(code, end):
    """후보 주소(응답 형식이 달라도 _find_rows 가 읽는다). 날짜 형식이 확실하지 않아 변형을 모두 시도한다."""
    base = f"https://api.stock.naver.com/chart/foreign/item/{code}/day"
    st = end - pd.Timedelta(days=420)
    yield base, {"startDateTime": st.strftime("%Y%m%d0000"), "endDateTime": end.strftime("%Y%m%d2359")}
    yield base, {"startDateTime": st.strftime("%Y%m%d000000"), "endDateTime": end.strftime("%Y%m%d235959")}
    yield base, {}
    yield "https://m.stock.naver.com/front-api/chart/pricesByPeriod", {
        "reutersCode": code, "category": "stock", "chartInfoType": "stock", "scriptChartType": "day"}


NAVER_SUFFIXES = (".O", ".N", ".P", ".A", ".K", "")  # 나스닥, 뉴욕, 아카, 아멕스, ETF, 접미사 없음


def fetch_naver_us(sym):
    """네이버 해외주식 일봉. 거래소 접미사와 후보 주소를 차례로 시도하고, 성공한 조합을 기억해 다음 종목에서 먼저 쓴다."""
    end = pd.Timestamp.today().normalize()
    errs = []
    combos = [(suf, k) for suf in (NAVER_SUFFIXES if "." not in sym else ("",)) for k in range(4)]
    if _NAVER_OK:  # 앞서 성공한 후보 주소 번호를 먼저
        combos.sort(key=lambda c: c[1] != _NAVER_OK[0])
    for suf, k in combos:
        url, params = list(_naver_urls(sym + suf, end))[k]
        tag = f"{suf or '-'}#{k + 1}"
        r = None
        try:
            r = requests.get(url, params=params, headers=HEAD, timeout=15)
            if r.status_code >= 400:
                errs.append(f"{tag} HTTP{r.status_code}")
                continue
            df = parse_naver_us(r.json())
            _NAVER_OK[:] = [k]
            NAVER_HIT[sym] = f"{suf or '-'}#{k + 1}"
            return df
        except Exception as e:
            errs.append(f"{tag} {type(e).__name__}" + (f" '{' '.join(r.text.split())[:40]}'" if r is not None else ""))
    raise RuntimeError("네이버 " + "; ".join(errs))


_NAVER_OK = []
NAVER_HIT = {}  # 종목 → 성공한 접미사#후보번호(진단용)


def fetch(sym, days="1y", log=print):
    """네이버(GitHub 서버에서 접속되는 것으로 확인된 곳) → 야후 → Stooq 순서. 모두 실패하면 오류를 모아 올린다."""
    errs = []
    if sym != "^GSPC":  # 네이버는 지수 주소를 몰라 지수는 SPY로 대체하게 둔다
        try:
            return fetch_naver_us(sym)
        except Exception as e:
            errs.append(str(e)[:400])
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


def proxy_index(frames):
    """지수를 못 받을 때: 받은 종목들을 첫 값 기준 100으로 맞춰 평균낸 대용 지수."""
    close = pd.concat([d["close"] / d["close"].iloc[0] * 100 for d in frames], axis=1).ffill().mean(axis=1)
    vol = pd.concat([d["volume"] for d in frames], axis=1).fillna(0).sum(axis=1)
    df = pd.DataFrame({"close": close, "volume": vol})
    df.attrs["proxy"] = True
    return df


def load_yahoo(tickers, delay=0.8, log=print):
    out, errs = {}, []
    for t in tickers:
        try:
            out[t] = fetch(t, log=log)
        except Exception as e:
            errs.append(f"{t}: {str(e)[:300]}")
            log(f"일봉 실패 {errs[-1]}")
            if not out and len(errs) >= 3:  # 처음 3개가 모두 실패하면 중단(출처 전체가 막힌 것)
                raise RuntimeError("미국 데이터를 받지 못함 — " + " | ".join(errs))
        time.sleep(delay)
    if len(out) < 2:
        raise RuntimeError("미국 종목 데이터를 받지 못함 — " + " | ".join(errs[:3]))
    idx = None
    for ref in ("SPY", "QQQ"):
        try:
            idx = fetch(ref, log=log)
            break
        except Exception as e:
            log(f"지수 {ref} 실패: {str(e)[:150]}")
    if idx is None:
        idx = proxy_index(list(out.values()))
        log("S&P500 대신 받은 종목들의 평균을 시황 지수로 사용")
    out["SPX"] = idx[["close", "volume"]]
    out["SPX"].attrs["proxy"] = bool(idx.attrs.get("proxy"))
    if NAVER_HIT:
        log("네이버 성공: " + ", ".join(f"{k}{v}" for k, v in NAVER_HIT.items()))
    return out
