"""네이버 금융 데이터 로더(해외 서버에서도 접근 가능). 일봉은 fchart XML, 외국인·기관 순매매는 종목 페이지 표.

수급은 '순매매량(주)' × 종가로 금액을 추정한다. 개인은 제공되지 않아 -(외국인+기관)으로 근사한다.
수급 조회가 실패해도 일봉 분석은 계속된다(수급 점수 항목만 빠진다).
"""
import re
import time
from io import StringIO

import pandas as pd
import requests

HEAD = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
ITEM = re.compile(r'<item data="(\d{8})\|([\d.]+)\|([\d.]+)\|([\d.]+)\|([\d.]+)\|([\d.]+)"')


def parse_chart(xml):
    rows = [m.groups() for m in ITEM.finditer(xml)]
    if not rows:
        raise ValueError("일봉 응답에 데이터 없음")
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("date"))
    return df.astype(float).sort_index()


def parse_flow(html):
    """종목 페이지 HTML → DataFrame(index=날짜, columns=inst_qty, foreign_qty). 기관/외국인 순매매량(주)."""
    for t in pd.read_html(StringIO(html)):
        cols = ["".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        if any("기관" in c for c in cols) and any("외국인" in c for c in cols) and any("날짜" in c for c in cols):
            t.columns = cols
            d = [c for c in cols if "날짜" in c][0]
            i = [c for c in cols if "기관" in c and "순매매" in c][0]
            f = [c for c in cols if "외국인" in c and "순매매" in c][0]
            t = t[[d, i, f]].dropna()
            t[d] = pd.to_datetime(t[d], format="%Y.%m.%d", errors="coerce")
            out = pd.DataFrame({"inst_qty": pd.to_numeric(t[i].astype(str).str.replace(",", "").str.replace("+", ""), errors="coerce").values,
                                "foreign_qty": pd.to_numeric(t[f].astype(str).str.replace(",", "").str.replace("+", ""), errors="coerce").values},
                               index=t[d].values)
            return out[out.index.notna()].dropna()
    raise ValueError("수급 표를 찾지 못함")


def _get(url, **params):
    r = requests.get(url, params=params, headers=HEAD, timeout=20)
    r.raise_for_status()
    return r


def add_flow(df, flow):
    """순매매량×종가로 금액 추정해 foreign/inst/individual 컬럼 추가(공통 날짜만, 최근 구간)."""
    j = df.join(flow, how="left")
    if j[["inst_qty", "foreign_qty"]].iloc[-5:].isna().any().any():
        return df  # 최근 5일 수급이 비면 사용하지 않음
    j[["inst_qty", "foreign_qty"]] = j[["inst_qty", "foreign_qty"]].fillna(0)
    j["inst"] = j["inst_qty"] * j["close"]
    j["foreign"] = j["foreign_qty"] * j["close"]
    j["individual"] = -(j["inst"] + j["foreign"])
    return j.drop(columns=["inst_qty", "foreign_qty"])


def load_naver(codes, days=250, delay=0.3, log=print):
    out = {}
    idx = parse_chart(_get("https://fchart.stock.naver.com/sise.nhn", symbol="KOSPI", timeframe="day",
                           count=days, requestType=0).text)
    out["KOSPI"] = idx[["close", "volume"]]
    for code in codes:
        try:
            df = parse_chart(_get("https://fchart.stock.naver.com/sise.nhn", symbol=code, timeframe="day",
                                  count=days, requestType=0).text)
        except Exception as e:
            log(f"일봉 실패 {code}: {type(e).__name__}")
            continue
        try:
            pages = []
            for p in (1, 2):
                r = _get("https://finance.naver.com/item/frgn.naver", code=code, page=p)
                r.encoding = "euc-kr"
                pages.append(parse_flow(r.text))
            df = add_flow(df, pd.concat(pages))
        except Exception as e:
            log(f"수급 실패 {code}: {type(e).__name__} (가격만 분석)")
        out[code] = df
        time.sleep(delay)
    if len(out) < 2:
        raise RuntimeError("네이버에서 종목 데이터를 받지 못함")
    return out
