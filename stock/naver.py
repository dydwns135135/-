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


def _num(x):
    x = x.replace(",", "").replace("+", "").strip()
    return float(x) if re.fullmatch(r"-?\d+(\.\d+)?", x) else None


def parse_flow(html):
    """종목 페이지 HTML → DataFrame(index=날짜, columns=inst_qty, foreign_qty). 기관/외국인 순매매량(주).

    행 구성(날짜, 종가, 전일비, 등락률, 거래량, 기관 순매매량, 외국인 순매매량, 보유주수, 보유율)을 위치로 읽는다.
    """
    import lxml.html

    rows = {}
    for tr in lxml.html.fromstring(html).iter("tr"):
        cells = [" ".join(td.text_content().split()) for td in tr.findall("td")]
        if len(cells) >= 7 and re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", cells[0]):
            inst, fgn = _num(cells[5]), _num(cells[6])
            if inst is not None and fgn is not None:
                rows[pd.Timestamp(cells[0].replace(".", "-"))] = (inst, fgn)
    if not rows:
        raise ValueError("수급 표를 찾지 못함")
    return pd.DataFrame(rows, index=["inst_qty", "foreign_qty"]).T.sort_index()


def parse_flow_json(data):
    """m.stock.naver.com/api/stock/<코드>/trend JSON → DataFrame(inst_qty, foreign_qty, indiv_qty)."""
    rows = {}
    for d in data:
        inst, fgn, ind = _num(str(d.get("organPureBuyQuant", ""))), _num(str(d.get("foreignerPureBuyQuant", ""))), \
            _num(str(d.get("individualPureBuyQuant", "")))
        day = str(d.get("bizdate", ""))
        if inst is not None and fgn is not None and re.fullmatch(r"\d{8}", day):
            rows[pd.Timestamp(day)] = (inst, fgn, ind if ind is not None else -(inst + fgn))
    if not rows:
        raise ValueError("JSON에 수급 값 없음")
    return pd.DataFrame(rows, index=["inst_qty", "foreign_qty", "indiv_qty"]).T.sort_index()


def _snippet(text, n=120):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))[:n]


def fetch_flow(code):
    """수급 조회: 모바일 JSON → 실패 시 PC 페이지 표. 둘 다 실패하면 응답 일부를 오류에 담는다."""
    errs = []
    try:
        r = requests.get(f"https://m.stock.naver.com/api/stock/{code}/trend", params={"pageSize": 30},
                         headers={**HEAD, "Referer": "https://m.stock.naver.com/"}, timeout=20)
        r.raise_for_status()
        return parse_flow_json(r.json())
    except Exception as e:
        errs.append(f"json {type(e).__name__}")
    try:
        pages = []
        for p in (1, 2):
            r = _get("https://finance.naver.com/item/frgn.naver", code=code, page=p)
            r.encoding = "euc-kr"
            pages.append(parse_flow(r.text))
        return pd.concat(pages)
    except Exception as e:
        body = locals().get("r")
        info = f"HTTP {body.status_code} len={len(body.text)} '{_snippet(body.text)}'" if body is not None else ""
        errs.append(f"html {type(e).__name__} {info}")
    raise ValueError(" / ".join(errs))


def _get(url, **params):
    r = requests.get(url, params=params, headers=HEAD, timeout=20)
    r.raise_for_status()
    return r


def add_flow(df, flow):
    """순매매량×종가로 금액 추정해 foreign/inst/individual 컬럼 추가(공통 날짜만, 최근 구간)."""
    j = df.join(flow, how="left")
    # 최근 5개 일봉 중 3일 이상 수급이 있으면 사용(당일 수급은 장중·직후엔 비어 있을 수 있어 0으로 채움)
    if j["inst_qty"].iloc[-5:].notna().sum() < min(3, len(j)):
        return df
    last_real = j["inst_qty"].last_valid_index()
    j[["inst_qty", "foreign_qty"]] = j[["inst_qty", "foreign_qty"]].fillna(0)
    j["inst"] = j["inst_qty"] * j["close"]
    j["foreign"] = j["foreign_qty"] * j["close"]
    if "indiv_qty" in j:
        j["individual"] = j["indiv_qty"].fillna(-(j["inst_qty"] + j["foreign_qty"])) * j["close"]
    else:
        j["individual"] = -(j["inst"] + j["foreign"])
    out = j.drop(columns=[c for c in ("inst_qty", "foreign_qty", "indiv_qty") if c in j])
    out.attrs["flow_last"] = f"{last_real:%Y-%m-%d}" if last_real is not None else None  # 당일 수급이 아직 없으면 어제 날짜
    return out


def load_naver(codes, days=250, delay=0.3, log=print, with_flow=True):
    out = {}
    idx = parse_chart(_get("https://fchart.stock.naver.com/sise.nhn", symbol="KOSPI", timeframe="day",
                           count=days, requestType=0).text)
    out["KOSPI"] = idx[["close", "volume"]]
    for code in codes:
        try:
            df = parse_chart(_get("https://fchart.stock.naver.com/sise.nhn", symbol=code, timeframe="day",
                                  count=days, requestType=0).text)
        except Exception as e:
            log(f"일봉 실패 {code}: {type(e).__name__}: {str(e)[:80]}")
            continue
        if not with_flow:  # 성과 확인(과거 가격 기반)에서는 수급을 쓰지 않는다
            out[code] = df
            time.sleep(delay)
            continue
        try:
            flow = fetch_flow(code)
            df = add_flow(df, flow)
            if "foreign" not in df:
                log(f"수급 미반영 {code}: 일봉 최신 {df.index[-1]:%Y-%m-%d}, 수급 최신 {flow.index[-1]:%Y-%m-%d}")
        except Exception as e:
            log(f"수급 실패 {code}: {type(e).__name__}: {str(e)[:300]} (가격만 분석)")
        out[code] = df
        time.sleep(delay)
    if len(out) < 2:
        raise RuntimeError("네이버에서 종목 데이터를 받지 못함")
    return out
