"""SEC EDGAR(미국 공시) 최근 공시·내부자 거래 조회. 무료·키 없음(SEC는 요청자를 알 수 있는 User-Agent를 요구: 환경변수 EDGAR_USER_AGENT로 바꿀 수 있음)."""
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import requests

UA_DEFAULT = "stock-analyzer research (github.com/dydwns135135)"
ITEM_NEGATIVE = {"1.03": "파산·회생 절차", "2.04": "채무 조기상환 사유 발생", "2.06": "자산 손상", "3.01": "상장 유지 요건 미달 통보",
                 "3.02": "미등록 주식 발행(희석 가능)", "4.01": "감사인 변경", "4.02": "재무제표 신뢰 불가(정정 필요)"}
ITEM_NOTABLE = {"1.01": "주요 계약 체결", "2.01": "자산 인수·처분 완료", "2.02": "실적 발표", "2.05": "구조조정 비용", "5.02": "임원 변동", "5.07": "주주총회 결과"}
_TICKERS = {}
LAST_ERROR = ""


def _get(url, ua=None):
    time.sleep(0.15)  # SEC 한도(초당 10회)보다 충분히 느리게
    r = requests.get(url, headers={"User-Agent": ua or os.environ.get("EDGAR_USER_AGENT") or UA_DEFAULT}, timeout=30)
    r.raise_for_status()
    return r


def ticker_ciks():
    """{티커: CIK(정수)}"""
    if not _TICKERS:
        j = _get("https://www.sec.gov/files/company_tickers.json").json()
        _TICKERS.update({v["ticker"].upper(): int(v["cik_str"]) for v in j.values()})
    return _TICKERS


def recent_filings(cik, days=90):
    """최근 days일 공시 목록: [{form, date, items, acc, doc}]"""
    j = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]
    since = (date.today() - timedelta(days=days)).isoformat()
    out = []
    for i, f in enumerate(j["form"]):
        if j["filingDate"][i] < since:
            continue
        out.append({"form": f, "date": j["filingDate"][i], "items": j.get("items", [""] * len(j["form"]))[i] or "",
                    "acc": j["accessionNumber"][i], "doc": j["primaryDocument"][i]})
    return out


def parse_form4(xml_text):
    """Form 4 XML → 공개시장 매수(P)·매도(S) 거래 [(코드, 주식수, 단가)]. 옵션 행사·증여 등 다른 코드는 제외."""
    root = ET.fromstring(xml_text)
    out = []
    for t in root.iter("nonDerivativeTransaction"):
        code = (t.findtext(".//transactionCode") or "").strip()
        if code not in ("P", "S"):
            continue
        try:
            sh = float(t.findtext(".//transactionShares/value") or 0)
            px = float(t.findtext(".//transactionPricePerShare/value") or 0)
        except ValueError:
            continue
        out.append((code, sh, px))
    return out


def summarize_8k(filings):
    """8-K 항목 번호로 분류. negative=주의 항목(악재성), notable=참고 항목."""
    eight = [f for f in filings if f["form"] in ("8-K", "8-K/A")]
    neg, notable = [], []
    for f in eight:
        for it in [x.strip() for x in f["items"].split(",") if x.strip()]:
            if it in ITEM_NEGATIVE:
                neg.append(f"{ITEM_NEGATIVE[it]}({f['date']})")
            elif it in ITEM_NOTABLE:
                notable.append(ITEM_NOTABLE[it])
    other = [f for f in filings if f["form"] in ("6-K", "20-F")]  # 해외 기업(TSMC 등)은 8-K 대신 이 양식을 낸다
    return {"positive": [], "negative": neg, "notable": notable, "count": len(eight) or len(other), "foreign": bool(other) and not eight}


def insider_trades(cik, filings, limit=15):
    """최근 Form 4에서 공개시장 매수·매도 합계. {buys, sells, buy_value, sell_value, n}"""
    res = {"buys": 0, "sells": 0, "buy_value": 0.0, "sell_value": 0.0, "n": 0}
    for f in [x for x in filings if x["form"] == "4"][:limit]:
        raw = re.sub(r"^.*/", "", f["doc"])  # xsl 폴더 없는 원본 XML
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{f['acc'].replace('-', '')}/{raw}"
        try:
            trades = parse_form4(_get(url).text)
        except Exception:
            continue
        res["n"] += 1
        for code, sh, px in trades:
            if code == "P":
                res["buys"] += 1
                res["buy_value"] += sh * px
            else:
                res["sells"] += 1
                res["sell_value"] += sh * px
    return res


def fetch_summaries(tickers, days=90, insiders=True):
    """{티커: {**summarize_8k, 'insider': {...}}}; 조회 실패한 종목은 빠지고 LAST_ERROR에 사유가 남는다."""
    global LAST_ERROR
    LAST_ERROR = ""
    if not tickers:
        return {}
    try:
        cks = ticker_ciks()
    except Exception as e:
        LAST_ERROR = f"티커 목록 실패: {type(e).__name__} {str(e)[:80]}"
        return {}
    out = {}
    for t in tickers:
        cik = cks.get(t.upper().replace("-", "."))  # BRK-B → BRK.B 식 표기 차이 대비
        if cik is None:
            LAST_ERROR = f"{t}: SEC 티커 목록에 없음"
            continue
        try:
            fl = recent_filings(cik, days)
            s = summarize_8k(fl)
            s["insider"] = insider_trades(cik, fl) if insiders else None
            out[t] = s
        except Exception as e:
            LAST_ERROR = f"{t}: {type(e).__name__} {str(e)[:80]}"
    return out


def fmt_money(v):
    return f"${v / 1e6:.1f}M" if abs(v) >= 1e6 else f"${v / 1e3:.0f}K"
