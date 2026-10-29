"""DART(전자공시) 최근 공시 조회·분류. API 키: https://opendart.fss.or.kr 에서 발급 → 환경변수 DART_API_KEY."""
import io
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import requests

BASE = "https://opendart.fss.or.kr/api"
POSITIVE = ["공급계약", "단일판매", "수주", "특허", "자기주식취득", "자기주식 취득", "무상증자", "배당", "흑자전환", "투자판단관련주요경영사항"]
NEGATIVE = ["유상증자", "전환사채", "신주인수권부사채", "교환사채", "감자", "횡령", "배임", "불성실공시", "상장폐지",
            "관리종목", "회생", "감사의견", "소송", "영업정지", "최대주주변경", "자본잠식"]


def classify_title(title):
    """공시 제목 → +1(호재 키워드) / -1(악재 키워드, 우선) / 0."""
    if any(k in title for k in NEGATIVE):
        return -1
    if any(k in title for k in POSITIVE):
        return 1
    return 0


def summarize(titles):
    pos = [t for t in titles if classify_title(t) > 0]
    neg = [t for t in titles if classify_title(t) < 0]
    return {"positive": pos, "negative": neg, "count": len(titles)}


_CC = {}
LAST_ERROR = ""  # 마지막 조회 실패 사유(리포트에 보여 원인을 바로 알 수 있게)


def corp_codes(key):
    """종목코드(6자리) → DART 고유번호(8자리) 매핑(한 번 받으면 재사용)."""
    if key in _CC:
        return _CC[key]
    r = requests.get(f"{BASE}/corpCode.xml", params={"crtfc_key": key}, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(zipfile.ZipFile(io.BytesIO(r.content)).read("CORPCODE.xml"))
    _CC[key] = {x.findtext("stock_code").strip(): x.findtext("corp_code")
                for x in root.iter("list") if (x.findtext("stock_code") or "").strip()}
    return _CC[key]


def recent_titles(key, corp_code, days=90):
    """최근 days일 공시 제목 목록."""
    end = date.today()
    p = {"crtfc_key": key, "corp_code": corp_code, "page_count": 100,
         "bgn_de": (end - timedelta(days=days)).strftime("%Y%m%d"), "end_de": end.strftime("%Y%m%d")}
    j = requests.get(f"{BASE}/list.json", params=p, timeout=30).json()
    if j.get("status") not in ("000", "013"):  # 013 = 조회된 데이터 없음(정상)
        raise RuntimeError(f"DART 응답 {j.get('status')}: {j.get('message')}")
    return [re.sub(r"\s+", " ", x["report_nm"]).strip() for x in j.get("list", [])]


def fetch_summaries(codes, key=None, days=90):
    """{종목코드: summarize(...)}; 키가 없거나 조회 실패한 종목은 빠진다."""
    global LAST_ERROR
    key = key or os.environ.get("DART_API_KEY")
    LAST_ERROR = ""
    if not key:
        LAST_ERROR = "DART_API_KEY 없음"
        return {}
    if not codes:
        return {}
    try:
        cc = corp_codes(key)
    except Exception as e:
        LAST_ERROR = f"회사 코드 목록 실패: {type(e).__name__} {str(e)[:80]}"
        return {}
    out = {}
    for code in codes:
        try:
            if code in cc:
                out[code] = summarize(recent_titles(key, cc[code], days))
        except Exception as e:
            LAST_ERROR = f"{type(e).__name__} {str(e)[:80]}"
    return out
