"""네이버 금융 한국 시장 데이터 출처 진단: 후보 주소마다 상태·인코딩·형태·앞부분을 한 번에 출력한다. 사용: python -m stock.diag_naver"""
import re

import requests

from stock.scan import HEAD, decode

F = "https://finance.naver.com"
M = "https://m.stock.naver.com"
CANDIDATES = [
    ("pc 테마 목록", F + "/sise/theme.naver", {}),
    ("pc 테마 목록 page=1", F + "/sise/theme.naver", {"page": 1}),
    ("pc 테마 상세 511", F + "/sise/sise_group_detail.naver", {"type": "theme", "no": 511}),
    ("pc 업종 목록", F + "/sise/sise_group.naver", {"type": "upjong"}),
    ("pc 시가총액 순위", F + "/sise/sise_market_sum.naver", {"sosok": 0, "page": 1}),
    ("pc 거래량 상위", F + "/sise/sise_quant.naver", {"sosok": 0}),
    ("mobile 시가총액 KOSPI", M + "/api/stocks/marketValue/KOSPI", {"page": 1, "pageSize": 5}),
    ("mobile 상승률 KOSPI", M + "/api/stocks/up/KOSPI", {"page": 1, "pageSize": 5}),
    ("mobile 거래량 KOSPI", M + "/api/stocks/quantTop/KOSPI", {"page": 1, "pageSize": 5}),
    ("mobile 거래대금 KOSPI", M + "/api/stocks/priceTop/KOSPI", {"page": 1, "pageSize": 5}),
    ("mobile 테마 1", M + "/api/stocks/theme", {"page": 1, "pageSize": 5}),
    ("mobile 테마 2", M + "/api/theme/list", {"page": 1, "pageSize": 5}),
    ("mobile 테마 3", M + "/api/stocks/themeGroup", {"page": 1, "pageSize": 5}),
    ("mobile 업종", M + "/api/stocks/industry", {"page": 1, "pageSize": 5}),
    ("api.stock 테마", "https://api.stock.naver.com/stock/theme", {"page": 1, "pageSize": 5}),
]


def describe(r):
    ct = r.headers.get("content-type", "")
    text = decode(r.content)
    out = f"[{r.status_code}] len={len(r.content)} type={ct.split(';')[0]} final={r.url[:90]}"
    if "json" in ct:
        return out + " :: " + re.sub(r"\s+", " ", text)[:260]
    flags = {"__NEXT_DATA__": "__NEXT_DATA__" in text, "sise_group_detail": text.count("sise_group_detail"),
             "item/main.naver": text.count("item/main.naver"), "<tr": text.count("<tr"), "theme": text.lower().count("theme")}
    plain = re.sub(r"\s+", " ", re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", text, flags=re.S))[:140]
    return out + f" flags={flags} text='{plain}'"


def main():
    for label, url, params in CANDIDATES:
        try:
            print(f"{label:<22}", describe(requests.get(url, params=params, headers=HEAD, timeout=15)))
        except Exception as e:
            print(f"{label:<22} [ERR] {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
