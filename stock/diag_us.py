"""미국 데이터 출처 진단: 후보 주소마다 상태 코드와 응답 앞부분을 한 번에 출력한다. 사용: python -m stock.diag_us"""
import re

import requests

from stock.yahoo import HEAD

NV = "https://api.stock.naver.com/chart/foreign/item/"
CANDIDATES = [
    ("naver NVDA.O 날짜12", NV + "NVDA.O/day", {"startDateTime": "202509010000", "endDateTime": "202610282359"}),
    ("naver NVDA.O 날짜14", NV + "NVDA.O/day", {"startDateTime": "20250901000000", "endDateTime": "20261028235959"}),
    ("naver NVDA.O 무파라미터", NV + "NVDA.O/day", {}),
    ("naver NVDA.O dayCandle", NV + "NVDA.O", {"periodType": "dayCandle", "range": "300"}),
    ("naver SPY.P 날짜12", NV + "SPY.P/day", {"startDateTime": "202509010000", "endDateTime": "202610282359"}),
    ("naver front-api NVDA.O", "https://m.stock.naver.com/front-api/chart/pricesByPeriod",
     {"reutersCode": "NVDA.O", "category": "stock", "chartInfoType": "stock", "scriptChartType": "day"}),
    ("naver LMT.N", NV + "LMT.N/day", {"startDateTime": "202509010000", "endDateTime": "202610282359"}),
    ("naver LMT 접미사없음", NV + "LMT/day", {"startDateTime": "202509010000", "endDateTime": "202610282359"}),
    ("naver ABBV.N", NV + "ABBV.N/day", {"startDateTime": "202509010000", "endDateTime": "202610282359"}),
    ("naver TSM.N", NV + "TSM.N/day", {"startDateTime": "202509010000", "endDateTime": "202610282359"}),
    ("naver basic LMT.N", "https://api.stock.naver.com/stock/LMT.N/basic", {}),
    ("naver 검색 LMT", "https://ac.stock.naver.com/ac", {"q": "LMT", "target": "stock,index,marketindicator"}),
    ("naver basic NVDA.O", "https://api.stock.naver.com/stock/NVDA.O/basic", {}),
    ("naver 검색 NVDA", "https://ac.stock.naver.com/ac", {"q": "NVDA", "target": "stock,index,marketindicator"}),
    ("yahoo query2 NVDA", "https://query2.finance.yahoo.com/v8/finance/chart/NVDA", {"range": "5d", "interval": "1d"}),
    ("stooq nvda.us", "https://stooq.com/q/d/l/", {"s": "nvda.us", "i": "d"}),
    ("nasdaq.com NVDA", "https://api.nasdaq.com/api/quote/NVDA/historical",
     {"assettype": "stocks", "fromdate": "2026-10-01", "limit": "10", "todate": "2026-10-27"}),
]


def main():
    for label, url, params in CANDIDATES:
        try:
            r = requests.get(url, params=params, headers=HEAD, timeout=15)
            body = re.sub(r"\s+", " ", r.text)[:160]
            print(f"[{r.status_code}] {label} len={len(r.text)} :: {body}")
        except Exception as e:
            print(f"[ERR] {label} :: {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
