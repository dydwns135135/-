"""데이터 로딩: pykrx(실데이터) / CSV 폴더 / 합성 데모.

종목 DataFrame 컬럼: open, high, low, close, volume, (선택) foreign, inst = 외국인/기관 일별 순매수 금액(원).
"""
import os

import numpy as np
import pandas as pd

from stock.themes import THEMES

OHLCV = ["open", "high", "low", "close", "volume"]


def load_csv_dir(path):
    """path/<코드>.csv (index=날짜 첫 열, 컬럼 위 표 참고). KOSPI.csv 가 있으면 시황으로 사용."""
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".csv"):
            out[f[:-4]] = pd.read_csv(os.path.join(path, f), index_col=0, parse_dates=True).sort_index()
    return out


def load_pykrx(codes, days=250, index_code="1001"):
    """pykrx 로 일봉 + 투자자별 순매수(외국인·기관). 설치: pip install pykrx"""
    from pykrx import stock as k

    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(days * 1.6))
    s, e = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    out = {}
    for code in codes:
        df = k.get_market_ohlcv(s, e, code).rename(
            columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"})[OHLCV]
        try:
            tv = k.get_market_trading_value_by_date(s, e, code)
            df["foreign"] = tv["외국인합계"]
            df["inst"] = tv["기관합계"]
        except Exception:
            pass
        out[code] = df
    idx = k.get_index_ohlcv(s, e, index_code).rename(columns={"종가": "close", "거래량": "volume"})
    out["KOSPI"] = idx[["close", "volume"]]
    return out


def synthetic_market(n=260, seed=7):
    """데모용 합성 시장. HBM=상승 초입(수급 유입), 2차전지=이미 급등(과열), 나머지=횡보/약세."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    plan = {"HBM": (0.0035, 0.0), "반도체 전공정": (0.0015, 0.0), "2차전지": (0.0, 1.0),
            "방산": (0.0003, 0.0), "바이오": (-0.0006, 0.0)}
    out = {}
    for theme, members in THEMES.items():
        drift, hot = plan[theme]
        theme_noise = rng.normal(0, 0.008, n)
        for j, code in enumerate(members):
            r = rng.normal(0, 0.014, n) + theme_noise
            k = 45 if hot == 0 else 30
            r[: n - k] += rng.normal(0, 0.0004)
            r[n - k:] += drift * (1.3 - 0.15 * j) if not hot else 0.008 * (1.25 - 0.1 * j)
            if hot:
                r[-5:] += 0.055 * (1.2 - 0.15 * j)  # 막판 급등 → 과열
            elif drift >= 0.003:
                r[-4:] += 0.004  # 초입 가속
            c = 50000 * np.cumprod(1 + r)
            o = np.r_[c[0], c[:-1]]
            h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
            l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
            v = rng.integers(8e5, 1.2e6, n).astype(float)
            ramp = drift != 0 or hot
            v[n - k:] *= 1 + (2.2 if ramp else 0.0) * np.linspace(0.3, 1, k) * (rng.random(k) + 0.5)
            if ramp and not hot:
                v[-4:] *= 2.2  # 거래량 터지기 시작
            val = c * v
            flow = rng.normal(0, 0.01, n) * val
            if ramp:
                flow[n - k:] += 0.06 * val[n - k:]
            out[code] = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v,
                                      "foreign": flow * 0.6, "inst": flow * 0.4}, index=idx)
    m = 2500 * np.cumprod(1 + rng.normal(0.0005, 0.008, n))
    out["KOSPI"] = pd.DataFrame({"close": m, "volume": rng.integers(3e8, 5e8, n).astype(float)}, index=idx)
    return out
