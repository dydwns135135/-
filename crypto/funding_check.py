"""Bitget 선물의 실제 펀딩비 확인(조회 전용): 롱을 들고 있을 때 연간 비용(+) 또는 수취(-)가 얼마인지.
python -m crypto.funding_check   (공개 시세만 사용, API 키·주문 없음)

백테스트의 펀딩비 가정(연 ~11%, 코인 기준)이 비(非)코인 상품에도 맞는지 확인하는 용도.
펀딩비가 양수면 롱이 숏에게 지불한다. 정산 간격(1·4·8시간)은 상품마다 달라 이력의 시각 간격으로 추정한다."""
from __future__ import annotations

import argparse
import os

import ccxt
import numpy as np
import pandas as pd

SYMBOLS = ("BTC", "ETH", "XAU", "XAUT", "PAXG", "XAG", "QQQ", "SPY", "NDX100", "SP500", "NVDA", "TSLA", "AAPL", "MSTR", "COIN", "TQQQ")
PAGES = 6  # 100개씩 최대 6번(최근 600회 정산)


def summarize(rows: list[dict]) -> dict:
    """펀딩비 이력(timestamp[ms], fundingRate) → 간격·평균·연환산(%)·양수 비율."""
    d = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
    if len(d) < 3:
        raise ValueError(f"이력이 너무 적음({len(d)}개)")
    interval_h = float(np.median(np.diff(d["timestamp"].to_numpy())) / 3_600_000)
    mean = float(d["fundingRate"].astype(float).mean())
    span_days = float((d["timestamp"].iloc[-1] - d["timestamp"].iloc[0]) / 86_400_000)
    return {
        "정산간격(h)": round(interval_h, 1),
        "기간(일)": round(span_days),
        "회당평균(%)": mean * 100,
        "연환산(%)": mean * (8760 / interval_h) * 100,   # 롱 기준: +면 비용, -면 수취
        "양수비율(%)": float((d["fundingRate"].astype(float) > 0).mean() * 100),
    }


def history(ex, symbol: str, pages: int = PAGES) -> list[dict]:
    """최근 이력부터 과거로 거슬러 이어 받는다."""
    rows: list[dict] = []
    until = None
    for _ in range(pages):
        params = {"until": until} if until else {}
        batch = ex.fetch_funding_rate_history(symbol, limit=100, params=params)
        if not batch:
            break
        rows = [{"timestamp": b["timestamp"], "fundingRate": b["fundingRate"]} for b in batch] + rows
        new_until = batch[0]["timestamp"] - 1
        if until is not None and new_until >= until:
            break
        until = new_until
        if len(batch) < 100:
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    out = []
    for base in SYMBOLS:
        sym = f"{base}/USDT:USDT"
        if sym not in ex.markets:
            out.append({"종목": base, "비고": "Bitget에 없음"})
            continue
        try:
            out.append({"종목": base, **summarize(history(ex, sym)), "비고": ""})
        except Exception as e:  # 한 종목 실패가 전체를 막지 않게(이유는 표에 남긴다)
            out.append({"종목": base, "비고": f"{type(e).__name__}: {str(e)[:60]}"})
    df = pd.DataFrame(out)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "funding_check.csv"), index=False)
    view = df.copy()
    for c in ("회당평균(%)", "연환산(%)", "양수비율(%)"):
        if c in view:
            view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:+.3f}" if c == "회당평균(%)" else f"{v:+.1f}" if c == "연환산(%)" else f"{v:.0f}")
    text = view.fillna("").to_string(index=False)
    note = ("\n\n해석: 연환산(%)이 +면 롱을 들고 있을 때 매년 그만큼 비용을 내고, -면 받습니다. 백테스트의 가정은 +11%였습니다.\n"
            "양수비율은 정산 중 롱이 지불한 비율입니다. 과거 평균이며 시장 상황에 따라 크게 바뀝니다.")
    print(text + note)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("### Bitget 실제 펀딩비\n\n```\n" + text + note + "\n```\n")


if __name__ == "__main__":
    main()
