"""스냅샷의 수급이 직전 거래일까지 들어왔는지 점검하고, 비율이 낮으면 텔레그램으로 알린다.
사용: python -m stock.flowcheck <snapshot.csv> <이름>

네이버가 당일 외국인·기관 수급을 저녁 8시 이후에도 주지 않는 것이 확인되어(2026-10-28), 수급은 '직전 거래일까지'면 정상으로 본다.
가격은 그날 종가, 수급은 하루 전까지로 매일 똑같이 정의하면 둘 다 그 시점에 알 수 있는 값이라 미래 정보를 쓰지 않는다.
"""
import sys

import numpy as np
import pandas as pd


def lag_ok(s):
    """flow_last 가 date 의 직전 영업일 이후(영업일 기준 하루 이내)인 행. 주말은 건너뛰고 공휴일은 고려하지 않는다."""
    fl = pd.to_datetime(s["flow_last"], errors="coerce")
    dt = pd.to_datetime(s["date"], errors="coerce")
    return pd.Series([bool(pd.notna(f) and pd.notna(d) and np.busday_count(f.date(), d.date()) <= 1) for f, d in zip(fl, dt)],
                     index=s.index)


def main(path, label):
    s = pd.read_csv(path)
    ok = int(lag_ok(s).sum())
    msg = f"{label} {s['date'].max()}: 수급이 직전 거래일까지 확인된 종목 {ok}/{len(s)}"
    print(msg)
    if ok < 0.8 * len(s):
        from trader import notify
        notify.send("⚠️ " + msg + " — 수급이 이틀 이상 늦습니다. 이 날 스냅샷은 시험에서 제외됩니다.")


if __name__ == "__main__":
    main(sys.argv[1], " ".join(sys.argv[2:]) or "스냅샷")
