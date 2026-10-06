"""CSV(date,open,high,low,close)로 백테스트: python -m trader.backtest_cli data.csv  (--demo 로 가상 데이터)"""
import argparse

from .backtest import backtest, load_csv, synthetic_prices
from .strategy import StrategyParams

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?")
ap.add_argument("--demo", action="store_true", help="가상 랜덤워크 데이터 사용(전략 성능과 무관)")
ap.add_argument("--cash", type=float, default=10_000)
a = ap.parse_args()
if not a.csv and not a.demo:
    ap.error("CSV 경로 또는 --demo 필요")
df = synthetic_prices() if a.demo else load_csv(a.csv)
r = backtest(df, StrategyParams(), cash=a.cash)
for k, v in r.metrics.items():
    print(f"{k}: {v:.2%}" if isinstance(v, float) else f"{k}: {v}")
