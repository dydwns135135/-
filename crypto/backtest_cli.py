"""python -m crypto.backtest_cli data.csv [--leverage 2] [--no-short]   (CSV: date,open,high,low,close, 4시간봉 권장)
--demo 는 가상 데이터(성능과 무관)."""
import argparse

from trader.backtest import load_csv, synthetic_prices

from .backtest import FuturesConfig, backtest_futures

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?")
ap.add_argument("--demo", action="store_true")
ap.add_argument("--leverage", type=float, default=2.0)
ap.add_argument("--margin-pct", type=float, default=0.25)
ap.add_argument("--stop-loss", type=float, default=0.03)
ap.add_argument("--bar-hours", type=float, default=4.0)
ap.add_argument("--no-short", action="store_true")
a = ap.parse_args()
if not a.csv and not a.demo:
    ap.error("CSV 경로 또는 --demo 필요")
df = synthetic_prices(3000, vol=0.01) if a.demo else load_csv(a.csv)
cfg = FuturesConfig(a.leverage, a.margin_pct, a.stop_loss, not a.no_short, bar_hours=a.bar_hours)
r = backtest_futures(df, cfg)
for k, v in r.items():
    if k not in ("equity", "trades"):
        print(f"{k}: {v:.2%}" if isinstance(v, float) else f"{k}: {v}")
