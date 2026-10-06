"""선물 백테스트: 롱/숏, 레버리지, 수수료, 펀딩비, 거래소 손절, 청산 반영.
신호는 봉 종가에 확정 → 다음 봉 시가 체결. 봉 안에서 고가/저가가 손절가·청산가에 닿으면 처리."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader.strategy import StrategyParams

from .strategy import compute_ls_signals


@dataclass
class FuturesConfig:
    leverage: float = 2.0
    margin_pct: float = 0.25  # 잔고 중 증거금으로 쓰는 비율
    stop_loss: float = 0.03  # 진입가 대비 가격 변동 %
    allow_short: bool = True
    fee_rate: float = 0.0006  # 편도 테이커 수수료 (실제 요율 확인 후 조정)
    slippage: float = 0.0005
    funding_per_8h: float = 0.0001  # 롱이 지불, 숏이 수취(상수 가정)
    bar_hours: float = 4.0
    maint_margin: float = 0.005  # 유지증거금률(단순화)


def backtest_futures(df: pd.DataFrame, cfg: FuturesConfig = FuturesConfig(),
                     params: StrategyParams = StrategyParams(), balance: float = 1000.0) -> dict:
    sig = compute_ls_signals(df, params)
    O, H, L, C = (sig[k].to_numpy() for k in ("open", "high", "low", "close"))
    le, lx, se, sx = (sig[k].to_numpy() for k in ("long_entry", "long_exit", "short_entry", "short_exit"))
    bal, pos, qty, entry, margin = balance, 0, 0.0, 0.0, 0.0
    pending, equity, trades, liquidations = None, [], [], 0
    fund = cfg.funding_per_8h * cfg.bar_hours / 8

    def close(px_raw: float, i: int, reason: str, slip_dir: int = 1):
        nonlocal bal, pos, qty
        px = px_raw * (1 - pos * cfg.slippage * slip_dir)
        pnl = pos * qty * (px - entry)
        bal += pnl - qty * px * cfg.fee_rate
        trades.append((sig.index[i], pos, entry, px, pnl, reason))
        pos, qty = 0, 0.0

    for i in range(len(sig)):
        if pending == "flat" and pos != 0:
            close(O[i], i, "signal")
        elif pending in ("long", "short") and pos == 0 and bal > 0:
            side = 1 if pending == "long" else -1
            margin = bal * cfg.margin_pct
            px = O[i] * (1 + side * cfg.slippage)
            qty = margin * cfg.leverage / px
            bal -= qty * px * cfg.fee_rate
            pos, entry = side, px
        pending = None

        if pos != 0:  # 봉 내부 손절/청산
            stop = entry * (1 - pos * cfg.stop_loss)
            liq = entry * (1 - pos * (1 / cfg.leverage - cfg.maint_margin))
            worst = L[i] if pos == 1 else H[i]
            first = max(stop, liq) if pos == 1 else min(stop, liq)  # 가격이 먼저 닿는 쪽
            if (pos == 1 and worst <= first) or (pos == -1 and worst >= first):
                if first == liq and abs(liq - entry) < abs(stop - entry):  # 청산이 손절보다 먼저
                    bal -= margin  # 증거금 전액 손실
                    trades.append((sig.index[i], pos, entry, liq, -margin, "liquidation"))
                    liquidations += 1
                    pos, qty = 0, 0.0
                else:
                    fill = min(stop, O[i]) if pos == 1 else max(stop, O[i])  # 갭이면 시가로 체결
                    close(fill, i, "stop")
        if pos != 0:
            bal -= pos * qty * C[i] * fund

        if pos == 1 and lx[i]:
            pending = "flat"
        elif pos == -1 and sx[i]:
            pending = "flat"
        elif pos == 0:
            if le[i] and not lx[i]:
                pending = "long"
            elif cfg.allow_short and se[i] and not sx[i]:
                pending = "short"
        equity.append(bal + (pos * qty * (C[i] - entry) if pos else 0.0))
        if bal <= 0:
            break

    eq = pd.Series(equity, index=sig.index[: len(equity)])
    tdf = pd.DataFrame(trades, columns=["date", "side", "entry", "exit", "pnl", "reason"])
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / balance - 1
    return {
        "equity": eq, "trades": tdf,
        "총수익률": total,
        "연환산수익률": (1 + total) ** (1 / years) - 1 if total > -1 else -1.0,
        "최대낙폭(MDD)": float((eq / eq.cummax() - 1).min()),
        "거래횟수": len(tdf), "청산횟수": liquidations,
        "승률": float((tdf["pnl"] > 0).mean()) if len(tdf) else float("nan"),
        "단순보유수익률": float(C[len(eq) - 1] / C[0] - 1),
    }
