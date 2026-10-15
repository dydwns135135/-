"""과거 5분봉 백테스트. 순수 함수(데이터는 리스트로 받음)라 MT5 없이 테스트 가능.
가정: 신호가 난 봉의 종가에 진입, 다음 봉부터 고가/저가로 SL/TP 판정(같은 봉에 둘 다 닿으면 SL 우선),
거래당 스프레드+슬리피지 비용(가격 단위)을 한 번 차감. 손익은 가격 단위(금은 달러/온스, 0.01랏 ≈ 1온스 기준 달러)."""
from .indicators import atr, ema, rsi
from .strategy import entry_signal


# ---------- 전략: 봉마다 'buy' / 'sell' / None 리스트를 돌려준다 ----------
def sig_rsi(h, l, c, oversold=30, overbought=70, trend_filter=False, rsi_period=14, ema_period=200):
    r, e = rsi(c, rsi_period), ema(c, ema_period)
    out = [None] * len(c)
    for i in range(1, len(c)):
        s = entry_signal(r[i - 1], r[i], oversold, overbought)
        if s and trend_filter:
            if e[i] is None or (s == "buy") != (c[i] > e[i]):
                s = None
        out[i] = s
    return out


def sig_donchian(h, l, c, n=20):
    """직전 n 개 봉의 최고가 돌파 → 매수, 최저가 이탈 → 매도 (추세선 돌파의 단순 버전)."""
    out = [None] * len(c)
    for i in range(n, len(c)):
        hi, lo = max(h[i - n:i]), min(l[i - n:i])
        if c[i] > hi:
            out[i] = "buy"
        elif c[i] < lo:
            out[i] = "sell"
    return out


def sig_ema_pullback(h, l, c, fast=20, mid=50, slow=200):
    """EMA50 > EMA200 인 상승 추세에서 종가가 EMA20 아래→위로 되돌아오면 매수(하락은 반대)."""
    ef, em, es = ema(c, fast), ema(c, mid), ema(c, slow)
    out = [None] * len(c)
    for i in range(1, len(c)):
        if None in (ef[i], ef[i - 1], em[i], es[i]):
            continue
        if em[i] > es[i] and c[i - 1] < ef[i - 1] and c[i] >= ef[i]:
            out[i] = "buy"
        elif em[i] < es[i] and c[i - 1] > ef[i - 1] and c[i] <= ef[i]:
            out[i] = "sell"
    return out


# ---------- 시뮬레이터 ----------
def simulate(h, l, c, signals, atrs, sl_mult, tp_mult, spread, rsi_exit=None):
    """한 번에 포지션 하나. rsi_exit 에 RSI 리스트를 주면 RSI 50 되돌파 청산도 사용. 거래 리스트 반환."""
    trades, pos = [], None
    for i in range(len(c)):
        if pos:
            side, entry, sl, tp, i0 = pos
            hit = None
            if side == "buy":
                if l[i] <= sl:
                    hit = sl
                elif h[i] >= tp:
                    hit = tp
            else:
                if h[i] >= sl:
                    hit = sl
                elif l[i] <= tp:
                    hit = tp
            if hit is None and rsi_exit is not None and rsi_exit[i] is not None:
                if (side == "buy" and rsi_exit[i] >= 50) or (side == "sell" and rsi_exit[i] <= 50):
                    hit = c[i]
            if hit is not None:
                pnl = ((hit - entry) if side == "buy" else (entry - hit)) - spread
                trades.append({"i": i0, "pnl": pnl})
                pos = None
        if pos is None and signals[i] and atrs[i]:
            d = atrs[i]
            if signals[i] == "buy":
                pos = ("buy", c[i], c[i] - sl_mult * d, c[i] + tp_mult * d, i)
            else:
                pos = ("sell", c[i], c[i] + sl_mult * d, c[i] - tp_mult * d, i)
    return trades


def stats(trades):
    n = len(trades)
    if n == 0:
        return {"n": 0, "win": 0.0, "total": 0.0, "avg": 0.0, "pf": 0.0, "maxdd": 0.0}
    pnls = [t["pnl"] for t in trades]
    wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p <= 0]
    eq = peak = dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    gross_loss = -sum(losses)
    return {"n": n, "win": len(wins) / n * 100, "total": sum(pnls), "avg": sum(pnls) / n,
            "pf": (sum(wins) / gross_loss) if gross_loss > 0 else float("inf"), "maxdd": dd}


def split_stats(trades, split_index):
    """학습 구간(앞 70%) / 검증 구간(뒤 30%) 으로 거래를 나눠 각각 통계. 설정을 과거에 끼워 맞추는 것(과최적화) 점검용."""
    return stats([t for t in trades if t["i"] < split_index]), stats([t for t in trades if t["i"] >= split_index])


def run_grid(h, l, c, spread, sl_mults=(1.0, 1.5, 2.0), tp_mults=(1.5, 3.0, 4.5), atr_period=14):
    """전략 × SL/TP 배수 조합 전부 실행. 결과 행 리스트."""
    a = atr(h, l, c, atr_period)
    r = rsi(c, 14)
    strategies = {
        "RSI 역추세(필터 없음)": (sig_rsi(h, l, c), r),
        "RSI 역추세+EMA200필터": (sig_rsi(h, l, c, trend_filter=True), r),
        "돌파 채널(20)": (sig_donchian(h, l, c, 20), None),
        "돌파 채널(50)": (sig_donchian(h, l, c, 50), None),
        "EMA 눌림목": (sig_ema_pullback(h, l, c), None),
    }
    split = int(len(c) * 0.7)
    rows = []
    for name, (sigs, rx) in strategies.items():
        for sl in sl_mults:
            for tp in tp_mults:
                t = simulate(h, l, c, sigs, a, sl, tp, spread, rsi_exit=rx)
                ins, oos = split_stats(t, split)
                rows.append({"name": name, "sl": sl, "tp": tp, "all": stats(t), "in": ins, "out": oos})
    return rows
