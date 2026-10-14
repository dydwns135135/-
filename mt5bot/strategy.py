"""RSI 과매수/과매도 진입·청산 판단 (순수 함수)."""


def entry_signal(rsi_prev, rsi_now, oversold=30.0, overbought=70.0):
    """과매도 구간을 아래→위로 벗어나면 'buy', 과매수 구간을 위→아래로 벗어나면 'sell'."""
    if rsi_prev is None or rsi_now is None:
        return None
    if rsi_prev < oversold <= rsi_now:
        return "buy"
    if rsi_prev > overbought >= rsi_now:
        return "sell"
    return None


def should_exit(side, rsi_now, exit_level=50.0):
    """RSI 가 중립선(50)을 넘어 되돌아오면 청산. SL/TP 와 별개의 추가 청산 조건."""
    if rsi_now is None:
        return False
    return rsi_now >= exit_level if side == "buy" else rsi_now <= exit_level


def sl_tp_prices(side, price, sl_dist, tp_dist, digits):
    """진입가 기준 고정 손절/익절 가격. 가격 단위 거리(XAUUSD 는 달러)."""
    if sl_dist <= 0 or tp_dist <= 0:
        raise ValueError("sl_dist, tp_dist 는 0보다 커야 합니다 (SL/TP 필수)")
    if side == "buy":
        sl, tp = price - sl_dist, price + tp_dist
    else:
        sl, tp = price + sl_dist, price - tp_dist
    return round(sl, digits), round(tp, digits)


def trend_ok(side, price, ema_value):
    """추세 필터: 장기 EMA 위에서는 매수만, 아래에서는 매도만 허용. EMA 미계산이면 거부."""
    if ema_value is None:
        return False
    return price > ema_value if side == "buy" else price < ema_value


def daily_loss_hit(day_start_equity, equity, max_loss_pct):
    """오늘 시작 자산 대비 손실이 한도(%) 이상이면 True. max_loss_pct<=0 이면 사용 안 함."""
    if max_loss_pct <= 0 or day_start_equity <= 0:
        return False
    return (day_start_equity - equity) / day_start_equity * 100.0 >= max_loss_pct
