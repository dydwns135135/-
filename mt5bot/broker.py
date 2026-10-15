"""MT5 터미널 연동 (Windows + MetaTrader5 패키지 필요). 임포트는 지연시켜 테스트가 가능하게 한다."""
import math
import os
from datetime import datetime

from .strategy import sl_tp_prices

MAGIC = 20260513
DEVIATION = 20  # 허용 슬리피지(포인트)


def _mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as e:
        raise SystemExit("MetaTrader5 패키지가 없습니다. Windows 에서 `pip install MetaTrader5` 후 MT5 터미널을 실행해 두세요.") from e
    return mt5


def connect(allow_live=False):
    """터미널 초기화 + (환경변수가 있으면) 로그인. 데모 계좌가 아니면 allow_live 없이는 거부."""
    mt5 = _mt5()
    kwargs = {}
    if os.environ.get("MT5_PATH"):
        kwargs["path"] = os.environ["MT5_PATH"]
    if os.environ.get("MT5_LOGIN"):
        kwargs.update(login=int(os.environ["MT5_LOGIN"]),
                      password=os.environ.get("MT5_PASSWORD", ""),
                      server=os.environ.get("MT5_SERVER", ""))
    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"MT5 초기화 실패: {mt5.last_error()}")
    info = mt5.account_info()
    if info is None:
        raise RuntimeError(f"계좌 정보 없음(로그인 확인): {mt5.last_error()}")
    if info.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO and not allow_live:
        mt5.shutdown()
        raise SystemExit("실계좌입니다. 데모에서 검증 후 --allow-live 를 명시해야 실행됩니다.")
    return info


def shutdown():
    _mt5().shutdown()


def account_summary():
    """잔고/자산/마진 요약."""
    info = _mt5().account_info()
    return {
        "login": info.login, "server": info.server, "currency": info.currency,
        "mode": "DEMO" if info.trade_mode == _mt5().ACCOUNT_TRADE_MODE_DEMO else "LIVE",
        "balance": info.balance, "equity": info.equity, "margin": info.margin,
        "free_margin": info.margin_free, "margin_level": info.margin_level, "leverage": info.leverage,
    }


def get_rates(symbol, count=600):
    """5분봉 확정봉만(진행 중인 마지막 봉 제외). (시각, 고가, 저가, 종가) 리스트 4개."""
    mt5 = _mt5()
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"심볼 선택 실패 {symbol}: {mt5.last_error()}")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count + 1)
    if rates is None or len(rates) < 2:
        raise RuntimeError(f"캔들 수집 실패: {mt5.last_error()}")
    rates = rates[:-1]
    return ([datetime.fromtimestamp(int(r["time"])) for r in rates],
            [float(r["high"]) for r in rates], [float(r["low"]) for r in rates],
            [float(r["close"]) for r in rates])


def get_history(symbol, count=30000):
    """백테스트용 과거 5분봉(확정봉). 요청 개수를 못 받으면 절반씩 줄여 재시도. (시각, 고, 저, 종) 리스트."""
    mt5 = _mt5()
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"심볼 선택 실패 {symbol}: {mt5.last_error()}")
    n = count
    while n >= 1000:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n + 1)
        if rates is not None and len(rates) > 1000:
            rates = rates[:-1]
            return ([datetime.fromtimestamp(int(r["time"])) for r in rates],
                    [float(r["high"]) for r in rates], [float(r["low"]) for r in rates],
                    [float(r["close"]) for r in rates])
        n //= 2
    raise RuntimeError(f"과거 데이터 수집 실패: {mt5.last_error()} (MT5 에서 GOLD 5분 차트를 열어 과거 데이터를 받은 뒤 다시 시도)")


def equity():
    return _mt5().account_info().equity


def positions(symbol):
    mt5 = _mt5()
    return [p for p in (mt5.positions_get(symbol=symbol) or []) if p.magic == MAGIC]


def calc_volume(symbol, sl_dist, risk_money):
    """손절 시 손실이 risk_money 이하가 되는 랏(최소/최대/스텝 보정, 내림)."""
    mt5 = _mt5()
    si = mt5.symbol_info(symbol)
    if si.trade_tick_size <= 0 or si.trade_tick_value <= 0:
        return si.volume_min
    loss_per_lot = sl_dist / si.trade_tick_size * si.trade_tick_value
    lots = risk_money / loss_per_lot
    lots = math.floor(lots / si.volume_step) * si.volume_step
    lots = min(max(lots, si.volume_min), si.volume_max)
    return round(lots, 2)


def min_volume(symbol):
    return _mt5().symbol_info(symbol).volume_min


def _filling(symbol):
    mt5 = _mt5()
    mode = mt5.symbol_info(symbol).filling_mode
    if mode & 1:
        return mt5.ORDER_FILLING_FOK
    if mode & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def open_order(symbol, side, volume, sl_dist, tp_dist):
    """시장가 진입. SL/TP 를 주문에 반드시 포함하며, 최소 거리(stops_level) 미달이면 주문하지 않는다."""
    mt5 = _mt5()
    si = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if side == "buy" else tick.bid
    min_dist = (si.trade_stops_level + si.spread) * si.point
    if sl_dist < min_dist or tp_dist < min_dist:
        raise ValueError(f"SL/TP 거리가 브로커 최소 거리({min_dist:.{si.digits}f})보다 작습니다")
    sl, tp = sl_tp_prices(side, price, sl_dist, tp_dist, si.digits)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "rsi-bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling(symbol),
    }
    check = mt5.order_check(request)
    if check is None or check.retcode != 0:
        raise RuntimeError(f"order_check 실패: {check}")
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"order_send 실패: {result}")
    slip = abs(result.price - price) / si.point
    return {"ticket": result.order, "requested": price, "filled": result.price,
            "slippage_points": slip, "sl": sl, "tp": tp, "volume": volume}


def close_position(pos):
    """보유 포지션 반대 매매로 청산."""
    mt5 = _mt5()
    tick = mt5.symbol_info_tick(pos.symbol)
    is_buy = pos.type == mt5.POSITION_TYPE_BUY
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
        "position": pos.ticket,
        "price": tick.bid if is_buy else tick.ask,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "rsi-bot close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling(pos.symbol),
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"청산 실패: {result}")
    return result
