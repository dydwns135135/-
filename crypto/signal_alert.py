"""알림 전용 모드(업비트 현물 등에서 직접 주문): 거래소 API 키·서버 없이 업비트 공개 시세로 앙상블 목표 비중을 계산하고,
조정이 필요한 날에만 '오늘 할 일'을 텔레그램으로 알린다. 주문은 사람이 앱에서 직접 한다.

python -m crypto.signal_alert --capital-krw 1000000
- 기준 비중(마지막으로 조정하라고 알린 비중)은 state 파일에 저장한다(Actions 캐시). 목표와 기준이 --step(기본 25%p) 이상 차이나거나
  목표가 0 이 되면 조정 알림. 알림을 받고도 주문하지 않으면 실제 보유와 어긋나므로, 매 메시지에 '목표 보유 금액'을 함께 적는다."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from trader import notify

from .asset_mix import check_daily
from .momentum_bot import ENSEMBLE_LOOKBACKS, ensemble_weight
from .upbit_data import closed_only, fetch_upbit_daily


def decide_alert(w: float, ref: float | None, step: float = 0.25) -> str:
    """'init'(첫 실행) | 'adjust' | 'hold'. 목표가 0 이 되었는데 기준이 양(+)이면 항상 조정."""
    if ref is None:
        return "init" if w > 0 else "hold"
    if w <= 0 and ref > 0:
        return "adjust"
    return "adjust" if abs(w - ref) >= step - 1e-12 else "hold"


def build_message(w: float, npos: int, ref: float | None, action: str, price: float, closes: pd.Series,
                  capital: float | None, alloc: float) -> str:
    n = len(ENSEMBLE_LOOKBACKS)
    lines = [f"📊 BTC 앙상블: {n}개 기간 중 {npos}개 양(+) → 목표 비중 {w:.0%}", f"BTC 현재가 {price:,.0f}원"]
    if capital:
        tgt = capital * alloc * w
        lines.append(f"목표 보유 금액 {tgt:,.0f}원 (투입금 {capital:,.0f}원 × {alloc:.0%} × {w:.0%}) ≈ {tgt / price:.6f} BTC")
    if action == "adjust" and ref is not None:
        diff = w - ref
        verb = "추가 매수" if diff > 0 else "일부 매도" if w > 0 else "전량 매도"
        amt = f" 약 {capital * alloc * abs(diff):,.0f}원" if capital else f" (비중 {abs(diff):.0%}p)"
        lines.insert(0, f"🔔 조정 필요: {verb}{amt}")
        lines.append(f"기준 비중 {ref:.0%} → {w:.0%} 로 갱신")
    elif action == "init":
        lines.insert(0, "🔔 첫 설정: 아래 목표 비중에 맞춰 보유량을 맞추세요")
    else:
        lines.insert(0, f"조정 없음 (기준 비중 {ref:.0%}, 차이 {abs(w - (ref or 0)):.0%}p)")
    lines.append("※ 알림을 받고 주문하지 않았다면 실제 보유와 어긋납니다. 목표 보유 금액을 기준으로 맞추세요.")
    return "\n".join(lines)


def load_state(path: Path) -> float | None:
    try:
        v = json.loads(path.read_text()).get("ref_weight")
        return float(v) if v is not None else None
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        return None


def save_state(path: Path, ref: float) -> None:
    path.write_text(json.dumps({"ref_weight": ref}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capital-krw", type=float, default=None, help="투입 원화 금액(선택). 있으면 목표 보유 금액을 계산")
    ap.add_argument("--alloc", type=float, default=1.0, help="투입금 중 앙상블 100%% 일 때 BTC 로 보유할 비율(기본 100%%)")
    ap.add_argument("--step", type=float, default=0.25, help="조정을 알리는 비중 차이(기본 25%%p)")
    ap.add_argument("--state", default="signal_state.json")
    a = ap.parse_args()
    if not 0 < a.alloc <= 1:
        raise SystemExit("--alloc 는 0 초과 1 이하여야 합니다(현물 1배).")
    try:
        df = closed_only(check_daily(fetch_upbit_daily("KRW-BTC", total=600)))
        res = ensemble_weight(df["close"])
        if res is None:
            raise SystemExit(f"일봉이 부족합니다({len(df)}개)")
        w, npos = res
        state = Path(a.state)
        ref = load_state(state)
        action = decide_alert(w, ref, a.step)
        msg = build_message(w, npos, ref, action, float(df["close"].iloc[-1]), df["close"], a.capital_krw, a.alloc)
        if action in ("init", "adjust"):
            save_state(state, w)
        elif ref is None:
            save_state(state, w)
    except Exception as e:
        notify.send(f"❌ 알림 전용 모드 오류: {type(e).__name__}: {str(e)[:200]}")
        raise
    print(msg)
    notify.send(msg)


if __name__ == "__main__":
    main()
