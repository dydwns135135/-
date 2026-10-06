"""주문 전 리스크 점검. 한도를 넘으면 주문을 줄이거나 거부한다."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class RiskLimits:
    max_order_usd: float = 500.0  # 1회 주문 최대 금액
    max_position_pct: float = 0.20  # 종목당 최대 비중(총자산 대비)
    max_positions: int = 5
    daily_loss_limit: float = 0.03  # 당일 시작 자산 대비 -3% 이면 신규 매수 중단
    stop_loss: float = 0.08  # 평단 대비 -8% 이면 매도


class RiskManager:
    def __init__(self, limits: RiskLimits, state_path: str = "state.json"):
        self.limits = limits
        self.state_path = Path(state_path)

    def _load(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def start_equity_today(self, equity: float) -> float:
        """오늘 처음 호출된 시점의 자산을 기준선으로 저장."""
        st, today = self._load(), date.today().isoformat()
        if st.get("date") != today:
            st = {"date": today, "start_equity": equity}
            self.state_path.write_text(json.dumps(st))
        return st["start_equity"]

    def halted(self, equity: float) -> bool:
        start = self.start_equity_today(equity)
        return equity < start * (1 - self.limits.daily_loss_limit)

    def stop_hit(self, avg_cost: float, price: float) -> bool:
        return price <= avg_cost * (1 - self.limits.stop_loss)

    def buy_qty(self, price: float, equity: float, cash: float, held_value: float, n_positions: int) -> int:
        """허용되는 매수 수량(0이면 매수 불가). 정수 주 단위."""
        lim = self.limits
        if price <= 0 or n_positions >= lim.max_positions or self.halted(equity):
            return 0
        budget = min(lim.max_order_usd, lim.max_position_pct * equity - held_value, cash)
        return max(0, math.floor(budget / price))
