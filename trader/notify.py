"""텔레그램 알림. TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 가 없으면 아무것도 하지 않는다."""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger("trader")


def send(text: str) -> None:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log.info("텔레그램 설정 없음, 알림 생략")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text[:4000]},
            timeout=10,
        )
        r.raise_for_status()
    except requests.RequestException as e:  # 알림 실패가 매매를 막으면 안 됨
        log.warning("텔레그램 전송 실패: %s", type(e).__name__)
