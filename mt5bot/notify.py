"""텔레그램 알림 (표준 라이브러리만 사용). TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 환경변수 필요.
토큰은 어떤 경우에도 출력하지 않는다."""
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request


CONFIG_FILE = Path(__file__).resolve().parent.parent / "telegram.txt"


def _config():
    """(토큰, chat id). 환경변수가 우선이고, 없으면 telegram.txt 의 첫 줄=토큰, 둘째 줄=chat id."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if (not token or not chat) and CONFIG_FILE.exists():
        lines = [x.strip() for x in CONFIG_FILE.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
        token = token or (lines[0] if lines else "")
        chat = chat or (lines[1] if len(lines) > 1 else "")
    return token, chat


def _call(method, **params):
    token, _ = _config()
    if not token:
        return {"ok": False, "description": "토큰이 설정되지 않았습니다 (telegram.txt 첫 줄 또는 TELEGRAM_BOT_TOKEN)"}
    url = f"https://api.telegram.org/bot{token}/{method}"
    try:
        data = urllib.parse.urlencode(params).encode()
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=15) as r:
            return json.load(r)
    except Exception as e:  # 예외 문구에 토큰(URL)이 들어갈 수 있어 종류만 출력
        return {"ok": False, "description": f"전송 실패({type(e).__name__})"}


def send(text):
    """알림 전송. 실패해도 예외를 내지 않고 결과(dict)를 돌려준다."""
    _, chat = _config()
    if not chat:
        return {"ok": False, "description": "chat id 가 설정되지 않았습니다 (telegram.txt 둘째 줄 또는 TELEGRAM_CHAT_ID)"}
    return _call("sendMessage", chat_id=chat, text=text[:4000])


def find_chat_ids():
    """봇이 최근 받은 메시지에서 chat id 를 찾는다(봇에게 먼저 메시지를 보내 둘 것)."""
    res = _call("getUpdates")
    ids = {}
    for u in res.get("result") or []:
        chat = (u.get("message") or {}).get("chat") or {}
        if "id" in chat:
            ids[chat["id"]] = chat.get("type", "?")
    return res, ids
