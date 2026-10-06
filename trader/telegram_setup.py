"""텔레그램 설정 도우미(수동 실행): 토큰이 맞는지, 내 chat id 가 무엇인지 찾고, 테스트 알림을 보낸다.

python -m trader.telegram_setup
- TELEGRAM_BOT_TOKEN 만 있으면: 봇이 받은 최근 메시지에서 chat id 를 찾아 출력한다(봇에게 먼저 메시지를 보내 둘 것).
- TELEGRAM_CHAT_ID 도 있으면: 그 번호로 테스트 알림을 보낸다.
토큰은 어떤 경우에도 출력하지 않는다(요청 주소에 토큰이 들어 있어 예외 메시지도 출력하지 않음)."""
from __future__ import annotations

import os

import requests

API = "https://api.telegram.org/bot{token}/{method}"


def _call(token: str, method: str, **params) -> dict:
    try:
        r = requests.get(API.format(token=token, method=method), params=params, timeout=15)
    except requests.RequestException as e:
        return {"ok": False, "description": f"네트워크 오류({type(e).__name__})"}
    try:
        return r.json()
    except ValueError:
        return {"ok": False, "description": f"응답을 읽지 못함(HTTP {r.status_code})"}


def find_chat_ids(updates: list[dict]) -> dict[int, str]:
    """getUpdates 결과에서 {chat id: 종류}. 이름 등 개인정보는 담지 않는다."""
    found: dict[int, str] = {}
    for u in updates:
        msg = u.get("message") or u.get("edited_message") or u.get("channel_post") or {}
        chat = msg.get("chat") or {}
        if "id" in chat:
            found[int(chat["id"])] = chat.get("type", "?")
    return found


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN 이 설정되지 않았습니다. GitHub Secrets 에 토큰을 먼저 넣으세요.")
        return 1

    me = _call(token, "getMe")
    if not me.get("ok"):
        print(f"❌ 토큰이 올바르지 않습니다: {me.get('description')}")
        print("   BotFather 에서 /revoke 로 새 토큰을 받아 Secret 을 다시 넣어 보세요.")
        return 1
    print(f"✅ 토큰 정상: 봇 @{me['result'].get('username')}")

    upd = _call(token, "getUpdates")
    ids = find_chat_ids(upd.get("result") or []) if upd.get("ok") else {}
    if ids:
        for cid, kind in ids.items():
            print(f"📌 chat id 후보: {cid}  ({kind})   ← 'private' 인 숫자가 내 번호입니다")
    else:
        print("ℹ️ 봇이 받은 메시지가 없습니다. 텔레그램에서 내 봇을 열고 '시작(Start)' 또는 아무 메시지를 보낸 뒤 다시 실행하세요.")

    if chat:
        sent = _call(token, "sendMessage", chat_id=chat, text="✅ 알림 테스트: 자동매매 봇 알림이 연결되었습니다.")
        if sent.get("ok"):
            print(f"✅ 테스트 알림 전송 성공 (chat id {chat}) — 텔레그램에 메시지가 왔는지 확인하세요.")
        else:
            print(f"❌ 테스트 알림 실패: {sent.get('description')}")
            print("   chat id 가 틀렸거나, 봇에게 먼저 메시지를 보내지 않았을 수 있습니다.")
            return 1
    else:
        print("ℹ️ TELEGRAM_CHAT_ID 가 아직 없습니다. 위 'chat id 후보'(private)를 Secret 으로 넣은 뒤 다시 실행하면 테스트 알림을 보냅니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
