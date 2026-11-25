"""자막 번역. 타이밍은 원본 문장 경계(세그먼트)를 유지한다.

번역 수단 우선순위: cache/translations.json(Claude Code가 작성) > SHORTS_LLM=anthropic > 원본 자막 유지.
translations.json 형식: {"<세그먼트 인덱스>": "한국어 문장"}  (translation_todo.json 이 필요한 목록)
"""
import json
from pathlib import Path

from .selector import _call_claude, _json_from, llm_enabled


def apply(segments, clips, lang, sub_lang, cache):
    """segments 각 항목에 'sub' 필드(화면 자막 텍스트)를 채우고 방식 기록을 돌려준다."""
    cache = Path(cache)
    for i, s in enumerate(segments):
        s["idx"] = i
        s["sub"] = s["text"]
    if sub_lang == "original" or lang.split("-")[0] == sub_lang:
        return {"method": "not-needed", "subtitle_language": lang if sub_lang == "original" else sub_lang}
    need = sorted({s["idx"] for c in clips for s in segments if s["end"] > c["start"] and s["start"] < c["end"]})
    tf = cache / "translations.json"
    tr = json.loads(tf.read_text(encoding="utf-8")) if tf.exists() else {}
    missing = [i for i in need if str(i) not in tr]
    method = "claude-code(translations.json)" if tr and len(missing) < len(need) else None
    if missing and llm_enabled():
        try:
            for a in range(0, len(missing), 60):
                chunk = missing[a:a + 60]
                prompt = ("다음 문장들을 자연스러운 한국어 자막으로 번역하세요. 고유명사는 보존하고 띄어쓰기·문장부호를 다듬으세요. "
                          'JSON 객체 {"인덱스":"번역"} 만 답하세요.\n' +
                          "\n".join(f"{i}: {segments[i]['text']}" for i in chunk))
                tr.update({str(k): v for k, v in (_json_from(_call_claude(prompt)) or {}).items()})
            tf.write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
            method = "llm-anthropic"
        except Exception:
            pass
    missing = [i for i in need if str(i) not in tr]
    for s in segments:
        if str(s["idx"]) in tr:
            s["sub"] = tr[str(s["idx"])]
    if missing:
        (cache / "translation_todo.json").write_text(json.dumps(
            {str(i): segments[i]["text"] for i in missing}, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"method": "none-original-fallback" if not method else method + "+partial-original",
                "note": f"번역 수단이 없어 {len(missing)}개 문장은 원본 언어 자막 유지 (translation_todo.json 참고)",
                "subtitle_language": lang}
    return {"method": method or "translations.json", "subtitle_language": sub_lang}
