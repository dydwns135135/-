"""숏츠 후보 선정: (1) 편집자가 확정한 edit_list.json (2) 선택적 LLM (3) API 키 없이 쓰는 휴리스틱."""
import json
import os
import re
import urllib.request
from pathlib import Path

from .util import ShortsError, cjk_ratio

MIN_D, MAX_D, GOOD_LO, GOOD_HI = 15.0, 75.0, 25.0, 60.0
PASS_SCORE = 2.5
HOOK_WORDS = re.compile(r"\b(why|how|never|secret|mistake|truth|actually|but|nobody|biggest|worst|best|stop)\b|"
                        r"왜|어떻게|사실|절대|비밀|실수|진짜|문제|아무도|최고|최악|충격|반전", re.I)
BAD_WORDS = re.compile(r"sponsor|subscribe|like and|hit the bell|patreon|promo code|협찬|구독|좋아요|광고|후원|제공받", re.I)
WEAK_START = re.compile(r"^(and|but|so|that|this|it|also|then|그래서|근데|그리고|그런데|이게|그게|그러니까|음|um|uh)\b", re.I)
TERMINAL = re.compile(r"[.!?。…다요죠]\s*[\"')\]]*$")


def _words(t):
    return re.findall(r"\w+", t.lower())


def _nwords(t):
    return max(1, round(len(re.sub(r"\s", "", t)) / 2.5)) if cjk_ratio(t) > 0.5 else len(t.split())


def units(segments):
    """전사 세그먼트를 문장 단위로 합친다(문장부호 없이 끊긴 조각 병합)."""
    out, cur = [], None
    for s in segments:
        if cur and not TERMINAL.search(cur["text"]) and s["start"] - cur["end"] < 0.8 and s["end"] - cur["start"] < 14:
            cur = {"start": cur["start"], "end": s["end"], "text": cur["text"] + " " + s["text"]}
        else:
            if cur:
                out.append(cur)
            cur = {"start": s["start"], "end": s["end"], "text": s["text"]}
    if cur:
        out.append(cur)
    return out


def _score(us, i, j):
    sub = us[i:j + 1]
    d = sub[-1]["end"] - sub[0]["start"]
    first, last = sub[0]["text"], sub[-1]["text"]
    s = 0.0
    if "?" in first:
        s += 2.0
    s += min(2, len(HOOK_WORDS.findall(first))) * 1.0
    if re.search(r"\d", first):
        s += 0.5
    if _nwords(first) <= 18:
        s += 0.5
    if WEAK_START.search(first.strip()):
        s -= 1.0
    wps = sum(_nwords(u["text"]) for u in sub) / d
    s += max(0.0, 1 - abs(wps - 2.7) / 2.7) * 1.5
    s += 1.0 if GOOD_LO <= d <= GOOD_HI else 0.0
    s += 1.0 if TERMINAL.search(last) else -0.5
    gaps = [sub[k + 1]["start"] - sub[k]["end"] for k in range(len(sub) - 1)]
    s -= 1.5 * sum(g > 2.0 for g in gaps)
    if sum(g for g in gaps if g > 0) / d > 0.25:
        s -= 1.0
    if BAD_WORDS.search(" ".join(u["text"] for u in sub)):
        s -= 3.0
    return s


def _reason(us, i, j, score):
    first = us[i]["text"]
    bits = []
    if "?" in first:
        bits.append("질문으로 시작해 궁금증을 유발")
    elif HOOK_WORDS.search(first):
        bits.append("강한 주장·반전 어휘로 시작")
    else:
        bits.append("첫 문장이 짧고 바로 본론으로 진입")
    d = us[j]["end"] - us[i]["start"]
    bits.append(f"{d:.0f}초 안에 문장 단위로 완결")
    return f"{', '.join(bits)} (휴리스틱 점수 {score:.1f})."


def _overlap(a, b):
    o = min(a[1], b[1]) - max(a[0], b[0])
    return max(0.0, o) / max(1e-6, min(a[1] - a[0], b[1] - b[0]))


def _jacc(a, b):
    a, b = set(_words(a)), set(_words(b))
    return len(a & b) / len(a | b) if a and b else 0.0


def pad_bounds(segments, start, end, pre=0.15, post=0.3):
    """발화 경계를 기준으로 앞뒤 여유를 주되 이웃 발화를 침범하지 않는다."""
    prev_end = max([s["end"] for s in segments if s["end"] <= start + 1e-3] or [0.0])
    next_start = min([s["start"] for s in segments if s["start"] >= end - 1e-3] or [end + post + 1])
    return max(prev_end + (start - prev_end) / 2 if start - prev_end < 2 * pre else start - pre, 0.0), \
        min(end + post, end + (next_start - end) / 2 if next_start - end < 2 * post else end + post)


def heuristic_select(segments, count):
    us = units(segments)
    cands = []
    for i in range(len(us)):
        for j in range(i, len(us)):
            d = us[j]["end"] - us[i]["start"]
            if d > MAX_D:
                break
            if d < MIN_D:
                continue
            cands.append((_score(us, i, j), i, j))
    cands.sort(reverse=True)
    chosen = []
    for sc, i, j in cands:
        if sc < PASS_SCORE or len(chosen) >= count:
            continue
        span = (us[i]["start"], us[j]["end"])
        text = " ".join(u["text"] for u in us[i:j + 1])
        if any(_overlap(span, c["span"]) > 0.2 or _jacc(text, c["text"]) > 0.55 for c in chosen):
            continue
        chosen.append({"span": span, "text": text, "score": sc, "i": i, "j": j})
    clips = []
    for k, c in enumerate(chosen, 1):
        st, en = pad_bounds(segments, *c["span"])
        first = re.split(r"(?<=[.!?。…])\s+", us[c["i"]]["text"].strip())[0]
        title = first if len(first) <= 30 else first[:29].rstrip() + "…"
        clips.append({"rank": k, "start": round(st, 2), "end": round(en, 2), "first_line": first,
                      "title": title, "reason": _reason(us, c["i"], c["j"], c["score"]),
                      "score": round(c["score"], 2)})
    return clips


# ---------- 선택적 LLM ----------
def llm_enabled():
    return os.environ.get("SHORTS_LLM", "").lower() == "anthropic" and bool(os.environ.get("ANTHROPIC_API_KEY"))


def _call_claude(prompt):
    body = json.dumps({"model": os.environ.get("SHORTS_LLM_MODEL", "claude-sonnet-5-5"), "max_tokens": 4000,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body, {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
        "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["content"][0]["text"]


def _json_from(text):
    m = re.search(r"\[.*\]|\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def llm_select(segments, count, window=600, overlap=60):
    """겹치는 구간으로 나눠 후보를 받고, 합쳐서 순위를 매긴다."""
    end_t = segments[-1]["end"]
    pool, t = [], 0.0
    while t < end_t:
        part = [s for s in segments if s["start"] >= t and s["end"] <= t + window]
        if part:
            lines = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in part)
            prompt = (f"아래는 영상 전사본의 일부입니다. 독립적으로 볼 만한 숏츠 후보(25~60초, 최대 15~75초)를 "
                      f"최대 {count}개 골라 JSON 배열로만 답하세요. 각 항목: start, end(전사 타임스탬프의 발화 경계), "
                      f"title(한국어 훅 제목), reason(한 문장), score(0~10). 인사·협찬·반복은 제외, 한 클립은 한 주장.\n\n{lines}")
            data = _json_from(_call_claude(prompt)) or []
            pool += [d for d in data if isinstance(d, dict) and "start" in d and "end" in d]
        t += window - overlap
    pool.sort(key=lambda d: -float(d.get("score", 0)))
    chosen = []
    for d in pool:
        span = (float(d["start"]), float(d["end"]))
        if not MIN_D <= span[1] - span[0] <= MAX_D + 5 or any(_overlap(span, c["span"]) > 0.2 for c in chosen):
            continue
        chosen.append({"span": span, **d})
        if len(chosen) >= count:
            break
    return [{"rank": k, "start": c["span"][0], "end": c["span"][1], "title": c.get("title", ""),
             "reason": c.get("reason", "LLM 선정"), "score": c.get("score")} for k, c in enumerate(chosen, 1)]


def snap(segments, start, end):
    """시작·종료를 가장 가까운 발화 경계로 맞춘다(문장·단어 중간 절단 방지)."""
    starts = [s["start"] for s in segments]
    ends = [s["end"] for s in segments]
    st = min(starts, key=lambda x: abs(x - start))
    en = min(ends, key=lambda x: abs(x - end))
    if abs(st - start) > 1.0:
        st = start
    if abs(en - end) > 1.0:
        en = end
    return pad_bounds(segments, st, en) if (st, en) != (start, end) else (start, end)


def select(segments, count, cache, edit_list=None):
    """반환 (clips, 방식 기록). 우선순위: edit_list.json > LLM > 휴리스틱."""
    f = Path(edit_list) if edit_list else Path(cache) / "edit_list.json"
    if f.exists():
        data = json.loads(f.read_text(encoding="utf-8"))
        clips = []
        for k, c in enumerate(data["clips"][:count], 1):
            st, en = snap(segments, float(c["start"]), float(c["end"]))
            clips.append({**c, "rank": k, "start": round(st, 2), "end": round(en, 2)})
        return clips, {"method": data.get("method", "edit-list (편집자 확정)"), "source": str(f)}
    if llm_enabled():
        try:
            clips = llm_select(segments, count)
            if clips:
                return clips, {"method": "llm-anthropic", "model": os.environ.get("SHORTS_LLM_MODEL", "claude-sonnet-5-5")}
        except Exception as e:  # 네트워크·파싱 실패 시 휴리스틱으로 대체
            clips = heuristic_select(segments, count)
            return clips, {"method": "heuristic", "note": f"LLM 선정 실패({type(e).__name__}) 후 대체"}
    return heuristic_select(segments, count), {"method": "heuristic (API 키·외부 서비스 미사용)"}
