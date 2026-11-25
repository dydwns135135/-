"""자막 큐 생성, ASS 번인, 9:16 세로 편집, -14 LUFS 정규화."""
import json
import re
import shutil
from pathlib import Path

from .util import ShortsError, cjk_ratio, run, video_info

W, H = 1080, 1920
FONT_CANDIDATES = ["NanumSquare Bold", "NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR",
                   "Apple SD Gothic Neo", "Malgun Gothic", "AppleGothic", "WenQuanYi Zen Hei"]


def pick_font():
    r = run(["fc-list", ":lang=ko", "family"], check=False)
    have = r.stdout if r.returncode == 0 else ""
    for f in FONT_CANDIDATES:
        if f.lower() in have.lower():
            return f
    raise ShortsError("한글 폰트가 없습니다. 설치: macOS는 기본 포함(Apple SD Gothic Neo), "
                      "Ubuntu: sudo apt install fonts-nanum fonts-noto-cjk, Windows는 맑은 고딕 사용")


# ---------- 자막 큐 ----------
def line_len(text):
    return 14 if cjk_ratio(text) > 0.3 else 26


def wrap2(text, n):
    """최대 2줄로 균형 있게 줄바꿈. 반환 (줄 리스트)"""
    text = " ".join(text.split())
    if len(text) <= n:
        return [text]
    words = text.split(" ")
    best, bi = None, None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        cost = max(len(a), len(b)) + abs(len(a) - len(b)) * 0.3
        if bi is None or cost < best:
            best, bi = cost, k
    if bi is None:  # 공백 없는 긴 문자열
        return [text[:n], text[n:2 * n]]
    return [" ".join(words[:bi]), " ".join(words[bi:])]


def chunk_text(text):
    """문장을 2줄 이내 의미 단위(쉼표·접속 지점 우선)로 나눈다."""
    n = line_len(text)
    limit = 2 * n
    text = " ".join(text.split())
    if len(text) <= limit:
        return [text]
    parts = re.split(r"(?<=[,.!?;:、。…])\s+", text)
    chunks, cur = [], ""
    for p in parts:
        while len(p) > limit:  # 구두점이 없는 긴 조각은 공백 기준으로 자름
            cut = p.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        if cur and len(cur) + 1 + len(p) > limit:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        chunks.append(cur)
    return chunks


def build_cues(segments, start, end, use_words):
    """클립 시작 기준(0초) 큐 목록. 번역 자막은 원본 문장 경계 타이밍을 따른다."""
    cues = []
    for s in segments:
        if s["end"] <= start or s["start"] >= end:
            continue
        text = s.get("sub", s["text"]).strip()
        if not text:
            continue
        a, b = max(s["start"], start), min(s["end"], end)
        chunks = chunk_text(text)
        words = s.get("words") or []
        if use_words and text == s["text"] and words and len(chunks) > 1:
            # 단어 타임스탬프로 청크 경계를 맞춘다
            wi, total = 0, len(chunks)
            for ci, ch in enumerate(chunks):
                n = len(ch.split())
                seg = words[wi:wi + n] if ci < total - 1 else words[wi:]
                wi += n
                if not seg:
                    continue
                ca, cb = max(seg[0]["start"], a), min(seg[-1]["end"], b)
                if cb > ca:
                    cues.append({"start": ca - start, "end": cb - start, "text": "\n".join(wrap2(ch, line_len(ch)))})
        else:
            lens = [max(1, len(c)) for c in chunks]
            tot, t = sum(lens), a
            for ch, ln in zip(chunks, lens):
                dur = (b - a) * ln / tot
                cues.append({"start": t - start, "end": t + dur - start, "text": "\n".join(wrap2(ch, line_len(ch)))})
                t += dur
    # 다듬기: 순서·비중첩·범위 보장, 너무 짧은 큐는 다음 큐 시작 전까지 연장
    cues.sort(key=lambda c: c["start"])
    dur = end - start
    for i, c in enumerate(cues):
        c["start"] = max(0.0, c["start"])
        nxt = cues[i + 1]["start"] if i + 1 < len(cues) else dur
        if c["end"] - c["start"] < 0.7:
            c["end"] = min(c["start"] + 0.7, nxt - 0.02, dur)
        c["end"] = min(c["end"], nxt - 0.02 if i + 1 < len(cues) else dur, dur)
    return [c for c in cues if c["end"] - c["start"] > 0.1]


def _ass_time(t):
    cs = int(round(t * 100))
    h, r = divmod(cs, 360000)
    m, r = divmod(r, 6000)
    s, c = divmod(r, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def write_ass(path, cues, hook, font, dur):
    esc = lambda t: t.replace("{", "(").replace("}", ")").replace("\n", "\\N")
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Sub,{font},62,&H00FFFFFF,&H00FFFFFF,&H00101010,&H80000000,1,0,0,0,100,100,0,0,1,6,1,2,70,160,420,1
Style: Hook,{font},74,&H0000E0FF,&H0000E0FF,&H00101010,&H80000000,1,0,0,0,100,100,0,0,1,7,2,8,70,70,170,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    ev = []
    if hook:
        ev.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(dur)},Hook,,0,0,0,,{esc(hook)}")
    for c in cues:
        ev.append(f"Dialogue: 0,{_ass_time(c['start'])},{_ass_time(c['end'])},Sub,,0,0,0,,{esc(c['text'])}")
    Path(path).write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


def hook_text(title):
    title = " ".join(title.split())
    lines = wrap2(title, 12 if cjk_ratio(title) > 0.3 else 22)
    if len(lines) == 2 and len(title) > 2 * (12 if cjk_ratio(title) > 0.3 else 22):
        lines[1] = lines[1][:-1] + "…"
    return "\n".join(lines[:2])


# ---------- 레이아웃 ----------
def detect_face_x(media, start, end, samples=8):
    """한 사람이 화면 중심이면 얼굴 중심 x(0~1)를, 아니면 None. OpenCV가 없으면 None(=블러 배경 맞춤)."""
    try:
        import cv2
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:  # OpenCV 없음/손상 -> 블러 배경 맞춤으로 대체
        return None
    cap = cv2.VideoCapture(str(media))
    xs = []
    for k in range(samples):
        t = start + (end - start) * (k + 0.5) / samples
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = casc.detectMultiScale(g, 1.1, 6, minSize=(int(fr.shape[0] * 0.08),) * 2)
        if len(faces) == 1 or (len(faces) > 1 and sorted(f[2] * f[3] for f in faces)[-2] < 0.4 * max(f[2] * f[3] for f in faces)):
            f = max(faces, key=lambda f: f[2] * f[3])
            xs.append((f[0] + f[2] / 2) / fr.shape[1])
        else:
            xs.append(None)
    cap.release()
    good = [x for x in xs if x is not None]
    if len(xs) < 3 or len(good) < 0.6 * len(xs):
        return None
    good.sort()
    if good[-1] - good[0] > 0.15:  # 위치가 많이 움직이면 불안정 -> 고정 구도(블러 맞춤)
        return None
    return good[len(good) // 2]


def build_filter(info, layout, face_x, crop_bottom_pct):
    pre = ""
    iw, ih = info["width"], info["height"]
    if crop_bottom_pct:
        ih2 = int(ih * (1 - crop_bottom_pct / 100) // 2 * 2)
        pre = f"crop={iw}:{ih2}:0:0,"
        ih = ih2
    if layout == "crop" or (layout == "auto" and face_x is not None):
        sw = int(round(iw * H / ih / 2) * 2)
        fx = 0.5 if face_x is None else face_x
        x = int(min(max(fx * sw - W / 2, 0), max(sw - W, 0)))
        return f"[0:v]{pre}scale={sw}:{H},crop={W}:{H}:{x}:0,setsar=1[base]", "crop"
    fg = f"scale={W}:{H}:force_original_aspect_ratio=decrease,setsar=1"
    return (f"[0:v]{pre}split=2[a][b];"
            f"[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,gblur=sigma=8,scale={W}:{H},eq=brightness=-0.12[bg];"
            f"[b]{fg}[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]"), "fit"


def measure_loudness(media, start, dur):
    r = run(["ffmpeg", "-hide_banner", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", media, "-vn",
             "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"], check=False)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if not m:
        return None
    d = json.loads(m.group(0))
    if d["input_i"] in ("-inf", "inf"):
        return None
    return d


def render_clip(media, info, clip, cues, out_mp4, work, font, layout="auto", crop_bottom_pct=0,
                preset="medium", crf=20):
    """클립 한 편 렌더링. 반환: 사용된 레이아웃."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    start, end = clip["start"], clip["end"]
    dur = end - start
    write_ass(work / "sub.ass", cues, hook_text(clip["title"]), font, dur)
    face_x = detect_face_x(media, start, end) if layout in ("auto", "crop") else None
    vf, used = build_filter(info, layout, face_x, crop_bottom_pct)
    af = "loudnorm=I=-14:TP=-1.5:LRA=11"
    m = measure_loudness(media, start, dur) if info["has_audio"] else None
    if m:
        af += (f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
               f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    cmd = ["ffmpeg", "-hide_banner", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", media,
           "-filter_complex", f"{vf};[base]ass=sub.ass[v]", "-map", "[v]"]
    if info["has_audio"]:
        cmd += ["-map", "0:a:0", "-af", af, "-ar", "48000", "-c:a", "aac", "-b:a", "160k"]
    fps = info["fps"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-r", f"{fps:.3f}" if fps else "30", "-movflags", "+faststart", "-t", f"{dur:.3f}",
            str(Path(out_mp4).resolve())]
    run(cmd, cwd=work)
    shutil.rmtree(work, ignore_errors=True)
    return used
