"""공용 유틸: 서브프로세스, ffprobe, 시간·SRT 처리."""
import hashlib
import json
import re
import subprocess
from pathlib import Path


class ShortsError(RuntimeError):
    pass


def run(cmd, cwd=None, check=True):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise ShortsError(f"명령 실패({r.returncode}): {' '.join(map(str, cmd))[:300]}\n{r.stderr[-1500:]}")
    return r


def ffprobe(path):
    r = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path])
    return json.loads(r.stdout)


def video_info(path):
    d = ffprobe(path)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise ShortsError(f"영상 스트림이 없습니다: {path}")
    num, _, den = v.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {
        "width": int(v["width"]), "height": int(v["height"]), "fps": fps,
        "vcodec": v["codec_name"], "acodec": a["codec_name"] if a else None,
        "duration": float(d["format"].get("duration", 0)),
        "has_audio": a is not None,
    }


def sha(*parts, n=12):
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode())
    return h.hexdigest()[:n]


def slugify(s, maxlen=60):
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_")
    return s[:maxlen] or "video"


def fmt_tc(t, ms=False):
    t = max(0.0, float(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    base = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    return base + (f",{int(round((s - int(s)) * 1000)) % 1000:03d}" if ms else "")


def srt_time(t):
    t = max(0.0, t)
    total = int(round(t * 1000))
    h, r = divmod(total, 3600000)
    m, r = divmod(r, 60000)
    s, ms = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_time(s):
    s = s.strip().replace(",", ".")
    parts = s.split(":")
    sec = 0.0
    for p in parts:
        sec = sec * 60 + float(p)
    return sec


def write_srt(path, cues):
    lines = []
    for i, c in enumerate(cues, 1):
        lines += [str(i), f"{srt_time(c['start'])} --> {srt_time(c['end'])}", c["text"], ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def read_srt(path):
    txt = Path(path).read_text(encoding="utf-8-sig")
    cues = []
    for block in re.split(r"\n\s*\n", txt.strip()):
        rows = block.strip().splitlines()
        for k, row in enumerate(rows):
            if "-->" in row:
                a, b = row.split("-->")
                cues.append({"start": parse_time(a), "end": parse_time(b.split()[0]),
                             "text": "\n".join(rows[k + 1:])})
                break
    return cues


def is_cjk(ch):
    o = ord(ch)
    return (0xAC00 <= o <= 0xD7A3 or 0x3130 <= o <= 0x318F or 0x4E00 <= o <= 0x9FFF
            or 0x3040 <= o <= 0x30FF)


def cjk_ratio(text):
    chars = [c for c in text if not c.isspace()]
    return sum(is_cjk(c) for c in chars) / len(chars) if chars else 0.0
