"""ffprobe·loudness·SRT·자막 번인 검증."""
import re
from pathlib import Path

from .util import ffprobe, read_srt, run, video_info


def loudness(path):
    r = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"], check=False)
    tail = r.stderr.split("Summary:")[-1]
    m = re.search(r"I:\s+(-?[\d.]+)\s+LUFS", tail)
    return float(m.group(1)) if m else None


def _white_pixels(path, t, y0, y1):
    """프레임의 y0~y1 영역에서 거의 흰색(>=235)인 픽셀 수."""
    r = __import__("subprocess").run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
         "-vf", f"crop=1080:{y1 - y0}:0:{y0},format=gray", "-f", "rawvideo", "-"], capture_output=True)
    return sum(1 for b in r.stdout if b >= 235)


def verify_clip(mp4, srt, expected_dur, transcript_segments, clip_start, preview=None):
    checks, ok = {}, True

    def put(name, passed, detail):
        nonlocal ok
        checks[name] = {"ok": bool(passed), "detail": detail}
        ok &= bool(passed)

    d = ffprobe(mp4)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)
    put("resolution", v and (v["width"], v["height"]) == (1080, 1920), f"{v and v['width']}x{v and v['height']}")
    put("codec", v and v["codec_name"] == "h264" and a and a["codec_name"] == "aac",
        f"video={v and v['codec_name']} audio={a and a['codec_name']}")
    real = float(d["format"]["duration"])
    put("duration", abs(real - expected_dur) <= 0.35, f"실제 {real:.2f}s / 예상 {expected_dur:.2f}s")
    if a:
        lufs = loudness(mp4)
        put("loudness", lufs is not None and abs(lufs - (-14)) <= 2.0, f"{lufs} LUFS (목표 -14±2)")
    cues = read_srt(srt)
    put("srt_nonempty", len(cues) > 0, f"{len(cues)} cues")
    bad = [c for c in cues if c["start"] < 0 or c["end"] > real + 0.05 or c["end"] <= c["start"]]
    overlap = [1 for x, y in zip(cues, cues[1:]) if y["start"] < x["end"] - 0.001]
    put("srt_range", not bad and not overlap and (not cues or cues[0]["start"] < real), f"범위 밖 {len(bad)}, 중첩 {len(overlap)}; 클립 시작 기준")
    # 싱크: 각 큐의 원본 시각(클립 시작+큐 시각)이 전사 발화 구간 안에 있어야 함
    miss = 0
    for c in cues:
        t = clip_start + (c["start"] + c["end"]) / 2
        if not any(s["start"] - 0.5 <= t <= s["end"] + 0.5 for s in transcript_segments):
            miss += 1
    put("srt_sync", miss == 0, f"전사 발화와 어긋난 큐 {miss}/{len(cues)}")
    if cues:
        c = cues[len(cues) // 2]
        t = min((c["start"] + c["end"]) / 2, real - 0.1)
        n = _white_pixels(mp4, t, 1920 - 420 - 260, 1920 - 420 + 40)
        put("subtitle_burned_in", n > 300, f"자막 영역 흰색 픽셀 {n}")
        if preview:
            Path(preview).parent.mkdir(parents=True, exist_ok=True)
            run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", mp4, "-frames:v", "1", "-vf", "scale=540:-2", preview])
    return ok, checks
