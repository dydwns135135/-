"""전사: 기존 자막 파싱 또는 로컬 Whisper(mlx-whisper / faster-whisper)."""
import json
import platform
import re
from pathlib import Path

from .util import ShortsError, parse_time, read_srt


def parse_vtt(path):
    txt = Path(path).read_text(encoding="utf-8-sig")
    cues, prev_lines = [], []
    for block in re.split(r"\n\s*\n", txt):
        rows = [r for r in block.strip().splitlines() if r.strip()]
        for k, row in enumerate(rows):
            if "-->" in row:
                a, b = row.split("-->")
                lines = [re.sub(r"<[^>]+>", "", x).strip() for x in rows[k + 1:]]
                lines = [x for x in lines if x]
                new = [x for x in lines if x not in prev_lines]  # 자동자막의 롤링 중복 제거
                prev_lines = lines or prev_lines
                if new:
                    cues.append({"start": parse_time(a), "end": parse_time(b.split()[0]),
                                 "text": " ".join(new)})
                break
    return cues


def from_subtitle_file(path):
    cues = parse_vtt(path) if str(path).endswith(".vtt") else read_srt(path)
    return [{"start": c["start"], "end": c["end"], "text": " ".join(c["text"].split())} for c in cues
            if c["text"].strip()]


def _whisper(media, language, model):
    mlx_ok = platform.system() == "Darwin" and platform.machine() == "arm64"
    if mlx_ok:
        try:
            import mlx_whisper
            repo = model if "/" in model else "mlx-community/whisper-large-v3-turbo"
            res = mlx_whisper.transcribe(str(media), path_or_hf_repo=repo, word_timestamps=True,
                                         language=language)
            segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip(),
                     "words": [{"start": w["start"], "end": w["end"], "word": w["word"]}
                               for w in s.get("words", [])]} for s in res["segments"]]
            return res.get("language", language), segs, "mlx-whisper"
        except ImportError:
            pass
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ShortsError("음성 인식 도구가 없습니다. Apple Silicon 맥: pip install mlx-whisper / "
                          "그 외: pip install faster-whisper")
    name = model if "/" not in model else "small"
    m = WhisperModel(name, compute_type="int8")
    it, info = m.transcribe(str(media), language=language, word_timestamps=True, vad_filter=True)
    segs = [{"start": s.start, "end": s.end, "text": s.text.strip(),
             "words": [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])]}
            for s in it]
    return info.language, segs, "faster-whisper"


def get_transcript(src, cache, language=None, model="small"):
    """캐시된 전사가 있으면 재사용. 반환 dict(language, source, segments)."""
    cache = Path(cache)
    f = cache / "transcript.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    subs = src["subs"]
    manual = [s for s in subs if not s["auto"] and (not language or (s["lang"] or "").startswith(language) or not s["lang"])]
    if manual:
        s = manual[0]
        segs = from_subtitle_file(s["path"])
        lang = s["lang"] or language or src.get("language") or "und"
        tr = {"language": lang.split("-")[0], "source": f"subtitle-file:{Path(s['path']).name}", "segments": segs}
    else:
        try:
            lang, segs, engine = _whisper(src["media"], language, model)
            tr = {"language": lang, "source": engine, "segments": segs}
        except ShortsError:
            auto = [s for s in subs if s["auto"]]
            if not auto:
                raise
            s = auto[0]  # 음성 인식이 불가할 때만 자동 자막(타이밍 불안정 가능)
            tr = {"language": (s["lang"] or "und").split("-")[0],
                  "source": f"auto-caption:{Path(s['path']).name} (타이밍 불안정 가능)",
                  "segments": from_subtitle_file(s["path"])}
    tr["segments"] = [s for s in tr["segments"] if s["text"].strip() and s["end"] > s["start"]]
    f.write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
    return tr
