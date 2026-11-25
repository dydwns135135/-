"""원본 확보: yt-dlp(URL) 또는 로컬 파일."""
import json
import re
import shutil
import sys
from pathlib import Path

from .util import ShortsError, run, slugify, sha

PLACEHOLDER = re.compile(r"여기에|\[.*입력.*\]|<.*url.*>", re.I)
BLOCK_HINTS = ("sign in", "confirm you", "not a bot", "private video", "members-only",
               "drm", "login", "log in", "age-restricted", "requires authentication", "premium")


def is_placeholder(s):
    return (not s) or bool(PLACEHOLDER.search(s)) or not re.match(r"https?://", s.strip())


def _ytdlp():
    try:
        import yt_dlp  # noqa: F401
        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        exe = shutil.which("yt-dlp")
        if exe:
            return [exe]
    raise ShortsError("yt-dlp가 없습니다. 설치: pip install yt-dlp")


def _explain(stderr):
    low = stderr.lower()
    if any(h in low for h in BLOCK_HINTS):
        raise ShortsError("원본에 로그인/DRM/접근 제한이 있어 URL로 가져올 수 없습니다. "
                          "원본 파일을 직접 받아 `--input-file <파일>` 로 실행하세요.\n" + stderr[-400:])
    raise ShortsError("yt-dlp 실패:\n" + stderr[-800:])


def fetch_url(url, workdir, language=None):
    """반환: dict(video_id, title, media, subs[{path,lang,auto}], url, uploader)"""
    base = _ytdlp()
    r = run(base + ["--dump-single-json", "--no-playlist", "--no-warnings", url], check=False)
    if r.returncode != 0:
        _explain(r.stderr)
    meta = json.loads(r.stdout)
    vid = meta["id"]
    d = Path(workdir) / vid / ".cache"
    d.mkdir(parents=True, exist_ok=True)
    media = d / "source.mp4"
    if not media.exists():
        tmpl = str(d / "source.%(ext)s")
        sub_args = ["--write-subs", "--write-auto-subs", "--sub-format", "vtt",
                    "--sub-langs", (language + ".*,en.*" if language else "all,-live_chat")]
        r = run(base + ["--no-playlist", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
                        "--merge-output-format", "mp4", "-o", tmpl, *sub_args, url], check=False)
        if r.returncode != 0 and not media.exists():
            _explain(r.stderr)
    manual = set((meta.get("subtitles") or {}).keys())
    subs = []
    for p in sorted(d.glob("source.*.vtt")):
        lang = p.name[len("source."):-len(".vtt")]
        subs.append({"path": str(p), "lang": lang, "auto": lang.split("-")[0] not in
                     {m.split("-")[0] for m in manual}})
    return {"video_id": vid, "title": meta.get("title", vid), "media": str(media), "subs": subs,
            "url": url, "uploader": meta.get("uploader") or meta.get("channel"),
            "language": meta.get("language"), "cache": str(d)}


def local_file(path, workdir):
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ShortsError(f"파일을 찾을 수 없습니다: {p}")
    vid = f"{slugify(p.stem, 40)}-{sha(str(p), p.stat().st_size, n=6)}"
    d = Path(workdir) / vid / ".cache"
    d.mkdir(parents=True, exist_ok=True)
    subs = []
    for ext in (".srt", ".vtt"):
        for cand in (p.with_suffix(ext), *p.parent.glob(p.stem + ".*" + ext)):
            if cand.exists():
                lang = cand.suffixes[-2][1:] if len(cand.suffixes) >= 2 and cand != p.with_suffix(ext) else None
                subs.append({"path": str(cand), "lang": lang, "auto": False})
    return {"video_id": vid, "title": p.stem, "media": str(p), "subs": subs, "url": None,
            "uploader": None, "language": None, "cache": str(d)}
