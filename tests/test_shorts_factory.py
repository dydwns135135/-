"""shorts_factory 단위 테스트 (ffmpeg 필요한 E2E는 ffmpeg 없으면 건너뜀)."""
import json
import shutil
import subprocess

import pytest

from shorts_factory import render, selector, translate
from shorts_factory.cli import parse_args
from shorts_factory.source import is_placeholder
from shorts_factory.util import read_srt, write_srt


def test_placeholder_urls():
    assert is_placeholder("[여기에 유튜브 링크 입력]")
    assert is_placeholder("")
    assert not is_placeholder("https://www.youtube.com/watch?v=abc")


def test_count_range():
    with pytest.raises(SystemExit):
        parse_args(["https://x.y/z", "--count", "9"])
    with pytest.raises(SystemExit):
        parse_args(["https://x.y/z", "--count", "21"])
    assert parse_args(["https://x.y/z"]).count == 15


def _segs():
    return [{"start": i * 7.0, "end": i * 7.0 + 6.5, "text": t} for i, t in enumerate(
        ["왜 손절이 어려울까요?", "사실 규칙이 없기 때문입니다.", "예를 들어 2% 손실선을 정합니다.",
         "그러면 감정이 줄어듭니다.", "결국 규칙이 핵심입니다."])]


def test_cues_within_clip_and_start_at_zero():
    cues = render.build_cues(_segs(), 7.0, 30.0, use_words=True)
    assert cues[0]["start"] >= 0 and all(c["end"] <= 23.0 + 1e-6 for c in cues)
    assert all(b["start"] >= a["end"] for a, b in zip(cues, cues[1:]))
    assert all(len(c["text"].split("\n")) <= 2 for c in cues)


def test_srt_roundtrip(tmp_path):
    p = tmp_path / "a.srt"
    write_srt(p, [{"start": 0.5, "end": 2.0, "text": "안녕\n하세요"}])
    assert read_srt(p)[0]["text"] == "안녕\n하세요"


def test_selector_no_padding_fill():
    clips, info = selector.select(_segs(), 10, "/nonexistent")
    assert len(clips) < 10 and info["method"].startswith("heuristic")


def test_translation_fallback_records_method(tmp_path):
    segs = [{"start": 0, "end": 5, "text": "Hello there."}]
    info = translate.apply(segs, [{"start": 0, "end": 5}], "en", "ko", tmp_path)
    assert info["method"] == "none-original-fallback" and segs[0]["sub"] == "Hello there."
    (tmp_path / "translations.json").write_text(json.dumps({"0": "안녕하세요."}, ensure_ascii=False))
    info = translate.apply(segs, [{"start": 0, "end": 5}], "en", "ko", tmp_path)
    assert segs[0]["sub"] == "안녕하세요." and "claude-code" in info["method"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg 없음")
def test_end_to_end_tiny(tmp_path):
    mp4 = tmp_path / "t.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24", "-f", "lavfi",
                    "-i", "sine=frequency=300", "-t", "40", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                    "-shortest", str(mp4)], check=True)
    write_srt(tmp_path / "t.srt", [{"start": 1 + i * 6.5, "end": 7 + i * 6.5, "text": t} for i, t in enumerate(
        ["왜 손절이 어려울까요?", "사실 규칙이 없기 때문입니다.", "예를 들어 2% 손실선을 정합니다.",
         "그러면 감정이 줄어듭니다.", "결국 규칙이 핵심입니다."])])
    from shorts_factory.cli import main
    out = tmp_path / "out"
    assert main([ "--input-file", str(mp4), "--language", "ko", "--output-dir", str(out), "--count", "10",
                 "--preset", "ultrafast"]) == 0
    man = json.loads(next(out.glob("*/manifest.json")).read_text(encoding="utf-8"))
    assert man["clips"] and all(c["status"] == "done" for c in man["clips"])
