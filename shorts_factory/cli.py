"""CLI 진입점: python -m shorts_factory "<YouTube URL>" """
import argparse
import json
import re
import sys
import time
from pathlib import Path

from . import render, selector, translate, transcript, verify
from .source import fetch_url, is_placeholder, local_file
from .util import ShortsError, fmt_tc, sha, video_info, write_srt


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="shorts_factory", description="유튜브 영상 -> 자막 입힌 9:16 숏츠 10~20편")
    p.add_argument("url", nargs="?", help="유튜브 URL (--input-file 과 택일)")
    p.add_argument("--count", type=int, default=15, help="요청 편수 10~20 (기본 15)")
    p.add_argument("--language", help="원본 언어 코드(예: en, ko). 기본: 자동 감지")
    p.add_argument("--subtitle-language", default="ko", help="화면 자막 언어. 기본 ko, 'original'이면 원본 언어 유지")
    p.add_argument("--output-dir", default="~/Desktop/shorts")
    p.add_argument("--dry-run", action="store_true", help="분석·편집 목록까지만, 렌더링 생략")
    p.add_argument("--resume", action="store_true", help="검증을 마친 클립은 재사용하고 실패·미완료만 처리")
    p.add_argument("--input-file", help="URL 대신 사용자가 제공한 로컬 원본 파일")
    p.add_argument("--edit-list", help="편집자가 확정한 edit_list.json 경로(기본: 캐시의 edit_list.json)")
    p.add_argument("--layout", choices=["auto", "crop", "fit"], default="auto")
    p.add_argument("--crop-bottom-pct", type=float, default=0, help="원본 하단 박힌 자막 제거용 잘라낼 비율(퍼센트)")
    p.add_argument("--credit", help="결과물·captions.md·summary.md에 넣을 크레딧/게시 조건 문구")
    p.add_argument("--whisper-model", default="small")
    p.add_argument("--preset", default="medium", help="x264 preset")
    a = p.parse_args(argv)
    if not 10 <= a.count <= 20:
        p.error("--count 는 10~20 사이여야 합니다")
    if bool(a.url) == bool(a.input_file):
        p.error("URL 또는 --input-file 중 하나만 지정하세요")
    return a


def hashtags_for(clip, lang):
    words = [w for w in re.findall(r"[\w가-힣]{2,}", clip.get("title", "")) if not w.isdigit()][:3]
    return ["#shorts", "#쇼츠", "#릴스"] + [f"#{w}" for w in words]


def write_reports(root, man):
    clips = man["clips"]
    done = [c for c in clips if c["status"] == "done"]
    L = [f"# 숏츠 제작 요약 — {man['source']['title']}", "",
         f"- 원본: {man['source']['url'] or man['source']['file']}",
         f"- 요청 {man['requested']}편 / 선정 {len(clips)}편 / 완성 {len(done)}편",
         f"- 후보 선정 방식: {man['selection']['method']}",
         f"- 전사: {man['transcript']['source']} (언어 {man['transcript']['language']})",
         f"- 자막 번역: {man['translation']['method']}" + (f" — {man['translation']['note']}" if man['translation'].get('note') else "")]
    if man.get("credit"):
        L.append(f"- 크레딧/게시 조건: {man['credit']}")
    if len(clips) < 10:
        L += ["", f"> 10편 미만({len(clips)}편): 기준(25~60초 독립 완결, 첫 문장 흡인력, 중복·협찬 제외)을 통과한 후보가 "
              "부족했습니다. 균등 분할로 편수를 채우지 않았습니다."]
    L += ["", "| # | 제목 | 원본 타임코드 | 길이 | 상태 |", "|---|---|---|---|---|"]
    for c in clips:
        L.append(f"| {c['rank']} | {c['title_ko'] if 'title_ko' in c else c['title']} | {fmt_tc(c['start'])}–{fmt_tc(c['end'])} | "
                 f"{c['duration']:.0f}초 | {c['status']} |")
    fails = [c for c in clips if c["status"] == "failed"]
    if fails:
        L += ["", "## 실패"] + [f"- 클립 {c['rank']}: {c.get('error', '')[:200]}" for c in fails]
    top = [c for c in done if c.get("score") is not None][:3] or done[:3]
    if top:
        L += ["", "## 먼저 올려볼 만한 클립"] + [f"- 클립 {c['rank']} «{c['title']}»: {c['reason']}" for c in top]
    (root / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    C = ["# 게시 캡션", ""]
    for c in clips:
        src = man["source"]["url"] or man["source"]["file"]
        tc = f"{fmt_tc(c['start'])}–{fmt_tc(c['end'])}"
        C += [f"## 클립 {c['rank']} — {c['title']}", "", c.get("caption") or c["first_line"][:120], "",
              " ".join(c["hashtags"]), "", f"출처: {src} ({tc})" + (f" / {man['credit']}" if man.get("credit") else ""), ""]
    (root / "captions.md").write_text("\n".join(C), encoding="utf-8")


def main(argv=None):
    a = parse_args(argv)
    if a.url and is_placeholder(a.url):
        print("URL이 비어 있거나 자리표시자입니다. 영상은 처리하지 않습니다. 실제 URL과 원저작자 허락을 확인한 뒤 다시 실행하세요.",
              file=sys.stderr)
        return 2
    out = Path(a.output_dir).expanduser()
    try:
        src = fetch_url(a.url, out, a.language) if a.url else local_file(a.input_file, out)
        root = out / src["video_id"]
        cache = Path(src["cache"])
        for d in ("clips", "subtitles"):
            (root / d).mkdir(parents=True, exist_ok=True)
        info = video_info(src["media"])
        print(f"[1/4] 원본 확보: {src['title']} ({info['width']}x{info['height']}, {info['duration']:.0f}s)")
        tr = transcript.get_transcript(src, cache, a.language, a.whisper_model)
        segs = tr["segments"]
        if not segs:
            raise ShortsError("전사 결과가 비어 있습니다.")
        print(f"[2/4] 전사: {tr['source']} / {tr['language']} / {len(segs)}개 세그먼트")
        clips, sel = selector.select(segs, a.count, cache, a.edit_list)
        print(f"[3/4] 후보 선정: {sel['method']} → {len(clips)}편")
        tinfo = translate.apply(segs, clips, tr["language"], a.subtitle_language, cache)
        print(f"      자막 번역: {tinfo['method']}")
        font = render.pick_font() if not a.dry_run else "-"

        mf = root / "manifest.json"
        prev = {}
        if a.resume and mf.exists():
            prev = {c["rank"]: c for c in json.loads(mf.read_text(encoding="utf-8")).get("clips", [])}
        man = {"source": {"title": src["title"], "url": src["url"], "file": None if src["url"] else src["media"],
                          "uploader": src["uploader"], "duration": info["duration"]},
               "requested": a.count, "selection": sel, "transcript": {"source": tr["source"], "language": tr["language"]},
               "translation": tinfo, "credit": a.credit, "dry_run": a.dry_run, "clips": []}
        entries = []
        for c in clips:
            r = c["rank"]
            st, en = float(c["start"]), float(c["end"])
            fp = sha(src["video_id"], st, en, c.get("title"), a.layout, a.crop_bottom_pct, a.subtitle_language,
                     [s.get("sub") for s in segs if s["end"] > st and s["start"] < en])
            cues = render.build_cues(segs, st, en, use_words=True)
            name = f"clip_{r:02d}"
            e = {"rank": r, "title": c.get("title") or c["first_line"], "start": st, "end": en, "duration": round(en - st, 2),
                 "first_line": c.get("first_line") or next((s["text"] for s in segs if s["end"] > st), ""),
                 "reason": c.get("reason", ""), "score": c.get("score"), "file": f"clips/{name}.mp4",
                 "srt": f"subtitles/{name}.srt", "fingerprint": fp, "status": "planned",
                 "caption": c.get("caption"), "hashtags": c.get("hashtags") or hashtags_for(c, tr["language"])}
            if a.dry_run:
                entries.append(e)
                continue
            old = prev.get(r)
            if old and old.get("status") == "done" and old.get("fingerprint") == fp and (root / e["file"]).exists():
                entries.append({**e, **{k: old[k] for k in ("status", "layout", "checks") if k in old}})
                print(f"      clip {r:02d}: 재사용")
                continue
            write_srt(root / e["srt"], cues)
            t0 = time.time()
            try:
                e["layout"] = render.render_clip(src["media"], info, {"start": st, "end": en, "title": e["title"]}, cues,
                                                 root / e["file"], cache / "work" / name, font, a.layout,
                                                 a.crop_bottom_pct, a.preset)
                ok, checks = verify.verify_clip(str(root / e["file"]), root / e["srt"], en - st, segs, st,
                                                preview=root / "previews" / f"{name}.jpg")
                e["checks"] = checks
                e["status"] = "done" if ok else "failed"
                if not ok:
                    e["error"] = "검증 실패: " + ", ".join(k for k, v in checks.items() if not v["ok"])
            except Exception as ex:  # 한 편 실패가 전체를 막지 않는다
                e["status"], e["error"] = "failed", str(ex)[-600:]
            entries.append(e)
            man["clips"] = entries
            mf.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"      clip {r:02d}: {e['status']} ({time.time() - t0:.0f}s) {e.get('error', '')[:80]}")
        man["clips"] = entries
        mf.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        write_reports(root, man)
        done = sum(c["status"] == "done" for c in entries)
        print(f"[4/4] 완료: {done}/{len(entries)}편 → {root}")
        return 0 if a.dry_run or done == len(entries) else 1
    except ShortsError as ex:
        print(f"오류: {ex}", file=sys.stderr)
        return 1
