#!/usr/bin/env python3
"""음식 쇼츠 자동 편집기 (ffmpeg + 파이썬 표준 라이브러리만 사용).

원본 영상(들)에서 움직임(뜨기/먹기)과 소리(씹는 소리)가 큰 구간을 골라
목표 길이(기본 60초)의 9:16 쇼츠를 만든다. 자막은 넣지 않는다.

출력 (out 폴더):
  final.mp4        전체를 이어 붙인 완성본
  clips/NN_*.mp4   컷별 개별 파일 -> 캡컷에 한꺼번에 불러와 순서대로 재편집 가능
  edit_list.csv    어느 원본의 몇 초 구간을 썼는지 기록

사용 예:
  python autoedit.py raw/ -o out
  python autoedit.py a.mp4 b.mp4 --target 60 --seg 3 --fit blur
"""
import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def run(cmd, capture=False):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        sys.exit(f"ffmpeg 오류:\n{' '.join(cmd)}\n{r.stderr[-1500:]}")
    return r


def probe_duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)])
    return float(r.stdout.strip())


def has_audio(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", str(path)])
    return bool(r.stdout.strip())


def motion_series(path, fps=5):
    """초당 fps개 프레임의 장면 변화량(0~1). 숟가락질/손동작이 크면 값이 커진다."""
    r = run(["ffmpeg", "-v", "error", "-i", str(path), "-an",
             "-vf", f"fps={fps},scale=160:-2,select='gte(scene,0)',metadata=print:file=-",
             "-f", "null", "-"])
    out = []
    t = None
    for line in r.stdout.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            t = float(m.group(1))
        m = re.search(r"lavfi.scene_score=([\d.]+)", line)
        if m and t is not None:
            out.append((t, float(m.group(1))))
    return out


def audio_series(path, step=0.1):
    """step초 단위 소리 크기(선형 RMS). 씹는 소리가 크면 값이 커진다."""
    n = int(16000 * step)
    r = run(["ffmpeg", "-v", "error", "-i", str(path), "-vn",
             "-af", f"aresample=16000,asetnsamples={n},astats=metadata=1:reset=1,"
                    "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
             "-f", "null", "-"])
    out = []
    t = None
    for line in r.stdout.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            t = float(m.group(1))
        m = re.search(r"RMS_level=(-?[\d.inf]+)", line)
        if m and t is not None:
            db = -90.0 if "inf" in m.group(1) else float(m.group(1))
            out.append((t, 10 ** (max(db, -90.0) / 20)))
    return out


def window_means(series, start, end):
    vals = [v for t, v in series if start <= t < end]
    return sum(vals) / len(vals) if vals else 0.0


def normalize(values):
    """상위 5% 값을 1로 맞춰 0~1로 정규화 (튀는 값 하나가 전체를 망치지 않게)."""
    if not values:
        return values
    top = sorted(values)[max(0, int(len(values) * 0.95) - 1)] or 1e-9
    return [min(v / top, 1.0) for v in values]


def score_windows(path, dur, seg, use_audio):
    motion = motion_series(path)
    audio = audio_series(path) if use_audio else []
    starts = [i * seg for i in range(int(dur // seg))]
    m = normalize([window_means(motion, s, s + seg) for s in starts])
    a = normalize([window_means(audio, s, s + seg) for s in starts]) if use_audio else [0] * len(starts)
    w_m, w_a = (0.5, 0.5) if use_audio else (1.0, 0.0)
    return [(s, w_m * mm + w_a * aa) for s, mm, aa in zip(starts, m, a)]


def pick(all_scores, files, seg, target, min_score):
    """영상별로 길이에 비례해 예산을 나누고, 구간 점수가 높은 순으로 고른다.
    인접한 구간은 피해서 장면이 다양하게 나오게 하고, 마지막엔 시간순으로 정렬한다."""
    total_dur = sum(d for _, d in files)
    n_total = max(1, round(target / seg))
    chosen = []
    for (path, dur), scores in zip(files, all_scores):
        quota = max(1, round(n_total * dur / total_dur))
        taken = []
        for s, sc in sorted(scores, key=lambda x: -x[1]):
            if len(taken) >= quota:
                break
            if sc < min_score:
                continue
            if any(abs(s - t) < seg * 1.5 for t in taken):  # 바로 옆 구간 제외
                continue
            taken.append(s)
        chosen += [(path, s) for s in sorted(taken)]
    return chosen


def vf_chain(fit, focus, w, h, fps):
    if fit == "blur":  # 가로 영상: 위아래를 흐린 배경으로 채우고 원본은 가운데 전체 표시
        return (f"split[a][b];[a]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},boxblur=30:5[bg];"
                f"[b]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,fps={fps},format=yuv420p")
    return (f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}:(iw-{w})*{focus}:(ih-{h})*0.5,fps={fps},format=yuv420p")


def render_clip(src, start, length, dst, fit, focus, w, h, fps, with_audio):
    graph = "[0:v]" + vf_chain(fit, focus, w, h, fps) + "[v]"
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
           "-i", str(src)]
    if not with_audio:  # 무음 영상이라도 이어붙이기가 되도록 무음 트랙을 넣는다
        cmd += ["-f", "lavfi", "-t", f"{length:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "0:a" if with_audio else "1:a"]
    if with_audio:
        cmd += ["-af", f"afade=t=in:d=0.03,afade=t=out:st={length - 0.03:.3f}:d=0.03"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-shortest", "-movflags", "+faststart", str(dst)]
    run(cmd)


def main():
    ap = argparse.ArgumentParser(description="음식 쇼츠 자동 편집기")
    ap.add_argument("inputs", nargs="+", help="영상 파일 또는 폴더")
    ap.add_argument("-o", "--out", default=None, help="출력 폴더 (기본: out_날짜시간, 이전 결과를 덮어쓰지 않음)")
    ap.add_argument("--target", type=float, default=60.0, help="목표 길이(초), 기본 60")
    ap.add_argument("--seg", type=float, default=3.0, help="컷 한 개 길이(초), 기본 3")
    ap.add_argument("--fit", choices=["crop", "blur"], default="crop",
                    help="crop: 가운데 확대 크롭 / blur: 흐린 배경 위에 전체 화면")
    ap.add_argument("--focus", type=float, default=0.5,
                    help="crop일 때 가로 위치 0(왼쪽)~1(오른쪽), 음식이 치우쳐 있으면 조절")
    ap.add_argument("--min-score", type=float, default=0.15, help="이보다 밋밋한 구간은 버림")
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="렌더 없이 고른 구간만 출력")
    args = ap.parse_args()

    paths = []
    for p in map(Path, args.inputs):
        paths += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT) if p.is_dir() else [p]
    if not paths:
        sys.exit("영상 파일이 없습니다.")

    files = [(p, probe_duration(p)) for p in paths]
    all_scores = []
    for p, d in files:
        print(f"분석 중: {p.name} ({d:.1f}초)")
        all_scores.append(score_windows(p, d, args.seg, has_audio(p)))

    chosen = pick(all_scores, files, args.seg, args.target, args.min_score)
    if not chosen:
        sys.exit("쓸 만한 구간을 못 찾았습니다. --min-score 를 낮춰보세요.")
    total = len(chosen) * args.seg
    print(f"선택된 컷 {len(chosen)}개, 총 {total:.1f}초 (목표 {args.target:.0f}초)")
    if total < args.target * 0.9:
        print("  ※ 원본이 짧거나 밋밋한 구간이 많아 목표보다 짧습니다. 원본을 더 넣거나 --min-score 를 낮추세요.")
    if args.dry_run:
        for p, s in chosen:
            print(f"  {p.name} {s:6.1f}s")
        return

    out = Path(args.out or time.strftime("out_%Y%m%d_%H%M%S"))
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rows, listing = [], []
    for i, (p, s) in enumerate(chosen, 1):
        dst = clips / f"{i:02d}_{p.stem}_{int(s)}s.mp4"
        render_clip(p, s, args.seg, dst, args.fit, args.focus,
                    args.width, args.height, args.fps, has_audio(p))
        rows.append([i, p.name, f"{s:.2f}", f"{s + args.seg:.2f}", dst.name])
        listing.append(f"file '{dst.resolve()}'")
        print(f"  컷 {i}/{len(chosen)} 완료")

    with open(out / "edit_list.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["순서", "원본", "시작(초)", "끝(초)", "컷 파일"])
        w.writerows(rows)

    concat = out / "concat.txt"
    concat.write_text("\n".join(listing) + "\n", encoding="utf-8")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
         "-c", "copy", "-movflags", "+faststart", str(out / "final.mp4")])
    concat.unlink()
    print(f"\n완료: {out / 'final.mp4'}  ({probe_duration(out / 'final.mp4'):.1f}초)")
    print(f"캡컷용 개별 컷: {clips}/")


if __name__ == "__main__":
    main()
