# shorts_factory — 유튜브 → 자막 입힌 세로 숏츠 10~20편

URL 하나(또는 로컬 파일)에서 독립적으로 볼 만한 구간을 골라 1080×1920 MP4(H.264+AAC, -14 LUFS)와 SRT를 만든다.
**원저작자 허락을 받은 영상에만 사용할 것.** 업로드/게시는 하지 않는다(파일 제작까지).

## 설치
```bash
brew install ffmpeg            # macOS (Ubuntu: sudo apt install ffmpeg fonts-nanum fonts-noto-cjk)
pip install -r requirements-shorts.txt
# Apple Silicon 맥 음성 인식: pip install mlx-whisper
```
한글 폰트가 없으면 실행 시 설치 방법을 안내한다(macOS는 Apple SD Gothic Neo 기본 포함).

## 실행
```bash
python -m shorts_factory "https://www.youtube.com/watch?v=XXXXXXXXXXX"              # 기본 15편
python -m shorts_factory "<URL>" --count 12 --subtitle-language original --language en
python -m shorts_factory --input-file ~/Movies/talk.mp4 --language ko               # 로컬 파일(talk.srt가 옆에 있으면 사용)
python -m shorts_factory "<URL>" --dry-run                                          # 분석·편집 목록까지만
python -m shorts_factory "<URL>" --credit "출처: 채널명, 게시는 승인 채널만"          # 크레딧/게시 조건 기록
```
옵션: `--count 10~20`, `--language`, `--subtitle-language ko|original`, `--output-dir`(기본 `~/Desktop/shorts`),
`--dry-run`, `--resume`, `--input-file`, `--edit-list`, `--layout auto|crop|fit`, `--crop-bottom-pct`(원본 박힌 자막 잘라내기), `--credit`.

## 결과물
```
~/Desktop/shorts/<video-id>/
  clips/clip_NN.mp4   subtitles/clip_NN.srt   previews/clip_NN.jpg
  captions.md  manifest.json  summary.md   .cache/(원본·전사·편집목록)
```
`manifest.json`에 클립별 제목·원본 시각·길이·첫 대사·선정 이유·렌더 상태·검증 결과, 그리고 후보 선정/전사/번역 방식이 기록된다.
검증: ffprobe(해상도·코덱·길이), ebur128 음량, SRT 범위·중첩·싱크, 자막 번인 여부.

## 후보 선정·번역 방식
- `.cache/edit_list.json`(편집자가 전사본을 읽고 확정) → 있으면 최우선.
- 환경변수 `SHORTS_LLM=anthropic` + `ANTHROPIC_API_KEY`를 **직접 설정한 경우에만** Claude API로 선정·번역(외부 전송).
- 그 외에는 API 키 없이 동작하는 휴리스틱(첫 문장 훅, 밀도, 문장 완결, 중복·협찬 제외). 기준 통과 후보가 적으면 적은 편수만 만든다.
- 외국어 → 한국어 자막: `.cache/translations.json` (`{"세그먼트 인덱스":"한국어"}`, 필요 목록은 `translation_todo.json`).
  번역 수단이 없으면 원본 언어 자막으로 만들고 `summary.md`/`manifest.json`에 기록한다. 타이밍은 원본 문장 경계 유지.

`edit_list.json` 예: `{"method":"claude-code","clips":[{"start":12.3,"end":55.1,"title":"한국어 훅 제목","reason":"한 문장","caption":"...","hashtags":["#쇼츠"]}]}`

## 오류 후 재개
같은 명령에 `--resume`을 붙이면 검증을 마친 클립(입력·설정 지문 동일)은 재사용하고 실패·미완료만 다시 렌더한다.
다운로드·전사·편집 목록은 `.cache`에서 재사용된다. 로그인/DRM 영상은 `--input-file`로 받아 둔 파일을 쓴다.

## 한계
- 얼굴 추적은 OpenCV(Haar)로 한 사람이 안정적일 때만 크롭하고, 그 외에는 원본을 살린 블러 배경 맞춤이다.
- 원본에 박힌 자막은 자동 감지하지 않는다. 프레임을 확인해 `--crop-bottom-pct`로 잘라낸다.
