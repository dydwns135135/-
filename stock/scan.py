"""시장 훑기(한국): 네이버 금융 '테마' 목록에서 최근 움직이는 테마를 고르고 그 구성 종목을 자동으로 모은다.

미리 정해 둔 종목 목록 밖에서도 후보가 나오게 하려는 것이다. 페이지 구조가 바뀌어도 견디도록 링크(종목코드·테마번호)와
퍼센트 숫자 위치만 읽는다. 읽지 못하면 응답 일부를 오류에 담는다.
"""
import re
import time

import requests

HEAD = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
BASE = "https://finance.naver.com"
LIST_URL = BASE + "/sise/theme.naver"
DETAIL_URL = BASE + "/sise/sise_group_detail.naver"
PCT = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")
CODE = re.compile(r"/item/main\.naver\?code=(\d{6})")
THEME_NO = re.compile(r"[?&]no=(\d+)")


def _get(url, **params):
    r = requests.get(url, params=params, headers=HEAD, timeout=20)
    r.raise_for_status()
    return decode(r.content)


def decode(content):
    """네이버 페이지는 UTF-8(새 페이지)과 EUC-KR(옛 페이지)이 섞여 있어 UTF-8 을 먼저 시도한다."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("euc-kr", errors="replace")


def _snippet(html, n=120):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))[:n]


def parse_theme_list(html):
    """테마 목록 → [{no, name, d1(전일대비 %), d3(최근 3일 %), leaders[(code, name)]}]."""
    import lxml.html

    out, seen = [], set()
    for tr in lxml.html.fromstring(html).iter("tr"):
        link = next((a for a in tr.iter("a") if "sise_group_detail" in a.get("href", "") and "type=theme" in a.get("href", "")), None)
        if link is None:
            continue
        m = THEME_NO.search(link.get("href", ""))
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        pcts = []
        for td in tr.findall("td"):
            pm = PCT.search(td.text_content())
            if pm:
                pcts.append(float(pm.group(1)))
        leaders = []
        for a in tr.iter("a"):
            cm = CODE.search(a.get("href", ""))
            if cm:
                leaders.append((cm.group(1), a.text_content().strip()))
        out.append({"no": m.group(1), "name": link.text_content().strip(), "d1": pcts[0] if pcts else None,
                    "d3": pcts[1] if len(pcts) > 1 else None, "leaders": leaders})
    return out


def parse_theme_members(html):
    """테마 상세 → 페이지 순서대로 [(종목코드, 종목명)] (중복 제거)."""
    import lxml.html

    out, seen = [], set()
    for a in lxml.html.fromstring(html).iter("a"):
        m = CODE.search(a.get("href", ""))
        name = a.text_content().strip()
        if m and name and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append((m.group(1), name))
    return out


def pick_themes(themes, n3=40, n1=20):
    """최근 3일 등락률 상위 n3개 + 전일 등락률 상위 n1개(합집합). 등락률을 못 읽은 테마는 제외."""
    by3 = sorted([t for t in themes if t["d3"] is not None], key=lambda t: t["d3"], reverse=True)[:n3]
    by1 = sorted([t for t in themes if t["d1"] is not None], key=lambda t: t["d1"], reverse=True)[:n1]
    seen, out = set(), []
    for t in by3 + by1:
        if t["no"] not in seen:
            seen.add(t["no"])
            out.append(t)
    return out


def build_scan(n3=40, n1=20, per_theme=10, max_pages=12, delay=0.3, get=_get, log=print):
    """(테마→{종목코드: 종목명}, 통계). get 은 테스트에서 가짜로 바꿀 수 있다."""
    themes, errs = [], []
    for page in range(1, max_pages + 1):
        try:
            html = get(LIST_URL, page=page)
        except Exception as e:
            errs.append(f"p{page} {type(e).__name__}")
            break
        rows = parse_theme_list(html)
        if not rows:
            if page == 1:
                raise RuntimeError(f"테마 목록을 읽지 못함 — 응답 앞부분: '{_snippet(html)}'")
            break
        new = [r for r in rows if r["no"] not in {t["no"] for t in themes}]
        if not new:
            break
        themes += new
        time.sleep(delay)
    picked = pick_themes(themes, n3, n1)
    if not picked:
        raise RuntimeError(f"등락률을 읽은 테마가 없음(전체 {len(themes)}개)")
    themes_map, empty = {}, 0
    for t in picked:
        try:
            members = parse_theme_members(get(DETAIL_URL, type="theme", no=t["no"]))
        except Exception as e:
            log(f"테마 상세 실패 {t['name']}: {type(e).__name__}")
            members = []
        merged = {}
        for code, name in t["leaders"] + members:  # 목록의 주도주를 앞에 두고, 종목코드 기준으로 중복 제거(표기가 달라도)
            merged.setdefault(code, name)
        merged = dict(list(merged.items())[:per_theme])
        if len(merged) >= 3:
            themes_map[t["name"]] = merged
        else:
            empty += 1
        time.sleep(delay)
    stats = {"themes_total": len(themes), "themes_picked": len(picked), "themes_used": len(themes_map),
             "themes_thin": empty, "stocks": len({c for m in themes_map.values() for c in m})}
    if not themes_map:
        raise RuntimeError(f"구성 종목을 읽은 테마가 없음(선택 {len(picked)}개, 구성 3종목 미만 {empty}개)")
    return themes_map, stats
