"""시장 훑기(한국): 네이버 모바일 JSON 에서 오늘 움직이는 테마와 구성 종목을 모은다.

PC 페이지는 스크립트로 그려지는 새 사이트로 바뀌어 읽을 수 없고, m.stock.naver.com/api 는 JSON 으로 응답한다.
구성 종목 주소는 확실하지 않아 후보를 차례로 시도하고, 모두 실패하면 거래대금·상승률 상위 종목으로 대신 훑으며
시도한 주소와 상태 코드를 통계에 남긴다.
"""
import time

import requests

HEAD = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Referer": "https://m.stock.naver.com/"}
M = "https://m.stock.naver.com"
MEMBER_CANDIDATES = (("/api/stocks/theme/{no}", {}), ("/api/stocks/theme/{no}/stocks", {}), ("/api/stocks/theme", {"no": "{no}"}))


def decode(content):
    """UTF-8 을 먼저, 안 되면 EUC-KR 로 읽는다(옛 페이지 호환)."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("euc-kr", errors="replace")


def _jget(path, **params):
    r = requests.get(M + path, params=params, headers=HEAD, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP{r.status_code}")
    return r.json()


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("+", ""))
    except ValueError:
        return None


def _walk(j):
    if isinstance(j, dict):
        yield j
        for v in j.values():
            yield from _walk(v)
    elif isinstance(j, list):
        for v in j:
            yield from _walk(v)


def parse_groups(j):
    """테마/업종 목록 JSON → [{no, name, total, rate, rise}] (구성 종목 수·등락률을 읽은 것만)."""
    out, seen = [], set()
    for d in _walk(j):
        if "no" in d and "name" in d and "totalCount" in d and d["no"] not in seen:
            seen.add(d["no"])
            out.append({"no": d["no"], "name": str(d["name"]), "total": int(_num(d["totalCount"]) or 0),
                        "rate": _num(d.get("changeRate")), "rise": int(_num(d.get("riseCount")) or 0)})
    return out


def find_stocks(j):
    """JSON 어디에 있든 itemCode 가 든 보통주를 순서대로 [(코드, 이름)]. ETF·ETN 등은 제외(stockEndType 이 있으면 'stock'만)."""
    out, seen = [], set()
    for d in _walk(j):
        code = d.get("itemCode")
        if not code or code in seen or len(str(code)) != 6:
            continue
        if d.get("stockEndType", "stock") != "stock":
            continue
        seen.add(code)
        out.append((str(code), str(d.get("stockName") or d.get("name") or code)))
    return out


def pick_themes(groups, n=30, min_total=3, min_breadth=0.5):
    """구성 종목이 min_total 개 이상이고 상승 종목 비율이 min_breadth 이상인 테마 중 등락률 상위 n개."""
    ok = [g for g in groups if g["total"] >= min_total and g["rate"] is not None and g["rise"] / g["total"] >= min_breadth]
    return sorted(ok, key=lambda g: g["rate"], reverse=True)[:n]


def fetch_members(no, get, size, errors):
    """구성 종목 후보 주소를 차례로 시도한다. 성공한 주소는 errors 에 'OK' 로 남긴다."""
    for path, params in MEMBER_CANDIDATES:
        p = {k: v.format(no=no) for k, v in params.items()}
        key = path.format(no="{no}") + ("?no=" if p else "")
        try:
            stocks = find_stocks(get(path.format(no=no), page=1, pageSize=size, **p))
        except Exception as e:
            errors.setdefault(key, str(e)[:40])
            continue
        if stocks:
            errors[key] = "OK"
            return stocks
        errors.setdefault(key, "빈 응답")
    return []


def ranking_universe(get, n=50, cap=100):
    """안전장치: 거래대금 상위와 상승률 상위(코스피·코스닥)에서 보통주를 모은다."""
    out = {}
    for kind in ("priceTop", "up"):
        for cat in ("KOSPI", "KOSDAQ"):
            try:
                for code, name in find_stocks(get(f"/api/stocks/{kind}/{cat}", page=1, pageSize=n)):
                    out.setdefault(code, name)
            except Exception:
                pass
    return dict(list(out.items())[:cap])


def build_scan(n_themes=30, per_theme=10, max_pages=6, delay=0.2, get=_jget, log=print):
    """(테마→{종목코드: 종목명}, 통계). get(path, **params) 는 테스트에서 가짜로 바꿀 수 있다."""
    groups = []
    for page in range(1, max_pages + 1):
        try:
            got = parse_groups(get("/api/stocks/theme", page=page, pageSize=100))
        except Exception as e:
            if page == 1:
                raise RuntimeError(f"테마 목록을 읽지 못함 — {type(e).__name__}: {str(e)[:80]}")
            break
        new = [g for g in got if g["no"] not in {x["no"] for x in groups}]
        if not new:
            break
        groups += new
        time.sleep(delay)
    if not groups:
        raise RuntimeError("테마 목록이 비어 있음")
    picked = pick_themes(groups, n_themes)
    themes_map, errors = {}, {}
    for g in picked:
        members = fetch_members(g["no"], get, per_theme * 2, errors)
        if len(members) >= 3:
            themes_map[g["name"]] = dict(members[:per_theme])
        time.sleep(delay)
        if not themes_map and len(errors) >= 3 and all(v != "OK" for v in errors.values()) and g is picked[min(2, len(picked) - 1)]:
            break  # 처음 몇 개 테마에서 모든 후보 주소가 실패하면 더 시도하지 않는다
    source = "theme"
    if not themes_map:  # 안전장치
        rank = ranking_universe(get)
        if not rank:
            raise RuntimeError(f"구성 종목도 순위도 읽지 못함 — 시도한 주소: {errors}")
        themes_map = {"시장 상위(거래대금·상승률)": rank}
        source = "ranking"
    stats = {"themes_total": len(groups), "themes_picked": len(picked), "themes_used": len(themes_map), "source": source,
             "stocks": len({c for m in themes_map.values() for c in m}), "member_errors": errors}
    return themes_map, stats
