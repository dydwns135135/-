"""업비트 원화마켓 여러 코인을 분석하고 '상승 추세인 강한 종목에 갈아타기' 전략을 백테스트한다.
python -m crypto.upbit_rotation   (공개 시세만 사용, API 키·주문 없음)

종목 점수 = 8개 기간(30~250일) 중 가격이 오른 기간의 비율(0~1) — BTC 앙상블과 같은 규칙을 종목마다 적용.
- 후보군: 그날 기준 최근 30일 평균 거래대금 상위 N개(점수 계산 가능한 종목만, 그날까지의 데이터만 사용)
- 로테이션: 후보군 중 90일 수익률 상위 K개를 골라 (점수/K)씩 보유, 점수가 낮으면 그만큼 현금. 주 1회 재조정
- 동일비중: 후보군 전체에 (점수/N)씩 보유
비용은 비중이 바뀐 만큼(회전율) 0.15%(수수료 0.05% + 슬리피지 0.10%).
주의: 후보 종목을 '지금 거래대금 상위'에서 뽑으므로 이미 살아남은 종목만 들어 있다(생존 편향) → 결과는 실제보다 좋게 나온다."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

from .momentum_bot import ENSEMBLE_LOOKBACKS
from .upbit_data import URL, fetch_upbit_candles

COST = 0.0015
N_UNIVERSE, K_PICK, REBAL_DAYS, RANK_LB = 10, 3, 7, 90
STABLES = {"USDT", "USDC", "USDE", "DAI"}
HDR = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}


def top_krw_markets(limit: int = 25, get=requests.get) -> list[tuple[str, float]]:
    """원화마켓 중 24시간 거래대금 상위(스테이블코인 제외): [(마켓, 거래대금(원))]."""
    r = get("https://api.upbit.com/v1/market/all", params={"isDetails": "false"}, headers=HDR, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Upbit HTTP {r.status_code}")
    mk = [m["market"] for m in r.json() if m["market"].startswith("KRW-") and m["market"][4:] not in STABLES]
    rows = []
    for i in range(0, len(mk), 100):
        t = get("https://api.upbit.com/v1/ticker", params={"markets": ",".join(mk[i:i + 100])}, headers=HDR, timeout=20)
        if t.status_code != 200:
            raise RuntimeError(f"Upbit ticker HTTP {t.status_code}")
        rows += [(x["market"], float(x["acc_trade_price_24h"])) for x in t.json()]
        time.sleep(0.15)
    return sorted(rows, key=lambda x: -x[1])[:limit]


def load_panel(markets: list[str], total: int = 2600, fetch=fetch_upbit_candles) -> tuple[dict[str, pd.DataFrame], list[str]]:
    data, failed = {}, []
    for m in markets:
        for attempt in (0, 1):
            try:
                data[m] = fetch(URL, m, total)
                break
            except RuntimeError as e:
                if attempt == 1:
                    failed.append(f"{m}: {e}")
                else:
                    time.sleep(2)
    return data, failed


def build_panel(data: dict[str, pd.DataFrame], now: pd.Timestamp | None = None):
    """진행 중인 오늘 봉을 빼고 (open, close, value) 표(날짜×종목)로 만든다."""
    now = now or pd.Timestamp.now(tz="UTC")
    o, c, v = {}, {}, {}
    for m, df in data.items():
        df = df[df.index + pd.Timedelta(days=1) <= now]
        o[m], c[m], v[m] = df["open"], df["close"], df["value"]
    idx = pd.DataFrame(c).index
    return pd.DataFrame(o).reindex(idx), pd.DataFrame(c), pd.DataFrame(v).reindex(idx)


def scores(close: pd.DataFrame, lookbacks=ENSEMBLE_LOOKBACKS) -> pd.DataFrame:
    """오른 기간의 비율(0~1). 가장 긴 기간만큼 데이터가 쌓이기 전에는 NaN(후보 제외)."""
    votes = [(close > close.shift(lb)).astype(float).where(close.shift(lb).notna()) for lb in lookbacks]
    return sum(votes) / len(votes)   # 하나라도 NaN 이면 NaN


def target_weights(close: pd.DataFrame, value: pd.DataFrame, mode: str = "rotation", n_univ: int = N_UNIVERSE,
                   k: int = K_PICK, rebal: int = REBAL_DAYS, rank_lb: int = RANK_LB) -> pd.DataFrame:
    sc = scores(close)
    v30 = value.rolling(30, min_periods=20).mean().where(sc.notna())
    uni = v30.rank(axis=1, ascending=False) <= n_univ
    if mode == "rotation":
        rel = close.pct_change(rank_lb).where(uni)
        pick = rel.rank(axis=1, ascending=False) <= k
        w = sc.where(pick, 0.0) / k
    elif mode == "equal":
        w = sc.where(uni, 0.0) / n_univ
    else:
        raise ValueError(mode)
    w = w.fillna(0.0)
    if rebal > 1:
        keep = np.arange(len(w)) % rebal == 0
        w = w.where(pd.Series(keep, index=w.index), np.nan).ffill().fillna(0.0)
    return w


def simulate(open_: pd.DataFrame, w: pd.DataFrame, cost: float = COST) -> pd.DataFrame:
    """비중은 봉 t 마감에 정해 t+1 시가에 체결, 시가→다음 시가 수익을 먹는다. 일별 수익·회전율 반환."""
    pos = w.shift(1).fillna(0.0)
    ret = (open_.shift(-1) / open_ - 1).fillna(0.0)
    turn = pos.diff().abs().sum(axis=1)
    turn.iloc[0] = pos.iloc[0].abs().sum()
    r = (pos * ret).sum(axis=1) - turn * cost
    return pd.DataFrame({"ret": r, "turn": turn, "exposure": pos.sum(axis=1)})


def seg_stats(sim: pd.DataFrame) -> dict:
    r = sim["ret"]
    eq = (1 + r).cumprod()
    years = max(len(r) / 365, 1e-9)
    final = float(eq.iloc[-1])
    return {"연환산": final ** (1 / years) - 1 if final > 0 else -1.0,
            "최대낙폭": float((eq / eq.cummax() - 1).min()),
            "샤프": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else float("nan"),
            "연회전율": float(sim["turn"].sum() / years), "평균투입": float(sim["exposure"].mean())}


def compare(data_panel: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], btc: str = "KRW-BTC") -> dict[str, pd.DataFrame]:
    o, c, v = data_panel
    sc = scores(c)
    start = sc.notna().sum(axis=1).ge(5).idxmax()   # 후보가 5개 이상 생긴 첫날
    sims = {
        "BTC 보유": simulate(o[[btc]], pd.DataFrame(1.0, index=o.index, columns=[btc])),
        "BTC 앙상블(주1회)": simulate(o[[btc]], target_weights(c[[btc]], v[[btc]], "equal", n_univ=1)),
        f"로테이션 상위{K_PICK}": simulate(o, target_weights(c, v, "rotation")),
        f"동일비중 후보{N_UNIVERSE}": simulate(o, target_weights(c, v, "equal")),
    }
    n = len(o.loc[start:])
    segs = {"전체": start, "뒤쪽 절반": o.index[len(o) - n // 2], "최근 2년": o.index[max(len(o) - 730, 0)]}
    out = {}
    for name, s0 in segs.items():
        s0 = max(s0, start)
        out[name] = pd.DataFrame({k: seg_stats(v_.loc[s0:]) for k, v_ in sims.items()}).T
        out[name].attrs["range"] = f"{s0:%Y-%m-%d} ~ {o.index[-1]:%Y-%m-%d} ({len(o.loc[s0:])}일)"
    return out


def today_table(close: pd.DataFrame, value: pd.DataFrame) -> pd.DataFrame:
    sc = scores(close)
    w = target_weights(close, value, "rotation", rebal=1)
    last = close.index[-1]
    rows = []
    for m in close.columns:
        s = close[m].dropna()
        if len(s) < 31:
            continue
        pr = lambda lb: float(s.iloc[-1] / s.iloc[-1 - lb] - 1) if len(s) > lb else float("nan")
        rows.append({"종목": m[4:], "30일": pr(30), "90일": pr(90), "180일": pr(180),
                     "점수": float(sc.loc[last, m]) if not np.isnan(sc.loc[last, m]) else float("nan"),
                     "거래대금(억/일)": float(value[m].iloc[-30:].mean() / 1e8), "오늘 목표비중": float(w.loc[last, m])})
    return pd.DataFrame(rows).sort_values("거래대금(억/일)", ascending=False).reset_index(drop=True)


def fmt_compare(tbl: pd.DataFrame) -> str:
    t = tbl.copy()
    for col in ("연환산", "최대낙폭", "평균투입"):
        t[col] = t[col].map(lambda x: f"{x * 100:+.0f}%" if col != "평균투입" else f"{x * 100:.0f}%")
    t["샤프"] = t["샤프"].map(lambda x: f"{x:.2f}")
    t["연회전율"] = t["연회전율"].map(lambda x: f"{x:.1f}배")
    return t.to_string()


def fmt_today(t: pd.DataFrame) -> str:
    x = t.copy()
    for col in ("30일", "90일", "180일"):
        x[col] = x[col].map(lambda v: "-" if np.isnan(v) else f"{v * 100:+.0f}%")
    x["점수"] = x["점수"].map(lambda v: "-" if np.isnan(v) else f"{v:.2f}")
    x["거래대금(억/일)"] = x["거래대금(억/일)"].map(lambda v: f"{v:,.0f}")
    x["오늘 목표비중"] = x["오늘 목표비중"].map(lambda v: f"{v * 100:.0f}%" if v > 0 else "-")
    return x.to_string(index=False)


def main() -> None:
    mk = top_krw_markets(25)
    data, failed = load_panel([m for m, _ in mk])
    if "KRW-BTC" not in data or len(data) < 8:
        raise SystemExit(f"데이터 부족: 받은 종목 {len(data)}개, 실패 {failed}")
    panel = build_panel(data)
    res = compare(panel)
    lines = [f"데이터: 업비트 원화 거래대금 상위 {len(data)}개 종목 일봉, {panel[1].index[0]:%Y-%m-%d}~{panel[1].index[-1]:%Y-%m-%d}"]
    if failed:
        lines.append("받지 못한 종목: " + "; ".join(failed))
    for name, tbl in res.items():
        lines += ["", f"=== {name}: {tbl.attrs['range']} ===", fmt_compare(tbl)]
    lines += ["", "=== 오늘의 종목 분석 (거래대금 순) ===", fmt_today(today_table(panel[1], panel[2])), "",
              f"비용 {COST * 100:.2f}%(회전율 기준). 후보를 '현재 거래대금 상위'에서 뽑아 생존 편향이 있어 실제보다 좋게 나옵니다.",
              "과거 데이터이며 수익을 보장하지 않습니다. 알트코인은 상장폐지·급락 위험이 BTC보다 큽니다."]
    text = "\n".join(lines)
    print(text)
    os.makedirs("data", exist_ok=True)
    open("data/upbit_rotation.txt", "w").write(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
