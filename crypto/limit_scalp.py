"""지정가 왕복 단타와 그리드 매매(일정 간격으로 지정가 주문 깔기) 시뮬레이션: 수수료와 가격 하락에 따라 이익이 남는지 본다.
python -m crypto.limit_scalp   (업비트 KRW-ETH 1분봉 공개 시세만 사용, API 키·주문 없음)

규칙(모의): 포지션이 없으면 직전 1분봉 종가에 지정가 매수를 낸다. 그 봉의 저가가 매수가를 '뚫고 내려가야'(엄격 체결) 체결로 본다.
체결되면 매수가 × (1 + 목표폭)에 지정가 매도를 내고, 이후 봉의 고가가 목표가를 '뚫고 올라가야' 체결로 본다.
hold 분 안에 못 팔면 그 봉 종가에서 시장가로 판다(슬리피지 포함). 1분봉으로는 호가 대기열을 알 수 없어 실제보다 낙관적일 수 있다.
수수료는 체결 한 번당(매수·매도 각각) 부과한다.

그리드(영상 속 방식): 시작가 아래로 STEP원 간격 GRID_N개 칸에 각 PER원어치 지정가 매수를 깔고, 체결되면 STEP원 위에 지정가 매도를 내며,
팔리면 같은 칸에 다시 매수를 낸다. 한 봉 안에서는 이미 들고 있던 칸만 판다(보수적). 가격이 격자 아래로 내려가면 전부 사 둔 채 평가손실이 쌓인다."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .upbit_data import fetch_upbit_minutes

GAINS = (0.001, 0.002, 0.003, 0.005)        # 목표 이익 폭(0.1%~0.5%)
FEES = (0.0, 0.0004, 0.0005, 0.0025)        # 편도 수수료: 0%, 0.04%, 0.05%, 0.25%
HOLD = 30                                   # 최대 보유(분)
SLIP = 0.0005                               # 시장가 청산 슬리피지
TOTAL = 43200                               # 약 30일치 1분봉


def simulate_limit_scalp(df: pd.DataFrame, gain: float, fee: float, hold: int = HOLD, slip: float = SLIP, strict: bool = True) -> pd.DataFrame:
    """거래 목록: entry(시각), bars(보유 분), kind('목표'|'시간초과'), ret(수수료·슬리피지 반영 순수익률)."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(df)
    rows = []
    i = 1
    while i < n:
        bid = c[i - 1]
        if (l[i] < bid) if strict else (l[i] <= bid):
            target = bid * (1 + gain)
            exit_k, kind, sell = None, "시간초과", None
            for k in range(i + 1, min(i + 1 + hold, n)):
                if (h[k] > target) if strict else (h[k] >= target):
                    exit_k, kind, sell = k, "목표", target
                    break
            if exit_k is None:
                exit_k = min(i + hold, n - 1)
                sell = c[exit_k] * (1 - slip)
            ret = (sell * (1 - fee)) / (bid * (1 + fee)) - 1
            rows.append({"entry": df.index[i], "bars": exit_k - i, "kind": kind, "ret": ret})
            i = exit_k + 1
        else:
            i += 1
    return pd.DataFrame(rows, columns=["entry", "bars", "kind", "ret"])


def summarize(trades: pd.DataFrame, days: float) -> dict:
    if trades.empty:
        return {"n": 0}
    r = trades["ret"].to_numpy()
    eq = np.cumprod(1 + r)
    return {"n": len(r), "per_day": len(r) / max(days, 1e-9), "hit": float((trades["kind"] == "목표").mean()),
            "avg": float(r.mean()), "win": float((r > 0).mean()), "avg_win": float(r[r > 0].mean()) if (r > 0).any() else 0.0,
            "avg_loss": float(r[r <= 0].mean()) if (r <= 0).any() else 0.0, "final": float(eq[-1]),
            "mdd": float((eq / np.maximum.accumulate(eq) - 1).min())}


STEP, PER, GRID_N = 3000, 300_000, 100       # 영상 댓글: 간격 3000원, 칸당 30만원, 100개


def simulate_grid(df: pd.DataFrame, fee: float, step: float = STEP, per: float = PER, n: int = GRID_N) -> dict:
    """시작가 아래 n칸 그리드. 반환: 체결 횟수, 닫힌 왕복 수, 확정이익, 평가손익 포함 최종 손익, 최대 평가손실, 마지막에 보유 중인 칸 수."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    levels = c[0] - step * np.arange(1, n + 1)
    levels = levels[levels > 0]
    qty = per / levels
    cost = per * (1 + fee)
    proceeds = qty * (levels + step) * (1 - fee)
    has = np.zeros(len(levels), dtype=bool)
    realized, fills, pairs = 0.0, 0, 0
    worst = 0.0
    for t in range(1, len(c)):
        sell = has & (h[t] > levels + step)          # 이전 봉까지 들고 있던 칸만 판다
        if sell.any():
            k = int(sell.sum())
            realized += float((proceeds[sell] - cost).sum())
            has &= ~sell
            fills += k
            pairs += k
        buy = (~has) & (l[t] < levels)               # 가격이 뚫고 내려가야 매수 체결
        if buy.any():
            fills += int(buy.sum())
            has |= buy
        inv = float((qty[has] * c[t] * (1 - fee) - cost).sum()) if has.any() else 0.0
        worst = min(worst, realized + inv)
    inv = float((qty[has] * c[-1] * (1 - fee) - cost).sum()) if has.any() else 0.0
    return {"fills": fills, "pairs": pairs, "realized": realized, "final": realized + inv, "worst": worst, "held": int(has.sum()), "capital": per * len(levels)}


def grid_report(df: pd.DataFrame, label: str = "ETH/KRW") -> str:
    days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    p0, p1, lo = float(df["close"].iloc[0]), float(df["close"].iloc[-1]), float(df["low"].min())
    lines = [f"[그리드 모의] {label} 1분봉 {len(df)}개 · {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d} ({days:.0f}일) · 시작 {p0:,.0f}원 → 끝 {p1:,.0f}원 ({p1 / p0 - 1:+.1%}) · 기간 최저 {lo:,.0f}원 ({lo / p0 - 1:+.1%})",
             f"설정: 시작가 아래로 {STEP:,}원 간격 {GRID_N}칸, 칸당 {PER:,}원 지정가 매수 → 체결되면 {STEP:,}원 위에 지정가 매도 → 팔리면 같은 칸에 재주문. 엄격 체결.",
             f"묶이는 자금(전 칸 체결 시): 약 {GRID_N * PER:,.0f}원", "",
             "수수료(편도)   체결횟수  왕복   확정이익(원)  평가손익 포함(원)  최대 평가손실(원)  끝에 들고 있는 칸  확정이익/묶인자금"]
    for f in FEES:
        r = simulate_grid(df, f)
        lines.append(f"{f:>8.2%}   {r['fills']:>8d} {r['pairs']:>5d} {r['realized']:>13,.0f} {r['final']:>16,.0f} {r['worst']:>17,.0f} {r['held']:>12d}칸 {r['realized'] / r['capital']:>14.2%}")
    lines.append("")
    lines.append("읽는 법: '확정이익'은 영상의 '추정 수익'처럼 팔려서 닫힌 왕복만 센 값이고, '평가손익 포함'은 아직 못 판 코인의 현재 손익까지 더한 값입니다. 가격이 격자 아래로 내려가면 확정이익이 쌓여도 평가손실이 더 커질 수 있습니다.")
    lines.append("한계: 1분봉만 쓰고 호가 대기열·부분 체결·지연을 반영하지 않았고, 지난 30일의 한 시장 상황일 뿐입니다. 과거 결과가 미래를 보장하지 않습니다.")
    return "\n".join(lines)


def report(df: pd.DataFrame, label: str = "ETH/KRW") -> str:
    days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    bh = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
    lines = [f"[지정가 왕복 단타 모의] {label} 1분봉 {len(df)}개 · {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d} ({days:.0f}일) · 같은 기간 단순 보유 {bh:+.1%}",
             f"규칙: 직전 종가에 지정가 매수 → 체결되면 +목표폭에 지정가 매도, {HOLD}분 안에 못 팔면 시장가 청산(슬리피지 {SLIP:.2%}). 엄격 체결(가격이 뚫고 지나가야 체결).",
             "", "목표폭  수수료(편도)   거래수 하루   목표달성  거래당 평균(순)  이긴 비율  평균이익/평균손실   기간 누적(순)  최대낙폭"]
    for g in GAINS:
        for f in FEES:
            s = summarize(simulate_limit_scalp(df, g, f), days)
            if not s["n"]:
                lines.append(f"{g:>6.2%}  {f:>8.2%}      거래 없음")
                continue
            lines.append(f"{g:>6.2%}  {f:>8.2%}  {s['n']:>7d} {s['per_day']:>5.0f}  {s['hit']:>7.0%}  {s['avg']:>+14.3%}  {s['win']:>8.0%}  {s['avg_win']:>+7.2%}/{s['avg_loss']:>+6.2%}  {s['final'] - 1:>+12.1%}  {s['mdd']:>7.1%}")
        lines.append("")
    lines.append("읽는 법: 수수료 0.25%는 왕복 0.5%라 목표폭 0.5% 이하에서는 이익이 사실상 남지 않습니다. 같은 신호가 수수료 0%에서만 좋아 보인다면 수수료가 이익을 먹은 것입니다.")
    lines.append("한계: 1분봉만 쓰고 호가 대기열·체결 지연·부분 체결·API 호출 제한·서버 장애를 반영하지 않았고, 지난 30일의 한 시장 상황일 뿐입니다. 과거 결과가 미래를 보장하지 않습니다.")
    return "\n".join(lines)


def main() -> None:
    df = fetch_upbit_minutes(1, market="KRW-ETH", total=TOTAL)
    df = df[df.index + pd.Timedelta(minutes=1) <= pd.Timestamp.now(tz="UTC")]
    text = grid_report(df) + "\n\n" + report(df)
    print(text)
    os.makedirs("data", exist_ok=True)
    open("data/limit_scalp.txt", "w").write(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
