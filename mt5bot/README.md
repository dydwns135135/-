# MT5 RSI 자동매매 (XAUUSD M5)

Windows + MetaTrader5 데스크톱 터미널 필요 (Python `MetaTrader5` 패키지는 Windows 전용).

```
pip install -r requirements-mt5.txt
set MT5_LOGIN=12345678 & set MT5_PASSWORD=... & set MT5_SERVER=브로커-Demo   (생략 시 터미널에 로그인된 계좌)
python -m mt5bot check      # 연결·계좌 종류·잔고/마진·최근 캔들·RSI 확인 (주문 없음)
python -m mt5bot run        # 자동매매 (데모 계좌만 허용)
```

- 진입: RSI(14)가 30 아래에서 위로 벗어나면 매수, 70 위에서 아래로 벗어나면 매도. 확정된 5분봉만 사용.
- 추세 필터: 종가가 EMA200 위일 때만 매수, 아래일 때만 매도 (`--no-trend-filter` 로 끔).
- 청산: SL/TP(기본 ATR14 × 1.5 / × 3.0, 주문에 항상 포함. `--sl --tp` 로 고정 거리 지정 가능) 또는 RSI 가 50 을 되돌파할 때.
- 일일 손실 한도: 하루 시작 자산 대비 2% 손실이면 그날 신규 진입 중지 (`--max-daily-loss-pct`, 봇을 재시작하면 기준 초기화).
- 랏: `--risk`(기본 10, 계좌 통화)만큼만 손실되도록 계산, `--lots` 로 고정 가능.
- 실계좌는 `--allow-live` 없이는 실행 거부. 슬리피지(요청가 vs 체결가)는 진입 로그에 출력되므로 데모에서 충분히 확인 후 사용.
- 심볼 이름은 브로커마다 다를 수 있음(`GOLD`, `XAUUSD.m` 등) → `--symbol`.
