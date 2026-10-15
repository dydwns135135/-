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

## 알림 모드 (폰 MT5 에서 수동 매매)
주문은 내지 않고 진입/청산 타이밍만 텔레그램으로 알려 준다. (PC 에서 켜 두고, 폰 MT5 앱으로 직접 주문)
```
set TELEGRAM_BOT_TOKEN=BotFather 에서 받은 토큰
python -m mt5bot telegram-test          # chat id 확인 + 테스트 알림 (봇에게 먼저 메시지 보낼 것)
set TELEGRAM_CHAT_ID=위에서 나온 private 숫자
python -m mt5bot alert --symbol GOLD
```

## 백테스트
과거 GOLD 5분봉으로 전략(RSI 역추세, EMA200 필터, 돌파 채널, EMA 눌림목) × SL/TP 배수를 비교한다. 비용(스프레드)을 빼고, 앞 70% 학습 / 뒤 30% 검증 구간으로 나눠 과최적화를 점검한다.
```
python -m mt5bot backtest --symbol GOLD --bars 30000 --spread 0.3
```

## 간편 실행 (Windows)
- `start_check.bat` 더블클릭: 연결/계좌/지표 확인 (주문 없음)
- `start_run.bat` 더블클릭: 자동매매 (기본값: GOLD, 0.01랏, EMA200 필터, SL ATR×2.0 / TP ATR×3.0)
- `start_alert.bat` 더블클릭: 텔레그램 알림만 (주문 없음). 같은 폴더에 `telegram.txt` 를 만들어 첫 줄=봇 토큰, 둘째 줄=chat id 를 적어 둔다(저장소에는 올라가지 않음).

## 텔레그램 알림 (자동매매 `run`)
`telegram.txt` 가 있으면 `run` 도 진입/청산/SL·TP 종료/오류와 6시간마다 "정상 작동 중" 메시지를 텔레그램으로 보낸다. 정상 작동 메시지가 끊기면 PC 나 봇이 멈춘 것이다(`--heartbeat-hours` 로 조정, 0 이면 끔).
