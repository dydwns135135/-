# 미국 주식 자동매매 (토스증권 OpenAPI)

일목균형표(추세 필터) + 볼린저밴드(눌림목 진입) 롱 온리 전략. 일봉 기준, 하루 1회 실행.

## 전략
- **매수**: 종가 > 구름 상단 & 전환선 > 기준선(강세)인데 %B ≤ 0.35 (눌림목)
- **매도**: 종가 < 기준선, 종가 < 구름 상단, %B ≥ 1.0(과열), 또는 평단 -8% 손절
- 리스크: 1회 주문 $500, 종목당 비중 20%, 최대 5종목, 일일 -3% 이면 신규 매수 중단 (`trader/risk.py`)

## 사용
```
pip install -r requirements-trader.txt
python -m pytest tests                       # 테스트
python -m trader.backtest_cli data.csv       # date,open,high,low,close CSV 백테스트
python -m trader.backtest_cli --demo         # 가상 데이터 (성능과 무관)

export TOSS_CLIENT_ID=... TOSS_CLIENT_SECRET=... TOSS_ACCOUNT=...
python -m trader AAPL MSFT NVDA              # 모의 모드: 주문 전송 없이 로그만
TRADER_CONFIRM_LIVE=yes python -m trader AAPL --live   # 실거래
```

## 실거래 전 필수 확인
1. `trader/toss_client.py` 의 엔드포인트·응답 필드는 **공식 문서로 검증되지 않은 가정**입니다(`TODO(verify)`). 공식 문서와 대조해 수정하세요.
2. 실제 시장 데이터(CSV)로 백테스트하고, 모의 모드를 최소 수 주 돌려 본 뒤 소액으로 시작하세요.
3. API 키는 환경변수로만 두고, 출금 권한은 주지 마세요. 수익을 보장하지 않습니다.

## GitHub Actions + 텔레그램 (휴대폰 운영)
`.github/workflows/trader.yml` 이 평일 미국장 마감 후(21:30 UTC, 한국 06:30/05:30) 하루 한 번 실행하고 결과를 텔레그램으로 보냅니다.

1. 텔레그램에서 `@BotFather` 로 봇을 만들어 토큰을 받고, 봇에게 메시지를 보낸 뒤
   `https://api.telegram.org/bot<토큰>/getUpdates` 에서 `chat.id` 확인
2. 저장소 Settings → Secrets and variables → Actions
   - **Secrets**: `TOSS_CLIENT_ID`, `TOSS_CLIENT_SECRET`, `TOSS_ACCOUNT`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
   - **Variables**: `TRADER_SYMBOLS`(예: `AAPL MSFT NVDA`). 실거래를 상시 켜려면 `TRADER_LIVE=true` (기본 없음 = 모의)
3. Actions 탭 → trader → Run workflow 로 수동 테스트(휴대폰 GitHub 앱/브라우저 가능). `live` 체크 시 실거래.

주의: 스케줄 실행은 지연되거나 가끔 건너뛸 수 있고, 모의 모드는 실행마다 가상 자산이 초기화되므로 "신호 알림" 용도입니다. 토스증권 API의 IP 제한 여부(GitHub IP는 고정이 아님)는 확인이 필요합니다.
