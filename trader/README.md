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
