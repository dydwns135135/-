# Bitget USDT 선물 자동매매

주식 봇과 같은 지표(일목균형표 + 볼린저밴드)를 롱/숏 대칭으로 쓴다. 기본: BTC, 4시간봉, 레버리지 2배, 격리마진, 증거금 25%, 손절 3%(거래소 주문), **데모 모드**.

## 위험
- 레버리지는 손실도 키운다. 증거금 전액 청산 가능. **잃어도 되는 금액으로 레버리지 1~3배**부터.
- 4시간마다 한 번 실행하므로 봉 사이 급변은 **거래소에 걸린 손절 주문**이 막는다. 손절 주문 형식은 미검증(`exchange.py` `TODO(verify)`) → 데모에서 포지션에 손절이 실제로 붙는지 Bitget 앱에서 꼭 확인.
- 백테스트 데모(`--demo`)는 가상 데이터라 성능과 무관. 실제 4시간봉 CSV로 검증할 것.

## 사용
```
python -m crypto.backtest_cli btc_4h.csv --leverage 2        # date,open,high,low,close
export BITGET_API_KEY=... BITGET_API_SECRET=... BITGET_API_PASSPHRASE=...
python -m crypto                                              # 데모 모드 1회 실행
CRYPTO_CONFIRM_LIVE=yes python -m crypto --live               # 실거래
```
데모 모드는 Bitget 데모 방식(v2 mix REST 직접 호출, 종목 ID `SBTCSUSDT`, 증거금 `SUSDT`, `PAPTRADING` 헤더)으로 주문하고(ccxt 종목 목록에 데모 종목이 없어 통합 심볼을 쓰지 않음), 신호용 시세는 실제 BTC/USDT를 쓴다. 데모 키에는 **선물 주문 + 선물 포지션 읽기/쓰기** 권한이 필요하다(포지션 쓰기가 없으면 오류 40014).

Bitget API 키는 **거래 권한만** 주고 출금 권한은 끄고, 가능하면 IP 제한을 건다. 데모는 데모 전용 키가 필요하다.

## GitHub Actions (`.github/workflows/crypto.yml`)
- Secrets: `BITGET_API_KEY`, `BITGET_API_SECRET`, `BITGET_API_PASSPHRASE` (+ 기존 텔레그램)
- Variables: `CRYPTO_ENABLED=true`(스케줄 켜기), 선택 `CRYPTO_SYMBOL`, `CRYPTO_LEVERAGE`, `CRYPTO_LIVE=true`(실거래, 기본 없음)
- 먼저 Actions 탭에서 수동 실행(데모)으로 확인.
