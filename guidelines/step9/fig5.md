# Figure 5: NPCA HARQ vs Primary HARQ — 유리한 조건

**연구 질문 (RQ5)**: HARQ retransmission을 NPCA channel에서 수행하는 것이 primary에서 기다리는 것보다 유리한 조건은 무엇인가?

**스크립트**: `harq_sim/run_step9_fig5.py`

**출력**: `manuscript/figure/fig5_*.{eps,png,pdf}`

---

## 실험 파라미터

| 항목 | 값 |
|---|---|
| **Sweep 변수 (주)** | `snr_db_mean` ∈ {8, 10, 12, 14, 16, 18, 20} |
| **교차 조건** | `obss_rate` ∈ {0.10, 0.30, 0.50} |
| `num_stas` | 5 |
| `snr_db_std` | 1.0 dB |
| `obss_min` | 20 슬롯 |
| `obss_max` | 200 슬롯 |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

## 비교 대상 (2개 기법)

| 기법 | npca_enabled | harq_enabled | 의미 |
|---|---|---|---|
| `harq_only` | False | True | Primary에서 HARQ 재전송 |
| `fixed_cw_npca_harq` | True | True | NPCA channel에서 HARQ 가능 |

## 측정 지표

- `aggregate_throughput` — 전달 패킷 수
- `mean_access_delay` — 평균 지연
- **HARQ gain** (계산):
  ```python
  gain = npca_harq_throughput - primary_harq_throughput
  gain_pct = gain / primary_harq_throughput * 100
  ```

## Figure 구성

**옵션 A (논문 공간 절약 — 권장)**:
```
Figure 5: 단일 패널 + inset
  주 패널: throughput gain(%) vs snr_db_mean
           3개 선 = 3개 obss_rate (0.10/0.30/0.50)
           y=0 기준선 표시 (양수 = NPCA HARQ 유리, 음수 = primary HARQ 유리)
           각 obss_rate 선에 색상 구분 + std 음영

  inset: mean_access_delay vs snr_db (2개 선, obss_rate=0.30만)
```

**옵션 B (상세 분석용)**:
```
Figure 5: 2×3 subplot
  행: throughput / mean_access_delay
  열: obss_rate = 0.10 / 0.30 / 0.50
  각 패널: 2개 선 (harq_only, fixed_cw_npca_harq)
```

## 예상 관찰

**저 SNR (8~12 dB) + 고 OBSS rate (0.30~0.50):**
- Primary HARQ는 OBSS 기간 대기 → combining 기회를 오래 기다림
- NPCA HARQ는 OBSS 중에도 NPCA channel에서 combining → 빠른 전달
- gain > 0, 이득 최대

**고 SNR (18~20 dB):**
- 첫 전송에서 이미 MCS7 → PHY 성공률 높음 → HARQ combining 불필요
- NPCA 전환 오버헤드 > combining 이득 가능
- gain → 0 또는 음수

**교차점(crossover SNR):**
- OBSS rate가 높을수록 교차점 SNR도 높아짐 (NPCA 우회 이득이 더 큰 SNR까지 유지)

## 출력 파일

```
manuscript/figure/
  fig5_harq_gain_vs_snr.eps
  fig5_harq_gain_vs_snr.png
  fig5_harq_gain_vs_snr.pdf

results/step9/fig5/
  data.csv            ← (snr_db, obss_rate, baseline, seed, metric, value)
```

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-25 | 초안 작성; 옵션 A/B 제시 |
