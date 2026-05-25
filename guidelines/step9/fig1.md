# Figure 1: NPCA-HARQ 조합 이득 검증

**연구 질문 (RQ1)**: NPCA-HARQ는 HARQ-only 또는 NPCA-only보다 throughput/PDR/delay 측면에서 이득이 있는가?

**스크립트**: `harq_sim/run_step9_fig1.py`

**출력**: `manuscript/figure/fig1_*.{eps,png,pdf}`

---

## 실험 파라미터

| 항목 | 값 |
|---|---|
| **Sweep 변수** | `obss_rate` ∈ {0.10, 0.20, 0.30, 0.50} |
| `num_stas` | 5 |
| `snr_db_mean` | 14.0 dB |
| `snr_db_std` | 2.0 dB |
| `obss_min` | 20 슬롯 |
| `obss_max` | 200 슬롯 |
| `num_slots` | 50,000 |
| `ppdu_duration` | 20 |
| Seeds | [42, 123, 456] |

## 비교 대상 (4개 기법)

| 기법 | npca_enabled | harq_enabled | adaptive_cw |
|---|---|---|---|
| `legacy_edca` | False | False | False |
| `arq_only_npca` | True | False | False |
| `harq_only` | False | True | False |
| `fixed_cw_npca_harq` | True | True | False |

## 측정 지표

- `aggregate_throughput` (패킷 수)
- `mean_access_delay` (슬롯)
- `packet_delivery_ratio` (PDR)

## Figure 구성

```
Figure 1: 3-row × 1-col subplot (공유 x축: obss_rate)
  Row A: aggregate_throughput vs obss_rate  — 4개 선 + std 음영
  Row B: mean_access_delay vs obss_rate
  Row C: packet_delivery_ratio vs obss_rate

  범례: legacy_edca | arq_only_npca | harq_only | fixed_cw_npca_harq
  x축 눈금: [0.10, 0.20, 0.30, 0.50]
  오차 표시: mean ± std (3 seeds)
```

## 예상 관찰

- OBSS rate 증가 → primary 채널 혼잡 → NPCA 우회 이득 커짐
- throughput 순서: `fixed_cw_npca_harq` ≥ `arq_only_npca` ≥ `harq_only` ≥ `legacy_edca`
- delay: NPCA 계열이 낮음 (OBSS 중 TX 가능)
- PDR: HARQ 계열이 높음 (combining으로 PHY 실패 구제)
- OBSS rate 낮을 때: NPCA overhead로 이득 작음 → 교차점 존재

## 출력 파일

```
manuscript/figure/
  fig1_combined.eps   ← LaTeX 삽입용
  fig1_combined.png   ← 미리보기 (300 dpi)
  fig1_combined.pdf   ← 고품질 벡터

results/step9/fig1/
  data.csv            ← (obss_rate, baseline, seed, metric, value)
```

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-25 | 초안 작성 |
