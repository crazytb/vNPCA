# Figure 4: Adaptive vs Fixed CW — Collision-Delay Tradeoff

**연구 질문 (RQ4)**: adaptive CW_npca_init은 NPCA collision을 줄이면서 delay 이득을 유지하는가?

**스크립트**: `harq_sim/run_step9_fig4.py`

**출력**: `manuscript/figure/fig4_*.{eps,png,pdf}`

---

## 실험 파라미터

| 항목 | 값 |
|---|---|
| **Sweep 변수** | `num_stas` ∈ {2, 5, 10, 15, 20} |
| `obss_rate` | 0.30 |
| `snr_db_mean` | 14.0 dB |
| `snr_db_std` | 2.0 dB |
| `obss_min` | 20 슬롯 |
| `obss_max` | 200 슬롯 |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

## 비교 대상 (2개 기법)

| 기법 | npca_enabled | harq_enabled | adaptive_cw | npca_qsrc |
|---|---|---|---|---|
| `fixed_cw_npca_harq` | True | True | False | 0 (CW=15) |
| `adaptive_cw_npca_harq` | True | True | True | 동적 조정 |

## 측정 지표

- `collision_probability_npca` — NPCA 충돌 확률
- `mean_access_delay` — 평균 지연
- `npca_transition_count` — 전환 횟수 (빈 전환 포함)
- `avg_npca_qsrc` (per-STA 집계) — 자동 선택된 qsrc 평균

## Figure 구성

```
Figure 4: 3-row subplot + 보조 패널 (공유 x축: num_stas)
  Row A: collision_probability_npca vs num_stas
          fixed (solid, blue) vs adaptive (dashed, orange) + std 음영
  Row B: mean_access_delay vs num_stas
  Row C: npca_transition_count vs num_stas
          (fixed에서는 대부분 TX 동반, adaptive에서는 빈 전환 증가 추세)

  보조 패널 (inset or 별도): avg_npca_qsrc vs num_stas
          adaptive의 자동 조정 결과 — STA 수 증가 시 qsrc 어떻게 변하는지
```

## 핵심 발견 사항 (Step 8에서 관찰된 성능 역전)

```
실험 조건: num_stas=10, SNR=14dB, obss_rate=0.05, obss_min=20, obss_max=200
결과:
  fixed_cw_npca_harq:    throughput=1792, npca_transitions=2520
  adaptive_cw_npca_harq: throughput=935,  npca_transitions=539
```

**원인 연쇄 (논문 기술 필요)**:
1. SNR=14dB → MCS3 경계 → PHY 실패 빈발 → `npca_failure_rate > 0.3` → `q += 1`
2. STA 많음 → `primary_cw >= 4×CW_MIN` → `q += 1`
3. 빈번한 전환 시도 → `num_recent_npca_transitions > 3` → `q += 1`
4. qsrc → 3~4 도달 → NPCA backoff 기대값 = 63~127 슬롯
5. NPCA timer 평균 = obss_remain - switch_back_delay ≈ 100 슬롯
6. backoff 완료 전 timer 만료 → switch-back → **빈 전환** (TX 없이 복귀)
7. transition_count는 많지만 실제 TX 없음 → throughput 급감

**Figure 4에서 STA 수 별로 이 현상이 어떻게 전개되는지 정량화**

## 예상 관찰

- STA 수 적을 때 (2~5): adaptive ≈ fixed (qsrc 조정 불필요한 조건)
- STA 수 증가 시 (10~20): adaptive가 qsrc 과도하게 올려 NPCA 기회 놓침
  - collision_probability_npca: adaptive < fixed (충돌 억제 효과 있음)
  - mean_access_delay: adaptive > fixed (빈 전환으로 실질 TX 감소)
  - **tradeoff 확인: collision 억제는 성공, delay 이득 유지는 실패**
- avg_npca_qsrc: STA 수 증가에 따라 adaptive의 qsrc가 올라가는 추세

## 출력 파일

```
manuscript/figure/
  fig4_adaptive_vs_fixed.eps
  fig4_adaptive_vs_fixed.png
  fig4_adaptive_vs_fixed.pdf

results/step9/fig4/
  data.csv            ← (num_stas, baseline, seed, metric, value)
```

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-25 | 초안 작성; Step 8 역전 현상 분석 포함 |
