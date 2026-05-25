# Figure 3: Fixed CW_npca_init(qsrc) → NPCA Collision Burst

**연구 질문 (RQ3)**: fixed small CW_npca_init은 NPCA collision burst를 유발하는가?

**스크립트**: `harq_sim/run_step9_fig3.py`

**출력**: `manuscript/figure/fig3_*.{eps,png,pdf}`

---

## 실험 파라미터

| 항목 | 값 |
|---|---|
| **Sweep 변수** | `npca_qsrc` ∈ {0, 1, 2, 3, 4, 5} |
| `num_stas` | 10 |
| `obss_rate` | 0.30 |
| `snr_db_mean` | 20.0 dB |
| `snr_db_std` | 0.0 dB |
| `obss_min` | 20 슬롯 |
| `obss_max` | 200 슬롯 |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

> **주의**: `snr_db=20dB` (고 SNR, 결정론적) 사용 이유 —  
> PHY 실패를 최소화하여 충돌 효과를 분리 관찰.  
> PHY 실패와 충돌이 혼재하면 qsrc 효과 해석이 어려움.

## 비교 대상 (1개 기법, qsrc만 변경)

| 기법 | npca_enabled | harq_enabled | adaptive_cw | npca_qsrc |
|---|---|---|---|---|
| `fixed_cw_npca_harq` | True | True | False | sweep |

## CW 변환 공식

```
npca_cw = 2^qsrc × (CW_MIN + 1) − 1 = 2^qsrc × 16 − 1
qsrc=0 → CW=15,  qsrc=1 → CW=31,  qsrc=2 → CW=63
qsrc=3 → CW=127, qsrc=4 → CW=255, qsrc=5 → CW=511
```

## 측정 지표

- `collision_probability_npca` — NPCA 채널 충돌 확률
- `aggregate_throughput` — 전달 패킷 수
- `mean_access_delay` — 평균 지연
- `npca_transition_count` — 전환 횟수

## Figure 구성

```
Figure 3: 2-row subplot (공유 x축: npca_qsrc)
  Row A: 이중 y축
    y-left:  collision_probability_npca (선 + 음영)
    y-right: aggregate_throughput (점선)
  Row B: 이중 y축
    y-left:  mean_access_delay
    y-right: npca_transition_count (점선)

  x축 눈금: [0, 1, 2, 3, 4, 5]
  x축 보조 레이블: CW 값 표시 [15, 31, 63, 127, 255, 511]
```

## 예상 관찰

- qsrc=0 (CW=15): NPCA 충돌 확률 최대 → throughput 감소
- qsrc 증가: 충돌 감소 → throughput 증가, delay 개선
- qsrc 과도하면 (4, 5): backoff 기대값 > NPCA timer 평균 → 빈 전환 발생
  → transition count 감소, throughput 다시 감소
- **역U자(∩) 형태의 throughput 곡선** → 최적 qsrc 존재 → 논문 기여 동기 부여
- collision_probability_npca는 qsrc 증가에 따라 단조 감소

## 출력 파일

```
manuscript/figure/
  fig3_qsrc_sweep.eps
  fig3_qsrc_sweep.png
  fig3_qsrc_sweep.pdf

results/step9/fig3/
  data.csv            ← (npca_qsrc, npca_cw, seed, metric, value)
```

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-25 | 초안 작성 |
