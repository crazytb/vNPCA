# Figure 11: PPDU Duration Effect on qsrc* — W_avail Unification

**연구 질문 (RQ11)**: qsrc*는 W와 PPDU_duration의 개별 값이 아닌, 유효 경쟁 창 W_avail = W − PPDU_duration에 의해 결정되는가?

**배경**: Fig 3은 qsrc*(N) N 의존성, Fig 10은 qsrc*(W) W 의존성을 각각 보였다. 그러나 두 실험 모두 PPDU_duration=20으로 고정되어 있었으므로, W 의존성이 사실상 (W − 20) 의존성일 가능성이 있다.

**핵심 이론**: NPCA 창 내에서 STA가 전송을 완료하려면 `backoff + PPDU ≤ W`, 즉 유효 backoff 예산은 `W_avail = W − PPDU`다. qsrc*는 N과 W_avail만의 함수일 것이다.

**핵심 설계**:
- Part 1: W=170 고정, PPDU ∈ {10,20,30,50,80,120} sweep → qsrc*(PPDU) 관찰
- Part 2: 동일한 W_avail을 다른 (W,PPDU) 조합으로 달성 → collapse 검증

**스크립트**: `harq_sim/run_step9_fig11.py`

**출력**: `manuscript/figure/fig11_ppdu_weff.{eps,png,pdf}`

---

## 실험 파라미터

### Part 1 — PPDU sweep (W=170 고정)

| 항목 | 값 |
|---|---|
| W (OBSS duration) | 170 슬롯 (고정) |
| **Sweep 변수** | `PPDU_duration` ∈ {10, 20, 30, 50, 80, 120} × `num_stas` ∈ {10, 20, 30, 50} |
| OBSS 점유율 | 50% (rate = 0.50 / (170 × 0.50) ≈ 0.00588) |
| `qsrc` sweep | {0, 1, 2, 3, 4, 5} |
| `snr_db_mean` | 20.0 dB |
| `snr_db_std` | 0.0 dB |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

총 Part 1: 6 × 4 × 6 × 3 = **432 runs**

### Part 2 — W_avail Collapse 검증 (N=30 고정)

동일한 W_avail을 서로 다른 (W, PPDU) 조합으로 달성하여 qsrc*가 수렴하는지 확인:

| W_avail 목표 | (W, PPDU) 조합 |
|---|---|
| 30  | (W=50, PPDU=20), (W=80, PPDU=50), (W=110, PPDU=80) |
| 80  | (W=100, PPDU=20), (W=130, PPDU=50), (W=160, PPDU=80) |
| 150 | (W=170, PPDU=20), (W=200, PPDU=50), (W=230, PPDU=80) |

각 조합마다 OBSS 점유율 50% 유지 (rate = 0.50/(W×0.50) — W별 재계산)

총 Part 2: 3 × 3 × 6 × 3 = **162 runs**

**총 실험 횟수: 432 + 162 = 594 runs**

---

## 비교 대상

| 기법 | npca_enabled | harq_enabled | adaptive_cw | npca_qsrc |
|---|---|---|---|---|
| `fixed_cw_npca_harq` | True | True | False | sweep (0~5) |

---

## CW 변환 공식

```
npca_cw = 2^qsrc × 16 − 1
qsrc=0 → CW=15,  qsrc=1 → CW=31,  qsrc=2 → CW=63
qsrc=3 → CW=127, qsrc=4 → CW=255, qsrc=5 → CW=511
```

---

## 측정 지표

- `aggregate_throughput` — 주 지표, qsrc* 결정 기준
- `collision_probability_npca` — NPCA 충돌 확률
- `mean_access_delay` — 평균 지연
- `npca_transition_count` — NPCA 전환 총 횟수

---

## Figure 구성

```
Figure 11: 3-panel

Panel (a): Throughput vs qsrc (N=30 고정, W=170 고정, lines per PPDU ∈ {10,20,50,80,120})
           각 선의 최대값에 ★ 마커
           핵심 메시지: PPDU 증가 → ★ 위치(qsrc*)가 왼쪽으로 이동

Panel (b): qsrc*(PPDU) — W=170 고정, lines per N ∈ {10, 20, 30, 50}
           x축: PPDU duration (slots), y축: qsrc* (0~5)
           계단 함수 형태 (drawstyle="steps-post")
           핵심 메시지: PPDU 증가 → qsrc* 감소; N 클수록 qsrc* baseline 높음

Panel (c): W_avail Collapse 검증 — N=30 고정
           배경선: Part 1 결과 (W=170, N=30) → qsrc*(W_avail) 연속 참조 곡선
           산점도: Part 2 결과 — 각 W_avail 목표값에서 3개 (W,PPDU) 쌍
                  PPDU=20: ● / PPDU=50: ■ / PPDU=80: ▲ 마커
           핵심 메시지: 다른 (W,PPDU) 쌍도 같은 W_avail이면 참조선 위에 위치
                       → W_avail이 (W, PPDU)를 대체하는 통합 예측 변수
```

---

## 예상 결과 (Part 1, W=170, N=30)

| PPDU | W_avail | 예상 qsrc* |
|---|---|---|
| 10 | 160 | 1~2 |
| 20 | 150 | 1 (Fig 10 확인값) |
| 30 | 140 | 1 |
| 50 | 120 | 1 |
| 80 | 90 | 0~1 |
| 120 | 50 | 0 |

### Collapse 예상 (Part 2, N=30)

| W_avail | (W=50,PPDU=20) | (W=80,PPDU=50) | (W=110,PPDU=80) |
|---|---|---|---|
| 30 | qsrc*=1 | qsrc*=1 | qsrc*=1 |

| W_avail | (W=100,PPDU=20) | (W=130,PPDU=50) | (W=160,PPDU=80) |
|---|---|---|---|
| 80 | qsrc*=1 | qsrc*=1 | qsrc*=1 |

| W_avail | (W=170,PPDU=20) | (W=200,PPDU=50) | (W=230,PPDU=80) |
|---|---|---|---|
| 150 | qsrc*=1 | qsrc*=1 | qsrc*=1 |

각 행에서 3개 값이 동일하면 collapse 검증 성공.

---

## 논문 내 위치

```
§ Problem   : Fig 3 (qsrc*(N) — N 의존성)
§ Analysis  : Fig 10 (qsrc*(W_eff) — W 의존성)
              Fig 11 (qsrc*(W_avail) — PPDU 포함 통합 이론 완성)
              Theorem 수정: qsrc*(N, W_eff) → qsrc*(N, W_avail = W_eff − PPDU)
§ Algorithm : Adaptive qsrc 수정 — 입력: W_est − PPDU_own (= W_avail 추정치)
```

---

## 출력 파일

```
manuscript/figure/
  fig11_ppdu_weff.eps / .png / .pdf

results/step9/fig11/
  data.csv    ← (part, W, ppdu_duration, W_avail, num_stas, npca_qsrc, npca_cw, seed,
                  collision_probability_npca, aggregate_throughput,
                  mean_access_delay, npca_transition_count)
  fig11_ppdu_weff_preview.png
```

---

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-29 | 초안 작성 — PPDU_duration 변수화 및 W_avail 통합 이론 검증 실험 설계 |
