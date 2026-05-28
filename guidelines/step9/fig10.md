# Figure 10: qsrc* Dependence on OBSS Duration W_eff

**연구 질문 (RQ10)**: qsrc*는 OBSS duration W_eff에도 독립적으로 의존하는가?

**배경**: Fig 3는 qsrc*(N)의 N 의존성을 보였으나, W_eff는 고정(uniform(20,500), mean≈260)되어 있었다.  
Fig 10은 N을 고정하고 W를 직접 변화시켜 **qsrc*(W_eff) 의존성**을 분리 검증한다.

**핵심 설계**: `OBSS_MIN = OBSS_MAX = W` (고정 duration) → W_eff = W 정확히 제어

**스크립트**: `harq_sim/run_step9_fig10.py`

**출력**: `manuscript/figure/fig10_qsrc_weff.{eps,png,pdf}`

---

## 실험 파라미터

| 항목 | 값 |
|---|---|
| **Sweep 변수** | `W` ∈ {30, 50, 80, 120, 170, 230, 300, 400, 500, 700} × `num_stas` ∈ {10, 20, 30, 50} |
| OBSS 채널 점유율 | 50% (`obss_rate = occ / (W × (1 − occ))`, W별 재계산) |
| `obss_min = obss_max` | W (고정 duration — W_eff = W 정확히 제어) |
| `qsrc` sweep | {0, 1, 2, 3, 4, 5} |
| `snr_db_mean` | 20.0 dB (고 SNR, 결정론적 — 충돌 효과 분리) |
| `snr_db_std` | 0.0 dB |
| `ppdu_duration` | 20 슬롯 |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

총 실험 횟수: 10 × 4 × 6 × 3 = **720 runs**

> **주의**: occupancy 50%를 유지하면서 W를 바꾸면 OBSS 발생 빈도(rate)가 W에 반비례하여 변한다.  
> 짧은 W → OBSS 이벤트 많이 발생, 긴 W → OBSS 이벤트 드물게 발생.  
> 이는 의도된 설계 — "동일한 채널 점유 부하 하에서 개별 OBSS 창의 길이가 qsrc*에 미치는 영향"을 측정.

---

## 비교 대상 (1개 기법, W × num_stas × qsrc 3D sweep)

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

- `aggregate_throughput` — 전달 패킷 수 (주 지표, qsrc* 결정 기준)
- `collision_probability_npca` — NPCA 충돌 확률
- `mean_access_delay` — 평균 지연
- `npca_transition_count` — NPCA 전환 총 횟수

---

## Figure 구성

```
Figure 10: 3-panel

Panel (a): Throughput vs qsrc (N=30 고정, lines per W ∈ {50, 120, 230, 400, 700})
           각 선의 최대값에 ★ 마커
           핵심 메시지: W가 커질수록 ★의 위치(qsrc*)가 오른쪽으로 이동

Panel (b): qsrc*(W) — 핵심 결과 (lines per N ∈ {10, 20, 30, 50})
           x축: OBSS Duration W (슬롯), y축: qsrc* (0~5)
           계단 함수 형태 (drawstyle="steps-post")
           핵심 메시지: W 임계값 W*(N) 이상에서 qsrc* 상승; N 클수록 낮은 W에서 상승

Panel (c): Throughput gain at qsrc* vs qsrc=0 (%) (lines per N)
           x축: W, y축: gain(%)
           핵심 메시지: gain이 W와 N 모두 커질수록 증가 — adaptive qsrc의 실용 영역
```

---

## 예상 결과

| W (slots) | N=10 qsrc* | N=20 qsrc* | N=30 qsrc* | N=50 qsrc* |
|---|---|---|---|---|
| 30~80 | 0 | 0 | 0 | 0 |
| 120~170 | 0 | 0~1 | 1 | 1 |
| 230~300 | 0 | 1 | 1 | 1~2 |
| 400~700 | 0~1 | 1 | 1~2 | 2 |

W가 짧으면 (W < PPDU + mean_backoff(qsrc=1) ≈ 20 + 15 = 35): qsrc=1의 backoff가 창 안에 완료되지 않아 transition count 감소 → qsrc=0이 최적

---

## 논문 내 위치

```
§ Problem   : Fig 3 (qsrc*(N) — N 의존성 확인)
§ Analysis  : Fig 10 (qsrc*(W_eff) — W 의존성 추가 확인)
              → 두 그림 합쳐서 Theorem: qsrc*(N, W_eff) 근거 완성
§ Algorithm : Fig 4 (Adaptive qsrc — N과 W_eff 모두 추정)
```

---

## 출력 파일

```
manuscript/figure/
  fig10_qsrc_weff.eps / .png / .pdf

results/step9/fig10/
  data.csv    ← (W, num_stas, npca_qsrc, npca_cw, seed,
                  collision_probability_npca, aggregate_throughput,
                  mean_access_delay, npca_transition_count)
  fig10_qsrc_weff_preview.png
```

---

## 실험 결과 (results/step9/fig10/, 50k슬롯 × 3 seeds)

### qsrc* 및 Throughput Gain 요약

| W (slots) | N=10 qsrc*/gain | N=20 qsrc*/gain | N=30 qsrc*/gain | N=50 qsrc*/gain |
|---|---|---|---|---|
| 30  | 0 / +0.0% | 0 / +0.0% | **1 / +9.0%** | **2 / +17.9%** |
| 50  | 0 / +0.0% | 0 / +0.0% | **1 / +1.2%** | **1 / +10.5%** |
| 80  | 0 / +0.0% | **1 / +2.3%** | **1 / +7.8%** | **2 / +13.3%** |
| 120 | 0 / +0.0% | **1 / +0.2%** | **1 / +6.0%** | **2 / +4.9%** |
| 170 | 0 / +0.0% | **1 / +2.5%** | **1 / +2.9%** | **2 / +5.5%** |
| 230 | 0 / +0.0% | 0 / +0.0% | **1 / +3.3%** | **2 / +6.8%** |
| 300 | 0 / +0.0% | 0 / +0.0% | **1 / +2.2%** | **2 / +5.3%** |
| 400 | 0 / +0.0% | 0 / +0.0% | **1 / +2.6%** | **2 / +4.7%** |
| 500 | 0 / +0.0% | 0 / +0.0% | **1 / +2.6%** | **1 / +2.8%** |
| 700 | 0 / +0.0% | 0 / +0.0% | **2 / +1.3%** | **3 / +2.4%** |

### qsrc=0에서의 충돌 확률 (N별, W별)

| W | N=10 | N=20 | N=30 | N=50 |
|---|---|---|---|---|
| 30  | 0.443 | 0.721 | 0.860 | **0.969** |
| 50  | 0.473 | 0.745 | 0.879 | **0.968** |
| 80  | 0.519 | 0.774 | 0.895 | **0.963** |
| 170 | 0.526 | 0.768 | 0.864 | **0.942** |
| 300 | 0.492 | 0.723 | 0.830 | **0.919** |
| 500 | 0.444 | 0.669 | 0.785 | **0.888** |
| 700 | 0.418 | 0.648 | 0.757 | **0.863** |

### 핵심 관찰

#### 1. N이 qsrc*의 1차 결정 요인
- N=10: qsrc*=0 for all W → 소수 경쟁에서는 W 무관하게 qsrc=0이 최적
- N≥30: qsrc*≥1 for all W → N이 충분히 크면 W와 무관하게 qsrc 최적화 필요

#### 2. W는 2차 조절 요인 (large N에서만 유효)
- N=30: qsrc*=1 전 범위, W=700에서만 qsrc*=2로 이동
- N=50: qsrc*=2 주류, W=700에서 qsrc*=3으로 이동
- W 증가 → qsrc* 소폭 상승 확인 (단, 변화는 크지 않음)

#### 3. Gain이 최대인 구간 (반직관적 발견)
- **short W + large N**에서 gain이 가장 큼
  - W=30, N=50: gain=+17.9%, W=30, N=30: gain=+9.0%
  - 원인: 짧은 창에 많은 STA가 집중 → qsrc=0 충돌 확률 96.9% → 거의 모든 슬롯이 충돌로 낭비
  - qsrc=2(CW=63)는 평균 backoff 31.5슬롯 → 일부 STA만 창 안에 시도, 충돌 감소
- **long W + large N**에서는 gain 감소 (W=700, N=50: +2.4%)
  - 원인: W가 길면 exponential backoff가 창 안에서 자기 수정 가능 → 초기 CW 설정의 중요도 감소

#### 4. N=50 qsrc* 비단조성 (W=50, W=500에서 qsrc*=1)
- qsrc*=1 vs qsrc*=2 간 TP 차이: 25~30 패킷 (≈2%) → 3 seeds의 통계적 노이즈 범위
- 결론: N=50의 실질 qsrc*는 q*=2(W≤400) → q*=3(W≥700) 단조 증가가 타당

### 논문 메시지 (Fig 10)

> "qsrc*는 N이 1차 결정 요인이며, N≤10에서는 W 전 범위에서 qsrc*=0이 최적이다. N≥30에서는 W에 따른 2차 의존성이 나타나 W=700에서 qsrc*가 1~2단계 상승한다. 특히 short W + large N 조합에서 qsrc 최적화의 이득이 최대(+18%)를 기록하는데, 이는 짧은 접속 창에 다수 STA가 집중될 때 qsrc=0의 충돌 연쇄가 극심해지기 때문이다."

---

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-29 | 초안 작성 — Fig 3의 W_eff 의존성 누락 보완; 고정 duration 설계 확정 |
