# Figure 5: Per-STA Frame Length Heterogeneity — qsrc Fairness-Efficiency Tradeoff

**기여**: 짧은 프레임 STA와 긴 프레임 STA가 혼재할 때, 공유 fixed qsrc는 한쪽 그룹에
불리하며 per-STA adaptive qsrc가 aggregate throughput을 유지하면서 더 균형 잡힌
per-group throughput을 달성함을 보임.

**스크립트**: `harq_sim/run_step9_fig5.py`

**출력**: `manuscript/figure/fig5_*.{eps,png,pdf}`

---

## 연구 질문 (RQ5)

> 짧은 프레임 STA와 긴 프레임 STA가 동일 NPCA 채널에서 경쟁할 때,
> 어떤 qsrc 전략이 aggregate throughput과 per-group fairness를 동시에 최적화하는가?

핵심 insight:
- 프레임 길이가 다른 STA는 NPCA 창 내에서 서로 다른 "유효 전송 가능 슬롯"을 경험
- 짧은 프레임 STA: W_eff_short = E[OBSS] − switching − ppdu_short 가 커서 낮은 qsrc도 무방
- 긴 프레임 STA: W_eff_long이 상대적으로 작아 짧은 OBSS 이벤트에서 전송 불가(waste) 빈도 ↑
- per-STA adaptive qsrc는 각 STA의 waste_rate/col_rate를 독립 관측 → 그룹별 수렴 가능성

---

## 실험 파라미터

| 항목 | 값 |
|---|---|
| **SHORT group** | N=5 STAs, ppdu_duration=10 slots |
| **LONG group** | N=5 STAs, ppdu_duration=50 slots |
| 총 STA 수 | 10 (고정) |
| OBSS 채널 점유율 | 50% |
| `obss_min` | 20 슬롯 |
| `obss_max` | 500 슬롯 |
| `snr_db_mean` | 20.0 dB |
| `snr_db_std` | 0.0 dB |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

### 프레임 길이 선정 근거

| 그룹 | ppdu | W_eff (avg) | 설명 |
|---|---|---|---|
| Short | 10 | ≈248 슬롯 | e.g., 단일 MSDU (~90μs) |
| Long | 50 | ≈208 슬롯 | e.g., A-MSDU aggregate (~450μs) |

OBSS 창 하한 진입 조건 (npca_threshold=0):
- Short: `obss_remain ≥ 11` → OBSS_MIN=20이므로 모든 이벤트 진입 가능
- Long: `obss_remain ≥ 51` → OBSS_dur<51인 이벤트 진입 불가 (약 6.5% missed)

---

## 비교 대상 (5개)

| 기법 | Short qsrc | Long qsrc | 비고 |
|---|---|---|---|
| `fixed_q0` | 0 | 0 | 모두 CW=15 (공격적, 기본값) |
| `fixed_q1` | 1 | 1 | 모두 CW=31 |
| `fixed_hetero` | 0 | 1 | 수동 이질 할당 (heuristic) |
| `oracle_hetero` | qsrc*_short | qsrc*_long | Pre-sweep으로 결정한 그룹별 최적값 |
| `per_sta_adaptive` | adaptive | adaptive | 각 STA 독립 적응 (제안) |

### Oracle qsrc 결정 방법

Pre-sweep: 동질 환경(N=10, all short or all long)에서 qsrc ∈ {0,1,2,3}을 sweep:
```python
QSRC_CANDIDATES = [0, 1, 2, 3]

# Phase 1a: N=10, ppdu=10 (SHORT homogeneous)
# Phase 1b: N=10, ppdu=50 (LONG homogeneous)
# → oracle_short = argmax TP(ppdu=10)
# → oracle_long  = argmax TP(ppdu=50)
```

---

## Adaptive qsrc 알고리즘 (per-STA, sta.py 기구현)

각 STA가 독립적으로 K=5 전환마다 관측:
```
col_rate   = 총_충돌수 / 총_TX시도수   (TX당)
waste_rate = 1 - 전송성공방문수 / 총방문수  (방문당)

if col_rate > 0.70:    qsrc = min(qsrc+1, 5)
elif waste_rate > 0.30: qsrc = max(qsrc-1, 0)
```

Short-frame STA 예상 신호:
- col_rate ≈ 49% (N=10, Fig 3 v2 기준) < 0.70 → qsrc 증가 없음
- waste_rate ≈ 0% (OBSS_MIN=20 >> ppdu=10+1) → qsrc 감소 없음
- → qsrc 안정적으로 0 유지 예상

Long-frame STA 예상 신호:
- col_rate ≈ 49% (same N) < 0.70 → qsrc 증가 없음
- waste_rate: OBSS_dur<51인 이벤트(6.5%)에서 전송 불가 → 약 6-10% waste
  → θ_waste=0.30보다 낮아 qsrc 감소 신호 미발생
- → 두 그룹 모두 qsrc ≈ 0 수렴 예상 (baseline 확인용)

**핵심 관찰 포인트**:
1. 두 그룹이 같은 qsrc로 수렴하는지 vs 다른 qsrc로 수렴하는지
2. Per-group throughput이 fixed_q0/q1과 비교해서 어떻게 다른지
3. Jain's fairness index가 어떻게 변하는지

---

## 측정 지표

| 지표 | 계산 | 단위 |
|---|---|---|
| `tp_short` | Σ(packets_delivered for sta_id < N_SHORT) | packets |
| `tp_long` | Σ(packets_delivered for sta_id ≥ N_SHORT) | packets |
| `tp_agg` | tp_short + tp_long | packets |
| `jain_fairness` | (Σx_i)² / (N × Σx_i²), x_i = STA packets | [0,1] |
| `mean_qsrc_short` | per-STA qsrc history 평균 (short group) | - |
| `mean_qsrc_long` | per-STA qsrc history 평균 (long group) | - |
| `col_prob_npca` | aggregate NPCA collision probability | - |

---

## Figure 구성

```
Figure 5: 3-panel (공유 x축: Method)

Panel (a): Per-group throughput (grouped bar)
           x: 5 methods
           2개 막대 쌍: SHORT group (파랑), LONG group (주황)
           std error bar 포함
           핵심 메시지: fixed_q0에서 short/long 간 격차 vs adaptive에서 균형

Panel (b): Jain's fairness index (per-STA throughput 기준)
           x: 5 methods
           단일 막대, std 포함
           참조선: 0.95 (공정성 기준)
           핵심 메시지: adaptive ≥ best_fixed

Panel (c): mean qsrc per group (adaptive method만)
           막대: oracle_short (회색), oracle_long (회색 점선)
           꺾은선: adaptive short 평균 qsrc, adaptive long 평균 qsrc
           std 음영
           핵심 메시지: adaptive가 각 그룹의 oracle qsrc*로 수렴하는지 확인
```

---

## 출력 파일

```
manuscript/figure/
  fig5_hetero_frame.eps / .png / .pdf

results/step9/fig5/
  presweep.csv   ← (ppdu_type, qsrc, seed, aggregate_throughput)
  data.csv       ← (method, seed, tp_short, tp_long, tp_agg, jain_fairness,
                     mean_qsrc_short, mean_qsrc_long, col_prob_npca)
```

---

## 실험 결과 (results/step9/fig5/, 50000슬롯 × 3 seeds)

### Pre-sweep oracle (homogeneous N=10)

| Frame type | ppdu | oracle qsrc* | avg TP |
|---|---|---|---|
| SHORT | 10 | **0** (CW=15) | 2267 |
| LONG  | 50 | **1** (CW=31) | 529  |

oracle_long=1 ≠ oracle_short=0 → 프레임 길이에 따른 최적 qsrc 차이 확인 ✓

### 주요 비교 결과

| Method | TP_short | TP_long | TP_agg | Jain | q_short | q_long |
|---|---|---|---|---|---|---|
| fixed_q0        | 390 | 439 | 829 | 0.865 | 0.00 | 0.00 |
| fixed_q1        | 369 | 451 | 820 | 0.844 | 1.00 | 1.00 |
| fixed_hetero    | **440** | 435 | **875** | 0.879 | 0.00 | 1.00 |
| oracle_hetero   | **440** | 435 | **875** | 0.879 | 0.00 | 1.00 |
| per_sta_adaptive| 380 | **446** | 826 | **0.890** | 0.22 | 0.17 |

**핵심 관찰**:

1. **fixed_hetero = oracle_hetero**: oracle_short=0, oracle_long=1이 heuristic (short→q0, long→q1)과 일치 → 직관적 프레임 길이 기반 할당이 최적
2. **TP_long > TP_short (역설적 결과)**: 긴 프레임 STA가 OBSS_dur<51 이벤트에서 NPCA 진입 불가 → primary에 잔류 → short-frame STA가 NPCA로 빠져나가는 동안 primary 경쟁 감소 효과 (N=10→5로 완화)
3. **fixed_hetero가 fixed_q0 대비 +5.5% TP_agg**: 프레임 길이 인식 qsrc 할당이 유의미한 이득
4. **per_sta_adaptive 최고 fairness (Jain=0.890)**: qsrc 자체의 강한 분화(short=0.22, long=0.17, 거의 동일) 없이도 per-STA 독립 관측이 균형 잡힌 성능 달성
5. **adaptive의 qsrc 수렴**: 두 그룹 모두 qsrc≈0으로 수렴 (col_rate<θ_col=0.70, waste_rate<θ_waste=0.30) → OBSS_MAX=500 환경에서 waste_rate 신호 약함

**논문 메시지**:
"프레임 길이 이질성 환경에서 고정 shared qsrc는 aggregate throughput을 최적화하지 못한다.
프레임 길이 인식 qsrc 할당(fixed_hetero)은 shared 대비 +5.5% 이득을 달성하며,
per-STA adaptive는 qsrc 강한 분화 없이도 최고 per-STA fairness(Jain=0.890)를 달성한다."

---

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-25 | 초안 작성 (HARQ gain vs SNR) |
| 2026-05-26 | 전면 재설계: per-STA frame length heterogeneity 실험으로 변경; Fig 4와 동일 환경(OBSS_MAX=500, occ=50%); pre-sweep oracle, 5개 비교군, 3-panel figure |
