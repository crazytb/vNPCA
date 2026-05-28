# Figure 9: qsrc Optimization Under Native NPCA Contention

**연구 질문 (RQ9)**: NPCA 채널을 primary로 사용하는 native STA들이 존재할 때,  
optimal qsrc*는 어떻게 달라지는가? 그리고 qsrc 최적화의 중요성은 얼마나 커지는가?

**스크립트**: `harq_sim/run_step9_fig9.py`

**출력**: `manuscript/figure/fig9_native_npca.{eps,png,pdf}`

---

## 실험 동기

현재 Fig 3·4 모델에서 NPCA 채널은 항상 비어 있다 (`obss_generation_rate=0.0`).  
NPCA 채널의 경쟁자는 primary 채널 OBSS를 피해 전환한 STA들뿐 — 전환이 시작되는 순간 모두 동시에 도착하는 "동기적 경쟁"이다.

그러나 실제 802.11bn 환경에서는 **다른 BSS가 NPCA 주파수를 자신의 primary 채널로 운용**할 수 있다.  
이 경우 전환 STA들은 NPCA 채널에 이미 상주 중인 native STA들과 CSMA/CA 경쟁을 해야 한다.

```
Primary CH (ch 0):           OBSS ████████████████████
                             ↓ (trigger)
Transitioning STAs (N_t):   ←── NPCA 채널로 전환 ──→
                                  ↕ CSMA/CA contention
Native STAs (N_n):          항상 NPCA CH에서 경쟁 중
```

**Fig 3와의 차이점**:
| 측면 | Fig 3 | Fig 9 |
|------|-------|-------|
| NPCA 경쟁자 | 전환 STA들만 (동시 도착) | 전환 STA (N_t) + native STA (N_n) |
| native STA의 CW 상태 | 해당 없음 | 이미 진행 중인 backoff (비동기) |
| 유효 경쟁자 수 | N_t | N_t + f(N_n) — f ≤ 1 (비동기 특성 반영) |
| qsrc* 예상 | max(0, ⌊log₂(N_t/16)⌋) | **더 큰 qsrc* — 실험으로 정량화** |

---

## 실험 파라미터

| 항목 | 값 |
|------|-----|
| **Sweep 변수** | `npca_qsrc` ∈ {0,1,2,3,4,5} × `N_native` ∈ {0, 5, 10, 20, 30} |
| 전환 STA 수 (고정) | `N_transition = 20` |
| OBSS 채널 점유율 | 30% (`obss_rate ≈ 0.0039`) |
| `obss_min` / `obss_max` | 20 / 200 슬롯 |
| `snr_db_mean` | 20.0 dB (고 SNR, 결정론적) |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

> **고 SNR 사용 이유**: PHY 실패를 제거하여 CW 충돌 효과만 관찰. Fig 3와 동일한 조건으로 비교 가능.

> **N_transition=20 고정 이유**: Fig 3 v3에서 N=20이 qsrc*=1로 막 이동하는 경계점.  
> native STA 추가 시 qsrc*가 추가로 이동하는지 관찰하기에 적합.

**Native STA 설정**:
- `primary_channel = npca_ch` (channel_id=1을 primary로 사용)
- `npca_enabled = False` (채널 전환 없이 항상 NPCA 채널에 상주)
- `npca_initial_qsrc = 0` (표준 CW_min=15 사용 — D1.2 기본값)
- `ap_on_primary = True` (native STA의 AP는 자신의 primary=NPCA 채널에 존재)

---

## 비교 대상

| 기법 | N_transition | N_native | npca_qsrc |
|------|-------------|---------|-----------|
| `trans_only` (baseline) | 20 | 0 | sweep (0~5) |
| `with_native_5` | 20 | 5 | sweep (0~5) |
| `with_native_10` | 20 | 10 | sweep (0~5) |
| `with_native_20` | 20 | 20 | sweep (0~5) |
| `with_native_30` | 20 | 30 | sweep (0~5) |

---

## 측정 지표

### 전환 STA 지표 (primary metrics)
- `trans_throughput`: 전환 STA들의 NPCA 채널 전달 패킷 수 / 에피소드
- `trans_collision_prob`: 전환 STA들의 NPCA 채널 충돌 확률
- `trans_npca_transition_count`: NPCA 전환 횟수

### Native STA 지표 (secondary metrics)
- `native_throughput`: native STA들의 전달 패킷 수
- `native_collision_prob`: native STA들의 충돌 확률

### 통합 지표
- `total_npca_throughput = trans_throughput + native_throughput`
- `fairness_index = (trans_throughput/N_t) / (native_throughput/N_n)` — per-STA 공정성

---

## Figure 구성 (3-panel)

```
Figure 9
  Panel (a): Throughput of transitioning STAs vs qsrc
             X축: npca_qsrc ∈ {0,1,2,3,4,5}
             Y축: trans_throughput (packets/episode)
             선: N_native ∈ {0, 5, 10, 20, 30}
             ★ 마커: 각 선의 최대값 (qsrc*)
             → 핵심 질문 1: qsrc*가 N_native 증가에 따라 오른쪽으로 이동하는가?

  Panel (b): Optimal qsrc* vs N_native
             X축: N_native ∈ {0, 5, 10, 20, 30}
             Y축: qsrc* (0~5)
             점선 참조선: Fig 3 공식 qsrc*(N_t + N_n) — 단순 합산 예측
             실선: 실제 측정된 qsrc*(N_native | N_t=20)
             → 핵심 질문 2: qsrc*가 N_t + N_n의 단순 합산 공식으로 예측 가능한가?
               비동기 native STA의 유효 기여도 α = qsrc*(actual) / qsrc*(sum) 정량화

  Panel (c): Throughput gain (qsrc=qsrc* vs qsrc=0) vs N_native
             X축: N_native
             Y축: 처리량 이득 (%)
             → 핵심 질문 3: native STA가 많을수록 qsrc 최적화의 중요성이 커지는가?
             전환 STA 이득 (파란선) + native STA 피해/이득 (주황선) 동시 표시
```

---

## 핵심 가설

### 가설 1: qsrc*는 N_native에 따라 단조 증가
- Native STA의 비동기 backoff가 전환 STA에게 추가 경쟁자로 작용
- N_native=30 추가 → 유효 경쟁자 증가 → qsrc* 상향 이동

### 가설 2: qsrc*(N_t, N_n) < qsrc*(N_t + N_n)
- Native STA는 **이미 진행 중인 backoff**를 가짐 → 전환 STA 도착 시 일부는 곧 접근 완료
- 실질 경쟁 강도는 "N_t + N_n명이 동시 도착"보다 약함
- 유효 기여도 α: `qsrc*(N_t + N_n) 공식보다 1 작은 값`이 실제 optimal

### 가설 3: qsrc 최적화 이득이 N_native에 따라 증가
- native STA가 없을 때: N=20, gain ≈ 1.5% (Fig 3 v3 결과)
- native STA 30명 추가: 유효 경쟁자 ≈ 50 수준 → gain ≈ 6% 수준으로 증가

---

## 구현 필요 사항

### 변경 파일: `harq_sim/simulator.py`

**문제**: 현재 충돌 해소 코드에서 TX 요청 경로가 `ChannelType` enum 기반으로 하드코딩됨.
```python
# 현재 (buggy for native NPCA STAs):
ch_id = self._primary_ch_id if req.channel_type == ChannelType.PRIMARY else self._npca_ch_id
```

**수정**: STA의 실제 `primary_channel.channel_id`를 기반으로 라우팅.
```python
# 수정 후:
if req.channel_type == ChannelType.PRIMARY:
    ch_id = sta.primary_channel.channel_id   # native NPCA STA면 1, 일반 STA면 0
else:
    ch_id = self._npca_ch_id                 # 전환 중인 STA의 NPCA TX는 항상 1
```

이 변경은 **하위 호환성을 유지**함:
- 기존 전환 STA (primary_channel.channel_id=0): 기존과 동일하게 동작
- native NPCA STA (primary_channel.channel_id=1): PRIMARY TX가 올바르게 ch_id=1로 라우팅

**AP-absence 체크**: native STA는 `ap_on_primary=True` → AP-absence 실패 없음. 추가 수정 불필요.

### 새 파일: `harq_sim/run_step9_fig9.py`

```python
# 환경 구성 예시
primary_ch = Channel(channel_id=0, obss_generation_rate=OBSS_RATE, ...)
npca_ch    = Channel(channel_id=1, obss_generation_rate=0.0)

# 전환 STA (N_t개)
trans_stas = [
    STA(sta_id=i,
        primary_channel=primary_ch,
        npca_channel=npca_ch,
        npca_enabled=True,
        npca_initial_qsrc=qsrc,  ← sweep 변수
        ...)
    for i in range(N_TRANSITION)
]

# Native NPCA STA (N_n개)
native_stas = [
    STA(sta_id=N_TRANSITION + i,
        primary_channel=npca_ch,  ← NPCA가 primary
        npca_channel=None,
        npca_enabled=False,       ← 채널 전환 없음
        npca_initial_qsrc=0,      ← 표준 CW
        ap_on_primary=True,
        ...)
    for i in range(N_NATIVE)
]

sim = Simulator(
    num_slots=NUM_SLOTS,
    stas=trans_stas + native_stas,
    channels=[primary_ch, npca_ch],
)
```

### CSV 출력 스키마

```
qsrc, N_native, seed,
trans_throughput, trans_collision_prob, trans_npca_transition_count,
native_throughput, native_collision_prob,
total_npca_throughput, fairness_index
```

---

## 실험 환경 설정 근거

### OBSS 파라미터: Fig 3 v3와 동일하게 유지
- OBSS_OCCUPANCY=0.30, OBSS_MAX=200 → N=20에서 qsrc*=1 (경계점)
- Native STA 추가로 qsrc*=2까지 이동하는지 관찰 가능

### N_native 범위: {0, 5, 10, 20, 30}
- 0: baseline (Fig 3와 비교용)
- 5~10: 소규모 native BSS (인접 AP + 소수 단말)
- 20~30: 중규모 native BSS (office 환경)
- N_n=30 → 유효 경쟁자 ≈ 50 수준: Fig 3에서 qsrc*=2인 지점

### N_transition=20 고정
- Fig 3 v3에서 qsrc*=1 경계점 — 변화가 가장 뚜렷하게 관찰 가능
- 추후 N_t ∈ {5,10,20,30}으로 확장 실험 가능 (Panel(b)의 다중 선)

---

## Fig 3와의 관계 및 논문 포지셔닝

```
Fig 3 (단일 BSS):  qsrc*(N_t)        = max(0, ⌊log₂(N_t/16)⌋)
Fig 9 (다중 BSS):  qsrc*(N_t, N_n)  = max(0, ⌊log₂((N_t + α·N_n)/16)⌋)
```

실험으로 `α` (native STA의 유효 기여도)를 추정:
- α = 1.0: native STA가 전환 STA와 동등하게 기여 (동기 모델 예측)
- α < 1.0: 비동기 backoff로 인해 실제 경쟁 강도가 더 낮음

> **논문 기여 문구 (예시)**:  
> "We show that the optimal qsrc* in the presence of N_n native NPCA STAs follows  
> qsrc*(N_t, N_n) ≈ max(0, ⌊log₂((N_t + α N_n)/16)⌋) where α ≈ 0.6–0.8,  
> indicating that native STAs contribute less than equivalently-sized transitioning groups  
> due to the asynchronous nature of their backoff state."

---

## 실험 결과 v1 (results/step9/fig9/, N_t=20, 50000슬롯 × 3 seeds)

### Trans STA Throughput (패킷 전달량, mean ± std)

| N_n | q=0 | q=1 | q=2 | q=3 | q=4 | q=5 |
|-----|-----|-----|-----|-----|-----|-----|
| 0  | 1285±29 | **1285**±26 | 1238±21 | 1197±31 | 1135±14 | 1041±26 |
| 5  | **1189**±23 | 1162±10 | 1088±29 | 1023±23 | 956±36 | 928±24 |
| 10 | **1165**±22 | 1161±37 | 1071±9  | 979±10  | 963±15 | 905±22 |
| 20 | **1136**±16 | 1095±27 | 1060±7  | 969±22  | 912±7  | 869±44 |
| 30 | **1134**±31 | 1094±11 | 1008±23 | 937±32  | 906±49 | 881±45 |

### Native STA Throughput (패킷 전달량, mean)

| N_n | q=0 | q=1 | q=2 | q=3 | q=4 | q=5 |
|-----|-----|-----|-----|-----|-----|-----|
| 5  | 883 | 933 | 1006 | 1076 | 1164 | 1203 |
| 10 | 910 | 984 | 1048 | 1135 | 1176 | 1223 |
| 20 | 948 | 984 | 1096 | 1120 | 1178 | 1201 |
| 30 | 929 | 975 | 1070 | 1122 | 1194 | 1202 |

### NPCA Collision Probability (trans STAs)

| N_n | q=0 | q=1 | q=2 | q=3 | q=4 | q=5 |
|-----|-----|-----|-----|-----|-----|-----|
| 0  | 0.770 | 0.515 | 0.297 | 0.167 | 0.064 | 0.038 |
| 5  | 0.796 | 0.595 | 0.404 | 0.315 | 0.267 | 0.233 |
| 10 | 0.812 | 0.617 | 0.435 | 0.384 | 0.343 | 0.280 |
| 20 | 0.825 | 0.653 | 0.518 | 0.443 | 0.380 | 0.415 |
| 30 | 0.842 | 0.667 | 0.566 | 0.500 | 0.407 | 0.416 |

### qsrc* 요약

| N_n | qsrc* (실측) | qsrc*(N_t+N_n) 공식 | Trans gain% |
|-----|------------|-------------------|------------|
| 0  | 1 (≈0, 동점) | 0 | ≈0% |
| 5  | 0 | 1 | 0% |
| 10 | 0 | 1 | 0% |
| 20 | 0 | 1 | 0% |
| 30 | 0 | 2 | 0% |

---

## 핵심 관찰 (v1 결과)

### 1. 가설 1 반증: qsrc*는 N_native 증가에 따라 감소 (증가 아님)

예측: N_native 증가 → 경쟁자 증가 → qsrc* 상향  
실측: N_native=0 → qsrc*≈1; N_native≥5 → qsrc*=0

**원인**: Native STA는 qsrc=0(CW=15)으로 항상 경쟁. Trans STA가 qsrc를 높이면(더 큰 CW),
native STA에게 채널 접근 우선권을 양보하는 효과. Native STA는 이 빈 시간을 채워
trans STA의 실질 NPCA 활용 기회가 오히려 줄어든다.

### 2. Trans/Native 처리량 역전 관계

qsrc 증가 시:
- Trans STA 처리량: **단조 감소** (qsrc↑ → trans TP↓)
- Native STA 처리량: **단조 증가** (qsrc↑ → native TP↑)

qsrc는 Trans vs. Native 간 처리량 **공정성 파라미터**로 재해석 가능.
- qsrc=0: Trans 유리, Native 불리
- qsrc=5: Native가 Trans와 거의 동등한 처리량 확보

### 3. 충돌 확률은 증가하지만 처리량 최적화 방향이 반대

N_native=30, q=0: collision 0.842 (높음)이지만 그래도 trans TP가 최대.
충돌이 많아도 trans STA 관점에서는 "더 많이 시도"가 "적게 충돌"보다 낫다.
이는 NPCA 창 길이(평균 110슬롯)가 짧아 qsrc=0으로도 충분히 접근 가능하기 때문.

### 4. 논문 포지셔닝 재정의

원래 가설 (qsrc*가 N_n과 함께 증가)은 틀렸다. 실제 결과는 더 흥미롭다:
- Native STA의 존재가 qsrc* 최적화 문제를 **역전**시킴
- 동기적 전환 STA 간 경쟁(trans-only): qsrc 증가가 균형적으로 collision 감소에 기여
- 비동기 native STA 추가 시: native STA가 trans STA의 CW 증가 "혜택"을 가로챔

> **논문 수정 기여 문구 (예시)**:  
> "When native NPCA STAs (using default CW_min) coexist with transitioning STAs,  
> the optimal qsrc* for transitioning STAs reverts to 0 regardless of N_native,  
> as native STAs' constant aggressive contention (CW_min=15) captures the channel  
> capacity freed by a higher qsrc. This reveals that qsrc* optimization benefits  
> transitioning STAs primarily in isolated NPCA channel scenarios."

---

## 수정 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-05-28 | 초안 작성. native NPCA STA 환경에서의 qsrc 최적화 실험 설계. |
| 2026-05-28 | 구현 완료. simulator.py TX 라우팅 수정 (sta.primary_channel.channel_id 사용, req.channel_type으로 실패 dispatch). run_step9_fig9.py 신규 작성. 정식 실험 완료 (results/step9/fig9/). v1 결과: qsrc*가 N_native 증가에 따라 감소(0으로 수렴)하는 반직관적 결과 — native STA의 qsrc=0 경쟁이 trans STA의 CW 증가 이득을 흡수. |
