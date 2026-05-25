# vNPCA LLM-DRL 프로젝트

## 프로젝트 개요

IEEE 802.11bn **vNPCA(virtualized Non-Primary Channel Access)** 저널 논문용 시뮬레이션 코드.
JCCI2026 단편논문(`manuscript/ref/JCCI2026-vNPCA_final.pdf`)의 구조를 저널로 확장하며,
LLM을 **Reward Designer** 역할로 사용하는 **Two-Timescale LLM-DRL** 프레임워크를 구현한다.

참고 논문: `manuscript/ref/Large_Language_Model-Enabled_Reinforcement_Learning_for_Wireless_Network_Optimization.pdf`

---

## 아키텍처: Two-Timescale LLM-DRL

```
느린 시간 단위 (매 LLM_UPDATE_INTERVAL=50 에피소드마다)
  AP 측 LLMRewardDesigner (llm_reward_designer.py)
    입력: 자연어 user intent + 최근 50 에피소드 네트워크 통계
    Claude API 호출 → {throughput_weight, latency_penalty, npca_switch_bonus, npca_switch_cost, qos_priority}
    → 모든 NPCA STA의 보상 파라미터 업데이트 + intent_code 브로드캐스트

빠른 시간 단위 (매 슬롯 ~9μs)
  STA 측 SemiMDPLearner (DQN)
    OBSS 감지 시 action: 0=stay primary, 1=switch NPCA
    보상 = throughput_weight × 성공TX슬롯
          - latency_penalty × 옵션지속시간
          + npca_switch_bonus × (action==1)
          - npca_switch_cost  × (action==1)
    상태 = [obss_remain, radio_transition_time, tx_duration, cw_index, intent_code]
```

vNPCA 구조 매핑:
- LLMRewardDesigner ↔ AP의 NPCA Policy Manager (제어 평면)
- 설계된 reward params ↔ beacon 브로드캐스트로 전파되는 정책 파라미터
- STA DRL 에이전트 ↔ vNPCA 모듈 (데이터 평면)

---

## 핵심 설계 결정 및 이론적 근거

### 1. LLM contribution: 왜 규칙 기반으로 대체 불가능한가

단순 네트워크 통계 기반 reward 조정은 `if OBSS > threshold: weight += 0.1` 수준으로 대체 가능.
**LLM이 필요한 진짜 이유**: 자연어로 표현된 이질적 사용자 의도(intent)를 reward 파라미터로 변환하는 것은
가능한 모든 요청 조합을 수동으로 프로그래밍하지 않고는 규칙 기반으로 일반화 불가능.

동일한 네트워크 조건에서 intent만 달라도 전혀 다른 파라미터가 나와야 함:
```
"화상회의, 지연 최소화"     → tw=0.8,  lp=0.15, sb=0.02, sc=0.02, qos=latency
"파일 다운로드, 속도 최대화" → tw=2.2,  lp=0.02, sb=0.25, sc=0.02, qos=throughput
"배터리 절약, 저전력"       → tw=1.0,  lp=0.08, sb=0.02, sc=0.18, qos=energy
"일반 웹 서핑"             → tw=1.2,  lp=0.08, sb=0.08, sc=0.05, qos=balanced
```

`npca_switch_cost`의 역할: NPCA 전환 시 소모되는 radio transition energy를 패널티로 반영.
`npca_switch_bonus - npca_switch_cost` = 순 전환 인센티브.
energy intent에서는 sc >> sb → 불필요한 NPCA 전환 억제 → 긴 OBSS일 때만 전환이 유리.

### 2. Reward 정규화 원칙

LLM이 reward 파라미터를 바꾸면 reward scale이 intent마다 달라짐 → **reward 직접 비교 금지**.

비교 지표는 intent-independent 지표로 분리:
- `throughput` (channel_occupancy_time): 처리량 우선 intent 검증
- `avg_option_duration` (tau 평균): 지연 우선 intent 검증
- `npca_switch_ratio`: 행동 변화 검증 (energy intent → 낮음, throughput intent → 높음)

DRL 학습 안정성: 현재 `normalized_R = R / num_slots_per_episode` 로 scale 바운딩.
policy 자체(argmax Q)는 scale-invariant이므로 최적 행동 순서는 유지됨.

### 3. Contextual MDP (CMDP) 공식화

Two-Timescale 구조가 CMDP를 자연스럽게 유도함:

- LLM 파라미터는 K=50 에피소드 동안 **불변** → 해당 구간에서 MDP가 stationary
- context c = intent로부터 도출된 {tw, lp, sb, qos_priority} 묶음
- MDP(c) = (S, A, R_c, P, γ) — 전이 확률 P는 불변, 보상 R_c만 context에 의존
- DRL policy: **π(a | s, c)** — context를 조건으로 하는 단일 policy

```
context c (K 에피소드 동안 고정)
  ├── Reward:  R(s, a; c) = tw(c)×tx - lp(c)×tau + sb(c)×switch
  └── State:   obs = [network_features..., intent_code(c)]
```

intent_code가 reward와 state 양쪽에 포함되는 이유:
- reward에만: 에이전트가 reward 변화 원인을 모름 → 학습 불안정
- state에만: reward가 안 바뀌면 intent 정보가 무의미
- **양쪽 포함**: 에이전트가 context를 보고, context에 맞는 reward를 받으며 일관된 학습 ✓

### 4. Zero-shot adaptation (훈련 후 LLM 불필요)

훈련 중 다양한 context를 경험한 policy는 배포 시 LLM 재호출 없이 intent_code만으로 즉시 대응:

| 단계 | LLM 역할 | DRL 역할 |
|------|----------|----------|
| **훈련** | intent → reward params 설계 + intent_code 생성 | MDP(c) 위에서 π(a\|s,c) 학습 |
| **배포** | 불필요 (intent_code 직접 지정) | intent_code 조건부 최적 행동 즉시 출력 |

논문 기여 문구:
> "The two-timescale structure induces a Contextual MDP where LLM-designed reward parameters serve as a slowly-varying context. The conditioned policy π(a|s,c) enables zero-shot adaptation to intent changes at deployment without LLM requery."

---

---

## Figure 생성 규칙 (Step 9+)

### 저장 포맷 — 필수 3종 동시 저장

모든 Figure 생성 스크립트는 반드시 아래 3가지 포맷을 `manuscript/figure/` 에 저장해야 한다:

```python
# 각 run_step9_figN.py 내부 공통 패턴
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "manuscript", "figure")
os.makedirs(FIG_DIR, exist_ok=True)

fig.savefig(os.path.join(FIG_DIR, f"{fig_name}.eps"), format="eps", bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, f"{fig_name}.png"), format="png", dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, f"{fig_name}.pdf"), format="pdf", bbox_inches="tight")
```

| 포맷 | 용도 | 해상도 |
|---|---|---|
| `.eps` | LaTeX (`\includegraphics`) 삽입 — 벡터 | — |
| `.png` | 미리보기, Slack/이메일 공유 | 300 dpi |
| `.pdf` | 고품질 벡터, PDF 논문 버전 | — |

### Figure → 스크립트 → 계획 문서 매핑

인덱스: `guidelines/step9_index.md`  
상세 계획: `guidelines/step9/figN.md`  
스크립트: `harq_sim/run_step9_figN.py`  
출력 데이터: `results/step9/figN/data.csv`

Figure를 수정해야 할 때는 해당 `guidelines/step9/figN.md`의 **수정 이력** 섹션에 변경 내용을 기록한다.

---

## 파일 구조

```
vNPCA/
├── CLAUDE.md                        ← 이 파일
├── .venv/                           ← 가상환경 (torch, pandas, anthropic 설치됨)
├── llm_reward_designer.py           ← LLMRewardDesigner 클래스
├── main_llm_vnpca_training.py       ← 학습 진입점 (5종 비교 실험)
├── main_semi_mdp_training.py        ← 기존 학습 진입점 (하위 호환 유지)
├── npca_semi_mdp_env.py             ← Gymnasium 환경 래퍼
├── experiments/
│   ├── exp1_convergence.py          ← Exp1: 수렴 분석 (training only)
│   ├── exp2_heterogeneous.py        ← Exp2: Heterogeneous 디바이스 (training only)
│   ├── exp_intent_eval.py           ← Per-Intent Train+Eval (단일 에이전트 + 에너지)
│   └── exp_channel_conditions.py   ← [신규] 채널 조건 스윕 (primary+NPCA OBSS 조합)
└── drl_framework/
    ├── configs.py                   ← [수정] LLM 설정 + 에너지 모델 상수
    ├── random_access.py             ← [수정] STA 보상 파라미터 + context_vec + 에너지 추적
    ├── train.py                     ← [수정] LLM 통합, evaluate_policy(), run_baseline_simulation()
    ├── network.py                   ← DQN 네트워크 (미수정)
    └── params.py                    ← 하이퍼파라미터 (미수정)
```

---

## 주요 수정 사항

### `drl_framework/random_access.py`
- `STA.__init__`에 `throughput_weight=1.0`, `latency_penalty=0.05` 파라미터 추가
- `_end_option()`에서 하드코딩 제거 → `self.throughput_weight`, `self.latency_penalty` 사용
- `STA.update_reward_params(throughput_weight, latency_penalty, context_vec=)` 메서드 추가
- `STA.episode_energy_uJ` 필드 추가 — per-slot 에너지 누적 (IEEE 802.11ax 기반)
  - `step()`: Listen 상태(backoff/frozen) → 0.495 μJ/slot, TX 상태 → 2.772 μJ/slot
  - `_begin_option()`: action=1(NPCA 전환) 시 +0.75 μJ (TX↔Listen 전환 이벤트)

### `drl_framework/train.py`
- `train_semi_mdp(... llm_designer=None)` — LLM 통합, 하위 호환 유지
- `run_baseline_simulation(... fixed_action_fn)` — DRL 없는 고정 전략 시뮬레이션
- `collect_network_stats(...)` — LLM 입력용 네트워크 통계 수집
- **`evaluate_policy(learner, channels, stas_config, ...)` (신규)**:
  - 훈련 완료 후 greedy evaluation (ε=0, gradient 없음)
  - `learner.eval_mode=True` + `learner.memory=None` 으로 pure greedy 실행
  - 반환: DataFrame[episode, throughput, avg_option_duration, total_energy_uJ, npca_switch_ratio]
- `SemiMDPLearner.select_action()`: `eval_mode=True`일 때 순수 greedy (탐색 없음)

### `drl_framework/configs.py`
```python
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_UPDATE_INTERVAL = 50
LLM_USE_MOCK = False

# 에너지 모델 (IEEE 802.11ax TG, 11-14-0980-16-00ax, 20MHz, 1.1V, NSS=1)
SLOT_DURATION_US = 9.0
ENERGY_TX_PER_SLOT_UJ = 2.772      # 280mA × 1.1V × 9μs
ENERGY_LISTEN_PER_SLOT_UJ = 0.495  # 50mA × 1.1V × 9μs (backoff/frozen/CCA)
ENERGY_NPCA_TRANSITION_UJ = 0.75   # TX↔Listen @ 75mW × 0.01ms per 전환 이벤트
```

### `llm_reward_designer.py` (신규)
- `LLMRewardDesigner(model, update_interval, use_mock)` 클래스
- Anthropic SDK + `cache_control: {"type": "ephemeral"}` 으로 system prompt 캐싱
- `use_mock=True`: API 호출 없이 결정론적 mock 반환 (테스트용)
- `ANTHROPIC_API_KEY` 환경변수 필요 (mock=False 시)

### `main_llm_vnpca_training.py` (신규)
5종 비교 실험 함수:
- `run_always_npca_experiment()` — fixed_action=1 항상
- `run_never_npca_experiment()` — fixed_action=0 항상
- `run_rule_based_experiment()` — obss_remain >= ppdu_duration이면 전환
- `run_fixed_experiment()` — DRL, 고정 보상 파라미터
- `run_llm_experiment()` — DRL + LLM 적응형 보상 (제안 기법)

---

## 비교 지표 설계 원칙

**보상값(reward) 직접 비교 금지**: LLM-DRL은 보상 파라미터가 주기적으로 변하므로
방법 간 reward 값 비교는 스케일이 달라 무의미함.

**기본 비교 지표**: `episode_throughputs` = ch1 STA의 성공 TX 슬롯 수/에피소드
(보상 파라미터와 무관한 실제 네트워크 성능)

`plot_comparison()` 3-패널 구성:
1. 처리량 학습 곡선 (5종, running average)
2. 최종 평균 처리량 bar chart
3. LLM-DRL 처리량 + LLM 파라미터 변화 오버레이

---

## 실행 방법

```bash
# 가상환경 활성화
source .venv/bin/activate   # 또는 .venv/bin/python 직접 사용

# Mock 모드 (API 키 불필요) — 빠른 테스트
.venv/bin/python main_llm_vnpca_training.py --mode compare --mock --episodes 50 --slots 1000

# 전체 5종 비교 (mock)
.venv/bin/python main_llm_vnpca_training.py --mode compare --mock

# 실제 Claude API 사용
ANTHROPIC_API_KEY=sk-ant-... .venv/bin/python main_llm_vnpca_training.py --mode compare

# OBSS 조건 스윕 (논문 핵심 실험)
.venv/bin/python main_llm_vnpca_training.py --mode full --mock

# 기존 코드 (하위 호환)
.venv/bin/python main_semi_mdp_training.py
```

CLI 옵션:
```
--mode      llm | fixed | compare | full
--mock      LLM API 미사용 (테스트용)
--obss-duration  OBSS 지속 시간 슬롯 수 (기본 100)
--episodes  학습 에피소드 수 (기본 1000)
--slots     에피소드당 슬롯 수 (기본 ~11111)
--ppdu      short | medium | long | extra_long
--results-dir  결과 저장 경로 (기본 ./results/llm_vnpca)
```

---

## 핵심 코드 위치

| 기능 | 파일:위치 |
|------|----------|
| OBSS 감지 → DRL 결정 포인트 | `drl_framework/random_access.py` `STA._handle_primary_backoff()` |
| 고정 전략 훅 | `drl_framework/random_access.py` `STA._fixed_action` (외부 주입) |
| 보상 계산 | `drl_framework/random_access.py` `STA._end_option()` |
| LLM 호출 주기 | `drl_framework/train.py` `train_semi_mdp()` — `episode % update_interval == 0` |
| 네트워크 통계 수집 | `drl_framework/train.py` `collect_network_stats()` |
| LLM 프롬프트 | `llm_reward_designer.py` `_SYSTEM_PROMPT` |
| Rule-based 전략 | `main_llm_vnpca_training.py` `run_rule_based_experiment()` |
| context_vec 생성 | `llm_reward_designer.py` `get_context_vec()` — 정규화된 (tw,lp,sb,sc) |
| context_vec → state | `drl_framework/random_access.py` `STA.get_obs()` / `obs_to_vec()` |
| context_vec 갱신 | `drl_framework/train.py` `train_semi_mdp()` — LLM 업데이트 후 `sta.update_reward_params(..., context_vec=)` |
| 에너지 per-slot 누적 | `drl_framework/random_access.py` `STA.step()` — `_LISTEN_STATES` / `_TX_STATES` |
| NPCA 전환 에너지 이벤트 | `drl_framework/random_access.py` `STA._begin_option()` — action=1 시 +ENERGY_NPCA_TRANSITION_UJ |
| Greedy evaluation | `drl_framework/train.py` `evaluate_policy()` — ε=0, memory=None |
| Single-agent 환경 설정 | `experiments/exp_intent_eval.py` `_make_channels_stas()` — ch0×4 + ch1×1(focal) |
| Baseline eval (고정 전략) | `experiments/exp_intent_eval.py` `_evaluate_baseline()` |
| 채널 조건 스윕 환경 설정 | `experiments/exp_channel_conditions.py` `_make_channels_stas()` — per-channel OBSS rate 독립 설정 |

---

## 실험 관찰 목표 (5가지 시나리오)

각 실험은 `experiments/exp{N}_*.py` 에 독립 파일로 구현. 모든 실험은 충분한 CSV 로그를 저장하여 향후 그래프 재편집 가능하도록 설계.

| # | 관찰 목표 | 구현 파일 | 핵심 지표 |
|---|-----------|-----------|-----------|
| 1 | LLM-DRL이 고정 policy 대비 잘 수렴하는가 | `exp1_convergence.py` | per-episode throughput, loss, epsilon, NPCA ratio |
| 2 | Heterogeneous device 요구조건(throughput vs latency)이 reward에 잘 반영되는가 | `exp2_heterogeneous.py` | per-group throughput/latency, LLM qos_priority 변화 |
| 3 | 특정 device의 요구조건이 다른 device에 악영향을 끼치지 않는가 | `exp3_fairness.py` | per-STA throughput, Jain's fairness index |
| 4 | 충분히 긴 horizon에서 QoS 요구사항 변화 시 LLM이 적응하는가 | `exp4_dynamic_qos.py` | QoS 전환 이벤트 타임라인, 적응 latency |
| 5 | Device intent가 LLM을 통해 올바르게 출력되는가 | `exp5_intent.py` | reasoning 텍스트, qos_priority 분류 정확도 |

### Exp1 로그 스키마 (CSV)
```
episode, method, run, throughput, reward, avg_loss, epsilon,
npca_ratio, throughput_weight, latency_penalty, npca_switch_bonus, qos_priority
```
(비-LLM 기법은 throughput_weight 등 NaN)

### Exp2 로그 스키마
```
episode, group (latency_sensitive|throughput_hungry), throughput,
avg_option_duration, npca_ratio, throughput_weight, latency_penalty, qos_priority
```

### Exp4 로그 스키마
```
episode, phase (phase 명), active_intent, throughput,
avg_option_duration, throughput_weight, latency_penalty, npca_switch_bonus, qos_priority
```

---

## 에너지 모델 (IEEE 802.11ax 기반)

참고 문서: `11-14-0980-16-00ax-simulation-scenarios.docx` (TGax Simulation Scenarios)

| 상태 | 전류 | 전력 (1.1V) | 에너지/슬롯 (9μs) |
|------|------|------------|-------------------|
| Transmit | 280 mA | 308 mW | **2.772 μJ** |
| Listen (CCA/backoff) | 50 mA | 55 mW | **0.495 μJ** |
| NPCA 전환 이벤트 | TX↔Listen @ 75mW × 0.01ms | — | **0.75 μJ** (per event) |

시뮬레이션 상태 매핑:
- `PRIMARY_BACKOFF`, `PRIMARY_FROZEN`, `NPCA_BACKOFF`, `NPCA_FROZEN` → Listen (0.495 μJ/slot)
- `PRIMARY_TX`, `NPCA_TX` → Transmit (2.772 μJ/slot)
- NPCA switch (action=1, `_begin_option` 호출 시) → +0.75 μJ (단발 이벤트)

**Energy sanity 검증**: always_npca(16,040 μJ) > never_npca(13,414 μJ) — NPCA 전환이 에너지를 추가로 소모함 ✅

---

## 신규 실험: Per-Intent Train+Eval (`experiments/exp_intent_eval.py`)

### 설계 동기
기존 Exp1/Exp2는 **training phase 측정만** 있어 논문 주장 검증에 부족:
1. 학습 중 ε-greedy 탐색이 섞여 있어 "수렴한 policy의 성능"을 직접 측정 불가
2. energy consumption 지표 없음 — energy_saving intent 효과 증명 불가
3. Multi-agent 공유 Q-network → single intent에 특화된 π(a|s) 학습

### Single-Agent 환경 구조
```
ch0: 4 background STAs (OBSS 채널 contenders, npca_enabled=False)
ch1: 1 focal STA (단독 learner, npca_enabled=True)
```
ch1 background 없음 → intra-BSS 경쟁 없음 → focal STA 결정이 OBSS에만 의존

### 실험 흐름
```
intents: latency_sensitive, throughput_hungry, energy_saving, balanced, fixed_drl
baselines: always_npca, never_npca, rule_based

for each method:
  [Training Phase]  train_semi_mdp() — 3000 ep, LLM 업데이트 every 50 ep
  [Evaluation Phase] evaluate_policy() — 200 ep, ε=0 greedy
  → 저장: {method}_train_run{N}.csv, {method}_eval_run{N}.csv
```

### 3 KPI (Evaluation Phase 기준)
| KPI | 지표 | 기대 방향 |
|-----|------|-----------|
| 처리량 | `throughput` (TX slots/ep) | throughput_hungry ≥ fixed_drl |
| 지연 | `avg_option_duration` τ (slots) | latency_sensitive ≤ fixed_drl |
| 에너지 | `total_energy_uJ` (μJ/ep) | energy_saving ≤ fixed_drl |

### 실행 방법
```bash
# Mock 테스트 (API 불필요, 빠름)
.venv/bin/python experiments/exp_intent_eval.py --mock --train-episodes 50 --eval-episodes 20

# 전체 실험 (3000 ep × 5 runs, 실제 Claude API)
set -a && source .env && set +a
.venv/bin/python experiments/exp_intent_eval.py \
    --train-episodes 3000 --eval-episodes 200 --runs 5 --results-dir ./results/exp_eval_v1
```

CLI 옵션:
```
--mock            LLM API 미사용 (mock 모드)
--train-episodes  학습 에피소드 수 (기본 3000)
--eval-episodes   평가 에피소드 수 (기본 200)
--runs            반복 횟수 (기본 5)
--obss-rate       OBSS 발생률 (기본 0.05)
--obss-min/max    OBSS 지속시간 범위 슬롯 (기본 20-200)
--results-dir     결과 저장 경로
```

출력 파일:
```
results/exp_eval_v1/
├── all_eval.csv               ← 전체 eval 결과 (method별 병합)
├── all_train.csv              ← 전체 training curves
├── {method}_train_run{N}.csv  ← per-method training log
├── {method}_eval_run{N}.csv   ← per-method eval results
├── {method}_llm_history_run{N}.json  ← LLM 호출 이력
├── eval_kpi_comparison.png    ← 3-panel bar chart (TP/τ/E)
└── training_curves.png        ← DRL methods 학습 곡선
```

---

## 채널 조건 스윕 실험 (`experiments/exp_channel_conditions.py`)

### 설계 동기
exp_intent_eval은 NPCA 채널이 항상 비어있다고 가정. 실제 환경에서는 NPCA 채널도 외부 점유(external occupancy)가 있을 수 있음. 이때 intent별 전략이 어떻게 달라지는지 관찰.

**핵심 설계 원칙**: NPCA 채널 상태는 STA의 관측 state에 포함하지 않음.
- Primary 채널에 있는 STA는 NPCA 채널을 CCA로 직접 감지 불가 (IEEE 802.11 물리적 제약)
- NPCA 채널 점유는 환경의 숨겨진 동역학 — 전환 후 결과(NPCA_FROZEN)를 통해 간접 학습
- state는 8-dim 그대로 유지 (변경 없음)

### 3 시나리오

| 시나리오 | Primary OBSS | NPCA occupancy | 특성 |
|---------|-------------|---------------|------|
| `free_npca` | 5% | 0% | NPCA 항상 사용 가능 (baseline) |
| `light_npca` | 5% | 3% | 간헐적 NPCA 충돌 |
| `congested_npca` | 8% | 5% | 양쪽 채널 혼잡 |

OBSS duration range: 20~200 slots (두 채널 동일)

### 실행 방법
```bash
# Mock 테스트
.venv/bin/python experiments/exp_channel_conditions.py \
    --mock --train-episodes 10 --eval-episodes 5 --runs 1

# 전체 실험 (3000 ep × 3 runs, 실제 Claude API)
set -a && source .env && set +a
.venv/bin/python experiments/exp_channel_conditions.py \
    --train-episodes 3000 --eval-episodes 200 --runs 3 \
    --results-dir ./results/exp_channel_v1
```

출력 파일:
```
results/exp_channel_v1/
├── all_eval.csv
├── all_train.csv
├── {free_npca,light_npca,congested_npca}/   ← 시나리오별 서브디렉토리
│   └── {method}_{train,eval}_run{N}.csv
├── kpi_by_scenario.png        ← 3×3 그리드 (KPI행 × 시나리오열)
├── npca_ratio_across_scenarios.png  ← method별 NPCA 전환율 추이
└── training_curves.png        ← 시나리오별 DRL 학습 곡선
```

### 실험 결과 (results/exp_channel_v1/, 3000ep×3runs, 실제 Claude API)

| 시나리오 | 가설 | 결과 | 유의성 |
|---------|------|------|--------|
| free_npca | TP hungry ≥ fixed_drl | ✅ +1.2% | p=0.016 ✅ |
| free_npca | τ latency ≤ fixed_drl | ✅ -0.7% | p=0.070 ⚠️ n.s. |
| free_npca | E saving ≤ fixed_drl | ✅ -3.8% | p=0.000 ✅ |
| light_npca | TP hungry ≥ fixed_drl | ❌ -4.5% | p=0.000 ✅ |
| light_npca | τ latency ≤ fixed_drl | ❌ +17.1% | p=0.000 ✅ |
| light_npca | E saving ≤ fixed_drl | ✅ -2.8% | p=0.000 ✅ |
| congested_npca | TP hungry ≥ fixed_drl | ✅ +1.0% | p=0.175 ⚠️ n.s. |
| congested_npca | τ latency ≤ fixed_drl | ✅ -4.0% | p=0.002 ✅ |
| congested_npca | E saving ≤ fixed_drl | ✅ -3.9% | p=0.000 ✅ |

**`light_npca` 역전 현상 분석**:
- latency/throughput intent의 NPCA ratio가 0.7(fixed) → 0.3으로 급감
- 원인: NPCA 충돌로 reward 감소 → LLM이 switch_cost 증가 → 에이전트 과보수적 전환 억제
- 결과: OBSS가 있는 primary에서 대기 → τ·TP 모두 악화
- 해석: **partially observable 환경에서 LLM이 NPCA 충돌 비용을 전환 행동 자체에 잘못 귀속**
  → 논문에서 한계(limitation)이자 흥미로운 발견으로 기술 가능

**`energy_saving`이 3개 시나리오 모두에서 일관된 에너지 절감** (유일하게 모든 시나리오 가설 통과):
- sc >> sb 구조가 전환 비용을 모든 상황에서 일관되게 패널티로 반영하기 때문

---

## IEEE 802.11bn D1.2 표준 부합도 분석

참고 문서: `manuscript/ref/Draft P802.11bn_D1.2_NPCA.pdf` (§37.18, Sep 2025)

### D1.2 핵심 NPCA 스펙 (§37.18)

| 항목 | D1.2 규정 |
|------|----------|
| 전환 트리거 조건 1 | inter-BSS PPDU + NPCA_PPDU_REM_DUR ≥ Min Duration Threshold + intra-BSS NAV = 0 |
| 전환 트리거 조건 2 | RTS/CTS 3-PPDU 시퀀스 감지 |
| 전환 후 채널 접근 | EDCA 경쟁 (동일 파라미터, AIFSN=0) |
| NPCA 운용 시간 | NPCA_TIMER = NPCA_PPDU_REM_DUR - switch_back_delay |
| 복귀 조건 | NPCA_TIMER 만료 |
| CW 관리 | 전환 시 저장, 복귀 시 복원 |
| 전송량 | NPCA_TIMER 동안 TXOP 기반 다수 MPDU 연속 전송 |
| 파라미터 협상 | OMP Request/Response (switching delay, switch back delay) |
| AP 역할 | Beacon으로 NPCA 파라미터 공고, Min Duration Threshold 설정 |
| 최소 대역폭 | BSS ≥ 80 MHz 필요 |

### 참조 논문 (Wi-Fi 8 Unveiled) vs D1.2: 부합도 ~70%

✅ RTS/CTS 트리거 (조건 2) — §37.18.3 Condition 2와 정확히 일치  
✅ TXOP 기반 다수 MPDU 전송 — full buffer TXOP 전송  
✅ BSS ≥ 80 MHz (bss1이 160 MHz 운용)  
✅ EDCA 경쟁 on NPCA channel  
⚠️ Min Duration Threshold — 명시적 NPCA_PPDU_REM_DUR 변수 미모델링  
⚠️ CW 저장/복원 — 언급하나 절차 생략  
❌ OMP 파라미터 협상 미구현  
❌ MU EDCA 상호작용 (§37.18.2) 미구현  
❌ PHYLEN / MOPLEN 모드 구분 없음  

### 내 시뮬레이션 vs D1.2: 부합도 ~45%

✅ OBSS 감지 → NPCA 결정 트리거 (조건 1) — `primary_channel.obss_remain` = NPCA_PPDU_REM_DUR  
✅ Min Duration Threshold — `rule_based`의 `obss_remain ≥ ppdu_duration` = §37.18.3.1.c.i와 동일 논리  
✅ Switching delay — `radio_transition_time = 1 slot` ≈ NPCA switching delay  
✅ NPCA_TIMER 만료 시 복귀 — `obss_remain == 0` 조건  
✅ Exponential backoff (CW)  
❌ TXOP 다수 MPDU 전송 — PPDU 1개만 전송 (핵심 누락, 논문 대비 처리량 차이의 주원인)  
❌ RTS/CTS 트리거 (조건 2) 미구현  
❌ CW 저장/복원 미구현 (복귀 시 리셋)  
❌ EDCA on NPCA channel 없음 (단일 STA)  
❌ AP 역할 없음 (LLM이 대신 파라미터 설계)  
❌ intra-BSS NAV 체크 없음  

**논문 포지셔닝**: 내 시뮬레이션은 표준의 "데이터 평면 전송"보다 **"전환 결정 로직(switching policy)"**을 충실히 추상화. LLM이 intent별 최적 NPCA Min Duration Threshold를 설계한다는 관점으로 기여를 정의하면 D1.2와의 괴리가 정당화됨.

---

## NPCA 전환 Trade-off 설계 원칙

### 핵심 trade-off: OBSS duration vs 전환 비용

NPCA 전환이 유리한 조건:
```
OBSS 잔여 시간 > switching_delay + PPDU_duration
(NPCA_PPDU_REM_DUR > NPCA Min Duration Threshold)
```

- OBSS가 짧으면(< threshold): 전환해 봤자 OBSS가 곧 끝나 primary로 돌아와야 함 → 전환 비용만 낭비
- OBSS가 길면(≥ threshold): NPCA에서 PPDU 완전 전송 후 복귀 가능 → 이득

### OBSS rate 증가가 연구를 단순화하는 이유

Primary 채널이 항상 점유 → action=1이 항상 최적 → DRL이 trivially "항상 전환"을 학습.  
의사결정이 단순해져 intent별 차별화가 불가능해짐.

**올바른 방향: OBSS duration 분포를 조정** (rate가 아니라)

| 방법 | 효과 |
|------|------|
| Bimodal duration (10~30 + 100~300 혼합) | 전환 손해/이득 케이스 균형, threshold 결정의 영향 극대화 |
| PPDU duration 증가 (50~80 slots) | threshold 상승 → 더 많은 OBSS가 "전환 불리" 범위에 포함 |
| Switching delay 증가 (3~5 slots) | 전환 비용 증가 → 더 긴 OBSS일 때만 전환 유리 |

현재 설정(`OBSS_RANGE = (20, 200)`, `ppdu_duration = 33`)에서:
- 전환 손해 구간: 20~33 slots = 전체의 **15%**만 → 대부분 전환이 옳아 trade-off 약함

### LLM 기여의 재정의: intent별 최적 threshold 설계

D1.2에서 AP가 고정값으로 공고하는 NPCA Min Duration Threshold를,  
**LLM이 user intent에 따라 동적으로 설계**하는 것이 핵심 novelty:

```
energy_saving  → 높은 threshold (긴 OBSS일 때만 전환) → 전환 횟수 ↓ → 에너지 ↓
throughput     → 낮은 threshold (중간 OBSS도 전환)   → 전송 기회 ↑
latency        → threshold ≈ ppdu_duration           → 짧은 option duration 유지
```

OBSS duration 변동성이 클수록 threshold 선택의 영향이 커지고 intent별 KPI 차이가 뚜렷해짐.

---

## 구현 이력: context_vec (state 8차원화)

### 변경 배경
초기 구현에서 LLM 보상 파라미터는 reward에만 반영되고 state에는 포함되지 않았음.
이 경우 동일한 state에서 LLM 업데이트마다 reward가 달라지므로 에이전트가 non-stationary MDP를 경험 → CMDP 이론과 불일치.

- **one-hot encoding 검토 후 기각**: 동일 intent라도 LLM이 네트워크 상태에 따라 다른 파라미터를 출력할 수 있으므로 intent 레이블보다 실제 파라미터 값 자체가 더 정확한 context.
- **실제 reward 파라미터를 정규화해 state에 포함**: `context_vec = [tw_norm, lp_norm, sb_norm, sc_norm] ∈ [0,1]⁴`

### 변경 내용 (2026-05-21)
| 파일 | 변경 내용 |
|------|----------|
| `llm_reward_designer.py` | `INTENT_CODES`, `_CTX_NORM` 추가; `get_context_vec()` 메서드 추가 |
| `drl_framework/random_access.py` | `STA.context_vec` 필드 추가; `get_obs()` / `obs_to_vec()` 8차원화; `update_reward_params()` `context_vec` 파라미터 추가 |
| `drl_framework/train.py` | `n_observations = 4 → 8`; LLM 업데이트 후 `sta.update_reward_params(..., context_vec=ctx_vec)` |

### 정규화 기준
```python
tw_norm = (throughput_weight - 0.5) / 2.5   # bounds [0.5, 3.0]
lp_norm = (latency_penalty - 0.01) / 0.19   # bounds [0.01, 0.2]
sb_norm = npca_switch_bonus / 0.5            # bounds [0.0, 0.5]
sc_norm = npca_switch_cost / 0.3             # bounds [0.0, 0.3]
# fixed_drl 기본값 → context_vec = [0.20, 0.21, 0.0, 0.0]
```

---

## 실험 결과 요약

### Exp1: 수렴 분석 (results/exp1_v2/)
**조건**: OBSS duration=100 슬롯, medium PPDU, 1000 에피소드 × 3 runs, 실제 Claude API (claude-haiku-4-5), intent 없음(balanced)

**목적 달성 여부**: 조건부 달성

| 지표 | 결과 | 판정 |
|------|------|------|
| DRL 수렴 여부 | epsilon 0.9→0.05, loss 94~95% 감소 | ✅ 정상 수렴 |
| LLM-DRL 최종 처리량 | **17,126 슬롯** (5방법 중 1위) | ✅ |
| Fixed DRL 대비 처리량 차이 | +185 슬롯 (+1.1%) | ✅ |
| t-test 유의성 (episode-level) | p=0.018 (유의미, α=0.05) | ✅ |
| t-test 유의성 (per-run, n=3) | p=0.062 (유의미하지 않음) | ⚠️ runs 부족 |
| 수렴 안정성 (CV) | 전 방법 CV<0.06 | ✅ |
| LLM 적응 동작 | ep700 이후 Fixed DRL 지속 역전 | ✅ |

**핵심 관찰**:
- LLM-DRL의 NPCA switch ratio가 일관되게 높음 (ep999: 0.62 vs Fixed DRL 0.50) — LLM이 NPCA 활용을 유도함
- ep700부터 LLM-DRL이 Fixed DRL을 안정적으로 추월 (+72~+170 슬롯) — LLM 파라미터 누적 적응 효과
- OverloadedError 발생 (57회 중 14회 실패) → retry 로직 추가 완료 (5s/15s/30s backoff)
- API 비용: 60회 호출 기준 $0.04 (전체 5개 실험 ~$0.60 예상)

**한계 및 해석**:
- intent 없는 "balanced" 조건에서는 LLM 효과가 미미 (+1.1%) — intent를 명시하면 효과가 뚜렷해질 것
- runs=3으로 per-run t-test 검정력 부족 → runs=5 이상 권장
- 통계적으로는 "LLM-DRL이 Fixed DRL보다 나쁘지 않음"은 확실히 보임
- **Exp2(heterogeneous intent)에서 LLM 기여의 핵심 차별점이 드러날 것으로 예상**

### Exp2: Heterogeneous 디바이스 (results/exp2_v2/, results/exp2_v3/)

#### v1/v2 (4-dim state, obss=100 고정, 10 STAs)
**문제**: intent 간 throughput/τ 차이가 미미했던 원인
1. OBSS 발생 빈도 낮음 (occupancy ≈ 3%) → NPCA 결정이 전체 TP의 3%에만 영향
2. 고정 OBSS duration=100 슬롯 → intent별 최적 action이 동일
3. 10 STAs aggregate metric이 개별 결정 희석

#### v2 개선 (obss_rate=0.05, obss_range=20~200, num_stas=4)
**결과** (results/exp2_v2/, 1000ep×3runs, 4-dim state):

| Group | AvgTP | τ | NPCA% | 수렴 파라미터 |
|---|---|---|---|---|
| latency_sensitive | 10,369 | 433.9 | 32.0% | lp=0.200(max), net=-0.256 |
| throughput_hungry | 10,287 | 431.1 | 41.0% | tw=3.000(max), net=+0.443 |
| energy_saving | 10,355 | 425.1 | 41.9% | sc=0.300(max), net=-0.293 |
| fixed_drl | 10,357 | 423.9 | 40.0% | — |

**핵심 관찰**:
- qos_priority 분류 정확도 **100%** (2,850 에피소드 오분류 0건) ✅
- latency_sensitive NPCA ratio: 초반 0.474 → 최종 **0.316** (-33% 감소) ✅
- 가설 검증: latency τ ≤ fixed_drl ✅, throughput TP ≥ fixed_drl ✅, energy NPCA ≤ fixed_drl ⚠️
- energy_saving NPCA ratio가 예상보다 높음 (0.419 vs fixed 0.412): switch_cost=0.3이지만 NPCA TX 보상이 더 커서 여전히 전환이 유리한 경우 많음

#### v3 (8-dim state = context_vec 포함, results/exp2_v3/)
**결과** (1000ep×3runs):

| Group | AvgTP | τ | NPCA% | 수렴 파라미터 |
|---|---|---|---|---|
| latency_sensitive | 10,351 | 430.1 | 32.8% | lp=0.200, net=-0.210 |
| throughput_hungry | 10,418 | 427.8 | 37.2% | tw=3.000, net=+0.443 |
| energy_saving | 10,387 | 423.8 | 43.0% | sc=0.300, net=-0.293 |
| fixed_drl | 10,377 | 439.6 | **27.9%** | context_vec=[0.20,0.21,0,0] |

**핵심 관찰**:
- latency_sensitive NPCA ratio: pre-LLM 0.460 → **final 0.353** — context_vec이 state에 포함되면서 파라미터 수렴과 함께 행동도 수렴 ✅
- throughput_hungry NPCA ratio > latency_sensitive ✅ (0.372 vs 0.328)
- fixed_drl NPCA ratio 대폭 감소 (0.412→0.279): 기본 context_vec [0.2,0.21,0,0]이 "낮은 보상 환경" 신호로 작용 → baseline 역할 변질 ⚠️
- **1,000 에피소드 부족**: context 변화마다 Q-network 입력이 달라져 수렴 지연 발생

### Exp1 재실험: 8-dim state (results/exp1_v3/)
**조건**: obss=100, 1000ep×3runs, 8-dim state

| | LLM-DRL | Fixed DRL | 차이 |
|---|---|---|---|
| 4-dim (v2) | 17,126 | 16,941 | **+185** ✅ |
| 8-dim (v3) | 16,953 | 16,985 | **-33** ⚠️ |

**해석**: context_vec이 state에 추가됨으로써 Q-network이 수렴하기 위한 경험이 더 많이 필요. 1,000 에피소드로는 8-dim 환경에서 LLM-DRL의 우위가 드러나지 않음.

---

## 구조적 한계 및 향후 방향

### 현재 실험 방식의 한계 (intent별 별도 훈련)
각 Exp2 group이 독립적으로 훈련되므로, 실제로 학습된 것은 **단일 intent에 특화된 π(a|s)** — 진정한 CMDP의 **π(a|s,c)** 가 아님. Zero-shot adaptation을 증명하려면:

1. **단일 훈련에서 여러 intent를 순환**: 매 K 에피소드마다 intent를 랜덤 교체하며 훈련 → 에이전트가 context_vec 변화에 반응하는 일반 정책 학습
2. **에피소드 수 증가** (1,000 → 3,000+): 8-dim state에서 수렴하기 위한 충분한 샘플
3. **배포 시 검증**: 훈련 후 intent_code(context_vec)만 바꿔서 즉시 다른 행동이 나오는지 확인

---

## 향후 작업 (논문)

- [x] Exp1: 수렴 분석 구현 및 실행 (results/exp1_v2, 4-dim, 1000ep×3runs)
- [x] Exp1 재실험: 8-dim state (results/exp1_v3, 1000ep×3runs)
- [x] Exp2: Heterogeneous 디바이스 구현 (experiments/exp2_heterogeneous.py)
- [x] Exp2 v2 실행: obss_rate=0.05, range=20-200, 4-STAs, 4-dim (results/exp2_v2)
- [x] Exp2 v3 실행: 동일 조건, 8-dim state (results/exp2_v3)
- [x] context_vec (8-dim state) 구현: random_access.py / train.py / llm_reward_designer.py
- [x] **에너지 모델 구현** (IEEE 802.11ax 기반): STA.episode_energy_uJ 추적 (configs.py / random_access.py)
- [x] **evaluate_policy() 구현** (train.py): greedy eval, ε=0, gradient 없음
- [x] **exp_intent_eval.py 구현**: single-agent + train+eval + 3 KPI (TP/τ/E)
- [x] **exp_channel_conditions.py 구현**: 채널 조건 스윕 (3 scenarios × 8 methods)
- [x] **exp_channel_v1 실행 완료**: 3000ep×3runs, 실제 Claude API (results/exp_channel_v1/)
- [x] **IEEE 802.11bn D1.2 표준 부합도 분석**: 논문 ~70%, 내 시뮬레이션 ~45%
- [ ] **exp_intent_eval 전체 실험 실행** (3000ep×5runs, 실제 Claude API, results/exp_eval_v1)
- [ ] **OBSS duration bimodal 실험**: OBSS rate 유지, duration 분포를 bimodal(short+long)로 변경 → trade-off 강화
- [ ] **Exp2 재설계**: 단일 훈련에서 intent 순환 → 진정한 π(a|s,c) 학습 검증
- [ ] Exp3: Fairness 실험 구현 및 실행
- [ ] Exp4: Dynamic QoS 적응 실험 구현 및 실행
- [ ] Exp5: Intent 캡처 정확도 실험 구현 및 실행
- [ ] 논문 작성: 시뮬레이션 결과 섹션
