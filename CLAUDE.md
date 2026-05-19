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
    입력: 최근 50 에피소드 네트워크 통계
    Claude API 호출 → {throughput_weight, latency_penalty}
    → 모든 NPCA STA의 보상 파라미터 업데이트

빠른 시간 단위 (매 슬롯 ~9μs)
  STA 측 SemiMDPLearner (DQN)
    OBSS 감지 시 action: 0=stay primary, 1=switch NPCA
    보상 = throughput_weight × 성공TX슬롯 - latency_penalty × 옵션지속시간
```

vNPCA 구조 매핑:
- LLMRewardDesigner ↔ AP의 NPCA Policy Manager (제어 평면)
- 설계된 reward params ↔ beacon 브로드캐스트로 전파되는 정책 파라미터
- STA DRL 에이전트 ↔ vNPCA 모듈 (데이터 평면)

---

## 파일 구조

```
vNPCA/
├── CLAUDE.md                        ← 이 파일
├── .venv/                           ← 가상환경 (torch, pandas, anthropic 설치됨)
├── llm_reward_designer.py           ← [신규] LLMRewardDesigner 클래스
├── main_llm_vnpca_training.py       ← [신규] 학습 진입점 (5종 비교 실험)
├── main_semi_mdp_training.py        ← 기존 학습 진입점 (하위 호환 유지)
├── npca_semi_mdp_env.py             ← Gymnasium 환경 래퍼
└── drl_framework/
    ├── configs.py                   ← [수정] LLM 설정 상수 추가
    ├── random_access.py             ← [수정] STA에 throughput_weight/latency_penalty 추가
    ├── train.py                     ← [수정] LLM 통합, collect_network_stats(), run_baseline_simulation()
    ├── network.py                   ← DQN 네트워크 (미수정)
    └── params.py                    ← 하이퍼파라미터 (미수정)
```

---

## 주요 수정 사항

### `drl_framework/random_access.py`
- `STA.__init__`에 `throughput_weight=1.0`, `latency_penalty=0.05` 파라미터 추가
- `_end_option()`에서 하드코딩 제거 → `self.throughput_weight`, `self.latency_penalty` 사용
- `STA.update_reward_params(throughput_weight, latency_penalty)` 메서드 추가

### `drl_framework/train.py`
- `train_semi_mdp(... llm_designer=None)` — LLM 통합, 하위 호환 유지
- `run_baseline_simulation(... fixed_action_fn)` — DRL 없는 고정 전략 시뮬레이션
- `collect_network_stats(...)` — LLM 입력용 네트워크 통계 수집
- 두 함수 모두 `episode_throughputs` 수집:
  - `train_semi_mdp`: `learner.episode_throughputs`에 저장
  - `run_baseline_simulation`: `(episode_rewards, episode_throughputs)` 튜플 반환

### `drl_framework/configs.py`
```python
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_UPDATE_INTERVAL = 50
LLM_USE_MOCK = False
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

## 향후 작업 (논문)

- [ ] Exp1: 수렴 분석 실험 구현 및 실행
- [ ] Exp2: Heterogeneous 디바이스 실험 구현 및 실행
- [ ] Exp3: Fairness 실험 구현 및 실행
- [ ] Exp4: Dynamic QoS 적응 실험 구현 및 실행
- [ ] Exp5: Intent 캡처 정확도 실험 구현 및 실행
- [ ] 실제 Claude API로 전체 실험 실행
- [ ] 논문 작성: 시뮬레이션 결과 섹션
