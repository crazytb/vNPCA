# Figure 7: LLM vs Grid-best vs Hand-crafted Reward

**연구 질문 (RQ7)**: LLM-designed reward profile은 hand-crafted reward 또는 grid-best reward에 비해 어느 정도 성능을 내는가?

**스크립트**: `harq_sim/run_step9_fig7.py`

**출력**: `manuscript/figure/fig7_*.{eps,png,pdf}`

---

## 배경

현재 아키텍처에서 reward profile은 **post-hoc 평가 도구** — 동일한 시뮬레이션 결과를  
어떤 intent 관점으로 보느냐를 결정한다.

| 방법 | Profile 결정 방법 |
|---|---|
| Hand-crafted | `INTENT_PROFILES`의 5개 profile 중 intent와 이름 가장 유사한 것 선택 |
| LLM-designed | LLM이 intent 문자열로부터 weight vector 생성 (mock 모드) |
| Grid-best | 모든 profile 순회 → reward 최대인 profile 선택 |

## 실험 파라미터

| 항목 | 값 |
|---|---|
| **고정 시뮬레이션** | `fixed_cw_npca_harq` (아래 파라미터) |
| `num_stas` | 5 |
| `obss_rate` | 0.30 |
| `snr_db_mean` | 14.0 dB |
| `snr_db_std` | 2.0 dB |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

## Intent 조건 (5가지)

```python
INTENTS = [
    "maximize throughput",
    "minimize delay for XR gaming applications",
    "balanced throughput and latency",
    "maximize energy efficiency and reduce power consumption",
    "ensure fair coexistence with legacy EDCA stations",
]
```

## 측정 지표

- `reward` 값 (각 방법 기준)
- `aggregate_throughput` (method-independent)
- `mean_access_delay` (method-independent)
- `profile_name` (어떤 profile이 선택됐는지)

## Figure 구성

```
Figure 7: 2-panel layout
  Panel A (주): grouped bar chart
    X 그룹: 5개 intent
    X 내 막대: 3개 방법 (hand-crafted / LLM / grid-best)
    Y: reward 값
    (grid-best는 항상 최고값 — upper bound 역할)

  Panel B (검증): intent별 선택된 profile name 히트맵
    행: 5개 intent
    열: 3개 방법
    셀: profile name (throughput / delay_sensitive / qos_aware / fair_coexistence / energy_aware)
    색상: 일치 여부 (LLM = grid-best → 초록, 불일치 → 빨강)
```

## 핵심 분석 질문

1. LLM이 선택한 profile이 grid-best와 얼마나 일치하는가?
2. LLM이 intent를 올바른 profile로 매핑하는가? (특히 energy, delay intent)
3. Hand-crafted "가장 유사한 이름" 전략이 얼마나 suboptimal인가?

## Mock LLM Profile 매핑 참조

```python
# harq_sim/llm_reward_designer.py (mock 모드)
# intent 문자열 → profile 매핑 확인 필요
# mock은 결정론적 — seed 불필요
```

## 출력 파일

```
manuscript/figure/
  fig7_llm_vs_gridvscraft.eps
  fig7_llm_vs_gridvscraft.png
  fig7_llm_vs_gridvscraft.pdf

results/step9/fig7/
  data.csv            ← (intent, method, profile_chosen, reward, seed)
```

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-25 | 초안 작성 |
