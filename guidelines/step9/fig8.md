# Figure 8: Intent별 NPCA/HARQ Policy 차이

**연구 질문 (RQ8)**: delay-sensitive, QoS-aware, fairness-aware intent에서 선택되는 NPCA/HARQ policy가 어떻게 달라지는가?

**스크립트**: `harq_sim/run_step9_fig8.py`

**출력**: `manuscript/figure/fig8_*.{eps,png,pdf}`

---

## 배경

Intent에 따라 최적 시뮬레이션 파라미터가 달라진다:

| Intent | 권장 전략 |
|---|---|
| throughput | 낮은 qsrc (빠른 NPCA 접근), 낮은 threshold |
| delay_sensitive | 낮은 qsrc, 낮은 threshold (즉시 전환) |
| qos_aware | 중간 qsrc, 중간 threshold, adaptive CW |
| fair_coexistence | 높은 qsrc (충돌 완화), 높은 threshold (legacy 보호) |
| energy_aware | 높은 qsrc, 높은 threshold (불필요한 전환 억제) |

## 실험 파라미터

| 항목 | 값 |
|---|---|
| `num_stas` | 10 |
| `obss_rate` | 0.30 |
| `snr_db_mean` | 14.0 dB |
| `snr_db_std` | 2.0 dB |
| `obss_min` | 20 슬롯 |
| `obss_max` | 200 슬롯 |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

## Intent별 시뮬레이션 설정

```python
INTENT_CONFIGS = {
    "throughput":       dict(npca_qsrc=0, npca_threshold=0,  adaptive_cw=False),
    "delay_sensitive":  dict(npca_qsrc=0, npca_threshold=0,  adaptive_cw=False),
    "qos_aware":        dict(npca_qsrc=1, npca_threshold=10, adaptive_cw=True),
    "fair_coexistence": dict(npca_qsrc=2, npca_threshold=20, adaptive_cw=False),
    "energy_aware":     dict(npca_qsrc=3, npca_threshold=30, adaptive_cw=False),
}
# 모두 npca_enabled=True, harq_enabled=True
```

## 측정 지표 (6개 — 정규화하여 Radar chart 표시)

| 지표 | 방향 | 정규화 기준 |
|---|---|---|
| `aggregate_throughput` | 높을수록 좋음 | max across intents |
| `mean_access_delay` | 낮을수록 좋음 | `1 − delay/max_delay` |
| `packet_delivery_ratio` | 높을수록 좋음 | 그대로 |
| `jain_fairness_index` | 높을수록 좋음 | 그대로 |
| `total_energy_uj` (역) | 낮을수록 좋음 | `1 − energy/max_energy` |
| `npca_transition_rate` | intent 의존 | 별도 bar chart |

## Figure 구성

```
Figure 8: 3-panel layout

  Panel A (Radar chart): 정규화 지표 5개 축
    5개 polygon = 5개 intent config
    축: throughput_norm / delay_norm / PDR / fairness / energy_norm
    → intent별로 "어느 축이 강하고 약한지" 직관적 시각화

  Panel B (Grouped bar): 실제 수치 비교
    X 그룹: 5개 intent
    Y: mean_access_delay (left) + aggregate_throughput (right, 2nd axis)
    → throughput-delay tradeoff 정량적 표시

  Panel C (Bar): Policy 지표 직접 비교
    X: 5개 intent
    Y-left: npca_transition_rate
    Y-right: avg_npca_qsrc (adaptive의 경우), 또는 고정 qsrc 값 표시
    → "어떤 intent가 더 공격적/보수적 NPCA 전략을 선택하는가"
```

## 예상 관찰

- `throughput` / `delay_sensitive`: Radar chart에서 throughput + delay 축 강함
  - transition rate 높음 (공격적 NPCA)
- `fair_coexistence`: fairness 축 강함, throughput 낮음
  - qsrc=2로 충돌 줄여 STA 간 공정한 채널 접근
- `energy_aware`: energy 축 강함, transition rate 최저
  - qsrc=3 + threshold=30 → 짧은 OBSS 시 전환 억제
- `qos_aware`: 중간적 특성 (adaptive CW 사용)

## 출력 파일

```
manuscript/figure/
  fig8a_radar.eps         ← Radar chart
  fig8a_radar.png
  fig8a_radar.pdf
  fig8b_bar.eps           ← throughput-delay grouped bar
  fig8b_bar.png
  fig8b_bar.pdf
  fig8c_policy.eps        ← transition rate + qsrc bar
  fig8c_policy.png
  fig8c_policy.pdf

results/step9/fig8/
  data.csv            ← (intent, seed, metric, value)
```

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-25 | 초안 작성 |
