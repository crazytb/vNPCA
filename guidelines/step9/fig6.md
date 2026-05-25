# Figure 6: HARQ Combining Gain vs Higher MCS Tradeoff

**연구 질문 (RQ6)**: HARQ combining gain과 higher-MCS fresh transmission gain 사이의 tradeoff는 NPCA에서 어떻게 달라지는가?

**스크립트**: `harq_sim/run_step9_fig6.py`

**출력**: `manuscript/figure/fig6_*.{eps,png,pdf}`

---

## 배경

| 전략 | MCS | 장점 | 단점 |
|---|---|---|---|
| HARQ-CC (`fixed_cw_npca_harq`) | 원본 MCS 고정 | 누적 SNR로 실패 구제 | SNR 높아도 MCS 못 올림 |
| ARQ (`arq_only_npca`) | 매 재시도마다 현재 SNR 기반 새 MCS | SNR 좋으면 MCS7 바로 선택 | combining 없음 |

**가설**: 저 SNR → HARQ combining gain 지배, 고 SNR → Higher MCS gain 지배  
→ 교차점(crossover SNR) 존재

## 실험 파라미터

| 항목 | 값 |
|---|---|
| **Sweep 변수** | `snr_db_mean` ∈ {6, 8, 10, 12, 14, 16, 18, 20, 25, 30} |
| `snr_db_std` | 0.0 dB (결정론적 — 효과 분리) |
| `num_stas` | 5 |
| `obss_rate` | 0.30 |
| `obss_min` | 20 슬롯 |
| `obss_max` | 200 슬롯 |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

> **주의**: `snr_db_std=0` 사용 이유 — SNR 변동 없이 각 SNR 포인트에서 HARQ vs ARQ 효과를 명확히 분리

## 비교 대상 (2개 기법)

| 기법 | npca_enabled | harq_enabled | MCS 정책 |
|---|---|---|---|
| `arq_only_npca` | True | False | 재시도마다 새 MCS 선택 |
| `fixed_cw_npca_harq` | True | True | HARQ_RETX = 원본 MCS 고정 |

## 측정 지표

- `aggregate_throughput` — 전달 패킷 수 (주 지표)
- HARQ success rate (계산):
  ```python
  # per-STA stats에서 집계
  total_harq_success = sum(s["harq_tx_success"] for sta_id, s in metrics.items()
                           if sta_id != "aggregate")
  total_harq_fail    = sum(s["harq_tx_fail"]    for ...)
  harq_success_rate  = total_harq_success / (total_harq_success + total_harq_fail + 1e-9)
  ```

## Figure 구성

```
Figure 6: 2-row subplot (공유 x축: snr_db_mean)
  Row A: aggregate_throughput vs snr_db
          arq_only_npca (solid) vs fixed_cw_npca_harq (dashed) + std 음영
          교차점 위치 화살표 또는 수직 점선으로 표시
          x축 보조 레이블: MCS 전환 경계 (8/11/14/17/20/23/26 dB) 표시

  Row B: harq_success_rate vs snr_db
          fixed_cw_npca_harq만 표시 (combining이 실제로 도움 되는 SNR 범위 시각화)
          y=0.5 기준선 (SNR = MCS threshold → p_success = 0.5)
```

## PHY 모델 참조 (MCS SNR thresholds)

```python
# harq_sim/phy.py
MCS_SNR_THRESHOLDS = {
    0:  5.0,  # BPSK  1/2
    1:  8.0,  # QPSK  1/2
    2: 11.0,  # QPSK  3/4
    3: 14.0,  # 16-QAM 1/2
    4: 17.0,  # 16-QAM 3/4
    5: 20.0,  # 64-QAM 2/3
    6: 23.0,  # 64-QAM 3/4
    7: 26.0,  # 64-QAM 5/6
}
```

## 예상 관찰

- SNR = 6~12 dB (MCS0~MCS2 영역):
  - ARQ 재시도도 낮은 MCS → combining 없이는 실패 반복
  - HARQ combining: SNR 누적 → p_success 급증 → `fixed_cw_npca_harq` 우위
  - harq_success_rate 높음

- SNR = 14~20 dB (MCS3~MCS5 경계):
  - ARQ: 재시도 시 더 높은 MCS 선택 가능 → 단번에 성공 가능
  - HARQ: 낮은 MCS에 묶임 → combining해도 throughput 한계
  - 교차점 근방

- SNR ≥ 23 dB (MCS6~MCS7):
  - ARQ 첫 전송에서 MCS6/7 → 거의 무조건 성공
  - HARQ combining 불필요 → `arq_only_npca` 우위
  - harq_success_rate 낮아짐 (combining 기회 자체가 없음)

## 출력 파일

```
manuscript/figure/
  fig6_harq_vs_arq_snr.eps
  fig6_harq_vs_arq_snr.png
  fig6_harq_vs_arq_snr.pdf

results/step9/fig6/
  data.csv            ← (snr_db, baseline, seed, metric, value)
```

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-25 | 초안 작성 |
