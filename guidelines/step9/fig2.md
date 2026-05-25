# Figure 2: Primary CW 크기 → NPCA Transition Delay 이득

**연구 질문 (RQ2)**: primary CW가 큰 상황에서 NPCA transition은 delay를 줄이는가?

**스크립트**: `harq_sim/run_step9_fig2.py`

**출력**: `manuscript/figure/fig2_*.{eps,png,pdf}`

---

## 실험 파라미터

| 항목 | 값 |
|---|---|
| **Sweep 변수** | `num_stas` ∈ {2, 5, 10, 15, 20} |
| OBSS 채널 점유율 | 30% (`obss_rate = _occupancy_to_rate(0.30) ≈ 0.0039`) |
| `snr_db_mean` | 14.0 dB |
| `snr_db_std` | 2.0 dB |
| `obss_min` | 20 슬롯 |
| `obss_max` | 200 슬롯 |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

## 비교 대상 (2개 기법)

| 기법 | npca_enabled | harq_enabled |
|---|---|---|
| `legacy_edca` | False | False |
| `arq_only_npca` | True | False |

## 측정 지표

- `mean_access_delay` (슬롯) — 주 지표
- `collision_probability_primary` — primary CW 증가의 간접 지표

## Figure 구성

```
Figure 2: 2-row subplot (공유 x축: num_stas)
  Row A (주): mean_access_delay vs num_stas
              legacy_edca (solid), arq_only_npca (dashed)
              std 음영 표시
  Row B (보조): collision_probability_primary vs num_stas
              STA 수 증가 → primary 충돌 증가 → CW 증가 확인

  x축 눈금: [2, 5, 10, 15, 20]
  Row A에 delay 차이(gap) 주석 표시 — NPCA 이득 정량화
```

## 예상 관찰

- STA 수 증가 → primary 충돌 증가 → primary CW exponential 증가 → legacy delay 급증
- NPCA 계열은 OBSS 기간 primary 경쟁 없이 TX → delay 증가 완만
- 교차점: STA 수 적을 때 NPCA overhead 있지만, STA 많아지면 NPCA 압도적 우위
- Row B 충돌 확률 곡선이 Row A delay 곡선과 유사한 추세 → 상관관계 확인

## 출력 파일

```
manuscript/figure/
  fig2_delay_vs_nstas.eps
  fig2_delay_vs_nstas.png
  fig2_delay_vs_nstas.pdf

results/step9/fig2/
  data.csv            ← (num_stas, baseline, seed, metric, value)
```

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-25 | 초안 작성 |
| 2026-05-26 | `harq_sim/run_step9_fig2.py` 구현 완료; Panel (b)를 `collision_probability_primary` → `per_sta_throughput`으로 교체 (슬롯 기반 시뮬레이터에서 MAC 충돌이 극히 드물어 변별력 없음 — 대신 per-STA throughput이 contention 효과를 직접 표시) |
| 2026-05-26 | `obss_rate=0.30` → OBSS 점유율 30%로 재정의. 기존 obss_rate=0.30은 per-slot 도착률로 실제 점유율 97%였음 — Legacy EDCA starvation 문제. `_occupancy_to_rate(0.30) ≈ 0.0039`로 변환하여 실제 30% 점유율 달성. |
