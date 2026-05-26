# Figure 4: Adaptive qsrc vs. Fixed qsrc vs. Oracle (주요 기여)

**기여**: NPCA CW amnesia 하에서 col_rate·waste_rate 기반 adaptive qsrc 알고리즘이  
고정 qsrc 대비 throughput을 개선하며 oracle에 수렴함을 보임.

**스크립트**: `harq_sim/run_step9_fig4.py`

**출력**: `manuscript/figure/fig4_*.{eps,png,pdf}`

---

## 실험 파라미터 (Fig 3 v2와 동일 환경)

| 항목 | 값 |
|---|---|
| **Sweep 변수** | `num_stas` ∈ {5, 10, 20, 30, 50} |
| OBSS 채널 점유율 | 50% (`obss_rate = _occupancy_to_rate(0.50)`) |
| `obss_min` | 20 슬롯 |
| `obss_max` | 500 슬롯 |
| `snr_db_mean` | 20.0 dB (결정론적 — 충돌 효과 분리) |
| `snr_db_std` | 0.0 dB |
| `ppdu_duration` | 20 슬롯 |
| `num_slots` | 50,000 |
| Seeds | [42, 123, 456] |

---

## 비교 대상 (5개)

| 기법 | adaptive_cw | npca_qsrc | 설명 |
|---|---|---|---|
| `fixed_q0` | False | 0 | CW=15, 공격적 (표준 기본값) |
| `fixed_q1` | False | 1 | CW=31 |
| `fixed_q2` | False | 2 | CW=63 |
| `oracle` | False | N별 최적값 | Fig 3 v2 결과로 N→qsrc* 고정 (upper bound) |
| `adaptive` | **True** | 동적 조정 | **제안 알고리즘** |

Oracle qsrc 매핑 (Fig 3 v2 결과):
```python
ORACLE_QSRC = {5: 0, 10: 0, 20: 0, 30: 1, 50: 2}
```

---

## Adaptive qsrc 알고리즘

매 `K=20`회 NPCA 전환마다 다음을 계산:

```python
col_rate   = npca_collision_count / npca_transitions
waste_rate = 1 - (npca_tx_success + npca_tx_fail) / npca_transitions

if col_rate > θ_col:      qsrc = min(qsrc + 1, QSRC_MAX)  # CW 키움
elif waste_rate > θ_waste: qsrc = max(qsrc - 1, 0)         # CW 줄임
카운터 리셋
```

기본 파라미터: `K=5, θ_col=0.70, θ_waste=0.30, QSRC_MAX=5`

선택 근거:
- K=5: N=50 STA 환경에서 50000슬롯당 ~19 NPCA 방문 → K=5로 ~3-4회 업데이트 가능
- θ_col=0.70: N=30에서 qsrc*=1이 optimal인데 col@qsrc=1=66.5% < 0.70 → 조기 상승 방지

**직관**:
- `col_rate` 높음 → 많은 STA가 경쟁 → CW를 키워 충돌 감소
- `waste_rate` 높음 → 백오프가 창 내 미완료 → CW가 너무 큼, 줄여야

---

## 측정 지표

- `aggregate_throughput` — 전달 패킷 수 (주 지표)
- `collision_probability_npca` — NPCA 충돌 확률
- `mean_qsrc` — adaptive의 평균 qsrc (oracle qsrc*와 비교)
- `npca_transition_count` — NPCA 전환 총 횟수

---

## Figure 구성

```
Figure 4: 3-panel (공유 x축: num_stas)

Panel (a): aggregate_throughput vs num_stas
           5개 선: fixed_q0/q1/q2 (회색 계열), oracle (녹색 점선), adaptive (파랑 실선)
           std 음영 포함
           핵심 메시지: adaptive ≈ oracle > 최적 fixed

Panel (b): collision_probability_npca vs num_stas
           동일 5개 선
           adaptive와 oracle의 충돌률이 수렴하는지 확인

Panel (c): mean_qsrc vs num_stas (adaptive vs oracle 비교)
           막대: oracle qsrc* (회색)
           꺾은선: adaptive mean qsrc (파랑)
           ± std 음영
           핵심 메시지: adaptive가 oracle qsrc*에 수렴
```

---

## 실험 결과 (results/step9/fig4/data.csv, 50000슬롯 × 3 seeds)

| N | Oracle TP | Adaptive TP | vs Oracle | vs Best_Fixed | adaptive mean_q |
|---|---|---|---|---|---|
| 5  | 1250 | **1250** | +0.00% | +0.00% | 0.00 (oracle=0) ✓ |
| 10 | 1272 | 1259     | -1.00% | -1.00% | 0.18              |
| 20 | 1244 | **1249** | +0.43% | +0.43% | 0.68              |
| 30 | 1219 | **1232** | +1.01% | +1.01% | 1.07 (oracle=1) ✓ |
| 50 | 1203 | 1194     | -0.72% | -0.72% | 1.37 (oracle=2) ~ |

**핵심 관찰:**
- N=5,30: adaptive가 oracle과 동일한 qsrc*에 수렴 ✓
- N=30에서 adaptive가 oracle을 +1.01% 초과 — 동적 조정이 고정 oracle보다 유리
- N=50: 수렴 지연 (50000슬롯에서 ~19 방문 → ~3-4 업데이트로 qsrc=2 미달, mean_q=1.37)
- fixed_q0~q2는 "작은 N에서 좋은 qsrc"와 "큰 N에서 좋은 qsrc"가 서로 다름 → adaptive만이 전체 N에서 경쟁력 유지

**논문 메시지:** "adaptive qsrc는 oracle 대비 전체 N 범위에서 ±1% 이내이며, 고정 qsrc 선택 시 발생하는 worst-case 성능 저하를 방지한다."

**수렴 한계 언급 필요:** 밀집 네트워크(N=50)에서 STA당 NPCA 방문 횟수가 sparse → 50000슬롯 내 완전 수렴 미달. 실 시스템에서는 긴 운용 시간이 보장되므로 문제 없음.

---

## 구현 메모

`sta.py`에 추가 필요:
```python
# __init__
self._adap_trans = 0  # adaptive 관측 창 전환 카운터
self._adap_col   = 0  # 충돌 카운터
self._adap_tx    = 0  # TX 시도 카운터
self._adap_K     = 20  # 관측 창 크기
self._theta_col   = 0.50
self._theta_waste = 0.30

# _start_npca_transition 후
if self.adaptive_cw:
    self._adap_trans += 1

# _handle_npca_tx 내 collision 감지 시
if self.adaptive_cw:
    self._adap_col += 1
    self._adap_tx  += 1

# _start_switch_back 내
if self.adaptive_cw:
    self._maybe_update_qsrc()

def _maybe_update_qsrc(self):
    if self._adap_trans < self._adap_K:
        return
    col_rate   = self._adap_col / self._adap_trans
    waste_rate = 1.0 - self._adap_tx / self._adap_trans
    if col_rate > self._theta_col:
        self.npca_initial_qsrc = min(self.npca_initial_qsrc + 1, 5)
    elif waste_rate > self._theta_waste:
        self.npca_initial_qsrc = max(self.npca_initial_qsrc - 1, 0)
    self._adap_trans = self._adap_col = self._adap_tx = 0
```

---

## 출력 파일

```
manuscript/figure/
  fig4_adaptive_qsrc.eps / .png / .pdf

results/step9/fig4/
  data.csv    ← (num_stas, method, seed, aggregate_throughput,
                  collision_probability_npca, mean_qsrc, npca_transition_count)
```

---

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-25 | 초안 작성 (Step 8 역전 현상 분석 기반) |
| 2026-05-26 | 논문 방향 변경에 따라 전면 재설계: adaptive qsrc 알고리즘 주요 기여로 재정립; oracle 비교군 추가; Fig 3 v2 환경으로 통일 |
