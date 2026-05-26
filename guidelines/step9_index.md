# Step 9: NPCA qsrc/CW 최적화 Figure 인덱스

**논문 방향 (2026-05-26 확정)**: NPCA CW Amnesia 문제 + 최적 qsrc 선택 + Adaptive 알고리즘  
LLM/intent 기여는 별도 논문으로 분리.

---

## Figure → 스크립트 → 출력 매핑

| Figure | 역할 | 스크립트 | 상세 계획 | 상태 |
|---|---|---|---|---|
| Fig 1 | NPCA-HARQ 조합 이득 (motivation) | `harq_sim/run_step9_fig1.py` | [fig1.md](step9/fig1.md) | ✅ 완료 |
| Fig 2 | Primary CW save/restore → NPCA delay 이득 (background) | `harq_sim/run_step9_fig2.py` | [fig2.md](step9/fig2.md) | ✅ 완료 |
| Fig 3 | Fixed qsrc sweep: qsrc*(N, W_eff) 의존성 확인 (문제 식별) | `harq_sim/run_step9_fig3.py` | [fig3.md](step9/fig3.md) | ✅ 완료 |
| Fig 4 | Adaptive qsrc vs. Fixed vs. Oracle (주요 기여) | `harq_sim/run_step9_fig4.py` | [fig4.md](step9/fig4.md) | ✅ 완료 |
| Fig 5 | Per-STA 이질성: frame length × qsrc fairness-efficiency tradeoff | `harq_sim/run_step9_fig5.py` | [fig5.md](step9/fig5.md) | ✅ 완료 |
| Fig 6 | HARQ combining gain vs Higher MCS crossover (부록/참고) | `harq_sim/run_step9_fig6.py` | [fig6.md](step9/fig6.md) | ✅ 완료 |

> Fig 7, 8 (LLM/Intent 기반): 별도 논문으로 이관. 스크립트 및 guidelines 보존.

상태: ⬜ 미구현 / 🔄 진행 중 / ✅ 완료

---

## 논문 기여 구조

```
§ Motivation  : Fig 1 (NPCA-HARQ gain), Fig 2 (primary CW save/restore)
§ Problem     : Fig 3 (fixed qsrc의 한계 — qsrc*가 N·W_eff에 의존)
§ Analysis    : Theorem: qsrc*(N, W_eff, ppdu) 도출 (Bianchi 확장)
§ Algorithm   : Adaptive qsrc (col_rate + waste_rate 기반)
§ Evaluation  : Fig 4 (adaptive vs fixed vs oracle)
§ Extension   : Fig 5 (per-STA heterogeneity), Fig 6 (HARQ interaction)
```

---

## 출력 위치

| 종류 | 경로 |
|---|---|
| 논문 삽입용 (eps/png/pdf) | `manuscript/figure/figN_*.{eps,png,pdf}` |
| 실험 데이터 (CSV) | `results/step9/figN/data.csv` |

---

## 공통 실험 파라미터

```python
BASE = dict(
    num_slots       = 50_000,
    ppdu_duration   = 20,
    harq_horizon    = 200,
    npca_threshold  = 0,
    enable_trace    = False,
)
SEEDS = [42, 123, 456]
# Fig 4 환경 (Fig 3 v2와 동일)
OBSS_MAX       = 500
OBSS_OCCUPANCY = 0.50
NUM_STAS_LIST  = [5, 10, 20, 30, 50]
```

---

## 실행 순서

```bash
source .venv/bin/activate

# Phase 1: 기초 (완료)
python harq_sim/run_step9_fig1.py --out-dir results/step9/fig1_v2
python harq_sim/run_step9_fig2.py --out-dir results/step9/fig2_v2
python harq_sim/run_step9_fig3.py --out-dir results/step9/fig3_v2

# Phase 2: 핵심 기여
python harq_sim/run_step9_fig4.py --out-dir results/step9/fig4

# Phase 3: 확장
python harq_sim/run_step9_fig5.py --out-dir results/step9/fig5
python harq_sim/run_step9_fig6.py --out-dir results/step9/fig6
```
