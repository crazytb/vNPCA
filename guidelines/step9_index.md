# Step 9: 연구 질문 검증 Figure 인덱스

Guidelines §30의 8개 연구 질문 → 8개 Figure 1대1 매핑.  
각 Figure의 상세 계획은 `guidelines/step9/figN.md` 참조.

---

## Figure → 스크립트 → 출력 매핑

| Figure | 연구 질문 요약 | 스크립트 | 상세 계획 | 상태 |
|---|---|---|---|---|
| Fig 1 | NPCA-HARQ 조합 이득 (RQ1) | `harq_sim/run_step9_fig1.py` | [fig1.md](step9/fig1.md) | ✅ 완료 |
| Fig 2 | Primary CW → NPCA delay 이득 (RQ2) | `harq_sim/run_step9_fig2.py` | [fig2.md](step9/fig2.md) | ✅ 완료 |
| Fig 3 | Fixed qsrc → NPCA 충돌 (RQ3) | `harq_sim/run_step9_fig3.py` | [fig3.md](step9/fig3.md) | ⬜ 미구현 |
| Fig 4 | Adaptive vs Fixed CW tradeoff (RQ4) | `harq_sim/run_step9_fig4.py` | [fig4.md](step9/fig4.md) | ⬜ 미구현 |
| Fig 5 | NPCA HARQ vs Primary HARQ 조건 (RQ5) | `harq_sim/run_step9_fig5.py` | [fig5.md](step9/fig5.md) | ⬜ 미구현 |
| Fig 6 | HARQ combining gain vs Higher MCS (RQ6) | `harq_sim/run_step9_fig6.py` | [fig6.md](step9/fig6.md) | ⬜ 미구현 |
| Fig 7 | LLM vs Grid-best vs Hand-crafted (RQ7) | `harq_sim/run_step9_fig7.py` | [fig7.md](step9/fig7.md) | ⬜ 미구현 |
| Fig 8 | Intent별 NPCA/HARQ policy 차이 (RQ8) | `harq_sim/run_step9_fig8.py` | [fig8.md](step9/fig8.md) | ⬜ 미구현 |

상태: ⬜ 미구현 / 🔄 진행 중 / ✅ 완료

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
    num_slots       = 50_000,   # 논문 품질
    ppdu_duration   = 20,
    harq_horizon    = 200,
    npca_threshold  = 0,
    enable_trace    = False,
    mock_llm        = True,
)
SEEDS = [42, 123, 456]   # 3 seeds → mean ± std 표시
FAST  = dict(**BASE, num_slots=5_000)  # --fast 플래그용
```

---

## 실행 순서

```bash
source .venv/bin/activate

# Phase 1: 기초 sweep (독립)
python harq_sim/run_step9_fig1.py [--fast] --out-dir results/step9/fig1
python harq_sim/run_step9_fig2.py [--fast] --out-dir results/step9/fig2
python harq_sim/run_step9_fig3.py [--fast] --out-dir results/step9/fig3

# Phase 2: Tradeoff 분석
python harq_sim/run_step9_fig4.py [--fast] --out-dir results/step9/fig4
python harq_sim/run_step9_fig5.py [--fast] --out-dir results/step9/fig5
python harq_sim/run_step9_fig6.py [--fast] --out-dir results/step9/fig6

# Phase 3: Reward 분석
python harq_sim/run_step9_fig7.py [--fast] --out-dir results/step9/fig7
python harq_sim/run_step9_fig8.py [--fast] --out-dir results/step9/fig8
```
