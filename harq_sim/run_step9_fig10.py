"""
Figure 10: qsrc* Dependence on OBSS Duration W_eff

연구 질문 (RQ10): qsrc*는 OBSS duration W_eff에도 독립적으로 의존하는가?
고정 W(OBSS_MIN=OBSS_MAX=W)로 W_eff를 직접 제어하여 분리 검증.

계획: guidelines/step9/fig10.md
출력:
  manuscript/figure/fig10_qsrc_weff.{eps,png,pdf}
  results/step9/fig10/data.csv

실행:
  python harq_sim/run_step9_fig10.py [--fast] [--out-dir results/step9/fig10]
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harq_sim.run_step8 import build_and_run

# ─────────────────────────────────────────────────────────────────────────────
# 실험 설정
# ─────────────────────────────────────────────────────────────────────────────

W_LIST        = [30, 50, 80, 120, 170, 230, 300, 400, 500, 700]
NUM_STAS_LIST = [10, 20, 30, 50]
QSRC_LIST     = [0, 1, 2, 3, 4, 5]
SEEDS         = [42, 123, 456]

OBSS_OCCUPANCY = 0.50
PPDU_DURATION  = 20

FULL_SLOTS = 50_000
FAST_SLOTS =  5_000

# fast 모드 subset
FAST_W_LIST = [50, 170, 500]
FAST_N_LIST = [10, 30]
FAST_SEEDS  = [42]

CW_MIN = 15

# N별 선 색상 (panels b, c)
_COLORS_N = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
# W subset 선 색상 (panel a)
_COLORS_W = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

# panel (a)에 표시할 W 값 (5개 대표)
W_PANEL_A = [50, 120, 230, 400, 700]


def _occupancy_to_rate(occ: float, W: int) -> float:
    """고정 duration W에서 점유율 occ를 달성하는 OBSS 발생률."""
    return occ / (W * (1.0 - occ))


def _qsrc_to_cw(qsrc: int) -> int:
    return 2 ** qsrc * (CW_MIN + 1) - 1


# ─────────────────────────────────────────────────────────────────────────────
# Sweep: W × num_stas × qsrc × seed
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(num_slots: int, w_list: list, n_list: list, seeds: list) -> list[dict]:
    rows: list[dict] = []
    total = len(w_list) * len(n_list) * len(QSRC_LIST) * len(seeds)
    done  = 0

    for W in w_list:
        obss_rate = _occupancy_to_rate(OBSS_OCCUPANCY, W)
        for num_stas in n_list:
            for qsrc in QSRC_LIST:
                cw = _qsrc_to_cw(qsrc)
                for seed in seeds:
                    done += 1
                    print(f"  [{done:3d}/{total}] W={W:3d}  N={num_stas:2d}  "
                          f"qsrc={qsrc}  CW={cw:<4}  seed={seed}", flush=True)

                    sim = build_and_run(
                        num_slots      = num_slots,
                        num_stas       = num_stas,
                        obss_rate      = obss_rate,
                        obss_min       = W,
                        obss_max       = W,
                        npca_qsrc      = qsrc,
                        npca_threshold = 0,
                        ppdu_duration  = PPDU_DURATION,
                        harq_horizon   = 200,
                        snr_db_mean    = 20.0,
                        snr_db_std     = 0.0,
                        npca_enabled   = True,
                        harq_enabled   = True,
                        adaptive_cw    = False,
                        seed           = seed,
                        enable_trace   = False,
                    )
                    agg = sim.compute_metrics()["aggregate"]
                    rows.append({
                        "W":                           W,
                        "num_stas":                    num_stas,
                        "npca_qsrc":                   qsrc,
                        "npca_cw":                     cw,
                        "seed":                        seed,
                        "collision_probability_npca":  agg["collision_probability_npca"],
                        "aggregate_throughput":        agg["aggregate_throughput"],
                        "mean_access_delay":           agg["mean_access_delay"],
                        "npca_transition_count":       agg["npca_transition_count"],
                    })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = ["W", "num_stas", "npca_qsrc", "npca_cw", "seed",
          "collision_probability_npca", "aggregate_throughput",
          "mean_access_delay", "npca_transition_count"]


def save_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats(rows: list[dict], W: int, num_stas: int, qsrc: int,
           metric: str) -> tuple[float, float]:
    vals = [r[metric] for r in rows
            if r["W"] == W and r["num_stas"] == num_stas and r["npca_qsrc"] == qsrc]
    if not vals:
        return 0.0, 0.0
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def _optimal_qsrc(rows: list[dict], W: int, num_stas: int) -> tuple[int, float]:
    best_qsrc, best_tp = 0, -1.0
    for q in QSRC_LIST:
        m, _ = _stats(rows, W, num_stas, q, "aggregate_throughput")
        if m > best_tp:
            best_tp, best_qsrc = m, q
    return best_qsrc, best_tp


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(rows: list[dict], fig_dir: str, w_list: list, n_list: list) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 9.5), sharex=False)
    fig.subplots_adjust(hspace=0.42)

    x_qsrc   = QSRC_LIST
    x_labels = [f"q={q}\n(CW={_qsrc_to_cw(q)})" for q in x_qsrc]

    # ── Panel (a): Throughput vs qsrc, N=30 고정, lines per W ─────────────────
    ax_a = axes[0]
    FIXED_N   = 30 if 30 in n_list else n_list[-1]
    w_show    = [w for w in W_PANEL_A if w in w_list]

    for idx, W in enumerate(w_show):
        means, stds = [], []
        for q in x_qsrc:
            m, s = _stats(rows, W, FIXED_N, q, "aggregate_throughput")
            means.append(m); stds.append(s)

        color = _COLORS_W[idx % len(_COLORS_W)]
        ax_a.plot(x_qsrc, means, color=color, ls="-", marker="o",
                  markersize=5, linewidth=1.6, label=f"W={W}")
        ax_a.fill_between(x_qsrc,
                          [m - s for m, s in zip(means, stds)],
                          [m + s for m, s in zip(means, stds)],
                          color=color, alpha=0.12)

        q_opt, tp_opt = _optimal_qsrc(rows, W, FIXED_N)
        ax_a.plot(q_opt, tp_opt, marker="*", color=color,
                  markersize=11, zorder=5)

    ax_a.set_ylabel("Aggregate Throughput\n(delivered packets)", fontsize=10)
    ax_a.set_xticks(x_qsrc)
    ax_a.set_xticklabels(x_labels, fontsize=8)
    ax_a.set_xlabel("NPCA Initial CW Exponent (qsrc)", fontsize=9)
    ax_a.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_a.legend(fontsize=9, loc="lower left", ncol=3,
                frameon=True, edgecolor="#cccccc",
                title=f"OBSS Duration W (N={FIXED_N})", title_fontsize=8)
    ax_a.text(0.02, 0.97, "(a)  ★ = optimal qsrc* per W",
              transform=ax_a.transAxes, fontsize=9, va="top")

    # ── Panel (b): qsrc*(W), lines per N ──────────────────────────────────────
    ax_b = axes[1]

    for idx, num_stas in enumerate(n_list):
        opt_qsrcs = [_optimal_qsrc(rows, W, num_stas)[0] for W in w_list]
        color = _COLORS_N[idx % len(_COLORS_N)]
        ax_b.plot(w_list, opt_qsrcs, color=color, ls="-", marker="s",
                  markersize=6, linewidth=1.6, label=f"N={num_stas}",
                  drawstyle="steps-post")

    ax_b.set_ylabel("Optimal qsrc*", fontsize=10)
    ax_b.set_xlabel("OBSS Duration W (slots)", fontsize=10)
    ax_b.set_yticks(QSRC_LIST)
    ax_b.set_yticklabels([f"q={q}  (CW={_qsrc_to_cw(q)})" for q in QSRC_LIST],
                         fontsize=8)
    ax_b.set_xticks(w_list)
    ax_b.set_xticklabels([str(w) for w in w_list], fontsize=8, rotation=45)
    ax_b.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_b.legend(fontsize=9, loc="upper left",
                frameon=True, edgecolor="#cccccc",
                title="Num STAs", title_fontsize=8)
    ax_b.text(0.02, 0.97, "(b)  qsrc*(W) — threshold W* increases with N",
              transform=ax_b.transAxes, fontsize=9, va="top")

    # ── Panel (c): Gain vs W, lines per N ─────────────────────────────────────
    ax_c = axes[2]

    for idx, num_stas in enumerate(n_list):
        gains = []
        for W in w_list:
            q_opt, tp_opt = _optimal_qsrc(rows, W, num_stas)
            tp_base, _    = _stats(rows, W, num_stas, 0, "aggregate_throughput")
            gain = (tp_opt - tp_base) / tp_base * 100 if tp_base > 0 else 0.0
            gains.append(gain)
        color = _COLORS_N[idx % len(_COLORS_N)]
        ax_c.plot(w_list, gains, color=color, ls="-", marker="^",
                  markersize=6, linewidth=1.6, label=f"N={num_stas}")

    ax_c.axhline(0, color="gray", lw=0.8, ls=":")
    ax_c.set_ylabel("Throughput Gain over\nqsrc=0 (%)", fontsize=10)
    ax_c.set_xlabel("OBSS Duration W (slots)", fontsize=10)
    ax_c.set_xticks(w_list)
    ax_c.set_xticklabels([str(w) for w in w_list], fontsize=8, rotation=45)
    ax_c.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_c.legend(fontsize=9, loc="upper left",
                frameon=True, edgecolor="#cccccc",
                title="Num STAs", title_fontsize=8)
    ax_c.text(0.02, 0.97, "(c)  Gain increases with W and N",
              transform=ax_c.transAxes, fontsize=9, va="top")

    fig.suptitle(
        "Fig. 10  qsrc* Dependence on OBSS Duration $W_{\\mathrm{eff}}$\n"
        "Fixed OBSS duration (OBSS\\_MIN=OBSS\\_MAX=W), Occupancy=50%",
        fontsize=11,
    )

    _save_figure(fig, fig_dir, "fig10_qsrc_weff")
    plt.close(fig)


def _save_figure(fig, fig_dir: str, name: str) -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ms_fig    = os.path.join(repo_root, "manuscript", "figure")
    os.makedirs(ms_fig, exist_ok=True)

    for ext, kwargs in [
        ("eps", dict(format="eps",  bbox_inches="tight")),
        ("png", dict(format="png",  bbox_inches="tight", dpi=300)),
        ("pdf", dict(format="pdf",  bbox_inches="tight")),
    ]:
        dest = os.path.join(ms_fig, f"{name}.{ext}")
        fig.savefig(dest, **kwargs)
        print(f"  Figure → {dest}")

    preview = os.path.join(fig_dir, f"{name}_preview.png")
    fig.savefig(preview, format="png", dpi=150, bbox_inches="tight")
    print(f"  Preview → {preview}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 10 — qsrc* vs OBSS duration W_eff")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_SLOTS} slots, subset of W/N/seeds")
    parser.add_argument("--out-dir", default="results/step9/fig10")
    args = parser.parse_args()

    num_slots = FAST_SLOTS if args.fast else FULL_SLOTS
    w_list    = FAST_W_LIST if args.fast else W_LIST
    n_list    = FAST_N_LIST if args.fast else NUM_STAS_LIST
    seeds     = FAST_SEEDS  if args.fast else SEEDS
    out_dir   = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    mode  = "FAST" if args.fast else "FULL"
    total = len(w_list) * len(n_list) * len(QSRC_LIST) * len(seeds)
    print(f"=== Figure 10 [{mode}] — {num_slots} slots × {len(seeds)} seeds ===")
    print(f"    W sweep={w_list}")
    print(f"    N sweep={n_list}")
    print(f"    OBSS occupancy={OBSS_OCCUPANCY:.0%}  (rate varies with W)")
    print(f"    {len(w_list)} × {len(n_list)} × {len(QSRC_LIST)} × {len(seeds)} = {total} runs")

    rows = run_sweep(num_slots, w_list, n_list, seeds)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    print("Plotting...")
    plot(rows, out_dir, w_list, n_list)

    print(f"\nFigure 10 완료")
    print(f"  데이터  : {csv_path}")
    print(f"  논문용  : manuscript/figure/fig10_qsrc_weff.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
