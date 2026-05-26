"""
Figure 4: Adaptive qsrc vs. Fixed qsrc vs. Oracle — 주요 기여 (RQ4)

계획: guidelines/step9/fig4.md
출력:
  manuscript/figure/fig4_adaptive_qsrc.{eps,png,pdf}
  results/step9/fig4/data.csv

실행:
  python harq_sim/run_step9_fig4.py [--fast] [--out-dir results/step9/fig4]
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

NUM_STAS_LIST = [5, 10, 20, 30, 50]
SEEDS         = [42, 123, 456]

OBSS_MIN       = 20
OBSS_MAX       = 500
OBSS_OCCUPANCY = 0.50

FULL_SLOTS = 50_000
FAST_SLOTS =  5_000

# Oracle: Fig 3 v2에서 도출된 num_stas별 최적 qsrc*
ORACLE_QSRC = {5: 0, 10: 0, 20: 0, 30: 1, 50: 2}

# 비교 기법 정의
METHODS = [
    {"name": "fixed_q0", "adaptive_cw": False, "npca_qsrc": 0, "label": "Fixed qsrc=0 (CW=15)"},
    {"name": "fixed_q1", "adaptive_cw": False, "npca_qsrc": 1, "label": "Fixed qsrc=1 (CW=31)"},
    {"name": "fixed_q2", "adaptive_cw": False, "npca_qsrc": 2, "label": "Fixed qsrc=2 (CW=63)"},
    {"name": "oracle",   "adaptive_cw": False, "npca_qsrc": None, "label": "Oracle qsrc* (upper bound)"},
    {"name": "adaptive", "adaptive_cw": True,  "npca_qsrc": 0,    "label": "Adaptive qsrc (proposed)"},
]

_COLORS = {
    "fixed_q0": "#aaaaaa",
    "fixed_q1": "#888888",
    "fixed_q2": "#555555",
    "oracle":   "#2ca02c",
    "adaptive": "#1f77b4",
}
_LS = {
    "fixed_q0": ":",
    "fixed_q1": "--",
    "fixed_q2": "-.",
    "oracle":   "--",
    "adaptive": "-",
}

CW_MIN = 15

def _occupancy_to_rate(occ: float) -> float:
    mean_dur = (OBSS_MIN + OBSS_MAX) / 2.0
    return occ / (mean_dur * (1.0 - occ))


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(num_slots: int) -> list[dict]:
    rows: list[dict] = []
    obss_rate = _occupancy_to_rate(OBSS_OCCUPANCY)
    total = len(NUM_STAS_LIST) * len(METHODS) * len(SEEDS)
    done  = 0

    for num_stas in NUM_STAS_LIST:
        for method in METHODS:
            qsrc = ORACLE_QSRC[num_stas] if method["npca_qsrc"] is None else method["npca_qsrc"]
            for seed in SEEDS:
                done += 1
                print(f"  [{done:3d}/{total}] N={num_stas:2d}  method={method['name']:<12}"
                      f"  qsrc={qsrc}  seed={seed}", flush=True)

                sim = build_and_run(
                    num_slots       = num_slots,
                    num_stas        = num_stas,
                    obss_rate       = obss_rate,
                    obss_min        = OBSS_MIN,
                    obss_max        = OBSS_MAX,
                    npca_qsrc       = qsrc,
                    npca_threshold  = 0,
                    ppdu_duration   = 20,
                    harq_horizon    = 200,
                    snr_db_mean     = 20.0,
                    snr_db_std      = 0.0,
                    npca_enabled    = True,
                    harq_enabled    = True,
                    adaptive_cw     = method["adaptive_cw"],
                    seed            = seed,
                    enable_trace    = False,
                )
                agg = sim.compute_metrics()["aggregate"]

                # mean_qsrc: adaptive는 전환 이력 기반 시간 평균, fixed/oracle은 설정값
                if method["adaptive_cw"]:
                    histories = [sta._npca_qsrc_history for sta in sim.stas
                                 if sta._npca_qsrc_history]
                    mean_qsrc = float(np.mean([q for h in histories for q in h])) if histories else 0.0
                else:
                    mean_qsrc = float(qsrc)

                rows.append({
                    "num_stas":                  num_stas,
                    "method":                    method["name"],
                    "seed":                      seed,
                    "aggregate_throughput":       agg["aggregate_throughput"],
                    "collision_probability_npca": agg["collision_probability_npca"],
                    "npca_transition_count":      agg["npca_transition_count"],
                    "mean_qsrc":                  mean_qsrc,
                })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = ["num_stas", "method", "seed",
          "aggregate_throughput", "collision_probability_npca",
          "npca_transition_count", "mean_qsrc"]

def save_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats(rows, num_stas, method, metric):
    vals = [r[metric] for r in rows
            if r["num_stas"] == num_stas and r["method"] == method]
    if not vals:
        return 0.0, 0.0
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(rows: list[dict], fig_dir: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 9.5), sharex=True)
    fig.subplots_adjust(hspace=0.25)

    x = NUM_STAS_LIST

    for method in METHODS:
        name  = method["name"]
        label = method["label"]
        color = _COLORS[name]
        ls    = _LS[name]
        lw    = 2.0 if name in ("adaptive", "oracle") else 1.2
        zord  = 4 if name in ("adaptive", "oracle") else 2

        # ── Panel (a): Throughput ─────────────────────────────────────────────
        means, stds = zip(*[_stats(rows, n, name, "aggregate_throughput") for n in x])
        axes[0].plot(x, means, color=color, ls=ls, lw=lw, marker="o",
                     markersize=5, label=label, zorder=zord)
        axes[0].fill_between(x,
                             [m - s for m, s in zip(means, stds)],
                             [m + s for m, s in zip(means, stds)],
                             color=color, alpha=0.10, zorder=zord - 1)

        # ── Panel (b): Collision prob ─────────────────────────────────────────
        means, stds = zip(*[_stats(rows, n, name, "collision_probability_npca") for n in x])
        axes[1].plot(x, means, color=color, ls=ls, lw=lw, marker="s",
                     markersize=5, label=label, zorder=zord)
        axes[1].fill_between(x,
                             [m - s for m, s in zip(means, stds)],
                             [m + s for m, s in zip(means, stds)],
                             color=color, alpha=0.10, zorder=zord - 1)

        # ── Panel (c): mean qsrc ──────────────────────────────────────────────
        means, stds = zip(*[_stats(rows, n, name, "mean_qsrc") for n in x])
        axes[2].plot(x, means, color=color, ls=ls, lw=lw, marker="^",
                     markersize=5, label=label, zorder=zord)
        if name == "adaptive":
            axes[2].fill_between(x,
                                 [m - s for m, s in zip(means, stds)],
                                 [m + s for m, s in zip(means, stds)],
                                 color=color, alpha=0.15, zorder=zord - 1)

    axes[0].set_ylabel("Aggregate Throughput\n(delivered packets)", fontsize=10)
    axes[0].grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    axes[0].legend(fontsize=8, loc="upper right", ncol=1,
                   frameon=True, edgecolor="#cccccc")
    axes[0].text(0.02, 0.97, "(a)", transform=axes[0].transAxes, fontsize=9, va="top")

    axes[1].set_ylabel("NPCA Collision Probability", fontsize=10)
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    axes[1].text(0.02, 0.97, "(b)", transform=axes[1].transAxes, fontsize=9, va="top")

    axes[2].set_ylabel("Mean qsrc\n(initial NPCA CW exponent)", fontsize=10)
    axes[2].set_yticks([0, 1, 2, 3, 4, 5])
    axes[2].set_yticklabels(
        [f"q={q} (CW={2**q*(CW_MIN+1)-1})" for q in range(6)], fontsize=8)
    axes[2].grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    axes[2].text(0.02, 0.97, "(c) Adaptive tracks Oracle qsrc*",
                 transform=axes[2].transAxes, fontsize=9, va="top")

    axes[2].set_xlabel("Number of STAs", fontsize=10)
    axes[2].set_xticks(NUM_STAS_LIST)

    fig.suptitle(
        "Fig. 4  Adaptive qsrc vs. Fixed qsrc vs. Oracle\n"
        f"(OBSS occupancy={int(OBSS_OCCUPANCY*100)}%, obss_max={OBSS_MAX})",
        fontsize=11,
    )

    _save_figure(fig, fig_dir, "fig4_adaptive_qsrc")
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
        description="Figure 4 — Adaptive qsrc vs Fixed qsrc vs Oracle")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_SLOTS} slots")
    parser.add_argument("--out-dir", default="results/step9/fig4")
    args = parser.parse_args()

    num_slots = FAST_SLOTS if args.fast else FULL_SLOTS
    out_dir   = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    mode = "FAST" if args.fast else "FULL"
    rate = _occupancy_to_rate(OBSS_OCCUPANCY)
    total_cond = len(NUM_STAS_LIST) * len(METHODS) * len(SEEDS)
    print(f"=== Figure 4 [{mode}] — {num_slots} slots × {len(SEEDS)} seeds ===")
    print(f"    OBSS occupancy={OBSS_OCCUPANCY:.0%}  (rate={rate:.5f})")
    print(f"    {len(NUM_STAS_LIST)} num_stas × {len(METHODS)} methods × {len(SEEDS)} seeds = {total_cond} runs")

    rows = run_sweep(num_slots)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    print("Plotting...")
    plot(rows, out_dir)

    print(f"\nFigure 4 완료")
    print(f"  데이터  : {csv_path}")
    print(f"  논문용  : manuscript/figure/fig4_adaptive_qsrc.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
