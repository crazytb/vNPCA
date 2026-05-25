"""
Figure 2: Primary CW 크기 → NPCA Transition Delay 이득 (RQ2)

Guidelines §30 RQ2: primary CW가 큰 상황에서 NPCA transition은 delay를 줄이는가?

계획: guidelines/step9/fig2.md
출력:
  manuscript/figure/fig2_delay_vs_nstas.{eps,png,pdf}
  results/step9/fig2/data.csv

실행:
  python harq_sim/run_step9_fig2.py [--fast] [--out-dir results/step9/fig2]
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

from harq_sim.run_step8 import build_and_run

# ─────────────────────────────────────────────────────────────────────────────
# 실험 설정
# ─────────────────────────────────────────────────────────────────────────────

NUM_STAS_LIST    = [2, 5, 10, 15, 20]
SEEDS            = [42, 123, 456]

OBSS_MIN         = 20
OBSS_MAX         = 200
OBSS_OCCUPANCY   = 0.30   # 목표 OBSS 채널 점유율 (30%)

def _occupancy_to_rate(occ: float) -> float:
    """목표 OBSS 점유율 → per-slot 도착률 변환."""
    mean_dur = (OBSS_MIN + OBSS_MAX) / 2.0
    return occ / (mean_dur * (1.0 - occ))

BASELINES = [
    dict(label="Legacy EDCA",   npca_enabled=False, harq_enabled=False,
         color="#4c72b0", ls="-",  marker="o"),
    dict(label="ARQ-only NPCA", npca_enabled=True,  harq_enabled=False,
         color="#dd8452", ls="--", marker="s"),
]

FULL_SLOTS = 50_000
FAST_SLOTS =  5_000

# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(num_slots: int) -> list[dict]:
    rows: list[dict] = []
    total = len(NUM_STAS_LIST) * len(BASELINES) * len(SEEDS)
    done  = 0

    for num_stas in NUM_STAS_LIST:
        for bl in BASELINES:
            for seed in SEEDS:
                done += 1
                print(f"  [{done}/{total}] num_stas={num_stas:2d}  "
                      f"{bl['label']:<16}  seed={seed}", flush=True)

                sim = build_and_run(
                    num_slots       = num_slots,
                    num_stas        = num_stas,
                    obss_rate       = _occupancy_to_rate(OBSS_OCCUPANCY),
                    obss_min        = OBSS_MIN,
                    obss_max        = OBSS_MAX,
                    npca_qsrc       = 0,
                    npca_threshold  = 0,
                    ppdu_duration   = 20,
                    harq_horizon    = 200,
                    snr_db_mean     = 14.0,
                    snr_db_std      = 2.0,
                    npca_enabled    = bl["npca_enabled"],
                    harq_enabled    = bl["harq_enabled"],
                    adaptive_cw     = False,
                    seed            = seed,
                    enable_trace    = False,
                )
                agg = sim.compute_metrics()["aggregate"]
                rows.append({
                    "num_stas":            num_stas,
                    "baseline":            bl["label"],
                    "seed":                seed,
                    "mean_access_delay":   agg["mean_access_delay"],
                    "per_sta_throughput":  agg["aggregate_throughput"] / num_stas,
                })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = ["num_stas", "baseline", "seed",
          "mean_access_delay", "per_sta_throughput"]

def save_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats(rows, num_stas, label, metric):
    vals = [r[metric] for r in rows
            if r["num_stas"] == num_stas and r["baseline"] == label]
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(rows: list[dict], fig_dir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    x = NUM_STAS_LIST

    # ── Row A: mean_access_delay ──────────────────────────────────────────
    ax_a = axes[0]
    delay_vals = {}
    for bl in BASELINES:
        means, stds = [], []
        for n in x:
            m, s = _stats(rows, n, bl["label"], "mean_access_delay")
            means.append(m)
            stds.append(s)
        delay_vals[bl["label"]] = means
        ax_a.plot(x, means, color=bl["color"], ls=bl["ls"],
                  marker=bl["marker"], markersize=6, linewidth=1.8,
                  label=bl["label"])
        ax_a.fill_between(x,
                          [m - s for m, s in zip(means, stds)],
                          [m + s for m, s in zip(means, stds)],
                          color=bl["color"], alpha=0.15)

    # gap 주석 (num_stas=20 지점)
    idx_last = len(x) - 1
    y_legacy = delay_vals["Legacy EDCA"][idx_last]
    y_npca   = delay_vals["ARQ-only NPCA"][idx_last]
    gap      = y_legacy - y_npca
    x_ann    = x[idx_last]
    ax_a.annotate(
        "",
        xy=(x_ann + 0.3, y_npca),
        xytext=(x_ann + 0.3, y_legacy),
        arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2),
    )
    ax_a.text(x_ann + 0.7, (y_legacy + y_npca) / 2,
              f"Δ = {gap:.0f}\nslots",
              va="center", ha="left", fontsize=8, color="#555555")

    ax_a.set_ylabel("Mean Access Delay (slots)", fontsize=10)
    ax_a.set_ylim(bottom=0)
    ax_a.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_a.tick_params(labelsize=9)
    ax_a.text(0.02, 0.97, "(a)", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold", va="top")
    ax_a.legend(fontsize=9, loc="upper left", frameon=True, edgecolor="#cccccc")

    # ── Row B: per-STA throughput ─────────────────────────────────────────
    ax_b = axes[1]
    for bl in BASELINES:
        means, stds = [], []
        for n in x:
            m, s = _stats(rows, n, bl["label"], "per_sta_throughput")
            means.append(m)
            stds.append(s)
        ax_b.plot(x, means, color=bl["color"], ls=bl["ls"],
                  marker=bl["marker"], markersize=6, linewidth=1.8,
                  label=bl["label"])
        ax_b.fill_between(x,
                          [m - s for m, s in zip(means, stds)],
                          [m + s for m, s in zip(means, stds)],
                          color=bl["color"], alpha=0.15)

    ax_b.set_ylabel("Per-STA Throughput\n(delivered packets / STA)", fontsize=10)
    ax_b.set_xlabel("Number of STAs", fontsize=10)
    ax_b.set_ylim(0, None)
    ax_b.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_b.tick_params(labelsize=9)
    ax_b.set_xticks(x)
    ax_b.text(0.02, 0.97, "(b)", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold", va="top")

    fig.suptitle("Fig. 2  Primary CW Growth vs. NPCA Delay Benefit", fontsize=11)

    _save_figure(fig, fig_dir, "fig2_delay_vs_nstas")
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
        description="Figure 2 — Primary CW growth vs NPCA delay benefit (RQ2)")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_SLOTS} slots")
    parser.add_argument("--out-dir", default="results/step9/fig2")
    args = parser.parse_args()

    num_slots = FAST_SLOTS if args.fast else FULL_SLOTS
    out_dir   = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    mode = "FAST" if args.fast else "FULL"
    print(f"=== Figure 2 [{mode}] — {num_slots} slots × {len(SEEDS)} seeds ===")

    rows = run_sweep(num_slots)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    print("Plotting...")
    plot(rows, out_dir)

    print(f"\nFigure 2 완료")
    print(f"  데이터  : {csv_path}")
    print(f"  논문용  : manuscript/figure/fig2_delay_vs_nstas.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
