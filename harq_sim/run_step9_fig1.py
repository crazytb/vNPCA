"""
Figure 1: NPCA-HARQ 조합 이득 검증 (RQ1)

Guidelines §30 RQ1: NPCA-HARQ는 HARQ-only 또는 NPCA-only보다
throughput/PDR/delay 측면에서 이득이 있는가?

계획: guidelines/step9/fig1.md
출력:
  manuscript/figure/fig1_combined.{eps,png,pdf}
  results/step9/fig1/data.csv

실행:
  python harq_sim/run_step9_fig1.py [--fast] [--out-dir results/step9/fig1]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from harq_sim.run_step8 import build_and_run

# ─────────────────────────────────────────────────────────────────────────────
# 실험 설정
# ─────────────────────────────────────────────────────────────────────────────

# OBSS 점유율(occupancy)로 설정 — 내부에서 per-slot 도착률로 변환
# obss_rate = occupancy / (mean_duration × (1 - occupancy))
OBSS_OCCUPANCIES = [0.10, 0.20, 0.30, 0.50]
SEEDS            = [42, 123, 456]

OBSS_MIN = 20
OBSS_MAX = 200

def _occupancy_to_rate(occ: float) -> float:
    """목표 OBSS 점유율 → per-slot 도착률 변환."""
    mean_dur = (OBSS_MIN + OBSS_MAX) / 2.0
    return occ / (mean_dur * (1.0 - occ))

BASELINES = [
    dict(label="Legacy EDCA",       npca_enabled=False, harq_enabled=False, adaptive_cw=False,
         color="#4c72b0", ls="-",  marker="o"),
    dict(label="ARQ-only NPCA",     npca_enabled=True,  harq_enabled=False, adaptive_cw=False,
         color="#dd8452", ls="--", marker="s"),
    dict(label="HARQ-only",         npca_enabled=False, harq_enabled=True,  adaptive_cw=False,
         color="#55a868", ls="-.", marker="^"),
    dict(label="Fixed-CW NPCA-HARQ",npca_enabled=True,  harq_enabled=True,  adaptive_cw=False,
         color="#c44e52", ls=":",  marker="D"),
]

FULL_SLOTS = 50_000
FAST_SLOTS =  5_000

# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(num_slots: int) -> list[dict]:
    """obss_rate × baseline × seed sweep. Returns list of result dicts."""
    rows: list[dict] = []
    total = len(OBSS_OCCUPANCIES) * len(BASELINES) * len(SEEDS)
    done  = 0

    for occ in OBSS_OCCUPANCIES:
        obss_rate = _occupancy_to_rate(occ)
        for bl in BASELINES:
            for seed in SEEDS:
                done += 1
                print(f"  [{done}/{total}] occupancy={occ:.0%}  "
                      f"(rate={obss_rate:.4f})  {bl['label']:<24}  seed={seed}", flush=True)

                sim = build_and_run(
                    num_slots       = num_slots,
                    num_stas        = 5,
                    obss_rate       = obss_rate,
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
                    adaptive_cw     = bl["adaptive_cw"],
                    seed            = seed,
                    enable_trace    = False,
                )
                agg = sim.compute_metrics()["aggregate"]
                rows.append({
                    "obss_rate":          occ,       # 점유율(occupancy) 저장
                    "baseline":           bl["label"],
                    "seed":               seed,
                    "aggregate_throughput": agg["aggregate_throughput"],
                    "mean_access_delay":    agg["mean_access_delay"],
                    "packet_delivery_ratio": agg["packet_delivery_ratio"],
                })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = ["obss_rate", "baseline", "seed",
          "aggregate_throughput", "mean_access_delay", "packet_delivery_ratio"]

def save_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _agg_stats(rows: list[dict], obss_rate: float, label: str,
               metric: str) -> tuple[float, float]:
    """Return (mean, std) over seeds for given (obss_rate, label, metric)."""
    vals = [r[metric] for r in rows
            if r["obss_rate"] == obss_rate and r["baseline"] == label]
    if not vals:
        return 0.0, 0.0
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(rows: list[dict], fig_dir: str) -> None:
    """Generate 3-row subplot and save to manuscript/figure/ and fig_dir."""
    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    x = OBSS_OCCUPANCIES

    metrics = [
        ("aggregate_throughput",   "Throughput\n(delivered packets)",   False),
        ("mean_access_delay",      "Mean Access Delay\n(slots)",        False),
        ("packet_delivery_ratio",  "Packet Delivery Ratio (PDR)",       True),
    ]

    for ax, (metric, ylabel, is_pct) in zip(axes, metrics):
        for bl in BASELINES:
            means, stds = [], []
            for r in x:
                m, s = _agg_stats(rows, r, bl["label"], metric)
                means.append(m)
                stds.append(s)

            ax.plot(x, means, color=bl["color"], ls=bl["ls"],
                    marker=bl["marker"], markersize=5, linewidth=1.5,
                    label=bl["label"])
            ax.fill_between(x,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=bl["color"], alpha=0.15)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax.tick_params(axis="both", labelsize=9)
        if is_pct:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _: f"{v:.2f}"))
            ax.set_ylim(0, 1.05)

    axes[-1].set_xlabel("OBSS Channel Occupancy", fontsize=10)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"{int(v*100)}%" for v in x])

    # 서브플롯 레이블
    for ax, letter in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.text(0.02, 0.97, letter, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

    # 범례 — 최상단 서브플롯 위
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.01),
               frameon=True, edgecolor="#cccccc")

    fig.suptitle("Fig. 1  NPCA-HARQ Combination Gain vs. Baselines",
                 fontsize=11, y=1.04)

    _save_figure(fig, fig_dir, "fig1_combined")
    plt.close(fig)


def _save_figure(fig, fig_dir: str, name: str) -> None:
    """Save eps/png/pdf to manuscript/figure/ and an extra png to fig_dir."""
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

    # fig_dir 에도 png 미리보기 저장
    preview = os.path.join(fig_dir, f"{name}_preview.png")
    fig.savefig(preview, format="png", dpi=150, bbox_inches="tight")
    print(f"  Preview → {preview}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 1 — NPCA-HARQ combination gain (RQ1)")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_SLOTS} slots instead of {FULL_SLOTS}")
    parser.add_argument("--out-dir", default="results/step9/fig1",
                        help="Directory for data.csv and preview png")
    args = parser.parse_args()

    num_slots = FAST_SLOTS if args.fast else FULL_SLOTS
    out_dir   = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    mode = "FAST" if args.fast else "FULL"
    print(f"=== Figure 1 [{mode}] — {num_slots} slots × {len(SEEDS)} seeds ===")

    rows = run_sweep(num_slots)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    print("Plotting...")
    plot(rows, out_dir)

    print(f"\nFigure 1 완료")
    print(f"  데이터  : {csv_path}")
    print(f"  논문용  : manuscript/figure/fig1_combined.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
