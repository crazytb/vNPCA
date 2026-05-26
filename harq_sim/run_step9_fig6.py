"""
Figure 6: HARQ-CC combining vs ARQ (higher MCS) tradeoff via SNR sweep.

비교:
  arq_only_npca      — NPCA 있음, HARQ 없음 (재시도마다 새 MCS)
  fixed_cw_npca_harq — NPCA 있음, HARQ 있음 (원본 MCS 고정 + combining)

sweep: snr_db_mean ∈ {6,8,10,12,14,16,18,20,25,30}
env  : num_stas=5, obss_rate=0.30, obss_max=200, 50000 slots × 3 seeds

출력:
  results/step9/fig6/data.csv
  manuscript/figure/fig6_harq_vs_arq_snr.{eps,png,pdf}
"""

import csv
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harq_sim.channel import Channel
from harq_sim.policy import NPCAHARQPolicy
from harq_sim.sta import STA
from harq_sim.simulator import Simulator

# ── 실험 상수 ─────────────────────────────────────────────────────────────────
SNR_LIST        = [6, 8, 10, 12, 14, 16, 18, 20, 25, 30]
SEEDS           = [42, 123, 456]
NUM_STAS        = 5
OBSS_RATE       = 0.30
OBSS_MIN        = 20
OBSS_MAX        = 200
PPDU_DURATION   = 20
HARQ_HORIZON    = 200
NPCA_THRESHOLD  = 0
NPCA_QSRC       = 0
NUM_SLOTS       = 50_000

METHODS = ["arq_only_npca", "fixed_cw_npca_harq"]

# MCS SNR thresholds — x축 보조 레이블용 (harq_sim/phy.py 기준)
MCS_BOUNDARIES = {0: 5, 1: 8, 2: 11, 3: 14, 4: 17, 5: 20, 6: 23, 7: 26}


# ── Simulation builder ────────────────────────────────────────────────────────

def build_and_run(
    num_slots: int,
    snr_db_mean: float,
    harq_enabled: bool,
    seed: int,
) -> Simulator:
    random.seed(seed)
    primary = Channel(
        channel_id=0,
        obss_generation_rate=OBSS_RATE,
        obss_duration_range=(OBSS_MIN, OBSS_MAX),
    )
    npca_ch = Channel(channel_id=1, obss_generation_rate=0.0)
    policy = NPCAHARQPolicy(adaptive_cw=False)
    stas = [
        STA(
            sta_id=i,
            primary_channel=primary,
            npca_channel=npca_ch,
            npca_enabled=True,
            ppdu_duration=PPDU_DURATION,
            switching_delay=1,
            switch_back_delay=1,
            npca_min_duration_threshold=NPCA_THRESHOLD,
            npca_initial_qsrc=NPCA_QSRC,
            retry_limit=7,
            infinite_queue=True,
            snr_db_mean=snr_db_mean,
            snr_db_std=0.0,
            harq_enabled=harq_enabled,
            harq_validity_horizon=HARQ_HORIZON,
            policy=policy,
            adaptive_cw=False,
        )
        for i in range(NUM_STAS)
    ]
    sim = Simulator(
        num_slots=num_slots,
        stas=stas,
        channels=[primary, npca_ch],
        enable_trace=False,
    )
    sim.run()
    return sim


def _extract_row(sim: Simulator, method: str, snr_db: float, seed: int) -> dict:
    metrics = sim.compute_metrics()
    agg = metrics["aggregate"]

    total_harq_success = sum(
        v.get("harq_tx_success", 0)
        for k, v in metrics.items() if k != "aggregate"
    )
    total_harq_fail = sum(
        v.get("harq_tx_fail", 0)
        for k, v in metrics.items() if k != "aggregate"
    )
    harq_success_rate = total_harq_success / (total_harq_success + total_harq_fail + 1e-9)

    return {
        "method":            method,
        "snr_db":            snr_db,
        "seed":              seed,
        "aggregate_throughput": agg.get("aggregate_throughput", 0),
        "harq_success_rate": harq_success_rate,
        "harq_tx_success":   total_harq_success,
        "harq_tx_fail":      total_harq_fail,
    }


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(num_slots: int = NUM_SLOTS) -> list[dict]:
    rows = []
    total = len(SNR_LIST) * len(METHODS) * len(SEEDS)
    done = 0
    for snr_db in SNR_LIST:
        for method in METHODS:
            harq_enabled = (method == "fixed_cw_npca_harq")
            for seed in SEEDS:
                sim = build_and_run(num_slots, snr_db, harq_enabled, seed)
                rows.append(_extract_row(sim, method, snr_db, seed))
                done += 1
                print(f"  [{done}/{total}] {method} SNR={snr_db}dB seed={seed} "
                      f"TP={rows[-1]['aggregate_throughput']}")
    return rows


# ── Plot ──────────────────────────────────────────────────────────────────────

def _mean_std(rows: list[dict], method: str, metric: str) -> tuple[list, list, list]:
    snrs = sorted({r["snr_db"] for r in rows})
    means, stds = [], []
    for snr in snrs:
        vals = [r[metric] for r in rows if r["method"] == method and r["snr_db"] == snr]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return snrs, means, stds


def _find_crossover(snrs_a, means_a, snrs_b, means_b) -> float | None:
    """Return first SNR where method A crosses below method B."""
    for i in range(len(snrs_a) - 1):
        diff_now  = means_a[i]   - means_b[i]
        diff_next = means_a[i+1] - means_b[i+1]
        if diff_now * diff_next < 0:
            # linear interpolation
            frac = diff_now / (diff_now - diff_next)
            return snrs_a[i] + frac * (snrs_a[i+1] - snrs_a[i])
    return None


def plot(rows: list[dict], fig_dir: str, out_dir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    style = {
        "arq_only_npca":       dict(color="#1f77b4", linestyle="-",  label="ARQ-only NPCA"),
        "fixed_cw_npca_harq":  dict(color="#d62728", linestyle="--", label="HARQ-CC NPCA"),
    }

    # ── Panel (a): aggregate throughput ──────────────────────────────────────
    ax = axes[0]
    crossover_snr = None
    tp_a_data, tp_b_data = None, None
    for method in METHODS:
        snrs, means, stds = _mean_std(rows, method, "aggregate_throughput")
        means_arr = np.array(means)
        stds_arr  = np.array(stds)
        s = style[method]
        ax.plot(snrs, means_arr, linestyle=s["linestyle"], color=s["color"],
                marker="o", label=s["label"])
        ax.fill_between(snrs, means_arr - stds_arr, means_arr + stds_arr,
                        color=s["color"], alpha=0.15)
        if method == "arq_only_npca":
            tp_a_data = (snrs, means)
        else:
            tp_b_data = (snrs, means)

    if tp_a_data and tp_b_data:
        crossover_snr = _find_crossover(tp_a_data[0], tp_a_data[1],
                                        tp_b_data[0], tp_b_data[1])
    if crossover_snr is not None:
        ax.axvline(crossover_snr, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Crossover ≈ {crossover_snr:.1f} dB")

    ax.set_ylabel("Aggregate Throughput (packets)")
    ax.set_title("(a) Throughput vs SNR")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    # MCS boundary tick marks on top x-axis
    ax2 = ax.twiny()
    mcs_snrs  = list(MCS_BOUNDARIES.values())
    mcs_labels = [f"MCS{k}" for k in MCS_BOUNDARIES]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(mcs_snrs)
    ax2.set_xticklabels(mcs_labels, fontsize=7)

    # ── Panel (b): HARQ success rate ─────────────────────────────────────────
    ax = axes[1]
    for method in METHODS:
        snrs, means, stds = _mean_std(rows, method, "harq_success_rate")
        means_arr = np.array(means)
        stds_arr  = np.array(stds)
        s = style[method]
        ax.plot(snrs, means_arr, linestyle=s["linestyle"], color=s["color"],
                marker="s", label=s["label"])
        ax.fill_between(snrs, means_arr - stds_arr, means_arr + stds_arr,
                        color=s["color"], alpha=0.15)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Rate = 0.5")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("HARQ Success Rate")
    ax.set_title("(b) HARQ Success Rate vs SNR")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()

    fig_name = "fig6_harq_vs_arq_snr"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    for fmt, kw in [("eps", {}), ("png", {"dpi": 300}), ("pdf", {})]:
        fig.savefig(os.path.join(fig_dir, f"{fig_name}.{fmt}"),
                    format=fmt, bbox_inches="tight", **kw)
    # preview copy in out_dir
    fig.savefig(os.path.join(out_dir, f"{fig_name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {fig_dir}/{fig_name}.{{eps,png,pdf}}")
    if crossover_snr is not None:
        print(f"Crossover SNR ≈ {crossover_snr:.1f} dB")
    else:
        print("Crossover: not found in sweep range")


# ── CSV save ──────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "data.csv")
    fields = ["method", "snr_db", "seed",
              "aggregate_throughput", "harq_success_rate",
              "harq_tx_success", "harq_tx_fail"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    out_dir = os.path.join(project_dir, "results", "step9", "fig6")
    fig_dir = os.path.join(project_dir, "manuscript", "figure")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=out_dir)
    parser.add_argument("--slots", type=int, default=NUM_SLOTS)
    args = parser.parse_args()

    print("=== Fig 6: HARQ-CC vs ARQ SNR sweep ===")
    print(f"SNR list: {SNR_LIST}")
    print(f"Methods: {METHODS}")
    print(f"Seeds: {SEEDS}, slots: {args.slots}")
    print()

    rows = run_sweep(args.slots)
    save_csv(rows, args.out_dir)
    plot(rows, fig_dir, args.out_dir)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
