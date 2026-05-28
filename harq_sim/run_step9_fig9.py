"""
Figure 9: qsrc Optimization Under Native NPCA Contention (RQ9)

NPCA 채널을 primary로 사용하는 native STA N_n명이 상주할 때
optimal qsrc*가 어떻게 달라지는지 관찰.

Guidelines: guidelines/step9/fig9.md
스크립트: harq_sim/run_step9_fig9.py
출력: manuscript/figure/fig9_native_npca.{eps,png,pdf}

실행:
  python harq_sim/run_step9_fig9.py [--fast] [--out-dir results/step9/fig9]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harq_sim.channel import Channel
from harq_sim.simulator import Simulator
from harq_sim.sta import STA

# ─────────────────────────────────────────────────────────────────────────────
# 실험 설정
# ─────────────────────────────────────────────────────────────────────────────

N_TRANSITION  = 20
N_NATIVE_LIST = [0, 5, 10, 20, 30]
QSRC_LIST     = [0, 1, 2, 3, 4, 5]
SEEDS         = [42, 123, 456]

OBSS_MIN       = 20
OBSS_MAX       = 200
OBSS_OCCUPANCY = 0.30
SNR_DB_MEAN    = 20.0

FULL_SLOTS = 50_000
FAST_SLOTS =  5_000

_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#9467bd"]

CW_MIN = 15


def _occupancy_to_rate(occ: float) -> float:
    mean_dur = (OBSS_MIN + OBSS_MAX) / 2.0
    return occ / (mean_dur * (1.0 - occ))


def _qsrc_to_cw(qsrc: int) -> int:
    return 2 ** qsrc * (CW_MIN + 1) - 1


def _formula_qsrc_star(n_total: int) -> int:
    """Fig 3 closed-form: qsrc* = max(0, round(log2(N/16)))"""
    if n_total <= 0:
        return 0
    return max(0, round(math.log2(n_total / 16)))


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(num_slots: int, qsrc: int, n_native: int, seed: int) -> dict:
    random.seed(seed)
    obss_rate = _occupancy_to_rate(OBSS_OCCUPANCY)

    primary_ch = Channel(
        channel_id=0,
        obss_generation_rate=obss_rate,
        obss_duration_range=(OBSS_MIN, OBSS_MAX),
    )
    npca_ch = Channel(channel_id=1, obss_generation_rate=0.0)

    trans_stas = [
        STA(
            sta_id=i,
            primary_channel=primary_ch,
            npca_channel=npca_ch,
            npca_enabled=True,
            ppdu_duration=20,
            switching_delay=1,
            switch_back_delay=1,
            npca_min_duration_threshold=0,
            npca_initial_qsrc=qsrc,
            retry_limit=7,
            infinite_queue=True,
            snr_db_mean=SNR_DB_MEAN,
            snr_db_std=0.0,
            harq_enabled=True,
            harq_validity_horizon=200,
        )
        for i in range(N_TRANSITION)
    ]

    native_stas = [
        STA(
            sta_id=N_TRANSITION + i,
            primary_channel=npca_ch,    # NPCA 채널이 native STA의 primary
            npca_channel=None,
            npca_enabled=False,
            ppdu_duration=20,
            switching_delay=1,
            switch_back_delay=1,
            npca_initial_qsrc=0,
            retry_limit=7,
            infinite_queue=True,
            snr_db_mean=SNR_DB_MEAN,
            snr_db_std=0.0,
            harq_enabled=True,
            harq_validity_horizon=200,
            ap_on_primary=True,
        )
        for i in range(n_native)
    ]

    sim = Simulator(
        num_slots=num_slots,
        stas=trans_stas + native_stas,
        channels=[primary_ch, npca_ch],
        enable_trace=False,
    )
    sim.run()

    metrics = sim.compute_metrics()

    # Trans STA 집계
    trans_delivered = sum(metrics[i]["packets_delivered"] for i in range(N_TRANSITION))

    trans_npca_tx = sum(
        sim.stas[i].stats["npca_tx_success"] + sim.stas[i].stats["npca_tx_fail"]
        for i in range(N_TRANSITION)
    )
    trans_npca_col = sum(
        sim.stas[i].stats.get("npca_collision_count", 0)
        for i in range(N_TRANSITION)
    )
    trans_col_prob = trans_npca_col / trans_npca_tx if trans_npca_tx > 0 else 0.0
    trans_transitions = sum(metrics[i]["npca_transitions"] for i in range(N_TRANSITION))

    # Native STA 집계
    native_delivered = 0
    native_col_prob  = 0.0
    if n_native > 0:
        native_delivered = sum(
            metrics[N_TRANSITION + i]["packets_delivered"] for i in range(n_native)
        )
        native_tx_total = sum(
            sim.stas[N_TRANSITION + i].stats["primary_tx_success"]
            + sim.stas[N_TRANSITION + i].stats["primary_tx_fail"]
            for i in range(n_native)
        )
        native_col = sum(
            sim.stas[N_TRANSITION + i].stats.get("primary_collision_count", 0)
            for i in range(n_native)
        )
        native_col_prob = native_col / native_tx_total if native_tx_total > 0 else 0.0

    return {
        "qsrc":                        qsrc,
        "N_native":                    n_native,
        "seed":                        seed,
        "trans_throughput":            trans_delivered,
        "trans_collision_prob":        trans_col_prob,
        "trans_npca_transition_count": trans_transitions,
        "native_throughput":           native_delivered,
        "native_collision_prob":       native_col_prob,
        "total_npca_throughput":       trans_delivered + native_delivered,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(num_slots: int) -> list[dict]:
    rows: list[dict] = []
    total = len(N_NATIVE_LIST) * len(QSRC_LIST) * len(SEEDS)
    done  = 0

    for n_native in N_NATIVE_LIST:
        for qsrc in QSRC_LIST:
            for seed in SEEDS:
                done += 1
                print(f"  [{done:3d}/{total}] N_native={n_native:2d}  "
                      f"qsrc={qsrc}  seed={seed}", flush=True)
                row = run_one(num_slots, qsrc, n_native, seed)
                rows.append(row)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = [
    "qsrc", "N_native", "seed",
    "trans_throughput", "trans_collision_prob", "trans_npca_transition_count",
    "native_throughput", "native_collision_prob",
    "total_npca_throughput",
]


def save_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mean_std(rows: list[dict], n_native: int, qsrc: int,
              metric: str) -> tuple[float, float]:
    vals = [r[metric] for r in rows
            if r["N_native"] == n_native and r["qsrc"] == qsrc]
    if not vals:
        return 0.0, 0.0
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def _optimal_qsrc(rows: list[dict], n_native: int) -> tuple[int, float]:
    best_q, best_tp = 0, -1.0
    for q in QSRC_LIST:
        m, _ = _mean_std(rows, n_native, q, "trans_throughput")
        if m > best_tp:
            best_tp, best_q = m, q
    return best_q, best_tp


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(rows: list[dict], fig_dir: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 9.5), sharex=False)
    fig.subplots_adjust(hspace=0.38)

    x_qsrc   = QSRC_LIST
    x_labels = [f"q={q}\n(CW={_qsrc_to_cw(q)})" for q in x_qsrc]

    # ── Panel (a): Trans throughput vs qsrc, lines per N_native ──────────────
    ax_a = axes[0]
    for idx, n_native in enumerate(N_NATIVE_LIST):
        means, stds = [], []
        for q in x_qsrc:
            m, s = _mean_std(rows, n_native, q, "trans_throughput")
            means.append(m)
            stds.append(s)

        color = _COLORS[idx]
        ax_a.plot(x_qsrc, means, color=color, ls="-", marker="o",
                  markersize=5, linewidth=1.6, label=f"$N_n$={n_native}")
        ax_a.fill_between(x_qsrc,
                          [m - s for m, s in zip(means, stds)],
                          [m + s for m, s in zip(means, stds)],
                          color=color, alpha=0.12)

        q_opt, tp_opt = _optimal_qsrc(rows, n_native)
        ax_a.plot(q_opt, tp_opt, marker="*", color=color,
                  markersize=12, zorder=5)

    ax_a.set_ylabel("Transitioning STA\nThroughput (delivered pkts)", fontsize=10)
    ax_a.set_xticks(x_qsrc)
    ax_a.set_xticklabels(x_labels, fontsize=8)
    ax_a.set_xlabel("NPCA Initial CW Exponent (qsrc)", fontsize=9)
    ax_a.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_a.legend(fontsize=9, loc="lower left", ncol=3,
                frameon=True, edgecolor="#cccccc",
                title=f"Native STAs ($N_n$),  $N_t$={N_TRANSITION}",
                title_fontsize=8)
    ax_a.text(0.02, 0.97, "(a)  ★ = optimal qsrc* per $N_n$",
              transform=ax_a.transAxes, fontsize=9, va="top")

    # ── Panel (b): qsrc* vs N_native + formula reference line ────────────────
    ax_b = axes[1]
    opt_qsrcs = [_optimal_qsrc(rows, n)[0] for n in N_NATIVE_LIST]
    ref_qsrcs = [_formula_qsrc_star(N_TRANSITION + n) for n in N_NATIVE_LIST]

    ax_b.plot(N_NATIVE_LIST, opt_qsrcs, color="#4c72b0", ls="-", marker="o",
              markersize=8, linewidth=2.0, label="Actual qsrc* (experiment)")
    ax_b.plot(N_NATIVE_LIST, ref_qsrcs, color="#999999", ls="--", marker="s",
              markersize=6, linewidth=1.4,
              label=r"Formula qsrc$^*(N_t + N_n)$")

    ax_b.set_ylabel("Optimal qsrc*", fontsize=10)
    ax_b.set_xlabel("Number of Native STAs ($N_n$)", fontsize=10)
    ax_b.set_xticks(N_NATIVE_LIST)
    ax_b.set_yticks(QSRC_LIST)
    ax_b.set_yticklabels([f"q={q} (CW={_qsrc_to_cw(q)})" for q in QSRC_LIST],
                         fontsize=8)
    ax_b.set_ylim(-0.3, max(max(opt_qsrcs), max(ref_qsrcs)) + 0.5)
    ax_b.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_b.legend(fontsize=9, loc="upper left", frameon=True, edgecolor="#cccccc")
    ax_b.text(0.02, 0.97, "(b)  qsrc* vs. native STA count",
              transform=ax_b.transAxes, fontsize=9, va="top")

    # ── Panel (c): Trans TP gain at qsrc* vs qsrc=0 per N_native ─────────────
    ax_c = axes[2]
    gains = []
    for n_native in N_NATIVE_LIST:
        q_opt, tp_opt = _optimal_qsrc(rows, n_native)
        tp_base, _    = _mean_std(rows, n_native, 0, "trans_throughput")
        gain = (tp_opt - tp_base) / tp_base * 100 if tp_base > 0 else 0.0
        gains.append(gain)

    bars = ax_c.bar(range(len(N_NATIVE_LIST)), gains, color="#4c72b0", alpha=0.75,
                    label="Trans STA gain (qsrc* vs. qsrc=0)")
    ax_c.axhline(0, color="black", linewidth=0.8, linestyle=":")

    for bar, g in zip(bars, gains):
        ax_c.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.1,
                  f"{g:.1f}%", ha="center", va="bottom", fontsize=8)

    ax_c.set_ylabel("Throughput Gain at qsrc*\nvs. qsrc=0 (%)", fontsize=10)
    ax_c.set_xlabel("Number of Native STAs ($N_n$)", fontsize=10)
    ax_c.set_xticks(range(len(N_NATIVE_LIST)))
    ax_c.set_xticklabels([f"$N_n$={n}" for n in N_NATIVE_LIST], fontsize=9)
    ax_c.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax_c.legend(fontsize=9, frameon=True, edgecolor="#cccccc")
    ax_c.text(0.02, 0.97, "(c)  Gain of optimal qsrc selection",
              transform=ax_c.transAxes, fontsize=9, va="top")

    fig.suptitle(
        "Fig. 9  qsrc Optimization Under Native NPCA Contention\n"
        f"($N_t$={N_TRANSITION} transitioning STAs, OBSS occ.={OBSS_OCCUPANCY:.0%})",
        fontsize=11,
    )

    _save_figure(fig, fig_dir, "fig9_native_npca")
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
        description="Figure 9 — qsrc* under native NPCA contention (RQ9)")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_SLOTS} slots")
    parser.add_argument("--out-dir", default="results/step9/fig9")
    args = parser.parse_args()

    num_slots = FAST_SLOTS if args.fast else FULL_SLOTS
    out_dir   = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    mode = "FAST" if args.fast else "FULL"
    rate = _occupancy_to_rate(OBSS_OCCUPANCY)
    print(f"=== Figure 9 [{mode}] — {num_slots} slots × {len(SEEDS)} seeds ===")
    print(f"    N_transition={N_TRANSITION},  N_native sweep={N_NATIVE_LIST}")
    print(f"    OBSS occupancy={OBSS_OCCUPANCY:.0%}  (rate={rate:.6f})")
    total = len(N_NATIVE_LIST) * len(QSRC_LIST) * len(SEEDS)
    print(f"    {len(N_NATIVE_LIST)} × {len(QSRC_LIST)} × {len(SEEDS)} = {total} runs")

    rows = run_sweep(num_slots)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    print("Plotting...")
    plot(rows, out_dir)

    print(f"\nFigure 9 완료")
    print(f"  데이터  : {csv_path}")
    print(f"  논문용  : manuscript/figure/fig9_native_npca.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
