"""
Figure 5: Per-STA Frame Length Heterogeneity — qsrc Fairness-Efficiency Tradeoff

계획: guidelines/step9/fig5.md
출력:
  manuscript/figure/fig5_hetero_frame.{eps,png,pdf}
  results/step9/fig5/presweep.csv
  results/step9/fig5/data.csv

실행:
  python harq_sim/run_step9_fig5.py [--fast] [--out-dir results/step9/fig5]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harq_sim.channel import Channel
from harq_sim.policy import NPCAHARQPolicy
from harq_sim.simulator import Simulator
from harq_sim.sta import STA

# ─────────────────────────────────────────────────────────────────────────────
# 실험 설정
# ─────────────────────────────────────────────────────────────────────────────

SHORT_PPDU = 10   # 짧은 프레임 STA (슬롯)
LONG_PPDU  = 50   # 긴 프레임 STA (슬롯)
N_SHORT    = 5    # 짧은 프레임 STA 수
N_LONG     = 5    # 긴 프레임 STA 수
N_TOTAL    = N_SHORT + N_LONG   # = 10

SEEDS         = [42, 123, 456]

OBSS_MIN       = 20
OBSS_MAX       = 500
OBSS_OCCUPANCY = 0.50

FULL_SLOTS = 50_000
FAST_SLOTS =  5_000

QSRC_CANDIDATES = [0, 1, 2, 3]   # pre-sweep 탐색 범위

CW_MIN = 15


def _occupancy_to_rate(occ: float) -> float:
    mean_dur = (OBSS_MIN + OBSS_MAX) / 2.0
    return occ / (mean_dur * (1.0 - occ))


# ─────────────────────────────────────────────────────────────────────────────
# Custom simulator builder (heterogeneous ppdu)
# ─────────────────────────────────────────────────────────────────────────────

def build_and_run_hetero(
    num_slots:      int,
    obss_rate:      float,
    ppdu_short:     int,
    ppdu_long:      int,
    n_short:        int,
    n_long:         int,
    qsrc_short:     int,
    qsrc_long:      int,
    adaptive_short: bool,
    adaptive_long:  bool,
    seed:           int,
) -> Simulator:
    """N_SHORT STAs with ppdu_short and N_LONG STAs with ppdu_long."""
    random.seed(seed)

    primary = Channel(
        channel_id=0,
        obss_generation_rate=obss_rate,
        obss_duration_range=(OBSS_MIN, OBSS_MAX),
    )
    npca_ch = Channel(channel_id=1, obss_generation_rate=0.0)

    stas: list[STA] = []
    for i in range(n_short):
        policy = NPCAHARQPolicy(adaptive_cw=adaptive_short)
        stas.append(STA(
            sta_id=i,
            primary_channel=primary,
            npca_channel=npca_ch,
            npca_enabled=True,
            ppdu_duration=ppdu_short,
            switching_delay=1,
            switch_back_delay=1,
            npca_min_duration_threshold=0,
            npca_initial_qsrc=qsrc_short,
            retry_limit=7,
            infinite_queue=True,
            snr_db_mean=20.0,
            snr_db_std=0.0,
            harq_enabled=True,
            harq_validity_horizon=200,
            policy=policy,
            adaptive_cw=adaptive_short,
        ))
    for i in range(n_long):
        policy = NPCAHARQPolicy(adaptive_cw=adaptive_long)
        stas.append(STA(
            sta_id=n_short + i,
            primary_channel=primary,
            npca_channel=npca_ch,
            npca_enabled=True,
            ppdu_duration=ppdu_long,
            switching_delay=1,
            switch_back_delay=1,
            npca_min_duration_threshold=0,
            npca_initial_qsrc=qsrc_long,
            retry_limit=7,
            infinite_queue=True,
            snr_db_mean=20.0,
            snr_db_std=0.0,
            harq_enabled=True,
            harq_validity_horizon=200,
            policy=policy,
            adaptive_cw=adaptive_long,
        ))

    sim = Simulator(
        num_slots=num_slots,
        stas=stas,
        channels=[primary, npca_ch],
        enable_trace=False,
    )
    sim.run()
    return sim


def build_and_run_homo(
    num_slots: int,
    obss_rate: float,
    ppdu:      int,
    n_stas:    int,
    qsrc:      int,
    seed:      int,
) -> Simulator:
    """Homogeneous: all N STAs with same ppdu and fixed qsrc (for pre-sweep)."""
    random.seed(seed)

    primary = Channel(
        channel_id=0,
        obss_generation_rate=obss_rate,
        obss_duration_range=(OBSS_MIN, OBSS_MAX),
    )
    npca_ch = Channel(channel_id=1, obss_generation_rate=0.0)

    stas = [
        STA(
            sta_id=i,
            primary_channel=primary,
            npca_channel=npca_ch,
            npca_enabled=True,
            ppdu_duration=ppdu,
            switching_delay=1,
            switch_back_delay=1,
            npca_min_duration_threshold=0,
            npca_initial_qsrc=qsrc,
            retry_limit=7,
            infinite_queue=True,
            snr_db_mean=20.0,
            snr_db_std=0.0,
            harq_enabled=True,
            harq_validity_horizon=200,
            policy=NPCAHARQPolicy(adaptive_cw=False),
            adaptive_cw=False,
        )
        for i in range(n_stas)
    ]

    sim = Simulator(
        num_slots=num_slots,
        stas=stas,
        channels=[primary, npca_ch],
        enable_trace=False,
    )
    sim.run()
    return sim


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Pre-sweep — oracle qsrc* per frame type
# ─────────────────────────────────────────────────────────────────────────────

def run_presweep(num_slots: int, seeds: list[int]) -> tuple[int, int, list[dict]]:
    """Homogeneous sweep to find oracle_short and oracle_long."""
    print("\n=== Phase 1: Pre-sweep (oracle qsrc determination) ===")
    obss_rate = _occupancy_to_rate(OBSS_OCCUPANCY)
    rows: list[dict] = []

    for ppdu_type, ppdu_val in [("short", SHORT_PPDU), ("long", LONG_PPDU)]:
        best_q    = 0
        best_mean = -1.0
        for qsrc in QSRC_CANDIDATES:
            tps = []
            for seed in seeds:
                sim = build_and_run_homo(
                    num_slots=num_slots,
                    obss_rate=obss_rate,
                    ppdu=ppdu_val,
                    n_stas=N_TOTAL,
                    qsrc=qsrc,
                    seed=seed,
                )
                tp = sim.compute_metrics()["aggregate"]["aggregate_throughput"]
                tps.append(tp)
                rows.append({
                    "ppdu_type":            ppdu_type,
                    "ppdu":                 ppdu_val,
                    "qsrc":                 qsrc,
                    "seed":                 seed,
                    "aggregate_throughput": tp,
                })
                print(f"  presweep  ppdu={ppdu_val:2d} ({ppdu_type:5s})  "
                      f"qsrc={qsrc}  seed={seed}  tp={tp}", flush=True)
            mean_tp = statistics.mean(tps)
            if mean_tp > best_mean:
                best_mean = mean_tp
                best_q    = qsrc
        if ppdu_type == "short":
            oracle_short = best_q
            print(f"  → oracle_short = {oracle_short}  (tp={best_mean:.1f})")
        else:
            oracle_long = best_q
            print(f"  → oracle_long  = {oracle_long}  (tp={best_mean:.1f})")

    return oracle_short, oracle_long, rows


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Main heterogeneous experiment
# ─────────────────────────────────────────────────────────────────────────────

def _extract_row(sim: Simulator, method: str, seed: int) -> dict:
    metrics = sim.compute_metrics()

    tp_short = sum(
        metrics[i]["packets_delivered"] for i in range(N_SHORT)
    )
    tp_long = sum(
        metrics[N_SHORT + i]["packets_delivered"] for i in range(N_LONG)
    )
    tp_agg = tp_short + tp_long

    # Jain's fairness on per-STA packets_delivered
    per_sta_tp = [metrics[i]["packets_delivered"] for i in range(N_TOTAL)]
    n = N_TOTAL
    sum_x  = sum(per_sta_tp)
    sum_x2 = sum(x * x for x in per_sta_tp)
    jain = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 1.0

    # mean qsrc per group
    def _mean_qsrc(sta_ids):
        histories = [sim.stas[i]._npca_qsrc_history for i in sta_ids
                     if sim.stas[i]._npca_qsrc_history]
        if not histories:
            # no adaptive history → return the fixed qsrc
            return float(sim.stas[sta_ids[0]].npca_initial_qsrc)
        return float(np.mean([q for h in histories for q in h]))

    mean_qsrc_short = _mean_qsrc(list(range(N_SHORT)))
    mean_qsrc_long  = _mean_qsrc(list(range(N_SHORT, N_TOTAL)))

    col_prob = metrics["aggregate"]["collision_probability_npca"]

    return {
        "method":           method,
        "seed":             seed,
        "tp_short":         tp_short,
        "tp_long":          tp_long,
        "tp_agg":           tp_agg,
        "jain_fairness":    jain,
        "mean_qsrc_short":  mean_qsrc_short,
        "mean_qsrc_long":   mean_qsrc_long,
        "col_prob_npca":    col_prob,
    }


def run_sweep(num_slots: int, oracle_short: int, oracle_long: int) -> list[dict]:
    print("\n=== Phase 2: Main heterogeneous experiment ===")
    obss_rate = _occupancy_to_rate(OBSS_OCCUPANCY)

    methods = [
        {"name": "fixed_q0",        "sq": 0,            "lq": 0,           "as": False, "al": False},
        {"name": "fixed_q1",        "sq": 1,            "lq": 1,           "as": False, "al": False},
        {"name": "fixed_hetero",    "sq": 0,            "lq": 1,           "as": False, "al": False},
        {"name": "oracle_hetero",   "sq": oracle_short, "lq": oracle_long, "as": False, "al": False},
        {"name": "per_sta_adaptive","sq": 0,            "lq": 0,           "as": True,  "al": True },
    ]

    rows: list[dict] = []
    total = len(methods) * len(SEEDS)
    done  = 0

    for m in methods:
        for seed in SEEDS:
            done += 1
            print(f"  [{done:2d}/{total}]  method={m['name']:<20}"
                  f"  sq={m['sq']}  lq={m['lq']}"
                  f"  adaptive={m['as']}/{m['al']}"
                  f"  seed={seed}", flush=True)
            sim = build_and_run_hetero(
                num_slots=num_slots,
                obss_rate=obss_rate,
                ppdu_short=SHORT_PPDU,
                ppdu_long=LONG_PPDU,
                n_short=N_SHORT,
                n_long=N_LONG,
                qsrc_short=m["sq"],
                qsrc_long=m["lq"],
                adaptive_short=m["as"],
                adaptive_long=m["al"],
                seed=seed,
            )
            rows.append(_extract_row(sim, m["name"], seed))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

PRESWEEP_FIELDS = ["ppdu_type", "ppdu", "qsrc", "seed", "aggregate_throughput"]
DATA_FIELDS     = ["method", "seed", "tp_short", "tp_long", "tp_agg",
                   "jain_fairness", "mean_qsrc_short", "mean_qsrc_long", "col_prob_npca"]


def save_csv(rows: list[dict], path: str, fields: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats(rows: list[dict], method: str, metric: str):
    vals = [r[metric] for r in rows if r["method"] == method]
    if not vals:
        return 0.0, 0.0
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

METHOD_ORDER = ["fixed_q0", "fixed_q1", "fixed_hetero", "oracle_hetero", "per_sta_adaptive"]
METHOD_LABELS = {
    "fixed_q0":         "Fixed\nq=0",
    "fixed_q1":         "Fixed\nq=1",
    "fixed_hetero":     "Fixed\nHetero\n(q=0/1)",
    "oracle_hetero":    "Oracle\nHetero",
    "per_sta_adaptive": "Adaptive\n(proposed)",
}


def plot(rows: list[dict], oracle_short: int, oracle_long: int, fig_dir: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 10.5), sharex=False)
    fig.subplots_adjust(hspace=0.40)

    x_pos   = np.arange(len(METHOD_ORDER))
    x_labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    bar_w   = 0.30
    colors  = {"short": "#1f77b4", "long": "#ff7f0e"}

    # ── Panel (a): Per-group throughput ──────────────────────────────────────
    ax = axes[0]
    for gi, (grp, metric) in enumerate([("short", "tp_short"), ("long", "tp_long")]):
        means = []
        stds  = []
        for m in METHOD_ORDER:
            mn, sd = _stats(rows, m, metric)
            means.append(mn)
            stds.append(sd)
        offset = (gi - 0.5) * bar_w
        ax.bar(x_pos + offset, means, bar_w, label=f"{grp.capitalize()} group (ppdu={SHORT_PPDU if grp=='short' else LONG_PPDU})",
               color=colors[grp], alpha=0.85,
               yerr=stds, capsize=4, error_kw={"elinewidth": 1.2})

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Group Throughput\n(delivered packets)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontsize=9, va="top")

    # ── Panel (b): Jain's fairness ───────────────────────────────────────────
    ax = axes[1]
    jain_means = []
    jain_stds  = []
    for m in METHOD_ORDER:
        mn, sd = _stats(rows, m, "jain_fairness")
        jain_means.append(mn)
        jain_stds.append(sd)
    bar_colors = ["#aaaaaa", "#888888", "#555555", "#2ca02c", "#1f77b4"]
    ax.bar(x_pos, jain_means, 0.55, color=bar_colors, alpha=0.85,
           yerr=jain_stds, capsize=4, error_kw={"elinewidth": 1.2})
    ax.axhline(0.95, color="red", lw=1.2, ls="--", label="Fairness threshold (0.95)")
    ax.set_ylim(0.5, 1.05)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Jain's Fairness Index\n(per-STA throughput)", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, fontsize=9, va="top")

    # ── Panel (c): mean qsrc for adaptive method ─────────────────────────────
    ax = axes[2]
    adap_rows = [r for r in rows if r["method"] == "per_sta_adaptive"]
    if adap_rows:
        q_short_vals = [r["mean_qsrc_short"] for r in adap_rows]
        q_long_vals  = [r["mean_qsrc_long"]  for r in adap_rows]
        q_short_mean = statistics.mean(q_short_vals)
        q_long_mean  = statistics.mean(q_long_vals)
        q_short_std  = statistics.stdev(q_short_vals) if len(q_short_vals) > 1 else 0.0
        q_long_std   = statistics.stdev(q_long_vals)  if len(q_long_vals)  > 1 else 0.0

        bar_pos = np.array([0.0, 1.5])
        ax.bar(bar_pos,
               [q_short_mean, q_long_mean],
               0.5,
               yerr=[q_short_std, q_long_std],
               capsize=5, error_kw={"elinewidth": 1.2},
               color=[colors["short"], colors["long"]], alpha=0.85,
               label=["Short group", "Long group"],
               zorder=3)
        ax.axhline(oracle_short, xmin=0.05, xmax=0.45,
                   color=colors["short"], lw=2.0, ls="--",
                   label=f"Oracle q*_short = {oracle_short}", zorder=4)
        ax.axhline(oracle_long, xmin=0.55, xmax=0.95,
                   color=colors["long"], lw=2.0, ls="--",
                   label=f"Oracle q*_long = {oracle_long}", zorder=4)

        ax.set_xticks(bar_pos)
        ax.set_xticklabels([f"Short group\n(ppdu={SHORT_PPDU})",
                            f"Long group\n(ppdu={LONG_PPDU})"], fontsize=10)
        ax.set_yticks(list(range(6)))
        ax.set_yticklabels(
            [f"q={q} (CW={2**q*(CW_MIN+1)-1})" for q in range(6)], fontsize=8)
        ax.set_ylabel("Adaptive mean qsrc", fontsize=10)
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
        ax.text(0.02, 0.97, "(c) Adaptive qsrc convergence (per group)",
                transform=ax.transAxes, fontsize=9, va="top")

    fig.suptitle(
        f"Fig. 5  Heterogeneous Frame Length: SHORT (ppdu={SHORT_PPDU}) + LONG (ppdu={LONG_PPDU})\n"
        f"(N={N_SHORT}+{N_LONG}, OBSS occupancy={int(OBSS_OCCUPANCY*100)}%, obss_max={OBSS_MAX})",
        fontsize=11,
    )

    _save_figure(fig, fig_dir, "fig5_hetero_frame")
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
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(rows: list[dict], oracle_short: int, oracle_long: int) -> None:
    print("\n=== Fig 5 핵심 결과 ===")
    print(f"Pre-sweep oracle:  oracle_short={oracle_short}  oracle_long={oracle_long}")
    print(f"\n{'Method':<22}  {'TP_short':>9}  {'TP_long':>9}  {'TP_agg':>9}"
          f"  {'Jain':>6}  {'q_short':>7}  {'q_long':>7}")
    print("-" * 82)
    for m in METHOD_ORDER:
        tp_s = _stats(rows, m, "tp_short")[0]
        tp_l = _stats(rows, m, "tp_long")[0]
        tp_a = _stats(rows, m, "tp_agg")[0]
        jain = _stats(rows, m, "jain_fairness")[0]
        qs   = _stats(rows, m, "mean_qsrc_short")[0]
        ql   = _stats(rows, m, "mean_qsrc_long")[0]
        print(f"  {m:<20}  {tp_s:>9.0f}  {tp_l:>9.0f}  {tp_a:>9.0f}"
              f"  {jain:>6.3f}  {qs:>7.2f}  {ql:>7.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 5 — Per-STA Frame Length Heterogeneity")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_SLOTS} slots, 1 seed")
    parser.add_argument("--out-dir", default="results/step9/fig5")
    args = parser.parse_args()

    num_slots = FAST_SLOTS if args.fast else FULL_SLOTS
    seeds     = [SEEDS[0]] if args.fast else SEEDS
    out_dir   = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    mode  = "FAST" if args.fast else "FULL"
    rate  = _occupancy_to_rate(OBSS_OCCUPANCY)
    print(f"=== Figure 5 [{mode}] — {num_slots} slots × {len(seeds)} seeds ===")
    print(f"    SHORT ppdu={SHORT_PPDU}  LONG ppdu={LONG_PPDU}  N={N_SHORT}+{N_LONG}")
    print(f"    OBSS occupancy={OBSS_OCCUPANCY:.0%}  (rate={rate:.5f})")

    # Phase 1: pre-sweep
    oracle_short, oracle_long, presweep_rows = run_presweep(num_slots, seeds)
    save_csv(presweep_rows,
             os.path.join(out_dir, "presweep.csv"),
             PRESWEEP_FIELDS)

    # Phase 2: main experiment
    data_rows = run_sweep(num_slots, oracle_short, oracle_long)
    save_csv(data_rows,
             os.path.join(out_dir, "data.csv"),
             DATA_FIELDS)

    print_summary(data_rows, oracle_short, oracle_long)

    print("\nPlotting...")
    plot(data_rows, oracle_short, oracle_long, out_dir)

    print(f"\nFigure 5 완료")
    print(f"  Pre-sweep : {out_dir}/presweep.csv")
    print(f"  데이터    : {out_dir}/data.csv")
    print(f"  논문용    : manuscript/figure/fig5_hetero_frame.{{eps,png,pdf}}")
    print(f"  oracle: short={oracle_short}, long={oracle_long}")


if __name__ == "__main__":
    main()
