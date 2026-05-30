"""
Figure 11: PPDU Duration Effect on qsrc* — W_avail = W − PPDU Unification

연구 질문 (RQ11): qsrc*는 W와 PPDU의 개별 값이 아닌,
  유효 경쟁 창 W_avail = W − PPDU_duration 에 의해 결정되는가?

두 파트:
  Part 1 — W=170 고정, PPDU × N sweep: qsrc*(PPDU) 관찰
  Part 2 — 동일 W_avail을 다른 (W,PPDU) 조합으로 달성: collapse 검증

계획: guidelines/step9/fig11.md
출력:
  manuscript/figure/fig11_ppdu_weff.{eps,png,pdf}
  results/step9/fig11/data.csv

실행:
  python harq_sim/run_step9_fig11.py [--fast] [--out-dir results/step9/fig11]
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

W_FIXED       = 170
PPDU_LIST     = [10, 20, 30, 50, 80, 120]
NUM_STAS_LIST = [10, 20, 30, 50]
QSRC_LIST     = [0, 1, 2, 3, 4, 5]
SEEDS         = [42, 123, 456]

COLLAPSE_N = 30
# (W_avail_target, [(W, PPDU), ...])
COLLAPSE_CONFIGS: list[tuple[int, list[tuple[int, int]]]] = [
    (30,  [(50, 20), (80, 50), (110, 80)]),
    (80,  [(100, 20), (130, 50), (160, 80)]),
    (150, [(170, 20), (200, 50), (230, 80)]),
]

OBSS_OCCUPANCY = 0.50
FULL_SLOTS = 50_000
FAST_SLOTS =  5_000

FAST_PPDU_LIST        = [20, 50, 80]
FAST_N_LIST           = [10, 30]
FAST_SEEDS            = [42]
FAST_COLLAPSE_CONFIGS: list[tuple[int, list[tuple[int, int]]]] = [
    (80, [(100, 20), (130, 50), (160, 80)]),
]

CW_MIN = 15

# Panel (a)에 표시할 PPDU 값 (5개 대표, PPDU=30 제외)
PPDU_PANEL_A = [10, 20, 50, 80, 120]

_COLORS_N    = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
_COLORS_PPDU = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
_COLLAPSE_COLORS = ["#e377c2", "#7f7f7f", "#bcbd22"]
_PPDU_MARKERS = {20: "o", 50: "s", 80: "^"}


def _occupancy_to_rate(occ: float, W: int) -> float:
    return occ / (W * (1.0 - occ))


def _qsrc_to_cw(qsrc: int) -> int:
    return 2 ** qsrc * (CW_MIN + 1) - 1


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: W=W_FIXED 고정, PPDU × N × qsrc × seed
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep_part1(num_slots: int, ppdu_list: list, n_list: list,
                    seeds: list) -> list[dict]:
    rows: list[dict] = []
    obss_rate = _occupancy_to_rate(OBSS_OCCUPANCY, W_FIXED)
    total = len(ppdu_list) * len(n_list) * len(QSRC_LIST) * len(seeds)
    done  = 0

    for ppdu in ppdu_list:
        for num_stas in n_list:
            for qsrc in QSRC_LIST:
                cw = _qsrc_to_cw(qsrc)
                for seed in seeds:
                    done += 1
                    print(f"  Part1 [{done:3d}/{total}]  W={W_FIXED}  "
                          f"PPDU={ppdu:3d}  N={num_stas:2d}  "
                          f"qsrc={qsrc}  seed={seed}", flush=True)

                    sim = build_and_run(
                        num_slots      = num_slots,
                        num_stas       = num_stas,
                        obss_rate      = obss_rate,
                        obss_min       = W_FIXED,
                        obss_max       = W_FIXED,
                        npca_qsrc      = qsrc,
                        npca_threshold = 0,
                        ppdu_duration  = ppdu,
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
                        "part":                       1,
                        "W":                          W_FIXED,
                        "ppdu_duration":              ppdu,
                        "W_avail":                    W_FIXED - ppdu,
                        "num_stas":                   num_stas,
                        "npca_qsrc":                  qsrc,
                        "npca_cw":                    cw,
                        "seed":                       seed,
                        "collision_probability_npca": agg["collision_probability_npca"],
                        "aggregate_throughput":       agg["aggregate_throughput"],
                        "mean_access_delay":          agg["mean_access_delay"],
                        "npca_transition_count":      agg["npca_transition_count"],
                    })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: W_avail Collapse 검증 — N=COLLAPSE_N, 다른 (W,PPDU) 쌍
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep_part2(num_slots: int, collapse_configs: list,
                    seeds: list) -> list[dict]:
    rows: list[dict] = []
    total = (sum(len(pairs) for _, pairs in collapse_configs)
             * len(QSRC_LIST) * len(seeds))
    done  = 0

    for w_avail_target, pairs in collapse_configs:
        for (W, ppdu) in pairs:
            obss_rate = _occupancy_to_rate(OBSS_OCCUPANCY, W)
            for qsrc in QSRC_LIST:
                cw = _qsrc_to_cw(qsrc)
                for seed in seeds:
                    done += 1
                    print(f"  Part2 [{done:3d}/{total}]  W_avail={w_avail_target}  "
                          f"W={W:3d}  PPDU={ppdu:3d}  "
                          f"qsrc={qsrc}  seed={seed}", flush=True)

                    sim = build_and_run(
                        num_slots      = num_slots,
                        num_stas       = COLLAPSE_N,
                        obss_rate      = obss_rate,
                        obss_min       = W,
                        obss_max       = W,
                        npca_qsrc      = qsrc,
                        npca_threshold = 0,
                        ppdu_duration  = ppdu,
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
                        "part":                       2,
                        "W":                          W,
                        "ppdu_duration":              ppdu,
                        "W_avail":                    W - ppdu,
                        "num_stas":                   COLLAPSE_N,
                        "npca_qsrc":                  qsrc,
                        "npca_cw":                    cw,
                        "seed":                       seed,
                        "collision_probability_npca": agg["collision_probability_npca"],
                        "aggregate_throughput":       agg["aggregate_throughput"],
                        "mean_access_delay":          agg["mean_access_delay"],
                        "npca_transition_count":      agg["npca_transition_count"],
                    })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = ["part", "W", "ppdu_duration", "W_avail", "num_stas",
          "npca_qsrc", "npca_cw", "seed",
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

def _stats(rows: list[dict], W: int, ppdu: int, num_stas: int, qsrc: int,
           metric: str) -> tuple[float, float]:
    vals = [r[metric] for r in rows
            if r["W"] == W and r["ppdu_duration"] == ppdu
            and r["num_stas"] == num_stas and r["npca_qsrc"] == qsrc]
    if not vals:
        return 0.0, 0.0
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def _optimal_qsrc(rows: list[dict], W: int, ppdu: int,
                  num_stas: int) -> tuple[int, float]:
    best_qsrc, best_tp = 0, -1.0
    for q in QSRC_LIST:
        m, _ = _stats(rows, W, ppdu, num_stas, q, "aggregate_throughput")
        if m > best_tp:
            best_tp, best_qsrc = m, q
    return best_qsrc, best_tp


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(rows: list[dict], fig_dir: str,
         ppdu_list: list, n_list: list,
         collapse_configs: list) -> None:
    rows_p1 = [r for r in rows if r["part"] == 1]
    rows_p2 = [r for r in rows if r["part"] == 2]

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 9.5), sharex=False)
    fig.subplots_adjust(hspace=0.45)

    x_qsrc   = QSRC_LIST
    x_labels = [f"q={q}\n(CW={_qsrc_to_cw(q)})" for q in x_qsrc]
    FIXED_N  = COLLAPSE_N if COLLAPSE_N in n_list else n_list[-1]

    # ── Panel (a): Throughput vs qsrc, N=30, W=170 fixed, lines per PPDU ─────
    ax_a = axes[0]
    ppdu_show = [p for p in PPDU_PANEL_A if p in ppdu_list]

    for idx, ppdu in enumerate(ppdu_show):
        means, stds = [], []
        for q in x_qsrc:
            m, s = _stats(rows_p1, W_FIXED, ppdu, FIXED_N, q,
                          "aggregate_throughput")
            means.append(m); stds.append(s)

        color = _COLORS_PPDU[idx % len(_COLORS_PPDU)]
        ax_a.plot(x_qsrc, means, color=color, ls="-", marker="o",
                  markersize=5, linewidth=1.6, label=f"PPDU={ppdu}")
        ax_a.fill_between(x_qsrc,
                          [m - s for m, s in zip(means, stds)],
                          [m + s for m, s in zip(means, stds)],
                          color=color, alpha=0.12)
        q_opt, tp_opt = _optimal_qsrc(rows_p1, W_FIXED, ppdu, FIXED_N)
        ax_a.plot(q_opt, tp_opt, marker="*", color=color,
                  markersize=11, zorder=5)

    ax_a.set_ylabel("Aggregate Throughput\n(delivered packets)", fontsize=10)
    ax_a.set_xticks(x_qsrc)
    ax_a.set_xticklabels(x_labels, fontsize=8)
    ax_a.set_xlabel("NPCA Initial CW Exponent (qsrc)", fontsize=9)
    ax_a.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_a.legend(fontsize=9, loc="upper right", ncol=3,
                frameon=True, edgecolor="#cccccc",
                title=f"PPDU Duration  (N={FIXED_N}, W={W_FIXED})",
                title_fontsize=8)
    ax_a.text(0.02, 0.97, "(a)  ★ = optimal qsrc* per PPDU",
              transform=ax_a.transAxes, fontsize=9, va="top")

    # ── Panel (b): qsrc*(PPDU), W=170 fixed, lines per N ─────────────────────
    ax_b = axes[1]

    for idx, num_stas in enumerate(n_list):
        opt_qsrcs = [_optimal_qsrc(rows_p1, W_FIXED, p, num_stas)[0]
                     for p in ppdu_list]
        color = _COLORS_N[idx % len(_COLORS_N)]
        ax_b.plot(ppdu_list, opt_qsrcs, color=color, ls="-", marker="s",
                  markersize=6, linewidth=1.6, label=f"N={num_stas}",
                  drawstyle="steps-post")

    ax_b.set_ylabel("Optimal qsrc*", fontsize=10)
    ax_b.set_xlabel("PPDU Duration (slots)", fontsize=10)
    ax_b.set_yticks(QSRC_LIST)
    ax_b.set_yticklabels([f"q={q}  (CW={_qsrc_to_cw(q)})" for q in QSRC_LIST],
                         fontsize=8)
    ax_b.set_xticks(ppdu_list)
    ax_b.set_xticklabels([str(p) for p in ppdu_list], fontsize=9)
    ax_b.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_b.legend(fontsize=9, loc="upper right",
                frameon=True, edgecolor="#cccccc",
                title=f"Num STAs  (W={W_FIXED})", title_fontsize=8)
    ax_b.text(0.02, 0.97,
              "(b)  qsrc*(PPDU) — larger PPDU leaves less room for backoff",
              transform=ax_b.transAxes, fontsize=9, va="top")

    # ── Panel (c): W_avail collapse 검증 ─────────────────────────────────────
    ax_c = axes[2]

    # 배경 참조선: Part 1, N=30, W_avail = W_FIXED − ppdu
    ref_pairs = sorted([(W_FIXED - p, _optimal_qsrc(rows_p1, W_FIXED, p, COLLAPSE_N)[0])
                        for p in ppdu_list], key=lambda x: x[0])
    ref_x = [v[0] for v in ref_pairs]
    ref_y = [v[1] for v in ref_pairs]
    ax_c.plot(ref_x, ref_y, color="#888888", ls="--", linewidth=1.4,
              drawstyle="steps-post", label=f"Part 1 ref (W={W_FIXED}, N={COLLAPSE_N})",
              zorder=2)

    # Part 2 산점도: 각 W_avail 목표에서 3개 (W,PPDU) 쌍
    plotted_ppdu_labels: set[int] = set()
    for idx_wa, (w_avail_target, pairs) in enumerate(collapse_configs):
        color = _COLLAPSE_COLORS[idx_wa % len(_COLLAPSE_COLORS)]
        for (W, ppdu) in pairs:
            marker = _PPDU_MARKERS.get(ppdu, "D")
            q_opt, _ = _optimal_qsrc(rows_p2, W, ppdu, COLLAPSE_N)
            label = f"PPDU={ppdu}" if ppdu not in plotted_ppdu_labels else None
            ax_c.scatter(w_avail_target, q_opt,
                         marker=marker, color=color, s=90, zorder=5,
                         label=label, edgecolors="black", linewidths=0.5)
            if ppdu not in plotted_ppdu_labels:
                plotted_ppdu_labels.add(ppdu)
            ax_c.annotate(f"W={W}", (w_avail_target, q_opt),
                          textcoords="offset points", xytext=(5, 3),
                          fontsize=6, color="#555555")

    ax_c.set_ylabel("Optimal qsrc*", fontsize=10)
    ax_c.set_xlabel(r"$W_\mathrm{avail} = W - \mathrm{PPDU}$ (slots)", fontsize=10)
    ax_c.set_yticks(QSRC_LIST)
    ax_c.set_yticklabels([f"q={q}  (CW={_qsrc_to_cw(q)})" for q in QSRC_LIST],
                         fontsize=8)
    ax_c.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_c.legend(fontsize=8, loc="upper left",
                frameon=True, edgecolor="#cccccc")
    ax_c.text(0.02, 0.97,
              r"(c)  Same $W_\mathrm{avail}$ → same qsrc* regardless of (W, PPDU) split",
              transform=ax_c.transAxes, fontsize=8.5, va="top")

    fig.suptitle(
        "Fig. 11  PPDU Duration Effect on qsrc* — "
        r"$W_\mathrm{avail} = W - \mathrm{PPDU}$ Unification" + "\n"
        "Fixed-duration OBSS (OBSS\\_MIN=OBSS\\_MAX), Occupancy=50%",
        fontsize=11,
    )

    _save_figure(fig, fig_dir, "fig11_ppdu_weff")
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
        description="Figure 11 — PPDU duration effect on qsrc*, W_avail unification")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_SLOTS} slots, subset")
    parser.add_argument("--out-dir", default="results/step9/fig11")
    args = parser.parse_args()

    num_slots     = FAST_SLOTS if args.fast else FULL_SLOTS
    ppdu_list     = FAST_PPDU_LIST        if args.fast else PPDU_LIST
    n_list        = FAST_N_LIST           if args.fast else NUM_STAS_LIST
    seeds         = FAST_SEEDS            if args.fast else SEEDS
    collapse_cfgs = FAST_COLLAPSE_CONFIGS if args.fast else COLLAPSE_CONFIGS
    out_dir       = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    mode     = "FAST" if args.fast else "FULL"
    total_p1 = len(ppdu_list) * len(n_list) * len(QSRC_LIST) * len(seeds)
    total_p2 = (sum(len(ps) for _, ps in collapse_cfgs)
                * len(QSRC_LIST) * len(seeds))
    print(f"=== Figure 11 [{mode}] — {num_slots} slots × {len(seeds)} seeds ===")
    print(f"    Part 1: W={W_FIXED} fixed, PPDU={ppdu_list}, N={n_list}")
    print(f"    Part 2: W_avail targets={[wa for wa, _ in collapse_cfgs]}, N={COLLAPSE_N}")
    print(f"    Part 1: {total_p1} runs  |  Part 2: {total_p2} runs  "
          f"(total {total_p1 + total_p2})")

    print("\n--- Part 1: PPDU sweep ---")
    rows_p1 = run_sweep_part1(num_slots, ppdu_list, n_list, seeds)

    print("\n--- Part 2: W_avail collapse ---")
    rows_p2 = run_sweep_part2(num_slots, collapse_cfgs, seeds)

    rows_all = rows_p1 + rows_p2
    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows_all, csv_path)

    print("\nPlotting...")
    plot(rows_all, out_dir, ppdu_list, n_list, collapse_cfgs)

    print(f"\nFigure 11 완료")
    print(f"  데이터  : {csv_path}")
    print(f"  논문용  : manuscript/figure/fig11_ppdu_weff.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
