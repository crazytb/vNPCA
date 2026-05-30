"""
Figure 12: Multi-Round NPCA Access Characterization

연구 질문 (RQ12): W_avail 통합(qsrc* = f(N, W_avail))은 다중 TX round
  (K = W_avail/PPDU >> 1) 환경에서도 유지되는가?

두 파트:
  Part 1 — PPDU=20 고정, W sweep → K = W_avail/PPDU 특성화
            측정: avg_tx_per_visit vs K
  Part 2 — 3개 K-tier에서 W_avail collapse 검증
            (동일 W_avail, 다른 (W,PPDU) → qsrc* 비교)

계획: guidelines/step9/fig12.md
출력:
  manuscript/figure/fig12_multiround.{eps,png,pdf}
  results/step9/fig12/data.csv

실행:
  python harq_sim/run_step9_fig12.py [--fast] [--out-dir results/step9/fig12]
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

# Part 1: K sweep (PPDU=20 고정, N=30)
PPDU_PART1    = 20
N_PART1       = 30
W_LIST_PART1  = [30, 50, 80, 120, 170, 230, 300, 420, 600]
# → K = W_avail/PPDU: {0.5, 1.5, 3.0, 5.0, 7.5, 10.5, 14.0, 20.0, 29.0}

QSRC_LIST     = [0, 1, 2, 3, 4, 5]
SEEDS         = [42, 123, 456]

# Part 2: K-tier collapse 검증 (N=30)
COLLAPSE_N    = 30
# (tier_label, w_avail_target, [(W, PPDU), ...])
COLLAPSE_CONFIGS: list[tuple[str, int, list[tuple[int, int]]]] = [
    ("A",  30, [(50, 20), (80, 50), (110, 80)]),
    ("B", 150, [(170, 20), (200, 50), (230, 80)]),
    ("C", 400, [(420, 20), (450, 50), (480, 80)]),
]
COLLAPSE_PPDU_LIST = [20, 50, 80]

OBSS_OCCUPANCY = 0.50
FULL_SLOTS     = 50_000
FAST_SLOTS     =  5_000

# Fast mode 서브셋
FAST_W_LIST_PART1 = [50, 170, 420]
FAST_QSRC_LIST    = [0, 1, 2, 3]
FAST_SEEDS        = [42]
FAST_COLLAPSE_CONFIGS: list[tuple[str, int, list[tuple[int, int]]]] = [
    ("B", 150, [(170, 20), (200, 50)]),
    ("C", 400, [(420, 20), (450, 50), (480, 80)]),
]

CW_MIN = 15

# 색상/마커 팔레트
_COLORS_QSRC  = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
_COLORS_TIER  = {"A": "#4c72b0", "B": "#dd8452", "C": "#55a868"}
_PPDU_COLORS  = {20: "#1f77b4", 50: "#ff7f0e", 80: "#2ca02c"}
_PPDU_HATCHES = {20: "", 50: "//", 80: "xx"}
_PPDU_MARKERS = {20: "o", 50: "s", 80: "^"}


def _occupancy_to_rate(occ: float, W: int) -> float:
    return occ / (W * (1.0 - occ))


def _qsrc_to_cw(qsrc: int) -> int:
    return 2 ** qsrc * (CW_MIN + 1) - 1


def _k_value(W: int, ppdu: int) -> float:
    w_avail = W - ppdu
    return w_avail / ppdu if ppdu > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: K sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep_part1(num_slots: int, w_list: list, qsrc_list: list,
                    seeds: list) -> list[dict]:
    rows: list[dict] = []
    total = len(w_list) * len(qsrc_list) * len(seeds)
    done  = 0

    for W in w_list:
        obss_rate = _occupancy_to_rate(OBSS_OCCUPANCY, W)
        w_avail   = W - PPDU_PART1
        K         = _k_value(W, PPDU_PART1)
        for qsrc in qsrc_list:
            cw = _qsrc_to_cw(qsrc)
            for seed in seeds:
                done += 1
                print(f"  Part1 [{done:3d}/{total}]  W={W:3d}  K={K:5.1f}  "
                      f"qsrc={qsrc}  seed={seed}", flush=True)

                sim = build_and_run(
                    num_slots      = num_slots,
                    num_stas       = N_PART1,
                    obss_rate      = obss_rate,
                    obss_min       = W,
                    obss_max       = W,
                    npca_qsrc      = qsrc,
                    npca_threshold = 0,
                    ppdu_duration  = PPDU_PART1,
                    harq_horizon   = 200,
                    snr_db_mean    = 20.0,
                    snr_db_std     = 0.0,
                    npca_enabled   = True,
                    harq_enabled   = True,
                    adaptive_cw    = False,
                    seed           = seed,
                    enable_trace   = False,
                )
                metrics = sim.compute_metrics()
                agg     = metrics["aggregate"]
                avg_tv  = _avg_tx_per_visit(metrics)
                rows.append({
                    "part":                       1,
                    "tier":                       "",
                    "W":                          W,
                    "ppdu_duration":              PPDU_PART1,
                    "W_avail":                    w_avail,
                    "K":                          round(K, 3),
                    "num_stas":                   N_PART1,
                    "npca_qsrc":                  qsrc,
                    "npca_cw":                    cw,
                    "seed":                       seed,
                    "collision_probability_npca": agg["collision_probability_npca"],
                    "aggregate_throughput":       agg["aggregate_throughput"],
                    "mean_access_delay":          agg["mean_access_delay"],
                    "npca_transition_count":      agg["npca_transition_count"],
                    "avg_tx_per_visit":           avg_tv,
                })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: K-tier collapse 검증
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep_part2(num_slots: int, collapse_configs: list,
                    seeds: list) -> list[dict]:
    rows: list[dict] = []
    total = (sum(len(pairs) for _, _, pairs in collapse_configs)
             * len(QSRC_LIST) * len(seeds))
    done  = 0

    for (tier, w_avail_target, pairs) in collapse_configs:
        for (W, ppdu) in pairs:
            obss_rate = _occupancy_to_rate(OBSS_OCCUPANCY, W)
            K         = _k_value(W, ppdu)
            for qsrc in QSRC_LIST:
                cw = _qsrc_to_cw(qsrc)
                for seed in seeds:
                    done += 1
                    print(f"  Part2 [{done:3d}/{total}]  Tier={tier}  "
                          f"W_avail={w_avail_target}  W={W:3d}  PPDU={ppdu:2d}  "
                          f"K={K:5.1f}  qsrc={qsrc}  seed={seed}", flush=True)

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
                    metrics = sim.compute_metrics()
                    agg     = metrics["aggregate"]
                    avg_tv  = _avg_tx_per_visit(metrics)
                    rows.append({
                        "part":                       2,
                        "tier":                       tier,
                        "W":                          W,
                        "ppdu_duration":              ppdu,
                        "W_avail":                    w_avail_target,
                        "K":                          round(K, 3),
                        "num_stas":                   COLLAPSE_N,
                        "npca_qsrc":                  qsrc,
                        "npca_cw":                    cw,
                        "seed":                       seed,
                        "collision_probability_npca": agg["collision_probability_npca"],
                        "aggregate_throughput":       agg["aggregate_throughput"],
                        "mean_access_delay":          agg["mean_access_delay"],
                        "npca_transition_count":      agg["npca_transition_count"],
                        "avg_tx_per_visit":           avg_tv,
                    })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 보조 함수
# ─────────────────────────────────────────────────────────────────────────────

def _avg_tx_per_visit(metrics: dict) -> float:
    """NPCA 방문당 평균 TX 횟수 (성공+실패 합산 / 전환 횟수)."""
    total_npca_tx = sum(
        v["npca_tx_success"] + v["npca_tx_fail"]
        for k, v in metrics.items()
        if k != "aggregate"
    )
    transitions = metrics["aggregate"]["npca_transition_count"]
    return total_npca_tx / transitions if transitions > 0 else 0.0


FIELDS = ["part", "tier", "W", "ppdu_duration", "W_avail", "K", "num_stas",
          "npca_qsrc", "npca_cw", "seed",
          "collision_probability_npca", "aggregate_throughput",
          "mean_access_delay", "npca_transition_count", "avg_tx_per_visit"]


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
                  num_stas: int, qsrc_list: list = None) -> tuple[int, float]:
    if qsrc_list is None:
        qsrc_list = QSRC_LIST
    best_qsrc, best_tp = 0, -1.0
    for q in qsrc_list:
        m, _ = _stats(rows, W, ppdu, num_stas, q, "aggregate_throughput")
        if m > best_tp:
            best_tp, best_qsrc = m, q
    return best_qsrc, best_tp


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(rows: list[dict], fig_dir: str,
         w_list_p1: list, qsrc_list: list,
         collapse_configs: list) -> None:
    rows_p1 = [r for r in rows if r["part"] == 1]
    rows_p2 = [r for r in rows if r["part"] == 2]

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 10.5), sharex=False)
    fig.subplots_adjust(hspace=0.50)

    # ── Panel (a): avg_tx_per_visit vs K ─────────────────────────────────────
    ax_a = axes[0]

    k_values = [_k_value(W, PPDU_PART1) for W in w_list_p1]
    for idx, qsrc in enumerate(qsrc_list):
        means, stds = [], []
        for W in w_list_p1:
            m, s = _stats(rows_p1, W, PPDU_PART1, N_PART1, qsrc, "avg_tx_per_visit")
            means.append(m)
            stds.append(s)
        color = _COLORS_QSRC[idx % len(_COLORS_QSRC)]
        cw    = _qsrc_to_cw(qsrc)
        ax_a.plot(k_values, means, color=color, ls="-", marker="o",
                  markersize=5, linewidth=1.6, label=f"qsrc={qsrc} (CW={cw})")
        ax_a.fill_between(k_values,
                          [max(0, m - s) for m, s in zip(means, stds)],
                          [m + s for m, s in zip(means, stds)],
                          color=color, alpha=0.10)

    # 이론 참조선: K 비례 (점선)
    k_ref = np.linspace(0, max(k_values) * 1.05, 100)
    ax_a.plot(k_ref, k_ref * 0.35, color="black", ls=":", linewidth=1.2,
              alpha=0.5, label="~ 0.35 K (ref.)")

    ax_a.set_ylabel("Avg TX per NPCA Visit", fontsize=10)
    ax_a.set_xlabel(r"$K = W_\mathrm{avail}/\mathrm{PPDU}$  (PPDU=20, N=30)",
                    fontsize=9)
    ax_a.set_xlim(left=-0.5)
    ax_a.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_a.legend(fontsize=8, loc="upper left", ncol=2,
                frameon=True, edgecolor="#cccccc")
    ax_a.text(0.02, 0.97, "(a)  Multi-round characterization: avg TX per NPCA visit vs K",
              transform=ax_a.transAxes, fontsize=8.5, va="top")

    # ── Panel (b): qsrc* collapse — K-tier별 (W,PPDU) 쌍 비교 ───────────────
    ax_b = axes[1]

    tier_labels = [cfg[0] for cfg in collapse_configs]
    w_avail_vals = [cfg[1] for cfg in collapse_configs]
    n_tiers  = len(collapse_configs)

    # PPDU 목록 동적 추출
    all_ppdu = sorted(set(ppdu for _, _, pairs in collapse_configs for _, ppdu in pairs))
    n_ppdu   = len(all_ppdu)
    bar_width = 0.22
    offsets   = np.linspace(-(n_ppdu - 1) / 2, (n_ppdu - 1) / 2, n_ppdu) * bar_width

    x_pos = np.arange(n_tiers)
    for i, ppdu in enumerate(all_ppdu):
        qsrc_stars = []
        for tier, w_avail, pairs in collapse_configs:
            # 해당 tier + ppdu 조합이 있는 경우만
            matching = [(W, p) for W, p in pairs if p == ppdu]
            if matching:
                W_pair = matching[0][0]
                q_opt, _ = _optimal_qsrc(rows_p2, W_pair, ppdu, COLLAPSE_N, qsrc_list)
                qsrc_stars.append(q_opt)
            else:
                qsrc_stars.append(np.nan)

        color  = _PPDU_COLORS.get(ppdu, "#999999")
        hatch  = _PPDU_HATCHES.get(ppdu, "")
        pos    = x_pos + offsets[i]
        bars   = ax_b.bar(pos, qsrc_stars, width=bar_width, label=f"PPDU={ppdu}",
                          color=color, hatch=hatch, edgecolor="black",
                          linewidth=0.8, alpha=0.85)

    # Part 1 참조선 (PPDU=20)
    ref_qsrc = []
    for tier, w_avail, pairs in collapse_configs:
        # Part 1에서 W_avail = w_avail인 W 찾기 (PPDU=20 고정)
        W_ref = w_avail + PPDU_PART1
        if W_ref in w_list_p1:
            q_ref, _ = _optimal_qsrc(rows_p1, W_ref, PPDU_PART1, N_PART1, qsrc_list)
        else:
            q_ref = np.nan
        ref_qsrc.append(q_ref)

    ax_b.step(x_pos, ref_qsrc, where="mid", color="#888888",
              ls="--", linewidth=1.8, label=f"Part 1 ref (PPDU=20)", zorder=5)
    ax_b.scatter(x_pos, ref_qsrc, marker="*", color="#888888", s=120, zorder=6)

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels([f"Tier {cfg[0]}\n(W_avail={cfg[1]})" for cfg in collapse_configs],
                          fontsize=9)
    ax_b.set_ylabel("Optimal qsrc*", fontsize=10)
    ax_b.set_yticks(QSRC_LIST)
    ax_b.set_yticklabels([f"q={q}" for q in QSRC_LIST], fontsize=8)
    ax_b.set_ylim(-0.5, max(QSRC_LIST) + 0.5)
    ax_b.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax_b.legend(fontsize=8, loc="upper left", ncol=2,
                frameon=True, edgecolor="#cccccc")
    ax_b.text(0.02, 0.97,
              "(b)  qsrc* collapse: same W_avail → same qsrc* across PPDU and K-tier",
              transform=ax_b.transAxes, fontsize=8.5, va="top")

    # ── Panel (c): Throughput at qsrc* vs PPDU (W_avail tier별) ─────────────
    ax_c = axes[2]

    for tier, w_avail, pairs in collapse_configs:
        ppdu_vals, tp_vals = [], []
        for (W, ppdu) in sorted(pairs, key=lambda x: x[1]):
            q_opt, tp_opt = _optimal_qsrc(rows_p2, W, ppdu, COLLAPSE_N, qsrc_list)
            ppdu_vals.append(ppdu)
            tp_vals.append(tp_opt)

        color = _COLORS_TIER.get(tier, "#999999")
        K_str_list = [f"{_k_value(W, ppdu):.1f}" for W, ppdu in
                      sorted(pairs, key=lambda x: x[1])]
        K_label = ", ".join(K_str_list)
        ax_c.plot(ppdu_vals, tp_vals, color=color, ls="-", linewidth=1.8,
                  label=f"Tier {tier} (W_avail={w_avail}, K={K_label})")
        for ppdu, tp in zip(ppdu_vals, tp_vals):
            m = _PPDU_MARKERS.get(ppdu, "D")
            ax_c.scatter(ppdu, tp, marker=m, color=color, s=70, zorder=5)

    ax_c.set_ylabel("Throughput at qsrc*\n(delivered packets)", fontsize=10)
    ax_c.set_xlabel("PPDU Duration (slots)", fontsize=10)
    ax_c.set_xticks(all_ppdu)
    ax_c.set_xticklabels([str(p) for p in all_ppdu], fontsize=9)
    ax_c.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_c.legend(fontsize=8, loc="upper right",
                frameon=True, edgecolor="#cccccc")
    ax_c.text(0.02, 0.97,
              r"(c)  Same $W_\mathrm{avail}$ + same qsrc*, but PPDU↑ → K↓ → throughput↓",
              transform=ax_c.transAxes, fontsize=8.5, va="top")

    fig.suptitle(
        "Fig. 12  Multi-Round NPCA Access — K = W_avail/PPDU Characterization\n"
        "Fixed-duration OBSS, N=30, Occupancy=50%, SNR=20 dB",
        fontsize=11,
    )

    _save_figure(fig, fig_dir, "fig12_multiround")
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
        description="Figure 12 — Multi-round NPCA characterization, K = W_avail/PPDU")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_SLOTS} slots, subset")
    parser.add_argument("--out-dir", default="results/step9/fig12")
    args = parser.parse_args()

    num_slots      = FAST_SLOTS if args.fast else FULL_SLOTS
    w_list_p1      = FAST_W_LIST_PART1    if args.fast else W_LIST_PART1
    qsrc_list      = FAST_QSRC_LIST       if args.fast else QSRC_LIST
    seeds          = FAST_SEEDS           if args.fast else SEEDS
    collapse_cfgs  = FAST_COLLAPSE_CONFIGS if args.fast else COLLAPSE_CONFIGS
    out_dir        = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    mode     = "FAST" if args.fast else "FULL"
    total_p1 = len(w_list_p1) * len(qsrc_list) * len(seeds)
    total_p2 = (sum(len(pairs) for _, _, pairs in collapse_cfgs)
                * len(QSRC_LIST) * len(seeds))
    print(f"=== Figure 12 [{mode}] — {num_slots} slots × {len(seeds)} seeds ===")
    print(f"    Part 1: PPDU={PPDU_PART1} fixed, W={w_list_p1}, N={N_PART1}")
    print(f"    Part 2: Tiers={[t for t,_,_ in collapse_cfgs]}, N={COLLAPSE_N}")
    print(f"    Part 1: {total_p1} runs  |  Part 2: {total_p2} runs  "
          f"(total {total_p1 + total_p2})")

    print("\n--- Part 1: K sweep (avg_tx_per_visit) ---")
    rows_p1 = run_sweep_part1(num_slots, w_list_p1, qsrc_list, seeds)

    print("\n--- Part 2: K-tier collapse ---")
    rows_p2 = run_sweep_part2(num_slots, collapse_cfgs, seeds)

    rows_all = rows_p1 + rows_p2
    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows_all, csv_path)

    print("\nPlotting...")
    plot(rows_all, out_dir, w_list_p1, qsrc_list, collapse_cfgs)

    print(f"\nFigure 12 완료")
    print(f"  데이터  : {csv_path}")
    print(f"  논문용  : manuscript/figure/fig12_multiround.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
