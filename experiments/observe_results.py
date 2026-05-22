#!/usr/bin/env python3
"""
Results Observer — load saved experiment CSVs and visualize/summarize them.

사용법:
  # Exp1 결과 보기 (기본 경로)
  .venv/bin/python experiments/observe_results.py

  # 경로 직접 지정
  .venv/bin/python experiments/observe_results.py --dir results/exp1_convergence

  # 터미널 요약만 (플롯 없음)
  .venv/bin/python experiments/observe_results.py --no-plot

  # 플롯 화면에 띄우기 (저장 대신)
  .venv/bin/python experiments/observe_results.py --show

  # LLM 호출 이력 상세 출력
  .venv/bin/python experiments/observe_results.py --llm-detail

  # 특정 obss_duration 필터
  .venv/bin/python experiments/observe_results.py --obss 100
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import json
import math
import textwrap

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _running_avg(data, w):
    out = []
    for i, _ in enumerate(data):
        s = max(0, i - w + 1)
        out.append(float(np.mean(data[s:i+1])))
    return out


def _discover_csv(directory: str) -> pd.DataFrame | None:
    """Load exp1_all.csv if present; otherwise merge individual per-method CSVs."""
    all_csv = os.path.join(directory, "exp1_all.csv")
    if os.path.exists(all_csv):
        df = pd.read_csv(all_csv)
        print(f"Loaded {all_csv}  ({len(df):,} rows)")
        return df

    patterns = glob.glob(os.path.join(directory, "exp1_*.csv"))
    if not patterns:
        return None
    dfs = []
    for p in sorted(patterns):
        if "_all" in p:
            continue
        dfs.append(pd.read_csv(p))
        print(f"  + {os.path.basename(p)}")
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    print(f"Merged {len(dfs)} CSV files  ({len(df):,} rows)")
    return df


def _discover_llm_json(directory: str) -> list[dict]:
    """Collect all LLM call-history JSON files in the directory."""
    jsons = glob.glob(os.path.join(directory, "exp1_llm_history_*.json"))
    histories = []
    for p in sorted(jsons):
        with open(p) as f:
            data = json.load(f)
        histories.extend(data)
    return histories


# ---------------------------------------------------------------------------
# terminal summary
# ---------------------------------------------------------------------------

METRIC_WIDTH = 75

def _hline(char="-"):
    print(char * METRIC_WIDTH)


def print_summary(df: pd.DataFrame, last_n: int = 50):
    """Print per-method summary table (last N episodes)."""
    print()
    _hline("=")
    print(f"  EXPERIMENT RESULTS SUMMARY  (last {last_n} episodes)")
    _hline("=")

    methods = df["method"].unique().tolist()
    obss_vals = sorted(df["obss_duration"].unique().tolist()) if "obss_duration" in df.columns else [None]

    for obss in obss_vals:
        sub = df[df["obss_duration"] == obss] if obss is not None else df
        if sub.empty:
            continue
        if obss is not None:
            print(f"\n  OBSS duration = {obss} slots")
        _hline()
        print(f"  {'Method':<20} {'AvgTP':>8} {'StdTP':>8} {'AvgTau':>8} "
              f"{'NPCA%':>7} {'Reward':>9} {'LLMcalls':>9}")
        _hline()
        for method in methods:
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            tail = msub[msub["episode"] >= msub["episode"].max() - last_n + 1]
            avg_tp   = tail["throughput"].mean()
            std_tp   = tail.groupby("run")["throughput"].mean().std() if tail["run"].nunique() > 1 else float("nan")
            avg_tau  = tail["avg_option_duration"].mean() if "avg_option_duration" in tail.columns else float("nan")
            avg_npca = tail["npca_ratio"].mean() * 100
            avg_rew  = tail["reward"].mean()
            llm_n    = msub["throughput_weight"].notna().sum() if "throughput_weight" in msub.columns else 0
            std_str  = f"{std_tp:>8.1f}" if not math.isnan(std_tp) else f"{'N/A':>8}"
            tau_str  = f"{avg_tau:>8.1f}" if not math.isnan(avg_tau) else f"{'N/A':>8}"
            print(f"  {method:<20} {avg_tp:>8.1f} {std_str} {tau_str} "
                  f"{avg_npca:>6.1f}% {avg_rew:>9.2f} {llm_n:>9}")
        _hline()

    # Best method by throughput
    tail_all = df[df["episode"] >= df.groupby("method")["episode"].transform("max") - last_n + 1]
    best_row = tail_all.groupby("method")["throughput"].mean().idxmax()
    print(f"\n  Best throughput: {best_row}")

    # LLM vs Fixed DRL comparison
    if "LLM-DRL" in methods and "Fixed DRL" in methods:
        llm_tp  = tail_all[tail_all["method"] == "LLM-DRL"]["throughput"].mean()
        fix_tp  = tail_all[tail_all["method"] == "Fixed DRL"]["throughput"].mean()
        delta   = (llm_tp - fix_tp) / max(fix_tp, 1e-9) * 100
        sign    = "+" if delta >= 0 else ""
        print(f"  LLM-DRL vs Fixed DRL throughput: {sign}{delta:.1f}%")
    _hline("=")


def print_llm_detail(histories: list[dict]):
    """Print LLM call-by-call detail."""
    if not histories:
        print("  No LLM call history found.")
        return
    print()
    _hline("=")
    print(f"  LLM CALL HISTORY  ({len(histories)} calls)")
    _hline("=")
    print(f"  {'#':>4} {'Episode':>8} {'QoS':>12} {'tw':>6} {'lp':>6} "
          f"{'sb':>6} {'sc':>6}  Reasoning")
    _hline()
    for h in histories:
        episode = h.get("episode", h.get("call_count", "?"))
        qp   = h.get("qos_priority", "?")[:12]
        tw   = h.get("throughput_weight", float("nan"))
        lp   = h.get("latency_penalty",  float("nan"))
        sb   = h.get("npca_switch_bonus", float("nan"))
        sc   = h.get("npca_switch_cost",  float("nan"))
        rsn  = h.get("reasoning", "")
        rsn_short = textwrap.shorten(rsn, width=45, placeholder="…")
        call_n = h.get("call_count", "?")
        print(f"  {call_n:>4} {str(episode):>8} {qp:>12} {tw:>6.3f} {lp:>6.3f} "
              f"{sb:>6.3f} {sc:>6.3f}  {rsn_short}")
    _hline("=")


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

COLORS = {
    "Always NPCA":  "royalblue",
    "Never NPCA":   "tomato",
    "Rule-based":   "forestgreen",
    "Fixed DRL":    "darkorange",
    "LLM-DRL":      "purple",
}
STYLES = {
    "Always NPCA":  "--",
    "Never NPCA":   "-.",
    "Rule-based":   ":",
    "Fixed DRL":    "-",
    "LLM-DRL":      "-",
}
QP_COLORS = {
    "throughput": "blue",
    "latency":    "red",
    "energy":     "purple",
    "balanced":   "green",
}


def plot_results(df: pd.DataFrame, save_dir: str, show: bool = False,
                 window: int = 50, obss_filter: int | None = None):
    if obss_filter is not None and "obss_duration" in df.columns:
        df = df[df["obss_duration"] == obss_filter]
        if df.empty:
            print(f"No data for obss_duration={obss_filter}")
            return

    os.makedirs(save_dir, exist_ok=True)
    methods    = df["method"].unique().tolist()
    drl_methods = [m for m in methods if "DRL" in m]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    obss_tag = f" (OBSS={obss_filter})" if obss_filter else ""
    fig.suptitle(f"Exp1: Convergence Analysis{obss_tag}", fontsize=13, fontweight="bold")

    # --- Panel 1: Throughput ---
    ax = axes[0, 0]
    for method in methods:
        sub  = df[df["method"] == method]
        mean = sub.groupby("episode")["throughput"].mean().values
        ra   = _running_avg(mean.tolist(), window)
        lw   = 2.5 if "DRL" in method else 1.5
        ax.plot(ra, color=COLORS.get(method, "gray"),
                linestyle=STYLES.get(method, "-"), linewidth=lw, label=method)
        if sub["run"].nunique() > 1:
            std  = sub.groupby("episode")["throughput"].std().fillna(0).values
            ra_s = _running_avg(std.tolist(), window)
            eps  = sorted(sub["episode"].unique())
            ax.fill_between(eps,
                            [r - s for r, s in zip(ra, ra_s)],
                            [r + s for r, s in zip(ra, ra_s)],
                            alpha=0.15, color=COLORS.get(method, "gray"))
    ax.set_title(f"Throughput Convergence (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Successful TX Slots (ch1 STAs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Avg Loss (DRL only) ---
    ax = axes[0, 1]
    for method in drl_methods:
        sub  = df[df["method"] == method]
        mean = sub.groupby("episode")["avg_loss"].mean().values
        ra   = _running_avg(mean.tolist(), window)
        ax.plot(ra, color=COLORS.get(method, "gray"),
                linestyle=STYLES.get(method, "-"), linewidth=2, label=method)
    ax.set_title("DRL Avg Loss per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Smooth-L1 Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Epsilon + NPCA ratio (DRL only) ---
    ax  = axes[1, 0]
    ax2 = ax.twinx()
    for method in drl_methods:
        sub      = df[df["method"] == method]
        eps_vals = sub.groupby("episode")["epsilon"].mean().values
        npca_v   = sub.groupby("episode")["npca_ratio"].mean().values
        ax.plot(eps_vals,
                color=COLORS.get(method, "gray"), linestyle="--",
                linewidth=1.5, alpha=0.7, label=f"{method} ε")
        ax2.plot(_running_avg(npca_v.tolist(), window),
                 color=COLORS.get(method, "gray"), linestyle="-",
                 linewidth=1.5, alpha=0.9, label=f"{method} NPCA ratio")
    ax.set_title("Epsilon Decay & NPCA Switch Ratio (DRL)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon", color="gray")
    ax2.set_ylabel("NPCA Switch Ratio")
    l1, b1 = ax.get_legend_handles_labels()
    l2, b2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, b1 + b2, fontsize=7, loc="center right")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: LLM param evolution ---
    ax   = axes[1, 1]
    sub  = df[df["method"] == "LLM-DRL"]
    if not sub.empty and "throughput_weight" in sub.columns:
        pr = sub[sub["throughput_weight"].notna()].copy()
        if not pr.empty:
            agg = pr.groupby("episode").agg(
                throughput_weight=("throughput_weight", "mean"),
                latency_penalty=("latency_penalty",   "mean"),
                npca_switch_bonus=("npca_switch_bonus", "mean"),
                npca_switch_cost=("npca_switch_cost",   "mean"),
            ).reset_index()
            eps   = agg["episode"].values
            ax2p  = ax.twinx()
            ax.step(eps, agg["throughput_weight"],  "b-o", markersize=5, linewidth=1.8, where="post", label="throughput_weight")
            ax.step(eps, agg["npca_switch_bonus"],  "g-^", markersize=5, linewidth=1.8, where="post", label="npca_switch_bonus")
            ax.step(eps, agg["npca_switch_cost"],   "m-D", markersize=5, linewidth=1.8, where="post", label="npca_switch_cost")
            ax2p.step(eps, agg["latency_penalty"],  "r-s", markersize=5, linewidth=1.8, where="post", label="latency_penalty")
            ax2p.set_ylabel("latency_penalty", color="red", fontsize=9)

            prev_qp = None
            for _, row in pr.drop_duplicates("episode").iterrows():
                qp = row.get("qos_priority")
                if pd.notna(qp) and qp != prev_qp:
                    ax.axvline(row["episode"], color=QP_COLORS.get(qp, "gray"),
                               alpha=0.3, linewidth=1)
                    prev_qp = qp

            l1, b1 = ax.get_legend_handles_labels()
            l2, b2 = ax2p.get_legend_handles_labels()
            ax.legend(l1 + l2, b1 + b2, fontsize=7)
    ax.set_title("LLM Reward Param Evolution\n(vertical lines = qos_priority change)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("throughput_weight / npca_switch_bonus / npca_switch_cost")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        tag  = f"_obss{obss_filter}" if obss_filter else ""
        path = os.path.join(save_dir, f"exp1_convergence{tag}_observed.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Figure saved → {path}")


def plot_final_bar(df: pd.DataFrame, save_dir: str, show: bool = False,
                   last_n: int = 50):
    """Bar chart of final avg throughput per method (+/- std across runs)."""
    tail = df[df["episode"] >= df.groupby("method")["episode"].transform("max") - last_n + 1]
    stats = (tail.groupby("method")["throughput"]
             .agg(["mean", "std"]).reset_index()
             .sort_values("mean", ascending=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    x    = range(len(stats))
    cols = [COLORS.get(m, "gray") for m in stats["method"]]
    bars = ax.bar(x, stats["mean"], color=cols, yerr=stats["std"].fillna(0),
                  capsize=5, ecolor="black", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(stats["method"], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Avg Successful TX Slots (last 50 ep)")
    ax.set_title("Final Throughput Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, stats["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        path = os.path.join(save_dir, "exp1_final_bar_observed.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Figure saved → {path}")


def plot_intent_metrics(df: pd.DataFrame, save_dir: str, show: bool = False,
                        window: int = 50):
    """Throughput / Avg option duration / NPCA ratio — intent-validation 3-panel."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Intent-Validation Metrics", fontsize=12, fontweight="bold")
    metrics = [
        ("throughput",         "Successful TX Slots",   "Throughput"),
        ("avg_option_duration","Avg Option Duration (slots)", "Latency proxy"),
        ("npca_ratio",         "NPCA Switch Ratio",     "Behavior"),
    ]
    methods = df["method"].unique().tolist()
    for ax, (col, ylabel, title) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        for method in methods:
            sub  = df[df["method"] == method]
            mean = sub.groupby("episode")[col].mean().values
            ra   = _running_avg(mean.tolist(), window)
            lw   = 2.5 if "DRL" in method else 1.5
            ax.plot(ra, color=COLORS.get(method, "gray"),
                    linestyle=STYLES.get(method, "-"), linewidth=lw, label=method)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        path = os.path.join(save_dir, "exp1_intent_metrics_observed.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Figure saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Observe and visualize saved experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dir", default="./results/exp1_convergence",
                        help="Results directory (default: ./results/exp1_convergence)")
    parser.add_argument("--no-plot",    action="store_true", help="Skip plot generation")
    parser.add_argument("--show",       action="store_true", help="Show plots interactively instead of saving")
    parser.add_argument("--llm-detail", action="store_true", help="Print LLM call-by-call history")
    parser.add_argument("--obss",       type=int, default=None, help="Filter by obss_duration value")
    parser.add_argument("--window",     type=int, default=50,   help="Running average window (default: 50)")
    parser.add_argument("--last-n",     type=int, default=50,   help="Episodes for final summary (default: 50)")
    args = parser.parse_args()

    directory = os.path.abspath(args.dir)
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        sys.exit(1)

    print(f"\nScanning: {directory}")
    df = _discover_csv(directory)
    if df is None:
        print("No CSV files found. Run an experiment first:")
        print("  .venv/bin/python experiments/exp1_convergence.py --mock --episodes 200")
        sys.exit(1)

    # Ensure required columns exist (backward compat with older CSVs)
    for col, default in [
        ("avg_option_duration", float("nan")),
        ("npca_switch_cost",    float("nan")),
        ("npca_switch_bonus",   float("nan")),
        ("run",                 0),
    ]:
        if col not in df.columns:
            df[col] = default

    print_summary(df, last_n=args.last_n)

    if args.llm_detail:
        histories = _discover_llm_json(directory)
        print_llm_detail(histories)

    if not args.no_plot:
        if args.show:
            matplotlib.use("TkAgg")  # switch to interactive backend
        print("\nGenerating plots…")
        plot_results(df, save_dir=directory, show=args.show,
                     window=args.window, obss_filter=args.obss)
        plot_final_bar(df, save_dir=directory, show=args.show, last_n=args.last_n)
        plot_intent_metrics(df, save_dir=directory, show=args.show, window=args.window)
        if not args.show:
            print(f"\nAll plots saved to: {directory}")


if __name__ == "__main__":
    main()
