#!/usr/bin/env python3
"""
Experiment 1: Convergence Analysis

관찰 목표: LLM-DRL이 고정 policy(Fixed-reward DRL, Always/Never/Rule-based) 대비
          잘 수렴하는가?

출력:
  results/exp1_convergence/
    ├── exp1_{method}_{obss}_{run}.csv   # per-episode 상세 로그
    ├── exp1_all.csv                     # 전체 통합 로그
    └── exp1_convergence.png             # 4-panel 수렴 그래프

CSV 컬럼:
  episode, method, run, obss_duration,
  throughput, reward, avg_loss, epsilon, npca_ratio,
  throughput_weight, latency_penalty, npca_switch_bonus, qos_priority

Usage:
  .venv/bin/python experiments/exp1_convergence.py --mock --episodes 200 --slots 2000
  .venv/bin/python experiments/exp1_convergence.py --episodes 1000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from drl_framework.random_access import Channel
from drl_framework.train import train_semi_mdp, run_baseline_simulation
from drl_framework.configs import (
    PPDU_DURATION_VARIANTS, RADIO_TRANSITION_TIME,
    OBSS_GENERATION_RATE, DEFAULT_NUM_STAS_CH0, DEFAULT_NUM_STAS_CH1,
    DEFAULT_NUM_EPISODES, DEFAULT_NUM_SLOTS_PER_EPISODE,
    LLM_MODEL, LLM_UPDATE_INTERVAL,
)
from llm_reward_designer import LLMRewardDesigner

NAN = float("nan")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_channels_stas(obss_duration: int, ppdu_variant: str = "medium"):
    ppdu = PPDU_DURATION_VARIANTS.get(ppdu_variant, 33)
    channels = [
        Channel(channel_id=0, obss_generation_rate=OBSS_GENERATION_RATE["secondary"]),
        Channel(channel_id=1,
                obss_generation_rate=OBSS_GENERATION_RATE["primary"],
                obss_duration_range=(obss_duration, obss_duration)),
    ]
    stas = []
    for i in range(DEFAULT_NUM_STAS_CH0):
        stas.append({"sta_id": i, "channel_id": 0, "npca_enabled": False,
                     "ppdu_duration": ppdu, "radio_transition_time": RADIO_TRANSITION_TIME})
    for i in range(DEFAULT_NUM_STAS_CH1):
        stas.append({"sta_id": DEFAULT_NUM_STAS_CH0 + i, "channel_id": 1, "npca_enabled": True,
                     "ppdu_duration": ppdu, "radio_transition_time": RADIO_TRANSITION_TIME})
    return channels, stas


def _running_avg(data, w):
    out = []
    for i in range(len(data)):
        s = max(0, i - w + 1)
        out.append(float(np.mean(data[s:i+1])))
    return out


def _build_df(method, run, obss_duration,
              rewards, throughputs, avg_option_durations, avg_losses, epsilons, npca_ratios,
              llm_log=None):
    """Build a per-episode DataFrame with unified schema."""
    n = len(rewards)
    # Build LLM param columns (NaN for non-LLM methods)
    tw_col  = [NAN] * n
    lp_col  = [NAN] * n
    sb_col  = [NAN] * n
    qp_col  = [None] * n

    if llm_log:
        for entry in llm_log:
            ep = entry["episode"]
            if ep < n:
                tw_col[ep]  = entry.get("throughput_weight", NAN)
                lp_col[ep]  = entry.get("latency_penalty", NAN)
                sb_col[ep]  = entry.get("npca_switch_bonus", NAN)
                qp_col[ep]  = entry.get("qos_priority", None)
        # Forward-fill from first LLM call
        last_tw, last_lp, last_sb, last_qp = NAN, NAN, NAN, None
        for i in range(n):
            if not math.isnan(tw_col[i]):
                last_tw, last_lp, last_sb, last_qp = tw_col[i], lp_col[i], sb_col[i], qp_col[i]
            else:
                tw_col[i], lp_col[i], sb_col[i], qp_col[i] = last_tw, last_lp, last_sb, last_qp

    return pd.DataFrame({
        "episode":               range(n),
        "method":                method,
        "run":                   run,
        "obss_duration":         obss_duration,
        "throughput":            throughputs,
        "avg_option_duration":   avg_option_durations,
        "reward":                rewards,
        "avg_loss":              avg_losses,
        "epsilon":               epsilons,
        "npca_ratio":            npca_ratios,
        "throughput_weight":     tw_col,
        "latency_penalty":       lp_col,
        "npca_switch_bonus":     sb_col,
        "qos_priority":          qp_col,
    })


# ---------------------------------------------------------------------------
# single-method runners (return DataFrame)
# ---------------------------------------------------------------------------

def _run_drl(label, obss_duration, num_episodes, num_slots, ppdu_variant,
             llm_designer, run_id, device):
    channels, stas = _make_channels_stas(obss_duration, ppdu_variant)
    rewards, _, learner = train_semi_mdp(
        channels=channels, stas_config=stas,
        num_episodes=num_episodes, num_slots_per_episode=num_slots,
        device=device, random_ppdu=True, llm_designer=llm_designer,
    )
    return _build_df(
        method=label, run=run_id, obss_duration=obss_duration,
        rewards=rewards,
        throughputs=learner.episode_throughputs,
        avg_option_durations=learner.episode_avg_option_durations,
        avg_losses=learner.episode_avg_losses,
        epsilons=learner.episode_epsilons,
        npca_ratios=learner.episode_npca_ratios,
        llm_log=getattr(learner, "llm_log", None),
    )


def _run_baseline(label, obss_duration, num_episodes, num_slots, ppdu_variant,
                  fixed_action_fn, run_id, device):
    channels, stas = _make_channels_stas(obss_duration, ppdu_variant)
    rewards, throughputs, option_durations = run_baseline_simulation(
        channels=channels, stas_config=stas,
        num_episodes=num_episodes, num_slots_per_episode=num_slots,
        fixed_action_fn=fixed_action_fn, random_ppdu=True, device=device,
    )
    dummy_zeros = [0.0] * len(rewards)
    dummy_nans  = [NAN] * len(rewards)
    return _build_df(
        method=label, run=run_id, obss_duration=obss_duration,
        rewards=rewards, throughputs=throughputs,
        avg_option_durations=option_durations,
        avg_losses=dummy_zeros, epsilons=dummy_nans, npca_ratios=dummy_zeros,
    )


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_exp1(
    obss_duration: int = 100,
    ppdu_variant: str = "medium",
    num_episodes: int = DEFAULT_NUM_EPISODES,
    num_slots: int = DEFAULT_NUM_SLOTS_PER_EPISODE,
    use_mock: bool = True,
    num_runs: int = 1,
    save_dir: str = "./results/exp1_convergence",
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_dfs = []

    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"[Exp1] Run {run+1}/{num_runs}  obss={obss_duration}  episodes={num_episodes}")
        print(f"{'='*60}")

        # 1. Always NPCA
        print("\n--- Always NPCA ---")
        df = _run_baseline("Always NPCA", obss_duration, num_episodes, num_slots,
                           ppdu_variant, lambda _: (lambda: 1), run, device)
        all_dfs.append(df)
        df.to_csv(os.path.join(save_dir, f"exp1_always_npca_obss{obss_duration}_run{run}.csv"),
                  index=False)

        # 2. Never NPCA
        print("\n--- Never NPCA ---")
        df = _run_baseline("Never NPCA", obss_duration, num_episodes, num_slots,
                           ppdu_variant, lambda _: (lambda: 0), run, device)
        all_dfs.append(df)
        df.to_csv(os.path.join(save_dir, f"exp1_never_npca_obss{obss_duration}_run{run}.csv"),
                  index=False)

        # 3. Rule-based
        print("\n--- Rule-based ---")
        def rule_fn(sta):
            def _a():
                return 1 if sta.primary_channel.obss_remain >= sta.ppdu_duration else 0
            return _a
        df = _run_baseline("Rule-based", obss_duration, num_episodes, num_slots,
                           ppdu_variant, rule_fn, run, device)
        all_dfs.append(df)
        df.to_csv(os.path.join(save_dir, f"exp1_rule_based_obss{obss_duration}_run{run}.csv"),
                  index=False)

        # 4. Fixed-reward DRL (no LLM)
        print("\n--- Fixed-reward DRL ---")
        df = _run_drl("Fixed DRL", obss_duration, num_episodes, num_slots,
                      ppdu_variant, llm_designer=None, run_id=run, device=device)
        all_dfs.append(df)
        df.to_csv(os.path.join(save_dir, f"exp1_fixed_drl_obss{obss_duration}_run{run}.csv"),
                  index=False)

        # 5. LLM-DRL (proposed)
        print("\n--- LLM-DRL (proposed) ---")
        designer = LLMRewardDesigner(model=LLM_MODEL, update_interval=LLM_UPDATE_INTERVAL,
                                     use_mock=use_mock)
        df = _run_drl("LLM-DRL", obss_duration, num_episodes, num_slots,
                      ppdu_variant, llm_designer=designer, run_id=run, device=device)
        all_dfs.append(df)
        df.to_csv(os.path.join(save_dir, f"exp1_llm_drl_obss{obss_duration}_run{run}.csv"),
                  index=False)
        if designer.call_history:
            designer.save_history(os.path.join(save_dir,
                                               f"exp1_llm_history_obss{obss_duration}_run{run}.json"))

    # Merge all runs
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(os.path.join(save_dir, "exp1_all.csv"), index=False)
    print(f"\nAll logs saved → {save_dir}/exp1_all.csv  ({len(full_df)} rows)")

    # Plot
    plot_exp1(full_df, save_dir=save_dir, window=max(10, num_episodes // 20))
    _print_exp1_summary(full_df)


# ---------------------------------------------------------------------------
# Plotting
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


def plot_exp1(df: pd.DataFrame, save_dir: str, window: int = 50):
    """
    4-panel figure:
      Panel 1: Throughput convergence curves (all methods)
      Panel 2: Per-episode avg loss (DRL methods only)
      Panel 3: Epsilon decay + NPCA ratio (DRL methods)
      Panel 4: LLM reward param evolution (LLM-DRL only)
    """
    os.makedirs(save_dir, exist_ok=True)
    methods = df["method"].unique().tolist()
    drl_methods = [m for m in methods if "DRL" in m]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Exp1: Convergence Analysis", fontsize=13, fontweight="bold")

    # ---- Panel 1: Throughput ----
    ax = axes[0, 0]
    for method in methods:
        sub = df[df["method"] == method]
        # Average across runs
        mean_tp = sub.groupby("episode")["throughput"].mean().values
        ra = _running_avg(mean_tp.tolist(), window)
        lw = 2.5 if "LLM" in method or "Fixed" in method else 1.5
        ax.plot(ra, color=COLORS.get(method, "gray"),
                linestyle=STYLES.get(method, "-"),
                linewidth=lw, label=method)
        # Std band across runs if num_runs > 1
        if sub["run"].nunique() > 1:
            std_tp = sub.groupby("episode")["throughput"].std().fillna(0).values
            ra_std = _running_avg(std_tp.tolist(), window)
            eps = sub["episode"].unique()
            ax.fill_between(eps,
                            [r - s for r, s in zip(ra, ra_std)],
                            [r + s for r, s in zip(ra, ra_std)],
                            alpha=0.15, color=COLORS.get(method, "gray"))
    ax.set_title(f"Throughput Convergence (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Successful TX Slots (ch1 STAs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Avg Loss (DRL only) ----
    ax = axes[0, 1]
    for method in drl_methods:
        sub = df[df["method"] == method]
        mean_loss = sub.groupby("episode")["avg_loss"].mean().values
        ra = _running_avg(mean_loss.tolist(), window)
        ax.plot(ra, color=COLORS.get(method, "gray"),
                linestyle=STYLES.get(method, "-"),
                linewidth=2, label=method)
    ax.set_title("DRL Avg Loss per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Smooth-L1 Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Epsilon + NPCA ratio (DRL only) ----
    ax = axes[1, 0]
    ax2 = ax.twinx()
    for method in drl_methods:
        sub = df[df["method"] == method]
        eps_vals = sub.groupby("episode")["epsilon"].mean().values
        npca_vals = sub.groupby("episode")["npca_ratio"].mean().values
        ax.plot(eps_vals, color=COLORS.get(method, "gray"),
                linestyle="--", linewidth=1.5, alpha=0.7, label=f"{method} ε")
        ax2.plot(_running_avg(npca_vals.tolist(), window),
                 color=COLORS.get(method, "gray"),
                 linestyle="-", linewidth=1.5, alpha=0.9, label=f"{method} NPCA ratio")
    ax.set_title("Epsilon Decay & NPCA Switch Ratio (DRL)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon", color="gray")
    ax2.set_ylabel("NPCA Switch Ratio")
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7, loc="center right")
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: LLM param evolution ----
    ax = axes[1, 1]
    llm_sub = df[df["method"] == "LLM-DRL"]
    if not llm_sub.empty:
        # Only rows where LLM params were updated (not NaN)
        param_rows = llm_sub[llm_sub["throughput_weight"].notna()].copy()
        if not param_rows.empty:
            mean_params = param_rows.groupby("episode").agg(
                throughput_weight=("throughput_weight", "mean"),
                latency_penalty=("latency_penalty", "mean"),
                npca_switch_bonus=("npca_switch_bonus", "mean"),
            ).reset_index()

            episodes = mean_params["episode"].values
            ax2p = ax.twinx()
            ax.step(episodes, mean_params["throughput_weight"], "b-o",
                    markersize=5, linewidth=1.8, where="post", label="throughput_weight")
            ax.step(episodes, mean_params["npca_switch_bonus"], "g-^",
                    markersize=5, linewidth=1.8, where="post", label="npca_switch_bonus")
            ax2p.step(episodes, mean_params["latency_penalty"], "r-s",
                      markersize=5, linewidth=1.8, where="post", label="latency_penalty")
            ax2p.set_ylabel("latency_penalty", color="red", fontsize=9)

            # qos_priority annotation
            qp_colors = {"throughput": "blue", "latency": "red", "balanced": "green"}
            prev_qp = None
            for _, row in param_rows.drop_duplicates("episode").iterrows():
                qp = row["qos_priority"]
                if pd.notna(qp) and qp != prev_qp:
                    ax.axvline(row["episode"], color=qp_colors.get(qp, "gray"),
                               alpha=0.3, linewidth=1)
                    prev_qp = qp

            lines1, lbl1 = ax.get_legend_handles_labels()
            lines2, lbl2 = ax2p.get_legend_handles_labels()
            ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7)

        llm_tp = llm_sub.groupby("episode")["throughput"].mean().values
        ax.plot([], [], color="purple", linewidth=0, label="(bg: LLM-DRL throughput)")
        ax_bg = axes[1, 1]
        ax_bg2 = ax_bg.twinx() if not hasattr(ax_bg, '_twin_created') else ax_bg
    ax.set_title("LLM Reward Param Evolution\n(vertical lines = qos_priority change)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("throughput_weight / npca_switch_bonus")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "exp1_convergence.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_exp1_summary(df: pd.DataFrame, last_n: int = 50):
    print("\n" + "=" * 75)
    print(f"{'Method':<20} {'AvgThroughput':>14} {'AvgReward':>11} {'AvgNPCA%':>10} {'LLMcalls':>10}")
    print("-" * 75)
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        tail = sub[sub["episode"] >= sub["episode"].max() - last_n + 1]
        avg_tp   = tail["throughput"].mean()
        avg_rew  = tail["reward"].mean()
        avg_npca = tail["npca_ratio"].mean() * 100
        # LLM call count = non-NaN throughput_weight rows (unique episode)
        llm_rows = sub[sub["throughput_weight"].notna()]["episode"].nunique()
        print(f"{method:<20} {avg_tp:>14.1f} {avg_rew:>11.2f} {avg_npca:>9.1f}% {llm_rows:>10}")
    print("=" * 75)
    print(f"  (last {last_n} episodes, reward-agnostic throughput)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp1: Convergence Analysis")
    parser.add_argument("--obss-duration", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--slots", type=int, default=DEFAULT_NUM_SLOTS_PER_EPISODE)
    parser.add_argument("--ppdu", choices=list(PPDU_DURATION_VARIANTS.keys()), default="medium")
    parser.add_argument("--mock", action="store_true", default=False)
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs (for averaging)")
    parser.add_argument("--results-dir", default="./results/exp1_convergence")
    args = parser.parse_args()

    run_exp1(
        obss_duration=args.obss_duration,
        ppdu_variant=args.ppdu,
        num_episodes=args.episodes,
        num_slots=args.slots,
        use_mock=args.mock,
        num_runs=args.runs,
        save_dir=args.results_dir,
    )
