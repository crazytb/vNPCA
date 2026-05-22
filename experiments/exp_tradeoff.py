"""
experiments/exp_tradeoff.py — DRL Sanity Check with Bimodal OBSS Trade-off

Tests whether DRL can learn an optimal threshold policy in an environment
designed to make the NPCA switching decision non-trivial.

Bimodal OBSS duration creates a clear trade-off:
  Short OBSS (5~25 slots < ppdu_duration=33): TX_NPCA = min(obss,33) < 33 → worse than waiting
  Long  OBSS (80~200 slots > ppdu_duration):  TX_NPCA = 33 → full PPDU → better than waiting

Optimal policy (≈ rule_based): switch iff obss_remain >= ppdu_duration.

Methods:
  never_npca  — action=0 always
  always_npca — action=1 always (wastes TX on short OBSS events)
  rule_based  — switch iff obss_remain >= ppdu_duration (optimal threshold)
  drl_npca    — DRL, pure throughput reward (no LLM, no latency/energy penalties)

Reward (DRL): reward = 1.0 × TX_slots_in_option  (throughput only)

Environment (single-agent):
  ch0: 4 background STAs (NPCA channel, no external OBSS)
  ch1: 1 focal STA (primary channel, bimodal OBSS)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
from scipy import stats as scipy_stats

from drl_framework.random_access import Channel, STA, Simulator
from drl_framework.train import evaluate_policy, train_semi_mdp

# ── Experiment constants ────────────────────────────────────────────────────
OBSS_SHORT     = (5, 25)      # < ppdu_duration → switching reduces TX → BAD
OBSS_LONG      = (80, 200)    # > ppdu_duration → switching gives full TX → GOOD
PPDU_DURATION  = 33           # threshold: switch iff obss_remain >= PPDU_DURATION
OBSS_RATE      = 0.05
TRAIN_EPISODES = 1000
EVAL_EPISODES  = 500
NUM_RUNS       = 5
NUM_SLOTS      = int(100_000 / 9)   # ~11 111 slots per episode

METHODS = ["never_npca", "always_npca", "rule_based", "drl_npca"]
METHOD_LABELS = {
    "never_npca":  "Never NPCA",
    "always_npca": "Always NPCA",
    "rule_based":  "Rule-Based\n(optimal)",
    "drl_npca":    "DRL NPCA\n(ours)",
}
METHOD_COLORS = {
    "never_npca":  "#888888",
    "always_npca": "#e67e22",
    "rule_based":  "#27ae60",
    "drl_npca":    "#2980b9",
}


# ── Bimodal OBSS sampler ────────────────────────────────────────────────────

def make_bimodal_sampler(short_range=OBSS_SHORT, long_range=OBSS_LONG, short_prob=0.5):
    """Return a callable that samples bimodal OBSS duration."""
    def _sample():
        if random.random() < short_prob:
            return random.randint(*short_range)
        return random.randint(*long_range)
    return _sample


# ── Environment factory ─────────────────────────────────────────────────────

def _make_channels_stas(
    obss_rate: float = OBSS_RATE,
    ppdu_duration: int = PPDU_DURATION,
    radio_transition_time: int = 1,
    num_slots_per_episode: int = NUM_SLOTS,
    bimodal: bool = True,
):
    """ch0: 4 background STAs (no OBSS), ch1: 1 focal NPCA STA (bimodal OBSS)."""
    sampler = make_bimodal_sampler() if bimodal else None
    channels = [
        Channel(channel_id=0, obss_generation_rate=0.0),
        Channel(channel_id=1, obss_generation_rate=obss_rate,
                obss_duration_sampler=sampler),
    ]
    stas_config = []
    for i in range(4):
        stas_config.append({
            "sta_id": i,
            "channel_id": 0,
            "npca_enabled": False,
            "ppdu_duration": ppdu_duration,
            "radio_transition_time": radio_transition_time,
        })
    stas_config.append({
        "sta_id": 4,
        "channel_id": 1,
        "npca_enabled": True,
        "ppdu_duration": ppdu_duration,
        "radio_transition_time": radio_transition_time,
        "throughput_weight":  1.0,  # pure throughput reward: no latency/energy terms
        "latency_penalty":    0.0,
        "npca_switch_bonus":  0.0,
        "npca_switch_cost":   0.0,
    })
    return channels, stas_config


def _reset_channels(channels):
    for ch in channels:
        ch.intra_occupied = False
        ch.intra_end_slot = 0
        ch.obss_traffic = []
        ch.occupied_remain = 0
        ch.obss_remain = 0


# ── Baseline evaluation (with per-decision logging) ─────────────────────────

def _evaluate_baseline_with_decisions(
    channels,
    stas_config: list,
    fixed_action_fn,
    n_eval_episodes: int = EVAL_EPISODES,
    num_slots_per_episode: int = NUM_SLOTS,
    device=None,
):
    """Evaluate a fixed-action strategy. Returns (episode_df, decision_records)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episode_records = []
    all_decisions = []

    for episode in range(n_eval_episodes):
        _reset_channels(channels)
        episode_decision_log = []
        stas = []
        for config in stas_config:
            is_npca = config.get("npca_enabled", False)
            sta = STA(
                sta_id=config["sta_id"],
                channel_id=config["channel_id"],
                primary_channel=channels[config["channel_id"]],
                npca_channel=channels[0] if config["channel_id"] == 1 else None,
                npca_enabled=is_npca,
                radio_transition_time=config.get("radio_transition_time", 1),
                ppdu_duration=config.get("ppdu_duration", PPDU_DURATION),
                learner=None,
                num_slots_per_episode=num_slots_per_episode,
                throughput_weight=1.0,
                latency_penalty=0.0,
                npca_switch_bonus=0.0,
                npca_switch_cost=0.0,
            )
            if is_npca and fixed_action_fn is not None:
                sta._fixed_action = fixed_action_fn(sta)
                sta.decision_log = episode_decision_log
                sta.current_episode = episode
            stas.append(sta)

        simulator = Simulator(num_slots=num_slots_per_episode, channels=channels, stas=stas)
        simulator.memory = None
        simulator.device = device
        simulator.run()

        focal_stas = [s for s in stas if s.channel_id == 1]
        throughput = sum(s.channel_occupancy_time for s in focal_stas)
        tau_vals = [d["tau"] for d in episode_decision_log if "tau" in d]
        avg_tau = sum(tau_vals) / len(tau_vals) if tau_vals else 0.0
        n_decs = len(episode_decision_log)
        npca_ratio = (sum(1 for d in episode_decision_log if d.get("action") == 1) / n_decs
                      if n_decs > 0 else 0.0)

        episode_records.append({
            "episode": episode,
            "throughput": throughput,
            "avg_option_duration": avg_tau,
            "npca_switch_ratio": npca_ratio,
        })
        for d in episode_decision_log:
            all_decisions.append({
                "obss_remain": d.get("primary_channel_obss_occupied_remained", 0),
                "action": d.get("action", 0),
                "tau": d.get("tau", 0),
            })

    return pd.DataFrame(episode_records), all_decisions


def _evaluate_drl_with_decisions(
    learner,
    channels,
    stas_config: list,
    n_eval_episodes: int = EVAL_EPISODES,
    num_slots_per_episode: int = NUM_SLOTS,
    device=None,
):
    """Greedy DRL evaluation, also returns per-decision records."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learner.eval_mode = True
    saved_memory = learner.memory
    learner.memory = None
    learner.policy_net.eval()

    episode_records = []
    all_decisions = []

    try:
        for episode in range(n_eval_episodes):
            _reset_channels(channels)
            episode_decision_log = []
            stas = []
            for config in stas_config:
                is_npca = config.get("npca_enabled", False)
                sta = STA(
                    sta_id=config["sta_id"],
                    channel_id=config["channel_id"],
                    primary_channel=channels[config["channel_id"]],
                    npca_channel=channels[0] if config["channel_id"] == 1 else None,
                    npca_enabled=is_npca,
                    radio_transition_time=config.get("radio_transition_time", 1),
                    ppdu_duration=config.get("ppdu_duration", PPDU_DURATION),
                    learner=learner if is_npca else None,
                    num_slots_per_episode=num_slots_per_episode,
                    throughput_weight=1.0,
                    latency_penalty=0.0,
                    npca_switch_bonus=0.0,
                    npca_switch_cost=0.0,
                )
                if is_npca:
                    sta.decision_log = episode_decision_log
                    sta.current_episode = episode
                stas.append(sta)

            simulator = Simulator(num_slots=num_slots_per_episode, channels=channels, stas=stas)
            simulator.memory = None
            simulator.device = device
            simulator.run()

            focal_stas = [s for s in stas if s.channel_id == 1]
            throughput = sum(s.channel_occupancy_time for s in focal_stas)
            tau_vals = [d["tau"] for d in episode_decision_log if "tau" in d]
            avg_tau = sum(tau_vals) / len(tau_vals) if tau_vals else 0.0
            n_decs = len(episode_decision_log)
            npca_ratio = (sum(1 for d in episode_decision_log if d.get("action") == 1) / n_decs
                          if n_decs > 0 else 0.0)

            episode_records.append({
                "episode": episode,
                "throughput": throughput,
                "avg_option_duration": avg_tau,
                "npca_switch_ratio": npca_ratio,
            })
            for d in episode_decision_log:
                all_decisions.append({
                    "obss_remain": d.get("primary_channel_obss_occupied_remained", 0),
                    "action": d.get("action", 0),
                    "tau": d.get("tau", 0),
                })
    finally:
        learner.eval_mode = False
        learner.memory = saved_memory
        learner.policy_net.train()

    return pd.DataFrame(episode_records), all_decisions


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_results(all_eval: pd.DataFrame, drl_train: pd.DataFrame, all_decisions: dict,
                 results_dir: str, ppdu_duration: int = PPDU_DURATION):
    """3-panel figure:
      Left:   training curve (DRL throughput, smoothed)
      Middle: eval throughput bar chart (4 methods)
      Right:  P(switch=1) vs OBSS duration at decision (threshold analysis)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("NPCA DRL Sanity Check — Bimodal OBSS Trade-off", fontsize=13, fontweight="bold")

    # ── Panel 1: DRL training curve ──────────────────────────────────────────
    ax = axes[0]
    win = 30
    for run_idx, df in drl_train.groupby("run"):
        tp = df["throughput"].values
        smooth = np.convolve(tp, np.ones(win)/win, mode="valid")
        ax.plot(smooth, alpha=0.4, color=METHOD_COLORS["drl_npca"], linewidth=0.8)
    # mean over runs
    mean_tp = drl_train.groupby("episode")["throughput"].mean().values
    smooth_mean = np.convolve(mean_tp, np.ones(win)/win, mode="valid")
    ax.plot(smooth_mean, color=METHOD_COLORS["drl_npca"], linewidth=2, label="DRL (mean)")
    # rule_based reference line
    rb_mean = all_eval[all_eval["method"] == "rule_based"]["throughput"].mean()
    ax.axhline(rb_mean, color=METHOD_COLORS["rule_based"], linestyle="--", linewidth=1.5,
               label=f"Rule-Based eval ({rb_mean:.0f})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Throughput (TX slots/ep)")
    ax.set_title("DRL Training Convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 2: Eval throughput bar chart ───────────────────────────────────
    ax = axes[1]
    means = all_eval.groupby("method")["throughput"].mean()
    stds  = all_eval.groupby("method")["throughput"].std()
    x = np.arange(len(METHODS))
    bars = ax.bar(x, [means.get(m, 0) for m in METHODS],
                  yerr=[stds.get(m, 0) for m in METHODS],
                  color=[METHOD_COLORS[m] for m in METHODS],
                  capsize=5, alpha=0.85, edgecolor="white")
    # annotate values
    for bar, m in zip(bars, METHODS):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + stds.get(m, 0) + 30,
                f"{h:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=9)
    ax.set_ylabel("Mean Throughput (TX slots/ep)")
    ax.set_title("Eval Throughput Comparison")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 3: Decision threshold plot ─────────────────────────────────────
    ax = axes[2]
    # Bins: short zone (5~33) and long zone (34~200)
    bins = [0, 10, 20, 33, 50, 80, 120, 200]
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    bin_labels  = [f"{bins[i]}~{bins[i+1]}" for i in range(len(bins)-1)]

    for method, decisions in all_decisions.items():
        if not decisions:
            continue
        df_d = pd.DataFrame(decisions)
        prob_switch = []
        for i in range(len(bins)-1):
            mask = (df_d["obss_remain"] >= bins[i]) & (df_d["obss_remain"] < bins[i+1])
            sub = df_d[mask]
            p = sub["action"].mean() if len(sub) > 0 else float("nan")
            prob_switch.append(p)
        ax.plot(bin_centers, prob_switch, marker="o", linewidth=2,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method], alpha=0.85)

    ax.axvline(ppdu_duration, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold\n(ppdu={ppdu_duration})")
    ax.set_xlabel("OBSS Remaining at Decision (slots)")
    ax.set_ylabel("P(Switch to NPCA)")
    ax.set_title("Decision Threshold Analysis")
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels, rotation=45, fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "tradeoff_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_npca_ratio(all_eval: pd.DataFrame, results_dir: str):
    """Bar chart: NPCA switch ratio per method."""
    fig, ax = plt.subplots(figsize=(7, 4))
    means = all_eval.groupby("method")["npca_switch_ratio"].mean()
    stds  = all_eval.groupby("method")["npca_switch_ratio"].std()
    x = np.arange(len(METHODS))
    ax.bar(x, [means.get(m, 0) for m in METHODS],
           yerr=[stds.get(m, 0) for m in METHODS],
           color=[METHOD_COLORS[m] for m in METHODS],
           capsize=5, alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=9)
    ax.set_ylabel("NPCA Switch Ratio")
    ax.set_title("Bimodal OBSS — NPCA Switch Ratio per Method")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "npca_ratio.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Hypothesis check ────────────────────────────────────────────────────────

def hypothesis_check(all_eval: pd.DataFrame):
    """Print whether DRL ≥ always_npca and DRL ≈ rule_based (within 5%)."""
    print("\n── Hypothesis Check ──────────────────────────────────────────")
    m = all_eval.groupby("method")["throughput"].mean()
    s = all_eval.groupby("method")["throughput"].std()
    for method in METHODS:
        print(f"  {method:15s}: TP = {m.get(method, float('nan')):.1f} ± {s.get(method, 0):.1f}")

    drl_tp    = all_eval[all_eval["method"] == "drl_npca"]["throughput"]
    rule_tp   = all_eval[all_eval["method"] == "rule_based"]["throughput"]
    always_tp = all_eval[all_eval["method"] == "always_npca"]["throughput"]
    never_tp  = all_eval[all_eval["method"] == "never_npca"]["throughput"]

    if len(drl_tp) > 0 and len(always_tp) > 0:
        t, p = scipy_stats.ttest_ind(drl_tp, always_tp)
        diff = drl_tp.mean() - always_tp.mean()
        print(f"\n  DRL vs Always-NPCA: Δ={diff:+.1f}, p={p:.3f} {'✅' if diff > 0 else '❌'}")

    if len(drl_tp) > 0 and len(rule_tp) > 0:
        t, p = scipy_stats.ttest_ind(drl_tp, rule_tp)
        diff = drl_tp.mean() - rule_tp.mean()
        gap  = abs(diff) / rule_tp.mean() * 100
        print(f"  DRL vs Rule-Based:  Δ={diff:+.1f} ({gap:.1f}% gap), p={p:.3f}"
              f"  {'≈ optimal ✅' if gap < 5 else '⚠️ not yet optimal'}")

    if len(drl_tp) > 0 and len(never_tp) > 0:
        diff = drl_tp.mean() - never_tp.mean()
        print(f"  DRL vs Never-NPCA:  Δ={diff:+.1f} {'✅' if diff > 0 else '❌'}")
    print("──────────────────────────────────────────────────────────────\n")


# ── Main experiment ─────────────────────────────────────────────────────────

def run_exp(
    results_dir:     str  = "./results/exp_tradeoff_v1",
    train_episodes:  int  = TRAIN_EPISODES,
    eval_episodes:   int  = EVAL_EPISODES,
    num_runs:        int  = NUM_RUNS,
    obss_rate:       float = OBSS_RATE,
    ppdu_duration:   int  = PPDU_DURATION,
    num_slots:       int  = NUM_SLOTS,
    device=None,
):
    os.makedirs(results_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Bimodal OBSS: short={OBSS_SHORT}, long={OBSS_LONG}, PPDU={ppdu_duration}")
    print(f"Train: {train_episodes} ep × {num_runs} runs | Eval: {eval_episodes} ep\n")

    channels, stas_config = _make_channels_stas(
        obss_rate=obss_rate, ppdu_duration=ppdu_duration,
        num_slots_per_episode=num_slots,
    )

    all_eval_dfs  = []
    all_train_dfs = []
    # per-method decision records (for threshold plot), aggregated over runs
    all_decisions = {m: [] for m in METHODS}

    # ── Baselines (no training needed) ──────────────────────────────────────
    baseline_fns = {
        "never_npca":  lambda sta: (lambda: 0),
        "always_npca": lambda sta: (lambda: 1),
        "rule_based":  lambda sta: (lambda s=sta: 1 if s.primary_channel.obss_remain >= ppdu_duration else 0),
    }
    for method in ["never_npca", "always_npca", "rule_based"]:
        print(f"[{method}] Evaluating…")
        run_dfs = []
        run_decisions = []
        for run_idx in range(num_runs):
            ep_df, decisions = _evaluate_baseline_with_decisions(
                channels, stas_config,
                fixed_action_fn=baseline_fns[method],
                n_eval_episodes=eval_episodes,
                num_slots_per_episode=num_slots,
                device=device,
            )
            ep_df["run"]    = run_idx
            ep_df["method"] = method
            run_dfs.append(ep_df)
            run_decisions.extend(decisions)
            path = os.path.join(results_dir, f"{method}_eval_run{run_idx}.csv")
            ep_df.to_csv(path, index=False)
        all_eval_dfs.extend(run_dfs)
        all_decisions[method] = run_decisions
        mean_tp = pd.concat(run_dfs)["throughput"].mean()
        mean_nr = pd.concat(run_dfs)["npca_switch_ratio"].mean()
        print(f"  → mean TP={mean_tp:.1f}, NPCA ratio={mean_nr:.3f}\n")

    # ── DRL training + evaluation ────────────────────────────────────────────
    print("[drl_npca] Training…")
    run_eval_dfs   = []
    run_train_dfs  = []
    run_decisions  = []

    for run_idx in range(num_runs):
        print(f"\n  Run {run_idx+1}/{num_runs}")
        _ep_rewards, _ep_losses, learner = train_semi_mdp(
            channels=channels,
            stas_config=stas_config,
            num_episodes=train_episodes,
            num_slots_per_episode=num_slots,
            device=device,
            llm_designer=None,
        )
        train_df = pd.DataFrame({
            "episode":             range(len(learner.episode_throughputs)),
            "run":                 run_idx,
            "method":              "drl_npca",
            "throughput":          learner.episode_throughputs,
            "avg_option_duration": learner.episode_avg_option_durations,
            "npca_switch_ratio":   learner.episode_npca_ratios,
            "avg_loss":            learner.episode_avg_losses,
            "epsilon":             learner.episode_epsilons,
        })
        run_train_dfs.append(train_df)
        path = os.path.join(results_dir, f"drl_npca_train_run{run_idx}.csv")
        train_df.to_csv(path, index=False)

        print(f"  Evaluating run {run_idx+1}…")
        ep_df, decisions = _evaluate_drl_with_decisions(
            learner=learner,
            channels=channels,
            stas_config=stas_config,
            n_eval_episodes=eval_episodes,
            num_slots_per_episode=num_slots,
            device=device,
        )
        ep_df["run"]    = run_idx
        ep_df["method"] = "drl_npca"
        run_eval_dfs.append(ep_df)
        run_decisions.extend(decisions)
        path = os.path.join(results_dir, f"drl_npca_eval_run{run_idx}.csv")
        ep_df.to_csv(path, index=False)

    all_eval_dfs.extend(run_eval_dfs)
    all_train_dfs.extend(run_train_dfs)
    all_decisions["drl_npca"] = run_decisions

    all_eval_df  = pd.concat(all_eval_dfs,  ignore_index=True)
    all_train_df = pd.concat(all_train_dfs, ignore_index=True)

    all_eval_df.to_csv(os.path.join(results_dir, "all_eval.csv"),  index=False)
    all_train_df.to_csv(os.path.join(results_dir, "all_train.csv"), index=False)

    hypothesis_check(all_eval_df)

    print("Generating plots…")
    plot_results(all_eval_df, all_train_df, all_decisions, results_dir, ppdu_duration)
    plot_npca_ratio(all_eval_df, results_dir)

    print(f"\nAll done → {results_dir}")
    return all_eval_df, all_train_df


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRL trade-off sanity check (bimodal OBSS)")
    parser.add_argument("--train-episodes", type=int,   default=TRAIN_EPISODES)
    parser.add_argument("--eval-episodes",  type=int,   default=EVAL_EPISODES)
    parser.add_argument("--runs",           type=int,   default=NUM_RUNS)
    parser.add_argument("--obss-rate",      type=float, default=OBSS_RATE)
    parser.add_argument("--ppdu",           type=int,   default=PPDU_DURATION)
    parser.add_argument("--slots",          type=int,   default=NUM_SLOTS)
    parser.add_argument("--results-dir",    type=str,   default="./results/exp_tradeoff_v1")
    args = parser.parse_args()

    run_exp(
        results_dir=args.results_dir,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        num_runs=args.runs,
        obss_rate=args.obss_rate,
        ppdu_duration=args.ppdu,
        num_slots=args.slots,
    )
