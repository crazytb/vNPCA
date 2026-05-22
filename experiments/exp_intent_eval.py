"""
experiments/exp_intent_eval.py — Per-Intent Train + Eval Experiment

Single focal STA learns the vNPCA policy under each LLM-designed intent.
After training, the trained policy is evaluated greedily (ε=0) for N episodes.

Environment setup (single-agent):
  ch0: 4 background STAs (NPCA channel contenders, npca_enabled=False)
  ch1: 1 focal STA (the learner, npca_enabled=True)

Three KPIs measured in eval phase:
  throughput       — successful TX slots per episode
  avg_option_duration (τ) — latency proxy (slots from OBSS decision to TX end)
  total_energy_uJ  — cumulative radio energy per episode (IEEE 802.11ax model)

Energy model (11-14-0980-16-00ax-simulation-scenarios.docx, 20MHz, 1.1V, NSS=1):
  TX:       280mA × 1.1V × 9μs = 2.772 μJ/slot
  Listen:   50mA  × 1.1V × 9μs = 0.495 μJ/slot  (backoff/frozen/CCA)
  Switch:   TX↔Listen @ 75mW × 0.01ms = 0.75 μJ  per NPCA transition event
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

from drl_framework.configs import (
    LLM_MODEL,
    LLM_UPDATE_INTERVAL,
)
from drl_framework.random_access import Channel, STA, Simulator
from drl_framework.train import evaluate_policy, train_semi_mdp
from llm_reward_designer import LLMRewardDesigner

# ── Experiment constants ────────────────────────────────────────────────────
TRAIN_EPISODES = 3000
EVAL_EPISODES  = 200
NUM_RUNS       = 5
OBSS_RATE      = 0.05
OBSS_RANGE     = (20, 200)
PPDU_DURATION  = 33      # medium
NUM_SLOTS      = int(100_000 / 9)  # ~11 111 slots per episode

LLM_INTENTS = [
    ("latency_sensitive", "화상회의, 지연 최소화"),
    ("throughput_hungry", "파일 다운로드, 속도 최대화"),
    ("energy_saving",     "배터리 절약, 저전력"),
    ("balanced",          "일반 웹 서핑"),
]
DRL_METHODS = ["fixed_drl"]
BASELINES    = ["always_npca", "never_npca", "rule_based"]

ALL_METHODS = [name for name, _ in LLM_INTENTS] + DRL_METHODS + BASELINES

METHOD_LABELS = {
    "latency_sensitive": "Latency\nSensitive",
    "throughput_hungry": "Throughput\nHungry",
    "energy_saving":     "Energy\nSaving",
    "balanced":          "Balanced",
    "fixed_drl":         "Fixed DRL",
    "always_npca":       "Always\nNPCA",
    "never_npca":        "Never\nNPCA",
    "rule_based":        "Rule\nBased",
}


# ── Environment factory ─────────────────────────────────────────────────────

def _make_channels_stas(
    obss_rate: float = OBSS_RATE,
    obss_range: tuple = OBSS_RANGE,
    ppdu_duration: int = PPDU_DURATION,
    radio_transition_time: int = 1,
    num_slots_per_episode: int = NUM_SLOTS,
):
    """Single-agent setup: ch0=4 background STAs, ch1=1 focal NPCA STA."""
    channels = [
        Channel(channel_id=0, obss_generation_rate=0.0),                            # NPCA channel
        Channel(channel_id=1, obss_generation_rate=obss_rate,
                obss_duration_range=obss_range),                                     # primary channel
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
    })
    return channels, stas_config


# ── Baseline evaluation ─────────────────────────────────────────────────────

def _get_baseline_fn(method_name: str):
    """Return fixed_action_fn for a non-DRL baseline."""
    if method_name == "always_npca":
        return lambda sta: (lambda: 1)
    if method_name == "never_npca":
        return lambda sta: (lambda: 0)
    if method_name == "rule_based":
        def _rule(sta):
            def _decide():
                return 1 if sta.primary_channel.obss_remain >= sta.ppdu_duration else 0
            return _decide
        return _rule
    raise ValueError(f"Unknown baseline: {method_name}")


def _evaluate_baseline(
    channels,
    stas_config: list,
    fixed_action_fn,
    n_eval_episodes: int = EVAL_EPISODES,
    num_slots_per_episode: int = NUM_SLOTS,
    device=None,
) -> pd.DataFrame:
    """Evaluate a fixed-action (non-DRL) strategy. Mirrors evaluate_policy() output format."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = []
    for episode in range(n_eval_episodes):
        for ch in channels:
            ch.intra_occupied = False
            ch.intra_end_slot = 0
            ch.obss_traffic = []
            ch.occupied_remain = 0
            ch.obss_remain = 0

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
                ppdu_duration=config.get("ppdu_duration", 33),
                learner=None,
                num_slots_per_episode=num_slots_per_episode,
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

        focal_stas = [sta for sta in stas if sta.channel_id == 1]
        throughput  = sum(sta.channel_occupancy_time for sta in focal_stas)
        energy_uJ   = sum(sta.episode_energy_uJ for sta in focal_stas)

        tau_vals = [d["tau"] for d in episode_decision_log if "tau" in d]
        avg_tau  = sum(tau_vals) / len(tau_vals) if tau_vals else 0.0

        n_decs     = len(episode_decision_log)
        npca_ratio = (sum(1 for d in episode_decision_log if d.get("action") == 1) / n_decs
                      if n_decs > 0 else 0.0)

        records.append({
            "episode":            episode,
            "throughput":         throughput,
            "avg_option_duration": avg_tau,
            "total_energy_uJ":    energy_uJ,
            "npca_switch_ratio":  npca_ratio,
        })

        if episode % 50 == 0:
            print(f"  Eval {episode:3d}: TP={throughput:.0f}, τ={avg_tau:.1f}, "
                  f"E={energy_uJ:.0f}μJ, NPCA={npca_ratio:.2f}")

    return pd.DataFrame(records)


# ── Main experiment ─────────────────────────────────────────────────────────

def run_exp(
    results_dir:     str   = "./results/exp_eval_v1",
    train_episodes:  int   = TRAIN_EPISODES,
    eval_episodes:   int   = EVAL_EPISODES,
    num_runs:        int   = NUM_RUNS,
    use_mock:        bool  = False,
    obss_rate:       float = OBSS_RATE,
    obss_range:      tuple = OBSS_RANGE,
    ppdu_duration:   int   = PPDU_DURATION,
    num_slots:       int   = NUM_SLOTS,
    device=None,
):
    os.makedirs(results_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channels, stas_config = _make_channels_stas(
        obss_rate=obss_rate, obss_range=obss_range,
        ppdu_duration=ppdu_duration, num_slots_per_episode=num_slots,
    )

    all_train_dfs = []
    all_eval_dfs  = []

    # ── LLM intents + fixed_drl ──────────────────────────────────────────
    for method_name, intent_str in LLM_INTENTS + [("fixed_drl", None)]:
        for run_idx in range(num_runs):
            print(f"\n{'='*60}")
            print(f"[{method_name}] run {run_idx+1}/{num_runs}  intent='{intent_str or ''}'")
            print(f"{'='*60}")

            llm_designer = None
            if method_name != "fixed_drl":
                llm_designer = LLMRewardDesigner(
                    model=LLM_MODEL,
                    update_interval=LLM_UPDATE_INTERVAL,
                    use_mock=use_mock,
                )
                llm_designer.set_intent(intent_str)

            # Training phase
            ep_rewards, ep_losses, learner = train_semi_mdp(
                channels=channels,
                stas_config=stas_config,
                num_episodes=train_episodes,
                num_slots_per_episode=num_slots,
                device=device,
                llm_designer=llm_designer,
            )

            # Save training metrics
            train_df = pd.DataFrame({
                "episode":             range(len(learner.episode_throughputs)),
                "method":              method_name,
                "run":                 run_idx,
                "intent":              intent_str or "",
                "throughput":          learner.episode_throughputs,
                "avg_option_duration": learner.episode_avg_option_durations,
                "npca_switch_ratio":   learner.episode_npca_ratios,
                "avg_loss":            learner.episode_avg_losses,
                "epsilon":             learner.episode_epsilons,
            })
            train_df.to_csv(f"{results_dir}/{method_name}_train_run{run_idx}.csv", index=False)
            all_train_dfs.append(train_df)

            if llm_designer is not None and learner.llm_log:
                with open(f"{results_dir}/{method_name}_llm_history_run{run_idx}.json", "w") as f:
                    json.dump(learner.llm_log, f, indent=2)

            # Evaluation phase (greedy, ε=0)
            print(f"\n  --- Evaluation phase ({eval_episodes} episodes, greedy) ---")
            eval_df = evaluate_policy(
                learner=learner,
                channels=channels,
                stas_config=stas_config,
                n_eval_episodes=eval_episodes,
                num_slots_per_episode=num_slots,
                device=device,
                llm_designer=llm_designer,
            )
            eval_df["method"] = method_name
            eval_df["run"]    = run_idx
            eval_df["intent"] = intent_str or ""
            eval_df.to_csv(f"{results_dir}/{method_name}_eval_run{run_idx}.csv", index=False)
            all_eval_dfs.append(eval_df)

            print(f"  → TP={eval_df['throughput'].mean():.0f}±{eval_df['throughput'].std():.0f}  "
                  f"τ={eval_df['avg_option_duration'].mean():.1f}  "
                  f"E={eval_df['total_energy_uJ'].mean():.0f}μJ  "
                  f"NPCA={eval_df['npca_switch_ratio'].mean():.2f}")

    # ── Baselines ────────────────────────────────────────────────────────
    for method_name in BASELINES:
        for run_idx in range(num_runs):
            print(f"\n{'='*60}")
            print(f"[{method_name}] run {run_idx+1}/{num_runs}")
            print(f"{'='*60}")

            baseline_fn = _get_baseline_fn(method_name)
            eval_df = _evaluate_baseline(
                channels=channels,
                stas_config=stas_config,
                fixed_action_fn=baseline_fn,
                n_eval_episodes=eval_episodes,
                num_slots_per_episode=num_slots,
                device=device,
            )
            eval_df["method"] = method_name
            eval_df["run"]    = run_idx
            eval_df["intent"] = ""
            eval_df.to_csv(f"{results_dir}/{method_name}_eval_run{run_idx}.csv", index=False)
            all_eval_dfs.append(eval_df)

            print(f"  → TP={eval_df['throughput'].mean():.0f}±{eval_df['throughput'].std():.0f}  "
                  f"τ={eval_df['avg_option_duration'].mean():.1f}  "
                  f"E={eval_df['total_energy_uJ'].mean():.0f}μJ  "
                  f"NPCA={eval_df['npca_switch_ratio'].mean():.2f}")

    # ── Save combined CSVs ───────────────────────────────────────────────
    combined_eval  = pd.concat(all_eval_dfs,  ignore_index=True)
    combined_train = pd.concat(all_train_dfs, ignore_index=True) if all_train_dfs else None

    combined_eval.to_csv(f"{results_dir}/all_eval.csv",   index=False)
    if combined_train is not None:
        combined_train.to_csv(f"{results_dir}/all_train.csv", index=False)

    print(f"\nAll results saved to {results_dir}/")

    plot_eval_kpi(combined_eval, results_dir)
    if combined_train is not None:
        plot_training_curves(combined_train, results_dir)

    _print_hypothesis_check(combined_eval)

    return combined_eval


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_eval_kpi(eval_df: pd.DataFrame, results_dir: str):
    """3-panel bar chart: throughput / delay / energy by method."""
    metrics = [
        ("throughput",          "Throughput (TX slots/ep)",  "↑ higher is better"),
        ("avg_option_duration", "Avg Option Duration τ (slots)", "↓ lower is better"),
        ("total_energy_uJ",     "Energy Consumption (μJ/ep)",    "↓ lower is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ALL_METHODS)))

    for ax, (col, ylabel, note) in zip(axes, metrics):
        means, errs, labels = [], [], []
        for i, m in enumerate(ALL_METHODS):
            subset = eval_df[eval_df["method"] == m][col].values
            if len(subset) == 0:
                continue
            means.append(subset.mean())
            errs.append(subset.std())
            labels.append(METHOD_LABELS.get(m, m))

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=errs, color=colors[:len(labels)], capsize=4, alpha=0.85, edgecolor="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{ylabel}\n({note})", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Evaluation KPIs by Method (greedy, ε=0)", fontsize=11, y=1.01)
    fig.tight_layout()
    out = f"{results_dir}/eval_kpi_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_training_curves(train_df: pd.DataFrame, results_dir: str, window: int = 50):
    """Training throughput learning curves for DRL methods."""
    drl_methods = [m for m, _ in LLM_INTENTS] + DRL_METHODS
    colors = plt.cm.tab10(np.linspace(0, 1, len(drl_methods)))

    fig, ax = plt.subplots(figsize=(11, 5))
    for method, color in zip(drl_methods, colors):
        df_m = train_df[train_df["method"] == method]
        if df_m.empty:
            continue
        pivot = df_m.pivot_table(index="episode", columns="run", values="throughput", aggfunc="mean")
        mean  = pivot.mean(axis=1).rolling(window, min_periods=1).mean()
        std   = pivot.std(axis=1).rolling(window, min_periods=1).mean().fillna(0)
        ax.plot(mean.index, mean.values, label=METHOD_LABELS.get(method, method), color=color)
        ax.fill_between(mean.index, (mean - std).values, (mean + std).values, alpha=0.12, color=color)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Throughput (TX slots)")
    ax.set_title(f"Training Curves (rolling mean, window={window})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = f"{results_dir}/training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Hypothesis check ────────────────────────────────────────────────────────

def _print_hypothesis_check(eval_df: pd.DataFrame):
    """Print hypothesis verification: throughput, latency, energy vs fixed_drl."""
    ref = eval_df[eval_df["method"] == "fixed_drl"]
    if ref.empty:
        return

    print("\n" + "="*60)
    print("Hypothesis Check (vs fixed_drl baseline)")
    print("="*60)

    checks = [
        ("throughput_hungry",   "throughput",          "≥", "throughput_hungry TP ≥ fixed_drl TP"),
        ("latency_sensitive",   "avg_option_duration", "≤", "latency_sensitive τ ≤ fixed_drl τ"),
        ("energy_saving",       "total_energy_uJ",     "≤", "energy_saving E ≤ fixed_drl E"),
    ]
    for method, col, direction, desc in checks:
        grp = eval_df[eval_df["method"] == method][col].values
        bsl = ref[col].values
        if len(grp) == 0 or len(bsl) == 0:
            continue
        t, p = scipy_stats.ttest_ind(grp, bsl)
        diff = grp.mean() - bsl.mean()
        pct  = diff / bsl.mean() * 100
        passed = (diff >= 0) if direction == "≥" else (diff <= 0)
        sig    = "✅ sig." if p < 0.05 else "⚠️  n.s."
        mark   = "✅" if passed else "❌"
        print(f"  {mark} {desc}: Δ={diff:+.1f} ({pct:+.1f}%), p={p:.3f} {sig}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-Intent Train+Eval Experiment")
    parser.add_argument("--mock",            action="store_true",  help="Mock LLM (no API)")
    parser.add_argument("--train-episodes",  type=int,   default=TRAIN_EPISODES)
    parser.add_argument("--eval-episodes",   type=int,   default=EVAL_EPISODES)
    parser.add_argument("--runs",            type=int,   default=NUM_RUNS)
    parser.add_argument("--obss-rate",       type=float, default=OBSS_RATE)
    parser.add_argument("--obss-min",        type=int,   default=OBSS_RANGE[0])
    parser.add_argument("--obss-max",        type=int,   default=OBSS_RANGE[1])
    parser.add_argument("--ppdu",            type=int,   default=PPDU_DURATION)
    parser.add_argument("--results-dir",     type=str,   default="./results/exp_eval_v1")
    args = parser.parse_args()

    run_exp(
        results_dir=args.results_dir,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        num_runs=args.runs,
        use_mock=args.mock,
        obss_rate=args.obss_rate,
        obss_range=(args.obss_min, args.obss_max),
        ppdu_duration=args.ppdu,
    )


if __name__ == "__main__":
    main()
