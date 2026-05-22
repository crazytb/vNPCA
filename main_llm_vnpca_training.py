#!/usr/bin/env python3
"""
LLM-Enhanced vNPCA DRL Training

Two-timescale LLM-DRL framework for IEEE 802.11bn NPCA channel access:
  - Slow timescale (every LLM_UPDATE_INTERVAL episodes):
      AP-side LLMRewardDesigner analyzes network stats → updates reward params
  - Fast timescale (per OBSS detection slot):
      STA-side DRL agent makes NPCA switching decision

Usage:
    # LLM-DRL experiment (mock mode, no API key needed)
    python main_llm_vnpca_training.py --mode llm --mock

    # LLM-DRL experiment (real Claude API)
    python main_llm_vnpca_training.py --mode llm

    # Compare LLM-DRL vs fixed-reward DRL
    python main_llm_vnpca_training.py --mode compare --mock

    # Full comparison across OBSS conditions
    python main_llm_vnpca_training.py --mode full --mock
"""

import os
import argparse
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from drl_framework.random_access import Channel
from drl_framework.train import train_semi_mdp, run_baseline_simulation
from drl_framework.configs import (
    PPDU_DURATION_VARIANTS,
    RADIO_TRANSITION_TIME,
    OBSS_GENERATION_RATE,
    OBSS_DURATION_RANGE,
    DEFAULT_NUM_EPISODES,
    DEFAULT_NUM_SLOTS_PER_EPISODE,
    DEFAULT_NUM_STAS_CH0,
    DEFAULT_NUM_STAS_CH1,
    LLM_MODEL,
    LLM_UPDATE_INTERVAL,
    LLM_USE_MOCK,
)
from llm_reward_designer import LLMRewardDesigner


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def make_channels_and_stas(obss_duration: int = 100, ppdu_variant: str = "medium"):
    ppdu_duration = PPDU_DURATION_VARIANTS.get(ppdu_variant, 33)
    channels = [
        Channel(channel_id=0, obss_generation_rate=OBSS_GENERATION_RATE["secondary"]),
        Channel(
            channel_id=1,
            obss_generation_rate=OBSS_GENERATION_RATE["primary"],
            obss_duration_range=(obss_duration, obss_duration),
        ),
    ]
    stas_config = []
    for i in range(DEFAULT_NUM_STAS_CH0):
        stas_config.append({
            "sta_id": i,
            "channel_id": 0,
            "npca_enabled": False,
            "ppdu_duration": ppdu_duration,
            "radio_transition_time": RADIO_TRANSITION_TIME,
        })
    for i in range(DEFAULT_NUM_STAS_CH1):
        stas_config.append({
            "sta_id": DEFAULT_NUM_STAS_CH0 + i,
            "channel_id": 1,
            "npca_enabled": True,
            "ppdu_duration": ppdu_duration,
            "radio_transition_time": RADIO_TRANSITION_TIME,
        })
    return channels, stas_config


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_llm_experiment(
    obss_duration: int = 100,
    ppdu_variant: str = "medium",
    num_episodes: int = DEFAULT_NUM_EPISODES,
    num_slots: int = DEFAULT_NUM_SLOTS_PER_EPISODE,
    use_mock: bool = LLM_USE_MOCK,
    save_dir: str = "./results/llm_drl",
    random_ppdu: bool = True,
    label: str = "LLM-DRL",
) -> dict:
    """Run LLM-enhanced DRL experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels, stas_config = make_channels_and_stas(obss_duration, ppdu_variant)

    designer = LLMRewardDesigner(
        model=LLM_MODEL,
        update_interval=LLM_UPDATE_INTERVAL,
        use_mock=use_mock,
    )

    print(f"\n{'='*60}")
    print(f"[{label}] obss_duration={obss_duration}, ppdu={ppdu_variant}, mock={use_mock}")
    print(f"{'='*60}")

    rewards, losses, learner = train_semi_mdp(
        channels=channels,
        stas_config=stas_config,
        num_episodes=num_episodes,
        num_slots_per_episode=num_slots,
        device=device,
        random_ppdu=random_ppdu,
        llm_designer=designer,
    )
    throughputs = learner.episode_throughputs
    option_durations = learner.episode_avg_option_durations

    os.makedirs(save_dir, exist_ok=True)
    _save_results(rewards, losses, throughputs, option_durations, learner, designer, save_dir, label)

    return {
        "label": label,
        "obss_duration": obss_duration,
        "episode_rewards": rewards,
        "episode_throughputs": throughputs,
        "episode_avg_option_durations": option_durations,
        "episode_losses": losses,
        "final_avg_throughput": float(np.mean(throughputs[-50:])),
        "final_avg_option_duration": float(np.mean(option_durations[-50:])) if option_durations else 0.0,
        "final_avg_reward": float(np.mean(rewards[-50:])),
        "steps_done": learner.steps_done,
        "llm_calls": designer.call_count,
        "_designer": designer,
    }


def run_always_npca_experiment(
    obss_duration: int = 100,
    ppdu_variant: str = "medium",
    num_episodes: int = DEFAULT_NUM_EPISODES,
    num_slots: int = DEFAULT_NUM_SLOTS_PER_EPISODE,
    save_dir: str = "./results/always_npca",
    random_ppdu: bool = True,
    label: str = "Always NPCA",
) -> dict:
    """Baseline: always switch to NPCA when OBSS detected."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels, stas_config = make_channels_and_stas(obss_duration, ppdu_variant)

    print(f"\n{'='*60}")
    print(f"[{label}] obss_duration={obss_duration}")
    print(f"{'='*60}")

    rewards, throughputs, option_durations = run_baseline_simulation(
        channels=channels,
        stas_config=stas_config,
        num_episodes=num_episodes,
        num_slots_per_episode=num_slots,
        fixed_action_fn=lambda _: (lambda: 1),
        random_ppdu=random_ppdu,
        device=device,
    )

    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame({"episode": range(len(rewards)), "reward": rewards,
                  "throughput": throughputs, "avg_option_duration": option_durations}).to_csv(
        os.path.join(save_dir, f"results_{label.replace(' ', '_')}.csv"), index=False
    )

    return {
        "label": label,
        "obss_duration": obss_duration,
        "episode_rewards": rewards,
        "episode_throughputs": throughputs,
        "episode_avg_option_durations": option_durations,
        "episode_losses": [],
        "final_avg_throughput": float(np.mean(throughputs[-50:])),
        "final_avg_option_duration": float(np.mean(option_durations[-50:])) if option_durations else 0.0,
        "final_avg_reward": float(np.mean(rewards[-50:])),
        "steps_done": 0,
        "llm_calls": 0,
    }


def run_never_npca_experiment(
    obss_duration: int = 100,
    ppdu_variant: str = "medium",
    num_episodes: int = DEFAULT_NUM_EPISODES,
    num_slots: int = DEFAULT_NUM_SLOTS_PER_EPISODE,
    save_dir: str = "./results/never_npca",
    random_ppdu: bool = True,
    label: str = "Never NPCA",
) -> dict:
    """Baseline: never switch to NPCA (always stay on primary)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels, stas_config = make_channels_and_stas(obss_duration, ppdu_variant)

    print(f"\n{'='*60}")
    print(f"[{label}] obss_duration={obss_duration}")
    print(f"{'='*60}")

    rewards, throughputs, option_durations = run_baseline_simulation(
        channels=channels,
        stas_config=stas_config,
        num_episodes=num_episodes,
        num_slots_per_episode=num_slots,
        fixed_action_fn=lambda _: (lambda: 0),
        random_ppdu=random_ppdu,
        device=device,
    )

    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame({"episode": range(len(rewards)), "reward": rewards,
                  "throughput": throughputs, "avg_option_duration": option_durations}).to_csv(
        os.path.join(save_dir, f"results_{label.replace(' ', '_')}.csv"), index=False
    )

    return {
        "label": label,
        "obss_duration": obss_duration,
        "episode_rewards": rewards,
        "episode_throughputs": throughputs,
        "episode_avg_option_durations": option_durations,
        "episode_losses": [],
        "final_avg_throughput": float(np.mean(throughputs[-50:])),
        "final_avg_option_duration": float(np.mean(option_durations[-50:])) if option_durations else 0.0,
        "final_avg_reward": float(np.mean(rewards[-50:])),
        "steps_done": 0,
        "llm_calls": 0,
    }


def run_rule_based_experiment(
    obss_duration: int = 100,
    ppdu_variant: str = "medium",
    num_episodes: int = DEFAULT_NUM_EPISODES,
    num_slots: int = DEFAULT_NUM_SLOTS_PER_EPISODE,
    save_dir: str = "./results/rule_based",
    random_ppdu: bool = True,
    label: str = "Rule-based",
) -> dict:
    """Baseline: switch to NPCA if OBSS remaining >= PPDU duration (greedy threshold rule)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels, stas_config = make_channels_and_stas(obss_duration, ppdu_variant)

    print(f"\n{'='*60}")
    print(f"[{label}] obss_duration={obss_duration}, rule: switch if obss_remain >= ppdu_duration")
    print(f"{'='*60}")

    def rule_fn(sta):
        def _action():
            return 1 if sta.primary_channel.obss_remain >= sta.ppdu_duration else 0
        return _action

    rewards, throughputs, option_durations = run_baseline_simulation(
        channels=channels,
        stas_config=stas_config,
        num_episodes=num_episodes,
        num_slots_per_episode=num_slots,
        fixed_action_fn=rule_fn,
        random_ppdu=random_ppdu,
        device=device,
    )

    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame({"episode": range(len(rewards)), "reward": rewards,
                  "throughput": throughputs, "avg_option_duration": option_durations}).to_csv(
        os.path.join(save_dir, f"results_{label.replace(' ', '_')}.csv"), index=False
    )

    return {
        "label": label,
        "obss_duration": obss_duration,
        "episode_rewards": rewards,
        "episode_throughputs": throughputs,
        "episode_avg_option_durations": option_durations,
        "episode_losses": [],
        "final_avg_throughput": float(np.mean(throughputs[-50:])),
        "final_avg_option_duration": float(np.mean(option_durations[-50:])) if option_durations else 0.0,
        "final_avg_reward": float(np.mean(rewards[-50:])),
        "steps_done": 0,
        "llm_calls": 0,
    }


def run_fixed_experiment(
    obss_duration: int = 100,
    ppdu_variant: str = "medium",
    num_episodes: int = DEFAULT_NUM_EPISODES,
    num_slots: int = DEFAULT_NUM_SLOTS_PER_EPISODE,
    save_dir: str = "./results/fixed_drl",
    random_ppdu: bool = True,
    label: str = "DRL (fixed reward)",
) -> dict:
    """Run baseline DRL with fixed reward parameters (no LLM)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels, stas_config = make_channels_and_stas(obss_duration, ppdu_variant)

    print(f"\n{'='*60}")
    print(f"[{label}] obss_duration={obss_duration}, ppdu={ppdu_variant}")
    print(f"{'='*60}")

    rewards, losses, learner = train_semi_mdp(
        channels=channels,
        stas_config=stas_config,
        num_episodes=num_episodes,
        num_slots_per_episode=num_slots,
        device=device,
        random_ppdu=random_ppdu,
        llm_designer=None,
    )
    throughputs = learner.episode_throughputs
    option_durations = learner.episode_avg_option_durations

    os.makedirs(save_dir, exist_ok=True)
    _save_results(rewards, losses, throughputs, option_durations, learner, None, save_dir, label)

    return {
        "label": label,
        "obss_duration": obss_duration,
        "episode_rewards": rewards,
        "episode_throughputs": throughputs,
        "episode_avg_option_durations": option_durations,
        "episode_losses": losses,
        "final_avg_throughput": float(np.mean(throughputs[-50:])),
        "final_avg_option_duration": float(np.mean(option_durations[-50:])) if option_durations else 0.0,
        "final_avg_reward": float(np.mean(rewards[-50:])),
        "steps_done": learner.steps_done,
        "llm_calls": 0,
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_results(rewards, losses, throughputs, option_durations, learner, designer, save_dir, label):
    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
    torch.save({
        "policy_net_state_dict": learner.policy_net.state_dict(),
        "target_net_state_dict": learner.target_net.state_dict(),
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "episode_rewards": rewards,
        "episode_throughputs": throughputs,
        "episode_avg_option_durations": option_durations,
        "episode_losses": losses,
        "steps_done": learner.steps_done,
        "label": label,
    }, os.path.join(save_dir, f"model_{safe_label}.pth"))

    pd.DataFrame({"episode": range(len(rewards)), "reward": rewards,
                  "throughput": throughputs,
                  "avg_option_duration": option_durations}).to_csv(
        os.path.join(save_dir, f"results_{safe_label}.csv"), index=False
    )

    if designer is not None and designer.call_history:
        designer.save_history(os.path.join(save_dir, f"llm_history_{safe_label}.json"))
        df = pd.DataFrame(designer.call_history)
        df.to_csv(os.path.join(save_dir, f"llm_decisions_{safe_label}.csv"), index=False)
        print(f"  LLM history saved ({designer.call_count} calls)")

    llm_log = getattr(learner, "llm_log", [])
    if llm_log:
        pd.DataFrame(llm_log).to_csv(
            os.path.join(save_dir, f"llm_log_{safe_label}.csv"), index=False
        )

    print(f"  Model saved → {save_dir}/model_{safe_label}.pth")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _running_avg(data, window):
    out = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(data[start: i + 1])))
    return out


def plot_comparison(results: list[dict], save_dir: str = "./results", window: int = 50):
    """
    Three-panel comparison figure using throughput as the primary metric.

    Panel 1: Throughput learning curves (reward-agnostic)
    Panel 2: Final avg throughput bar chart
    Panel 3: LLM reward param evolution overlaid on LLM-DRL throughput
    """
    os.makedirs(save_dir, exist_ok=True)
    colors = ["royalblue", "tomato", "forestgreen", "darkorange", "purple"]
    styles = ["-", "--", "-.", ":", "-"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Throughput learning curves ---
    ax = axes[0]
    for i, res in enumerate(results):
        tp = res.get("episode_throughputs", [])
        if not tp:
            continue
        ra = _running_avg(tp, window)
        ax.plot(ra, color=colors[i % len(colors)], linestyle=styles[i % len(styles)],
                linewidth=2, label=res["label"])
    ax.set_title(f"Throughput Convergence\n(running avg, window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Successful TX Slots (ch1 STAs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Final throughput bar ---
    ax = axes[1]
    labels = [r["label"] for r in results]
    finals = [r.get("final_avg_throughput", 0.0) for r in results]
    bars = ax.bar(range(len(labels)), finals, color=colors[: len(labels)])
    ax.set_title("Final Avg Throughput\n(last 50 episodes)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Avg Successful TX Slots")
    for bar, v in zip(bars, finals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(finals) * 0.01,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: LLM param evolution overlaid on LLM-DRL throughput ---
    ax = axes[2]
    llm_res = next((r for r in results if "LLM" in r["label"]), None)

    if llm_res and llm_res.get("episode_throughputs"):
        tp = llm_res["episode_throughputs"]
        ra = _running_avg(tp, window)
        ax.plot(ra, color="purple", linewidth=2, alpha=0.6, label="LLM-DRL throughput")
        ax.set_ylabel("Successful TX Slots", color="purple")
        ax.set_xlabel("Episode")

        # Overlay LLM param changes from saved designer
        designer = llm_res.get("_designer")
        if designer and designer.call_history:
            episodes = [h["network_stats"].get("episode_progress", 0) *
                        len(tp) for h in designer.call_history]
            tw_vals = [h["throughput_weight"] for h in designer.call_history]
            lp_vals = [h["latency_penalty"] for h in designer.call_history]
            sb_vals = [h.get("npca_switch_bonus", 0.0) for h in designer.call_history]
            sc_vals = [h.get("npca_switch_cost", 0.0) for h in designer.call_history]
            qp_vals = [h.get("qos_priority", "balanced") for h in designer.call_history]

            ax2 = ax.twinx()
            ax2.step(episodes, tw_vals, "b-o", markersize=5, linewidth=1.5,
                     where="post", label="throughput_weight")
            ax2.step(episodes, lp_vals, "r-s", markersize=5, linewidth=1.5,
                     where="post", label="latency_penalty")
            ax2.step(episodes, sb_vals, "g-^", markersize=5, linewidth=1.5,
                     where="post", label="npca_switch_bonus")
            ax2.step(episodes, sc_vals, "m-D", markersize=5, linewidth=1.5,
                     where="post", label="npca_switch_cost")
            ax2.set_ylabel("LLM reward params", fontsize=9)

            # Annotate qos_priority transitions
            qp_colors = {"throughput": "blue", "latency": "red", "energy": "purple", "balanced": "green"}
            prev_qp = None
            for ep, qp in zip(episodes, qp_vals):
                if qp != prev_qp:
                    ax.axvline(ep, color=qp_colors.get(qp, "gray"), alpha=0.25, linewidth=1)
                    ax.text(ep, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 0,
                            qp[0].upper(), fontsize=6, color=qp_colors.get(qp, "gray"), va="bottom")
                    prev_qp = qp

            lines1, lbl1 = ax.get_legend_handles_labels()
            lines2, lbl2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7, loc="lower right")
        else:
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "LLM-DRL data unavailable",
                transform=ax.transAxes, ha="center", va="center")

    ax.set_title("LLM-DRL: Throughput & Reward Param Evolution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nComparison plot saved → {path}")


def plot_obss_sweep(all_results: list[dict], save_dir: str = "./results/sweep"):
    """Final throughput vs OBSS duration for all methods (reward-agnostic comparison)."""
    os.makedirs(save_dir, exist_ok=True)

    # Group by label prefix (strip OBSS tag)
    from collections import defaultdict
    groups: dict = defaultdict(lambda: {"durations": [], "finals": []})
    for r in all_results:
        base = r["label"].split("(")[0].strip()
        groups[base]["durations"].append(r["obss_duration"])
        groups[base]["finals"].append(r.get("final_avg_throughput", 0.0))

    colors = ["purple", "royalblue", "tomato", "forestgreen", "darkorange"]
    styles = ["-o", "--s", "-.^", ":D", "-*"]

    plt.figure(figsize=(9, 5))
    for (base, data), col, sty in zip(sorted(groups.items()), colors, styles):
        xs = data["durations"]
        ys = data["finals"]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        plt.plot([xs[i] for i in order], [ys[i] for i in order],
                 sty, color=col, linewidth=2, markersize=7, label=base)

    plt.xlabel("OBSS Duration (slots)")
    plt.ylabel("Final Avg Throughput — Successful TX Slots\n(last 50 episodes)")
    plt.title("Throughput vs OBSS Duration: All Methods")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    path = os.path.join(save_dir, "obss_sweep_throughput.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"OBSS sweep plot saved → {path}")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-Enhanced vNPCA DRL Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["llm", "fixed", "compare", "full"],
        default="compare",
        help=(
            "llm: LLM-DRL only | fixed: fixed-reward DRL only | "
            "compare: both on single OBSS condition | full: sweep OBSS durations"
        ),
    )
    parser.add_argument("--mock", action="store_true", default=False,
                        help="Use mock LLM (no API key needed)")
    parser.add_argument("--obss-duration", type=int, default=100,
                        help="OBSS duration in slots (used in llm/fixed/compare modes)")
    parser.add_argument("--episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--slots", type=int, default=DEFAULT_NUM_SLOTS_PER_EPISODE)
    parser.add_argument("--ppdu", choices=list(PPDU_DURATION_VARIANTS.keys()), default="medium")
    parser.add_argument("--no-random-ppdu", dest="random_ppdu", action="store_false", default=True)
    parser.add_argument("--results-dir", default="./results/llm_vnpca")
    args = parser.parse_args()

    use_mock = args.mock or LLM_USE_MOCK

    print("=" * 60)
    print("LLM-Enhanced vNPCA DRL Framework")
    print("=" * 60)
    print(f"Mode: {args.mode} | Mock LLM: {use_mock}")
    print(f"Episodes: {args.episodes} | Slots/ep: {args.slots}")

    if args.mode == "llm":
        result = run_llm_experiment(
            obss_duration=args.obss_duration,
            ppdu_variant=args.ppdu,
            num_episodes=args.episodes,
            num_slots=args.slots,
            use_mock=use_mock,
            save_dir=args.results_dir,
            random_ppdu=args.random_ppdu,
        )
        print(f"\nFinal avg reward: {result['final_avg_reward']:.2f}")
        print(f"LLM calls made: {result['llm_calls']}")

    elif args.mode == "fixed":
        result = run_fixed_experiment(
            obss_duration=args.obss_duration,
            ppdu_variant=args.ppdu,
            num_episodes=args.episodes,
            num_slots=args.slots,
            save_dir=args.results_dir,
            random_ppdu=args.random_ppdu,
        )
        print(f"\nFinal avg reward: {result['final_avg_reward']:.2f}")

    elif args.mode == "compare":
        results = []
        shared = dict(
            obss_duration=args.obss_duration,
            ppdu_variant=args.ppdu,
            num_episodes=args.episodes,
            num_slots=args.slots,
            save_dir=args.results_dir,
            random_ppdu=args.random_ppdu,
        )
        results.append(run_always_npca_experiment(**shared, label="Always NPCA"))
        results.append(run_never_npca_experiment(**shared, label="Never NPCA"))
        results.append(run_rule_based_experiment(**shared, label="Rule-based"))
        results.append(run_fixed_experiment(**shared, label="DRL (fixed reward)"))
        results.append(run_llm_experiment(**shared, use_mock=use_mock, label="LLM-DRL (proposed)"))
        plot_comparison(results, save_dir=args.results_dir)
        _print_summary(results)

    elif args.mode == "full":
        obss_durations = [20, 50, 100, 200]
        all_results = []
        for od in obss_durations:
            all_results.append(run_llm_experiment(
                obss_duration=od,
                ppdu_variant=args.ppdu,
                num_episodes=args.episodes,
                num_slots=args.slots,
                use_mock=use_mock,
                save_dir=os.path.join(args.results_dir, f"obss_{od}"),
                random_ppdu=args.random_ppdu,
                label=f"LLM-DRL (OBSS={od})",
            ))
            all_results.append(run_fixed_experiment(
                obss_duration=od,
                ppdu_variant=args.ppdu,
                num_episodes=args.episodes,
                num_slots=args.slots,
                save_dir=os.path.join(args.results_dir, f"obss_{od}"),
                random_ppdu=args.random_ppdu,
                label=f"DRL fixed (OBSS={od})",
            ))
        plot_obss_sweep(all_results, save_dir=os.path.join(args.results_dir, "sweep"))
        _print_summary(all_results)

    print("\nDone.")


def _print_summary(results: list[dict]):
    print("\n" + "=" * 72)
    print(f"{'Method':<30} {'OBSS':>6} {'AvgThroughput':>14} {'AvgReward':>10} {'LLMcalls':>9}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['label']:<30} {str(r['obss_duration']):>6} "
            f"{r.get('final_avg_throughput', 0.0):>14.1f} "
            f"{r['final_avg_reward']:>10.2f} "
            f"{r['llm_calls']:>9}"
        )
    print("=" * 72)
    print("  AvgThroughput = avg successful TX slots/episode (last 50 eps, reward-agnostic)")


if __name__ == "__main__":
    main()
