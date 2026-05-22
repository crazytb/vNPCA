#!/usr/bin/env python3
"""
Experiment 2: Heterogeneous Device Requirements

관찰 목표: Heterogeneous 디바이스 요구조건(throughput vs latency vs energy)이
          LLM에 의해 reward 파라미터에 잘 반영되는가?

핵심 가설:
  - latency_sensitive  → 높은 latency_penalty → 짧은 옵션 지속 시간 (OBSS 블로킹 최소화)
  - throughput_hungry  → 높은 throughput_weight + npca_switch_bonus → 높은 처리량 + 높은 NPCA 비율
  - energy_saving      → 높은 npca_switch_cost → 낮은 NPCA 전환 비율 (에너지 절약)
  - balanced           → 통계 기반 중간 동작
  - fixed_drl          → LLM 없이 고정 파라미터 (baseline)

출력:
  results/exp2_heterogeneous/
    ├── exp2_{group}_run{N}.csv    # per-episode 상세 로그 (group별)
    ├── exp2_all.csv               # 전체 통합 로그
    ├── exp2_convergence.png       # 4-panel 비교 그래프
    └── exp2_llm_history_{group}_run{N}.json  # LLM 호출 이력

CSV 컬럼:
  episode, group, run, obss_duration,
  throughput, avg_option_duration, reward, avg_loss, epsilon, npca_ratio,
  throughput_weight, latency_penalty, npca_switch_bonus, npca_switch_cost, qos_priority

Usage:
  .venv/bin/python experiments/exp2_heterogeneous.py --mock --episodes 200 --slots 2000
  .venv/bin/python experiments/exp2_heterogeneous.py --episodes 1000 --runs 3
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
# Intent definitions (group label → natural language intent string)
# ---------------------------------------------------------------------------

GROUPS = {
    "latency_sensitive":  "화상회의, 지연 최소화 (video call, minimize delay)",
    "throughput_hungry":  "파일 다운로드, 속도 최대화 (file download, maximize throughput)",
    "energy_saving":      "배터리 절약, 저전력 (battery saving, low power)",
    "balanced":           "",           # no intent — stat-driven
    "fixed_drl":          None,         # None = no LLM designer at all (baseline)
}

GROUP_COLORS = {
    "latency_sensitive": "tomato",
    "throughput_hungry": "royalblue",
    "energy_saving":     "forestgreen",
    "balanced":          "darkorange",
    "fixed_drl":         "gray",
}

GROUP_STYLES = {
    "latency_sensitive": "-",
    "throughput_hungry": "-",
    "energy_saving":     "-",
    "balanced":          "--",
    "fixed_drl":         ":",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_channels_stas(
    obss_duration_range: tuple = (20, 200),
    obss_rate: float = 0.05,
    num_stas: int = 4,
    ppdu_variant: str = "medium",
):
    ppdu = PPDU_DURATION_VARIANTS.get(ppdu_variant, 33)
    channels = [
        Channel(channel_id=0, obss_generation_rate=OBSS_GENERATION_RATE["secondary"]),
        Channel(channel_id=1,
                obss_generation_rate=obss_rate,
                obss_duration_range=obss_duration_range),
    ]
    stas = []
    for i in range(num_stas):
        stas.append({"sta_id": i, "channel_id": 0, "npca_enabled": False,
                     "ppdu_duration": ppdu, "radio_transition_time": RADIO_TRANSITION_TIME})
    for i in range(num_stas):
        stas.append({"sta_id": num_stas + i, "channel_id": 1, "npca_enabled": True,
                     "ppdu_duration": ppdu, "radio_transition_time": RADIO_TRANSITION_TIME})
    return channels, stas


def _running_avg(data, w):
    out = []
    for i in range(len(data)):
        s = max(0, i - w + 1)
        out.append(float(np.mean(data[s:i+1])))
    return out


def _build_df(group, run, obss_duration,
              rewards, throughputs, avg_option_durations, avg_losses, epsilons, npca_ratios,
              llm_log=None):
    n = len(rewards)
    tw_col  = [NAN] * n
    lp_col  = [NAN] * n
    sb_col  = [NAN] * n
    sc_col  = [NAN] * n
    qp_col  = [None] * n

    if llm_log:
        for entry in llm_log:
            ep = entry["episode"]
            if ep < n:
                tw_col[ep] = entry.get("throughput_weight", NAN)
                lp_col[ep] = entry.get("latency_penalty", NAN)
                sb_col[ep] = entry.get("npca_switch_bonus", NAN)
                sc_col[ep] = entry.get("npca_switch_cost", NAN)
                qp_col[ep] = entry.get("qos_priority", None)
        # Forward-fill from first LLM call
        last_tw, last_lp, last_sb, last_sc, last_qp = NAN, NAN, NAN, NAN, None
        for i in range(n):
            if not math.isnan(tw_col[i]):
                last_tw, last_lp, last_sb, last_sc, last_qp = (
                    tw_col[i], lp_col[i], sb_col[i], sc_col[i], qp_col[i])
            else:
                tw_col[i], lp_col[i], sb_col[i], sc_col[i], qp_col[i] = (
                    last_tw, last_lp, last_sb, last_sc, last_qp)

    return pd.DataFrame({
        "episode":             range(n),
        "group":               group,
        "run":                 run,
        "obss_duration":       obss_duration,
        "throughput":          throughputs,
        "avg_option_duration": avg_option_durations,
        "reward":              rewards,
        "avg_loss":            avg_losses,
        "epsilon":             epsilons,
        "npca_ratio":          npca_ratios,
        "throughput_weight":   tw_col,
        "latency_penalty":     lp_col,
        "npca_switch_bonus":   sb_col,
        "npca_switch_cost":    sc_col,
        "qos_priority":        qp_col,
    })


# ---------------------------------------------------------------------------
# Single-group runners
# ---------------------------------------------------------------------------

def _run_llm_group(group, intent, obss_duration_range, obss_rate, num_stas,
                   num_episodes, num_slots, ppdu_variant, use_mock, run_id, device):
    """Run DRL + LLM-designed reward with a given intent string."""
    channels, stas = _make_channels_stas(obss_duration_range, obss_rate, num_stas, ppdu_variant)
    designer = LLMRewardDesigner(model=LLM_MODEL, update_interval=LLM_UPDATE_INTERVAL,
                                 use_mock=use_mock)
    if intent:
        designer.set_intent(intent)

    rewards, _, learner = train_semi_mdp(
        channels=channels, stas_config=stas,
        num_episodes=num_episodes, num_slots_per_episode=num_slots,
        device=device, random_ppdu=True, llm_designer=designer,
    )
    obss_label = f"{obss_duration_range[0]}-{obss_duration_range[1]}"
    return (
        _build_df(
            group=group, run=run_id, obss_duration=obss_label,
            rewards=rewards,
            throughputs=learner.episode_throughputs,
            avg_option_durations=learner.episode_avg_option_durations,
            avg_losses=learner.episode_avg_losses,
            epsilons=learner.episode_epsilons,
            npca_ratios=learner.episode_npca_ratios,
            llm_log=getattr(learner, "llm_log", None),
        ),
        designer,
    )


def _run_fixed_group(obss_duration_range, obss_rate, num_stas,
                     num_episodes, num_slots, ppdu_variant, run_id, device):
    """Run DRL with fixed (no LLM) reward parameters as baseline."""
    channels, stas = _make_channels_stas(obss_duration_range, obss_rate, num_stas, ppdu_variant)
    rewards, _, learner = train_semi_mdp(
        channels=channels, stas_config=stas,
        num_episodes=num_episodes, num_slots_per_episode=num_slots,
        device=device, random_ppdu=True, llm_designer=None,
    )
    obss_label = f"{obss_duration_range[0]}-{obss_duration_range[1]}"
    return _build_df(
        group="fixed_drl", run=run_id, obss_duration=obss_label,
        rewards=rewards,
        throughputs=learner.episode_throughputs,
        avg_option_durations=learner.episode_avg_option_durations,
        avg_losses=learner.episode_avg_losses,
        epsilons=learner.episode_epsilons,
        npca_ratios=learner.episode_npca_ratios,
    )


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_exp2(
    obss_duration_range: tuple = (20, 200),
    obss_rate: float = 0.05,
    num_stas: int = 4,
    ppdu_variant: str = "medium",
    num_episodes: int = DEFAULT_NUM_EPISODES,
    num_slots: int = DEFAULT_NUM_SLOTS_PER_EPISODE,
    use_mock: bool = True,
    num_runs: int = 1,
    save_dir: str = "./results/exp2_v2",
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_dfs = []

    obss_label = f"{obss_duration_range[0]}-{obss_duration_range[1]}"
    print(f"Exp2 config: obss_rate={obss_rate}, obss_range={obss_label}, "
          f"num_stas={num_stas}, episodes={num_episodes}, runs={num_runs}")

    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"[Exp2] Run {run+1}/{num_runs}  obss_range={obss_label}  episodes={num_episodes}")
        print(f"{'='*60}")

        for group, intent in GROUPS.items():
            print(f"\n--- Group: {group} ---")
            if intent is not None:
                if intent:
                    print(f"    Intent: {intent}")
                else:
                    print(f"    Intent: (none — stat-driven)")
                df, designer = _run_llm_group(
                    group=group, intent=intent,
                    obss_duration_range=obss_duration_range, obss_rate=obss_rate,
                    num_stas=num_stas, num_episodes=num_episodes,
                    num_slots=num_slots, ppdu_variant=ppdu_variant,
                    use_mock=use_mock, run_id=run, device=device,
                )
                if designer.call_history:
                    hist_path = os.path.join(
                        save_dir, f"exp2_llm_history_{group}_run{run}.json")
                    designer.save_history(hist_path)
            else:
                # fixed_drl — no LLM
                df = _run_fixed_group(
                    obss_duration_range=obss_duration_range, obss_rate=obss_rate,
                    num_stas=num_stas, num_episodes=num_episodes,
                    num_slots=num_slots, ppdu_variant=ppdu_variant,
                    run_id=run, device=device,
                )

            csv_path = os.path.join(save_dir, f"exp2_{group}_run{run}.csv")
            df.to_csv(csv_path, index=False)
            all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(os.path.join(save_dir, "exp2_all.csv"), index=False)
    print(f"\nAll logs saved → {save_dir}/exp2_all.csv  ({len(full_df)} rows)")

    plot_exp2(full_df, save_dir=save_dir, window=max(10, num_episodes // 20))
    _print_exp2_summary(full_df)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_exp2(df: pd.DataFrame, save_dir: str, window: int = 50):
    """
    4-panel figure:
      Panel 1: Throughput convergence per group
      Panel 2: Avg option duration per group (latency proxy)
      Panel 3: NPCA switch ratio per group
      Panel 4: Final LLM reward params per group (bar chart)
    """
    os.makedirs(save_dir, exist_ok=True)
    groups = list(GROUPS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Exp2: Heterogeneous Device Requirements\n"
        "(LLM translates user intent → reward parameters → agent behavior)",
        fontsize=12, fontweight="bold")

    # ---- Panel 1: Throughput ----
    ax = axes[0, 0]
    for group in groups:
        sub = df[df["group"] == group]
        if sub.empty:
            continue
        mean_tp = sub.groupby("episode")["throughput"].mean().values
        ra = _running_avg(mean_tp.tolist(), window)
        lw = 2.5 if group != "fixed_drl" else 1.5
        ax.plot(ra, color=GROUP_COLORS.get(group, "gray"),
                linestyle=GROUP_STYLES.get(group, "-"),
                linewidth=lw, label=group)
        if sub["run"].nunique() > 1:
            std_tp = sub.groupby("episode")["throughput"].std().fillna(0).values
            ra_std = _running_avg(std_tp.tolist(), window)
            eps = sub["episode"].unique()
            ax.fill_between(eps,
                            [r - s for r, s in zip(ra, ra_std)],
                            [r + s for r, s in zip(ra, ra_std)],
                            alpha=0.12, color=GROUP_COLORS.get(group, "gray"))
    ax.set_title(f"Throughput (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Successful TX Slots (ch1 STAs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Avg option duration (latency proxy) ----
    ax = axes[0, 1]
    for group in groups:
        sub = df[df["group"] == group]
        if sub.empty:
            continue
        mean_tau = sub.groupby("episode")["avg_option_duration"].mean().values
        ra = _running_avg(mean_tau.tolist(), window)
        ax.plot(ra, color=GROUP_COLORS.get(group, "gray"),
                linestyle=GROUP_STYLES.get(group, "-"),
                linewidth=2, label=group)
    ax.set_title("Avg Option Duration τ (Latency Proxy)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Option Duration (slots)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: NPCA switch ratio ----
    ax = axes[1, 0]
    for group in groups:
        sub = df[df["group"] == group]
        if sub.empty:
            continue
        mean_npca = sub.groupby("episode")["npca_ratio"].mean().values
        ra = _running_avg(mean_npca.tolist(), window)
        ax.plot(ra, color=GROUP_COLORS.get(group, "gray"),
                linestyle=GROUP_STYLES.get(group, "-"),
                linewidth=2, label=group)
    ax.set_title("NPCA Switch Ratio")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Fraction of OBSS decisions → NPCA")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: Final LLM reward params (grouped bar) ----
    ax = axes[1, 1]
    llm_groups = [g for g in groups if g != "fixed_drl"]
    params_to_plot = ["throughput_weight", "latency_penalty", "npca_switch_bonus", "npca_switch_cost"]
    param_labels   = ["tw", "lp", "sb", "sc"]

    last_n = 50
    final_params = {}
    for group in llm_groups:
        sub = df[(df["group"] == group) & df["throughput_weight"].notna()]
        if sub.empty:
            final_params[group] = [NAN] * 4
        else:
            tail = sub[sub["episode"] >= sub["episode"].max() - last_n + 1]
            final_params[group] = [tail[p].mean() for p in params_to_plot]

    x = np.arange(len(params_to_plot))
    bar_width = 0.18
    offsets = np.linspace(-(len(llm_groups)-1)/2, (len(llm_groups)-1)/2, len(llm_groups)) * bar_width
    for i, group in enumerate(llm_groups):
        vals = final_params[group]
        bars = ax.bar(x + offsets[i], vals, bar_width,
                      label=group, color=GROUP_COLORS.get(group, "gray"), alpha=0.8)
    ax.set_title(f"Final LLM Reward Params (avg last {last_n} ep)")
    ax.set_xticks(x)
    ax.set_xticklabels(param_labels)
    ax.set_ylabel("Parameter Value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(save_dir, "exp2_convergence.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {path}")

    # Extra figure: expected vs actual param direction (hypothesis verification)
    _plot_hypothesis_check(df, save_dir, last_n=50)


def _plot_hypothesis_check(df: pd.DataFrame, save_dir: str, last_n: int = 50):
    """
    3-panel figure comparing key metrics across intents at convergence.
    Hypothesis direction annotations included.
    """
    groups = list(GROUPS.keys())
    llm_groups = [g for g in groups if g != "fixed_drl"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Exp2: Hypothesis Verification — Final Convergence Metrics",
                 fontsize=12, fontweight="bold")

    metrics = [
        ("throughput",          "Final Throughput (slots)",        "↑ throughput_hungry"),
        ("avg_option_duration", "Final Avg Option Duration (slots)","↓ latency_sensitive"),
        ("npca_ratio",          "Final NPCA Switch Ratio",          "↓ energy_saving"),
    ]
    for ax, (col, ylabel, hint) in zip(axes, metrics):
        vals, errs, colors = [], [], []
        for group in groups:
            sub = df[df["group"] == group]
            tail = sub[sub["episode"] >= sub["episode"].max() - last_n + 1]
            m = tail[col].mean()
            s = tail[col].std()
            vals.append(m if not math.isnan(m) else 0.0)
            errs.append(s if not math.isnan(s) else 0.0)
            colors.append(GROUP_COLORS.get(group, "gray"))

        bars = ax.bar(range(len(groups)), vals, color=colors, alpha=0.8, width=0.6)
        ax.errorbar(range(len(groups)), vals, yerr=errs, fmt="none",
                    color="black", capsize=4, linewidth=1.5)
        ax.set_title(f"{ylabel}\n({hint})", fontsize=10)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(save_dir, "exp2_hypothesis_check.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {path}")


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def _print_exp2_summary(df: pd.DataFrame, last_n: int = 50):
    print("\n" + "=" * 95)
    print(f"{'Group':<22} {'AvgTP':>10} {'AvgTau':>10} {'AvgNPCA%':>10} "
          f"{'tw':>6} {'lp':>6} {'sb':>6} {'sc':>6} {'qos_priority':<14}")
    print("-" * 95)
    for group in GROUPS.keys():
        sub = df[df["group"] == group]
        if sub.empty:
            continue
        tail = sub[sub["episode"] >= sub["episode"].max() - last_n + 1]
        avg_tp   = tail["throughput"].mean()
        avg_tau  = tail["avg_option_duration"].mean()
        avg_npca = tail["npca_ratio"].mean() * 100
        tw = tail["throughput_weight"].mean()
        lp = tail["latency_penalty"].mean()
        sb = tail["npca_switch_bonus"].mean()
        sc = tail["npca_switch_cost"].mean()
        qp_counts = tail["qos_priority"].value_counts()
        qp_top = qp_counts.index[0] if len(qp_counts) > 0 else "—"
        print(f"{group:<22} {avg_tp:>10.1f} {avg_tau:>10.1f} {avg_npca:>9.1f}% "
              f"{tw:>6.3f} {lp:>6.4f} {sb:>6.3f} {sc:>6.3f} {qp_top:<14}")
    print("=" * 95)
    print(f"  (last {last_n} episodes per group)\n")

    # Hypothesis check
    print("Hypothesis verification:")
    groups_present = df["group"].unique().tolist()
    def _tail_mean(group, col):
        sub = df[df["group"] == group]
        if sub.empty:
            return NAN
        tail = sub[sub["episode"] >= sub["episode"].max() - last_n + 1]
        return tail[col].mean()

    if "throughput_hungry" in groups_present and "fixed_drl" in groups_present:
        tp_hungry = _tail_mean("throughput_hungry", "throughput")
        tp_fixed  = _tail_mean("fixed_drl", "throughput")
        ok = "✅" if tp_hungry >= tp_fixed else "⚠️"
        print(f"  {ok} throughput_hungry TP ({tp_hungry:.1f}) >= fixed_drl TP ({tp_fixed:.1f})?")

    if "latency_sensitive" in groups_present and "fixed_drl" in groups_present:
        tau_lat   = _tail_mean("latency_sensitive", "avg_option_duration")
        tau_fixed = _tail_mean("fixed_drl",         "avg_option_duration")
        ok = "✅" if tau_lat <= tau_fixed else "⚠️"
        print(f"  {ok} latency_sensitive τ ({tau_lat:.1f}) <= fixed_drl τ ({tau_fixed:.1f})?")

    if "energy_saving" in groups_present and "fixed_drl" in groups_present:
        nr_energy = _tail_mean("energy_saving", "npca_ratio")
        nr_fixed  = _tail_mean("fixed_drl",     "npca_ratio")
        ok = "✅" if nr_energy <= nr_fixed else "⚠️"
        print(f"  {ok} energy_saving NPCA ratio ({nr_energy:.3f}) <= fixed_drl ({nr_fixed:.3f})?")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp2: Heterogeneous Device Requirements")
    parser.add_argument("--obss-min", type=int, default=20,
                        help="Min OBSS duration in slots (default 20)")
    parser.add_argument("--obss-max", type=int, default=200,
                        help="Max OBSS duration in slots (default 200)")
    parser.add_argument("--obss-rate", type=float, default=0.05,
                        help="OBSS generation rate (default 0.05)")
    parser.add_argument("--num-stas", type=int, default=4,
                        help="STAs per channel (default 4)")
    parser.add_argument("--episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--slots", type=int, default=DEFAULT_NUM_SLOTS_PER_EPISODE)
    parser.add_argument("--ppdu", choices=list(PPDU_DURATION_VARIANTS.keys()), default="medium")
    parser.add_argument("--mock", action="store_true", default=False)
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of independent runs per group")
    parser.add_argument("--results-dir", default="./results/exp2_v2")
    args = parser.parse_args()

    run_exp2(
        obss_duration_range=(args.obss_min, args.obss_max),
        obss_rate=args.obss_rate,
        num_stas=args.num_stas,
        ppdu_variant=args.ppdu,
        num_episodes=args.episodes,
        num_slots=args.slots,
        use_mock=args.mock,
        num_runs=args.runs,
        save_dir=args.results_dir,
    )
