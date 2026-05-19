import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import count
from collections import namedtuple

from drl_framework.network import ReplayMemory, DQN
from drl_framework.params import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'cum_reward', 'tau', 'done'))

class SemiMDPLearner:
    def __init__(self, n_observations, n_actions, device, memory_capacity=10000, lr=LR):
        self.device = device
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.steps_done = 0
        
        # DQN 네트워크 초기화
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer 및 Memory 초기화
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(memory_capacity)
    
    def select_action(self, state_tensor):
        """Epsilon-greedy action selection for Semi-MDP"""
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state_tensor).max(1)[1].item()
        else:
            return random.randint(0, self.n_actions - 1)
    
    def optimize_model(self):
        """Semi-MDP용 최적화 함수 - 옵션 기반 학습"""
        if len(self.memory) < BATCH_SIZE:
            return
            
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).long().unsqueeze(1)
        R_batch = torch.tensor(batch.cum_reward, device=self.device).float()  # 옵션 누적 보상
        tau_batch = torch.tensor(batch.tau, device=self.device).float()       # 옵션 길이(슬롯 수)
        done_batch = torch.tensor(batch.done, device=self.device).float()     # 1.0 if done else 0.0

        # Q(s_t, a)
        q_sa = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        # max_a' Q_target(s', a') for non-terminal only
        non_final_mask = (done_batch == 0)
        
        next_state_values = torch.zeros(len(state_batch), device=self.device)
        if non_final_mask.sum() > 0:  # non-terminal states가 있는 경우에만 처리
            non_final_next_states = torch.stack(
                [s for s, d in zip(batch.next_state, batch.done) if not d]
            ).to(self.device)
            
            with torch.no_grad():
                if non_final_next_states.numel() > 0:
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Semi-MDP TD Target: R + γ^τ * V(s') for non-terminal states
        # tau 클리핑으로 극도로 작은 할인 인수 방지
        tau_clipped = torch.clamp(tau_batch, max=20.0)  # 최대 20 슬롯으로 제한
        td_target = R_batch + (GAMMA ** tau_clipped) * next_state_values * (1.0 - done_batch)
        loss = F.smooth_l1_loss(q_sa, td_target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Soft update of target network"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
                                         target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)

def select_action(state, policy_net, env, steps_done, device):
    """Epsilon-greedy action selection"""
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model(policy_net, target_net, memory, optimizer, device):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
    #                               device=device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch  = torch.stack(batch.state).to(device)                # [B, obs_dim]
    action_batch = torch.tensor(batch.action, device=device).long().unsqueeze(1)
    R_batch      = torch.tensor(batch.cum_reward, device=device).float()  # 옵션 누적 보상
    tau_batch    = torch.tensor(batch.tau, device=device).float()         # 옵션 길이(슬롯 수)
    done_batch   = torch.tensor(batch.done, device=device).float()         # 1.0 if done else 0.0

    # Q(s_t, a)
    # state_action_values = policy_net(state_batch).gather(1, action_batch)
    q_sa = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    # # V(s_{t+1})
    # next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # with torch.no_grad():
    #     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # max_a' Q_target(s', a') for non-terminal only
    non_final_mask = (done_batch == 0)
    non_final_next_states = torch.stack(
        [s for s, d in zip(batch.next_state, batch.done) if not d]
    ).to(device)

    next_state_values = torch.zeros(len(state_batch), device=device)
    with torch.no_grad():
        if non_final_next_states.numel() > 0:
            # (DDQN 원하면 여기서 policy_net.argmax로 a' 뽑아 target_net.gather 사용)
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Loss
    # criterion = nn.SmoothL1Loss()
    # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    td_target = R_batch + (GAMMA ** tau_batch) * next_state_values * (1.0 - done_batch)
    loss = F.smooth_l1_loss(q_sa, td_target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def train(env, policy_net, target_net, optimizer, device, num_episodes=50):
    memory = ReplayMemory(10000)
    steps_done = 0
    episode_rewards = []

    for i_episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(env.flatten_dict_values(state),
                             dtype=torch.float32, device=device).unsqueeze(0)

        total_reward = 0
        for t in count():
            action = select_action(state, policy_net, env, steps_done, device)
            steps_done += 1

            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            total_reward += reward.item()
            done = terminated or truncated

            if not done:
                next_state = torch.tensor(env.flatten_dict_values(observation),
                                          dtype=torch.float32, device=device).unsqueeze(0)
            else:
                next_state = None

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model(policy_net, target_net, memory, optimizer, device)

            # Soft update target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
                                             target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_rewards.append(total_reward)
                print(f"Episode {i_episode}: total reward = {total_reward}")
                break

    return episode_rewards

def collect_network_stats(
    episode_rewards: list,
    decision_log: list,
    simulator,
    num_slots_per_episode: int,
    current_epsilon: float,
    current_episode: int,
    num_episodes: int,
    window: int = 50,
) -> dict:
    """
    Collect network statistics for the LLM reward designer.
    Called once every LLM_UPDATE_INTERVAL episodes inside train_semi_mdp().
    """
    win_rewards = episode_rewards[-window:] if len(episode_rewards) >= 1 else [0.0]
    avg_reward = sum(win_rewards) / len(win_rewards)

    recent_eps = [d for d in decision_log if d.get("episode", 0) >= current_episode - window]
    n_decisions = len(recent_eps)

    if n_decisions > 0:
        npca_switches = sum(1 for d in recent_eps if d.get("action", 0) == 1)
        npca_switch_ratio = npca_switches / n_decisions
        tau_vals = [d["tau"] for d in recent_eps if "tau" in d]
        avg_tau = sum(tau_vals) / len(tau_vals) if tau_vals else 1.0
        # reward per decision (normalised back to per-slot scale)
        rew_vals = [d["reward"] * num_slots_per_episode for d in recent_eps if "reward" in d]
        avg_throughput = sum(max(0.0, r) for r in rew_vals) / max(1, len(rew_vals))
    else:
        npca_switch_ratio = 0.5
        avg_tau = 10.0
        avg_throughput = 0.0

    # OBSS occupancy from the last episode's simulator log (channel_1 = primary)
    if simulator.log:
        obss_slots = sum(
            1 for row in simulator.log if row.get("channel_1_obss_occupied_remained", 0) > 0
        )
        obss_rate = obss_slots / len(simulator.log)
    else:
        obss_rate = 0.03

    return {
        "window_episodes": window,
        "obss_occupancy_rate": obss_rate,
        "avg_episode_reward": avg_reward,
        "npca_switch_ratio": npca_switch_ratio,
        "avg_throughput_slots": avg_throughput,
        "avg_option_duration": avg_tau,
        "current_epsilon": current_epsilon,
        "episode_progress": (current_episode + 1) / max(1, num_episodes),
    }


def run_baseline_simulation(
    channels,
    stas_config,
    num_episodes: int = 100,
    num_slots_per_episode: int = 1000,
    fixed_action_fn=None,
    random_ppdu: bool = False,
    device=None,
):
    """
    Run simulation with a fixed (non-DRL) action strategy.

    fixed_action_fn: callable(sta) -> callable()
        Given a STA, returns a zero-arg function that returns 0 or 1 each call.
        Used for Always-NPCA, Never-NPCA, and rule-based baselines.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from drl_framework.random_access import STA, Simulator

    episode_rewards = []
    episode_throughputs = []          # reward-agnostic: successful TX slots of ch1 STAs
    episode_avg_option_durations = [] # avg option duration (tau) per episode — latency proxy

    for episode in range(num_episodes):
        for ch in channels:
            ch.intra_occupied = False
            ch.intra_end_slot = 0
            ch.obss_traffic = []
            ch.occupied_remain = 0
            ch.obss_remain = 0

        episode_decision_log = []
        stas = []
        for config in stas_config:
            sta = STA(
                sta_id=config["sta_id"],
                channel_id=config["channel_id"],
                primary_channel=channels[config["channel_id"]],
                npca_channel=channels[0] if config["channel_id"] == 1 else None,
                npca_enabled=config.get("npca_enabled", False),
                radio_transition_time=config.get("radio_transition_time", 1),
                ppdu_duration=config.get("ppdu_duration", 33),
                random_ppdu=random_ppdu,
                learner=None,
                num_slots_per_episode=num_slots_per_episode,
            )
            if config.get("npca_enabled", False) and fixed_action_fn is not None:
                sta._fixed_action = fixed_action_fn(sta)
                sta.decision_log = episode_decision_log
                sta.current_episode = episode
            stas.append(sta)

        simulator = Simulator(num_slots=num_slots_per_episode, channels=channels, stas=stas)
        # No replay memory for baselines; _finalize_pending handles memory=None gracefully
        simulator.memory = None
        simulator.device = device
        simulator.run()

        total_reward = sum(sta.new_episode_reward for sta in stas if sta.npca_enabled)
        episode_rewards.append(total_reward)

        # channel_occupancy_time: successful TX slots regardless of reward params
        throughput = sum(sta.channel_occupancy_time for sta in stas if sta.channel_id == 1)
        episode_throughputs.append(throughput)

        # avg option duration (tau) per episode
        tau_vals = [d["tau"] for d in episode_decision_log if "tau" in d]
        ep_avg_tau = sum(tau_vals) / len(tau_vals) if tau_vals else 0.0
        episode_avg_option_durations.append(ep_avg_tau)

        if episode % 100 == 0:
            avg_tp = sum(episode_throughputs[-10:]) / min(10, len(episode_throughputs))
            avg_tau = sum(episode_avg_option_durations[-10:]) / min(10, len(episode_avg_option_durations))
            print(f"  Episode {episode:3d}: Avg Throughput = {avg_tp:6.1f} slots, Avg Tau = {avg_tau:.1f}")

    print("Baseline simulation completed!")
    return episode_rewards, episode_throughputs, episode_avg_option_durations


def train_semi_mdp(channels, stas_config, num_episodes=100, num_slots_per_episode=1000, device=None, random_ppdu=False, llm_designer=None):
    """
    Semi-MDP를 사용한 NPCA STA 학습 함수
    
    Args:
        channels: 채널 리스트
        stas_config: STA 설정 리스트
        num_episodes: 학습 에피소드 수
        num_slots_per_episode: 에피소드당 슬롯 수
        device: torch device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # SemiMDPLearner 초기화
    n_observations = 4  # obs feature 개수
    n_actions = 2       # 0=StayPrimary, 1=GoNPCA
    learner = SemiMDPLearner(n_observations, n_actions, device)
    
    episode_rewards = []
    episode_losses = []
    episode_throughputs = []          # reward-agnostic: successful TX slots of ch1 STAs
    episode_epsilons = []             # epsilon at start of each episode
    episode_npca_ratios = []          # NPCA switch ratio per episode
    episode_avg_losses = []           # average loss per episode
    episode_avg_option_durations = [] # avg option duration (tau) per episode — latency proxy
    llm_log = []                      # LLM reward-design call records

    # CSV 로깅을 위한 리스트들
    decision_log = []  # 모든 결정 시점 기록

    print(f"Starting Semi-MDP training on {device}")
    print(f"Episodes: {num_episodes}, Slots per episode: {num_slots_per_episode}")
    if llm_designer is not None:
        print(f"LLM Reward Designer enabled (update every {llm_designer.update_interval} episodes, "
              f"mock={llm_designer.use_mock})")
    
    for episode in range(num_episodes):
        # epsilon at episode start
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * learner.steps_done / EPS_DECAY)
        episode_epsilons.append(epsilon)
        losses_before = len(episode_losses)

        # 채널 상태 초기화
        for ch in channels:
            ch.intra_occupied = False
            ch.intra_end_slot = 0
            ch.obss_traffic = []
            ch.occupied_remain = 0
            ch.obss_remain = 0
        
        # STA 생성 및 초기화 (각 에피소드마다 새로 생성)  
        from drl_framework.random_access import STA, Simulator
        stas = []
        for config in stas_config:
            sta = STA(
                sta_id=config["sta_id"],
                channel_id=config["channel_id"],
                primary_channel=channels[config["channel_id"]],
                npca_channel=channels[0] if config["channel_id"] == 1 else None,
                npca_enabled=config.get("npca_enabled", False),
                radio_transition_time=config.get("radio_transition_time", 1),
                ppdu_duration=config.get("ppdu_duration", 33),
                random_ppdu=random_ppdu,
                learner=learner if config.get("npca_enabled", False) else None,
                num_slots_per_episode=num_slots_per_episode
            )
            
            # CSV 로깅을 위한 설정 (NPCA enabled STA만)
            if config.get("npca_enabled", False):
                sta.decision_log = decision_log
                sta.current_episode = episode
            
            stas.append(sta)
        
        # 시뮬레이터 실행
        simulator = Simulator(num_slots=num_slots_per_episode, channels=channels, stas=stas)
        simulator.memory = learner.memory
        simulator.device = device
        simulator.run()
        
        # 에피소드별 총 보상 수집 - new_episode_reward 사용
        total_reward = 0
        for sta in stas:
            if sta.npca_enabled:
                total_reward += sta.new_episode_reward
            sta.new_episode_reward = 0.0  # Reset for next episode

        episode_rewards.append(total_reward)

        # 처리량: 보상 파라미터와 무관한 ch1 STA의 성공 TX 슬롯 수
        throughput = sum(sta.channel_occupancy_time for sta in stas if sta.channel_id == 1)
        episode_throughputs.append(throughput)

        # per-episode NPCA switch ratio and avg option duration
        ep_decisions = [d for d in decision_log if d.get("episode") == episode]
        if ep_decisions:
            ep_npca = sum(1 for d in ep_decisions if d.get("action") == 1) / len(ep_decisions)
            tau_vals = [d["tau"] for d in ep_decisions if "tau" in d]
            ep_avg_tau = sum(tau_vals) / len(tau_vals) if tau_vals else 0.0
        else:
            ep_npca = 0.0
            ep_avg_tau = 0.0
        episode_npca_ratios.append(ep_npca)
        episode_avg_option_durations.append(ep_avg_tau)

        # 학습 수행 - 빈도 증가로 학습 속도 개선
        if len(learner.memory) >= BATCH_SIZE:
            # 에피소드당 3번 학습으로 학습 속도 향상
            for _ in range(3):
                loss = learner.optimize_model()
                if loss is not None:
                    episode_losses.append(loss)

            # Target network를 매 에피소드마다 업데이트 (빈도 증가)
            learner.update_target_network()

        # per-episode average loss
        losses_this_ep = episode_losses[losses_before:]
        episode_avg_losses.append(
            sum(losses_this_ep) / len(losses_this_ep) if losses_this_ep else 0.0
        )
        
        # LLM reward designer 업데이트 (control plane: slow timescale)
        if (llm_designer is not None
                and episode > 0
                and episode % llm_designer.update_interval == 0):
            stats = collect_network_stats(
                episode_rewards=episode_rewards,
                decision_log=decision_log,
                simulator=simulator,
                num_slots_per_episode=num_slots_per_episode,
                current_epsilon=epsilon,
                current_episode=episode,
                num_episodes=num_episodes,
                window=llm_designer.update_interval,
            )
            new_params = llm_designer.design_reward_params(stats)
            for sta in stas:
                if sta.npca_enabled:
                    sta.update_reward_params(
                        throughput_weight=new_params["throughput_weight"],
                        latency_penalty=new_params["latency_penalty"],
                        npca_switch_bonus=new_params["npca_switch_bonus"],
                    )
            llm_entry = {
                "episode": episode,
                **new_params,
                **{f"stat_{k}": v for k, v in stats.items()},
            }
            llm_log.append(llm_entry)
            print(f"  [LLM] ep={episode}: tw={new_params['throughput_weight']:.3f}, "
                  f"lp={new_params['latency_penalty']:.4f}, "
                  f"sb={new_params['npca_switch_bonus']:.3f}, "
                  f"qos={new_params['qos_priority']} | {new_params['reasoning']}")

        # 진행 상황 출력
        if episode % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            avg_tp = sum(episode_throughputs[-10:]) / min(10, len(episode_throughputs))
            avg_loss = sum(episode_avg_losses[-10:]) / max(1, min(10, len(episode_avg_losses)))
            print(f"Episode {episode:3d}: Avg Reward = {avg_reward:6.2f}, "
                  f"Throughput = {avg_tp:6.1f} slots, "
                  f"Loss = {avg_loss:.4f}, Epsilon = {epsilon:.3f}")
    
    print("Training completed!")

    # CSV 파일로 결정 로그 저장
    if decision_log:
        decision_df = pd.DataFrame(decision_log)
        os.makedirs("./semi_mdp_results", exist_ok=True)
        csv_path = "./semi_mdp_results/decision_log.csv"
        decision_df.to_csv(csv_path, index=False)
        print(f"Decision log saved to {csv_path}")
        print(f"Total decision points logged: {len(decision_log)}")

    learner.llm_log = llm_log
    learner.episode_throughputs = episode_throughputs
    learner.episode_epsilons = episode_epsilons
    learner.episode_npca_ratios = episode_npca_ratios
    learner.episode_avg_losses = episode_avg_losses
    learner.episode_avg_option_durations = episode_avg_option_durations
    return episode_rewards, episode_losses, learner

def train_semi_mdp_with_env(num_episodes=500, num_slots_per_episode=3000, device=None, random_env=True):
    """
    NPCASemiMDPEnv 환경을 직접 사용하는 Semi-MDP 학습 함수
    
    Args:
        num_episodes: 학습 에피소드 수
        num_slots_per_episode: 에피소드당 슬롯 수
        device: torch device
        random_env: 랜덤 환경 사용 여부
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Semi-MDP 환경 생성
    from npca_semi_mdp_env import NPCASemiMDPEnv
    
    env = NPCASemiMDPEnv(
        num_stas=2,
        num_slots=num_slots_per_episode,
        obss_generation_rate=0.05,  # Base rate (will be randomized if random_env=True)
        npca_enabled=True,
        throughput_weight=10.0,
        latency_penalty_weight=0.1,
        random_env=random_env
    )
    
    # 기존 4차원 관찰 공간 사용 (호환성 유지)
    obs_dim = 4  # [obss_remaining, current_slot, tx_duration, cw_index]
    n_actions = env.action_space.n
    
    # SemiMDPLearner 초기화
    learner = SemiMDPLearner(obs_dim, n_actions, device)
    
    episode_rewards = []
    episode_losses = []
    
    print(f"Starting Semi-MDP Environment training on {device}")
    print(f"Episodes: {num_episodes}, Slots per episode: {num_slots_per_episode}")
    print(f"Random environment: {random_env}")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        # 기존 4차원 관찰 벡터로 변환 (호환성 유지)
        obs_vector = dict_to_legacy_vector(obs)
        obs_tensor = torch.tensor(obs_vector, dtype=torch.float32, device=device).unsqueeze(0)
        
        episode_reward = 0.0
        done = False
        step_count = 0
        max_steps = 1000  # Prevent infinite episodes
        
        while not done and step_count < max_steps:
            # Action selection
            action = learner.select_action(obs_tensor)
            learner.steps_done += 1
            
            # Environment step
            try:
                next_obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                if not done and not truncated:
                    # 기존 4차원 관찰 벡터로 변환 (호환성 유지)
                    next_obs_vector = dict_to_legacy_vector(next_obs)
                    next_obs_tensor = torch.tensor(next_obs_vector, dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    next_obs_tensor = None
                
                # Store transition
                duration = info.get('duration', 1)  # Semi-MDP duration
                learner.memory.push(
                    obs_tensor.squeeze(0),
                    action,
                    next_obs_tensor.squeeze(0) if next_obs_tensor is not None else None,
                    reward,
                    duration,
                    done or truncated
                )
                
                obs_tensor = next_obs_tensor
                step_count += 1
                
            except Exception as e:
                if "step() called when not at decision point" in str(e):
                    # No decision point found, end episode
                    done = True
                else:
                    raise e
        
        episode_rewards.append(episode_reward)
        
        # Training
        if len(learner.memory) >= BATCH_SIZE:
            for _ in range(3):  # Multiple training steps per episode
                loss = learner.optimize_model()
                if loss is not None:
                    episode_losses.append(loss)
            
            learner.update_target_network()
        
        # Progress logging
        if episode % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            avg_loss = sum(episode_losses[-10:]) / max(1, len(episode_losses[-10:]))
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * learner.steps_done / EPS_DECAY)
            print(f"Episode {episode:3d}: Avg Reward = {avg_reward:6.2f}, "
                  f"Avg Loss = {avg_loss:.4f}, Epsilon = {epsilon:.3f}, "
                  f"Memory Size = {len(learner.memory)}")
    
    print("Training completed!")
    
    return episode_rewards, episode_losses, learner

def dict_to_legacy_vector(obs_dict):
    """Semi-MDP Dict 관찰을 기존 4차원 벡터로 변환 (호환성 유지)"""
    return [
        float(obs_dict.get('obss_remaining', 0)),     # OBSS 남은 시간
        float(obs_dict.get('current_slot', 1)),       # 현재 슬롯 (radio transition time 대용)
        33.0,  # tx_duration (고정값, PPDU_DURATION)
        float(obs_dict.get('cw_index', 0))            # CW 인덱스
    ]

def flatten_dict_observation(obs_dict):
    """Dict 관찰을 flat vector로 변환"""
    result = []
    
    # Scalar values
    for key in ['current_slot', 'backoff_counter', 'cw_index', 'obss_remaining',
                'channel_busy_intra', 'channel_busy_obss', 'npca_channel_busy',
                'current_obss_duration', 'current_ppdu_duration']:
        result.append(float(obs_dict.get(key, 0)))
    
    # Array values
    for key in ['primary_busy_history', 'obss_busy_history', 'npca_busy_history']:
        if key in obs_dict:
            result.extend(obs_dict[key].flatten())
        else:
            result.extend([0.0] * 10)  # Default history length
    
    # Single-element arrays
    for key in ['obss_frequency', 'avg_obss_duration']:
        if key in obs_dict:
            result.append(float(obs_dict[key][0]))
        else:
            result.append(0.0)
    
    return result
