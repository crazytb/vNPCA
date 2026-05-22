import pandas as pd
import torch
import random
import math
from typing import List

# Copy-pasting the given code components into the environment to enable simulation
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple
from drl_framework.params import EPS_START, EPS_END, EPS_DECAY
from drl_framework.configs import ENERGY_TX_PER_SLOT_UJ, ENERGY_LISTEN_PER_SLOT_UJ, ENERGY_NPCA_TRANSITION_UJ
from collections import defaultdict

# Fix random seed for reproducibility
random.seed(42)

# Constants
CONTENTION_WINDOW = [2 ** (i + 4) - 1 for i in range(7)]  # CW from 15 to 1023
SLOTTIME = 9  # μs

class STAState(Enum):
    PRIMARY_BACKOFF = auto()
    PRIMARY_FROZEN = auto()
    PRIMARY_TX = auto()
    NPCA_BACKOFF = auto()
    NPCA_FROZEN = auto()
    NPCA_TX = auto()

# Pre-computed state sets for energy model (avoid set construction per slot)
_LISTEN_STATES = frozenset({STAState.PRIMARY_BACKOFF, STAState.PRIMARY_FROZEN,
                             STAState.NPCA_BACKOFF, STAState.NPCA_FROZEN})
_TX_STATES = frozenset({STAState.PRIMARY_TX, STAState.NPCA_TX})

@dataclass
class OccupyRequest:
    channel_id: int
    duration: int
    is_obss: bool = False

class Channel:
    def __init__(self, channel_id: int, obss_generation_rate: float = 0.0, obss_duration_range: Tuple[int, int] = (20, 40),
                 obss_duration_sampler=None):
        self.channel_id = channel_id
        self.obss_generation_rate = obss_generation_rate
        self.obss_duration_range = obss_duration_range
        self.obss_duration_sampler = obss_duration_sampler  # callable() -> int; overrides uniform randint if set

        self.intra_occupied = False
        self.intra_end_slot = 0

        self.obss_traffic: List[Tuple[str, int, int, int]] = []
        
        self.occupied_remain = 0
        self.obss_remain = 0

    def occupy(self, slot: int, duration: int, sta_id: int):
        self.intra_occupied = True
        self.intra_end_slot = slot + duration
        self.occupied_remain = duration

    def add_obss_traffic(self, req: OccupyRequest, slot: int):
        obss_tuple = (
            f"obss_gen_{self.channel_id}_slot{slot}",
            slot,
            req.duration,
            req.source_bss if hasattr(req, "source_bss") else -1
        )
        self.obss_traffic.append(obss_tuple)

    def is_busy_by_intra_bss(self, slot: int) -> bool:
        return self.occupied_remain > 0

    def is_busy_by_obss(self, slot: int) -> bool:
        return self.obss_remain > 0

    def is_busy(self, slot: int) -> bool:
        return (self.occupied_remain > 0) or (self.obss_remain > 0)

    def update(self, slot: int):
        if self.intra_occupied and self.intra_end_slot <= slot:
            self.intra_occupied = False

        self.obss_traffic = [t for t in self.obss_traffic if t[1] + t[2] > slot]

        self.occupied_remain = max(0, self.intra_end_slot - slot) if self.intra_occupied else 0

        active_obss = [start + dur - slot for _, start, dur, _ in self.obss_traffic if start <= slot < start + dur]
        self.obss_remain = max(active_obss) if active_obss else 0

    def generate_obss(self, slot: int):
        if self.obss_generation_rate == 0:
            return

        if not self.is_busy(slot):
            if random.random() < self.obss_generation_rate:
                duration = (self.obss_duration_sampler()
                            if self.obss_duration_sampler is not None
                            else random.randint(*self.obss_duration_range))
                obss_tuple = (
                    f"obss_gen_{self.channel_id}_slot{slot}",
                    slot,
                    duration,
                    -1
                )
                self.obss_traffic.append(obss_tuple)
                
    def get_latest_obss(self, slot: int) -> Optional[Tuple[str, int, int, int]]:
        active = [
            obss for obss in self.obss_traffic
            if obss[1] <= slot < obss[1] + obss[2]
        ]
        if not active:
            return None
        return max(active, key=lambda x: x[1])

class STA:
    def __init__(self,
                 sta_id: int,
                 channel_id: int,
                 primary_channel: Channel,
                 npca_channel: Optional[Channel] = None,
                 npca_enabled: bool = False,
                 radio_transition_time: int = 1,
                 ppdu_duration: int = 10,
                 random_ppdu: bool = False,
                 learner=None,
                 num_slots_per_episode: int = 1000,
                 throughput_weight: float = 1.0,
                 latency_penalty: float = 0.05,
                 npca_switch_bonus: float = 0.0,
                 npca_switch_cost: float = 0.0):
        self.sta_id = sta_id
        self.channel_id = channel_id
        self.primary_channel = primary_channel
        self.npca_channel = npca_channel
        self.npca_enabled = npca_enabled
        self.radio_transition_time = radio_transition_time
        self.random_ppdu = random_ppdu
        self.learner = learner
        self.num_slots_per_episode = num_slots_per_episode
        self.throughput_weight = throughput_weight
        self.latency_penalty = latency_penalty
        self.npca_switch_bonus = npca_switch_bonus
        self.npca_switch_cost = npca_switch_cost
        # Normalized reward params as context vector [tw, lp, sb, sc] ∈ [0,1]⁴
        # Inserted into DQN state so the agent conditions on current reward structure.
        # Default = fixed_drl baseline: tw=1.0→0.2, lp=0.05→0.21, sb=0.0, sc=0.0
        self.context_vec: list[float] = [0.2, 0.21, 0.0, 0.0]

        self.occupy_request: Optional[OccupyRequest] = None
        self.state = STAState.PRIMARY_BACKOFF
        self.next_state = self.state
        self.cw_index = 0
        self.backoff = self.generate_backoff() + 1
        self.tx_remaining = 0
        self.ppdu_duration = ppdu_duration
        self.current_obss = None
        self.intent = None
        self.current_tx_duration = 0

        self._opt_active = False
        self._opt_s = None
        self._opt_a = None
        self._opt_R = 0.0
        self._opt_tau = 0
        self._pending = None
        
        self.new_episode_reward = 0.0
        self._initial_occupancy_time = 0.0

        self.channel_occupancy_time = 0
        self.total_episode_slots = 0
        self.episode_energy_uJ: float = 0.0  # cumulative energy this episode (μJ)

    def generate_backoff(self) -> int:
        cw = CONTENTION_WINDOW[self.cw_index]
        return random.randint(0, cw)
    
    def handle_collision(self):
        self.cw_index = min(self.cw_index + 1, len(CONTENTION_WINDOW) - 1)
        self.backoff = self.generate_backoff()
        self.tx_remaining = 0
        self.next_state = STAState.PRIMARY_BACKOFF

    def handle_success(self):
        self.cw_index = 0
        self.backoff = self.generate_backoff()
        self.next_state = STAState.PRIMARY_BACKOFF
        if self.random_ppdu:
            self.ppdu_duration = random.randint(10, 200)
    
    def decide_action(self, slot):
        self.intent = None
        if self.state == STAState.PRIMARY_BACKOFF and self.backoff == 0:
            self.intent = "primary_tx"
        return self.intent

    def get_tx_duration(self, is_npca=False) -> int:
        if is_npca:
            return min(self.primary_channel.obss_remain, self.ppdu_duration)
        return self.ppdu_duration
    
    def get_obs(self):
        return {
            "primary_channel_obss_occupied_remained": self.primary_channel.obss_remain,
            "radio_transition_time": self.radio_transition_time,
            "tx_duration": self.get_tx_duration(),
            "cw_index": self.cw_index,
            # context: normalized reward params — agent conditions on current reward structure
            "ctx_tw": self.context_vec[0],
            "ctx_lp": self.context_vec[1],
            "ctx_sb": self.context_vec[2],
            "ctx_sc": self.context_vec[3],
        }

    def obs_to_vec(self, obs: dict, normalize: bool = False, caps=None):
        FEATURE_ORDER = (
            "primary_channel_obss_occupied_remained",
            "radio_transition_time",
            "tx_duration",
            "cw_index",
            "ctx_tw", "ctx_lp", "ctx_sb", "ctx_sc",
        )
        x = [float(obs[k]) for k in FEATURE_ORDER]
        if not normalize:
            return x
        caps = caps or {"slots": 1024, "cw_stage_max": 8}
        x[0] = min(x[0], caps["slots"]) / caps["slots"]
        x[1] = min(x[1], caps["slots"]) / caps["slots"]
        x[2] = min(x[2], caps["slots"]) / caps["slots"]
        x[3] = min(x[3], caps["cw_stage_max"]) / caps["cw_stage_max"]
        # x[4..7]: context_vec already normalized to [0,1]
        return x

    def step(self, slot: int):
        if self._opt_active:
            self._opt_tau += 1

        if self.state in _LISTEN_STATES:
            self.episode_energy_uJ += ENERGY_LISTEN_PER_SLOT_UJ
        elif self.state in _TX_STATES:
            self.episode_energy_uJ += ENERGY_TX_PER_SLOT_UJ

        if self.state == STAState.PRIMARY_BACKOFF:
            self._handle_primary_backoff(slot)
        elif self.state == STAState.PRIMARY_FROZEN:
            self._handle_primary_frozen(slot)
        elif self.state == STAState.PRIMARY_TX:
            self._handle_primary_tx(slot)
        elif self.state == STAState.NPCA_BACKOFF:
            self._handle_npca_backoff(slot)
        elif self.state == STAState.NPCA_FROZEN:
            self._handle_npca_frozen(slot)
        elif self.state == STAState.NPCA_TX:
            self._handle_npca_tx(slot)

    def _handle_primary_backoff(self, slot: int):
        if self.primary_channel.is_busy_by_intra_bss(slot):
            self.next_state = STAState.PRIMARY_FROZEN
        elif self.primary_channel.is_busy_by_obss(slot):
            if self.npca_enabled and self.npca_channel and (self.learner or hasattr(self, '_fixed_action')):
                if not self._opt_active:
                    obs_dict = self.get_obs()
                    obs_vec = self.obs_to_vec(obs_dict, normalize=True)

                    if self.learner:
                        self._finalize_pending_with_next_state(
                            next_obs_vec=obs_vec,
                            memory=self.learner.memory,
                            done=False,
                            device=self.learner.device
                        )

                    if hasattr(self, '_fixed_action'):
                        action = self._fixed_action()
                    else:
                        state_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=self.learner.device).unsqueeze(0)
                        action = self.learner.select_action(state_tensor)
                        self.learner.steps_done += 1

                    if hasattr(self, 'decision_log'):
                        log_entry = {}
                        if hasattr(self, '_fixed_action'):
                            log_entry = {
                                'episode': getattr(self, 'current_episode', -1),
                                'slot': slot,
                                'sta_id': self.sta_id,
                                'primary_channel_obss_occupied_remained': obs_dict.get('primary_channel_obss_occupied_remained', 0),
                                'radio_transition_time': obs_dict.get('radio_transition_time', 0),
                                'tx_duration': obs_dict.get('tx_duration', 0),
                                'cw_index': obs_dict.get('cw_index', 0),
                                'action': int(action),
                                'strategy': 'fixed'
                            }
                        elif self.learner:
                            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.learner.steps_done / EPS_DECAY)
                            log_entry = {
                                'episode': getattr(self, 'current_episode', -1),
                                'slot': slot,
                                'sta_id': self.sta_id,
                                'primary_channel_obss_occupied_remained': obs_dict.get('primary_channel_obss_occupied_remained', 0),
                                'radio_transition_time': obs_dict.get('radio_transition_time', 0),
                                'tx_duration': obs_dict.get('tx_duration', 0),
                                'cw_index': obs_dict.get('cw_index', 0),
                                'action': int(action),
                                'epsilon': epsilon,
                                'steps_done': self.learner.steps_done
                            }
                        self.decision_log.append(log_entry)
                    
                    self._begin_option(obs_dict, int(action))

                    self.current_obss = self.primary_channel.get_latest_obss(slot)
                    
                    if action == 0:
                        self.backoff = self.generate_backoff()
                        self.next_state = STAState.PRIMARY_FROZEN
                    else:
                        self.cw_index = 0
                        self.backoff = self.generate_backoff()
                        if self.npca_channel.is_busy_by_intra_bss(slot):
                            self.next_state = STAState.NPCA_FROZEN
                        else:
                            self.next_state = STAState.NPCA_BACKOFF
                else:
                    if self._opt_a == 0:
                        self.next_state = STAState.PRIMARY_FROZEN
                    else:
                        if self.npca_channel.is_busy_by_intra_bss(slot):
                            self.next_state = STAState.NPCA_FROZEN
                        else:
                            self.next_state = STAState.NPCA_BACKOFF
            else:
                self.next_state = STAState.PRIMARY_FROZEN
        else:
            if (self.backoff == 0) and not self.primary_channel.is_busy(slot):
                self.current_tx_duration = self.get_tx_duration()
                self.tx_remaining = self.current_tx_duration
                self.occupy_request = OccupyRequest(
                    channel_id=self.primary_channel.channel_id, 
                    duration=self.tx_remaining, 
                    is_obss=False)
                self.next_state = STAState.PRIMARY_TX
            else:
                self.backoff -= 1 if self.backoff > 0 else 0
    
    def _handle_primary_frozen(self, slot: int):
        if not self.primary_channel.is_busy(slot):
            self.next_state = STAState.PRIMARY_BACKOFF

    def _handle_primary_tx(self, slot: int):
        if self.primary_channel.is_busy_by_obss(slot):
            self.tx_success = False
        else:
            self.tx_success = True

        if self.tx_remaining > 0:
            self.tx_remaining -= 1

        if self.tx_remaining == 0:
            self._end_option()  # Move before success/collision handling
            
            if self.tx_success:
                self.channel_occupancy_time += self.current_tx_duration
                self.handle_success()
            else:
                self.handle_collision()

    def _handle_npca_backoff(self, slot: int):
        if self.npca_channel.is_busy(slot):
            self.next_state = STAState.NPCA_FROZEN
        else:
            if (self.backoff == 0) and not self.npca_channel.is_busy(slot):
                self.current_tx_duration = self.get_tx_duration(is_npca=True)
                self.tx_remaining = self.current_tx_duration
                self.occupy_request = OccupyRequest(
                    channel_id=self.npca_channel.channel_id,
                    duration=self.tx_remaining,
                    is_obss=True
                )
                self.next_state = STAState.NPCA_TX
            else:
                self.backoff -= 1 if self.backoff > 0 else 0

    def _handle_npca_frozen(self, slot: int):
        if self.primary_channel.obss_remain == 0:
            self.cw_index = 0
            self.backoff = self.generate_backoff()
            self.next_state = STAState.PRIMARY_BACKOFF

        if not self.npca_channel.is_busy(slot):
            self.next_state = STAState.NPCA_BACKOFF

    def _handle_npca_tx(self, slot: int):
        if self.tx_remaining > 0:
            self.tx_remaining -= 1
            return

        if self.tx_remaining == 0:
            self._end_option()  # Move before success/collision handling
            
            if self.tx_success:
                self.channel_occupancy_time += self.current_tx_duration
                self.handle_success()
            else:
                self.handle_collision()
            
            self.current_obss = None
            self.next_state = STAState.PRIMARY_BACKOFF
            return
        
    def _begin_option(self, s_dict, a_int):
        assert not self._opt_active, "Option already active"
        self._opt_active = True
        self._opt_s = s_dict
        self._opt_a = int(a_int)
        self._opt_R = 0.0
        self._opt_tau = 0
        self._initial_occupancy_time = self.channel_occupancy_time
        if int(a_int) == 1:
            self.episode_energy_uJ += ENERGY_NPCA_TRANSITION_UJ

    def update_reward_params(self, throughput_weight: float, latency_penalty: float,
                             npca_switch_bonus: float = 0.0, npca_switch_cost: float = 0.0,
                             context_vec: list[float] | None = None):
        """Update reward params and context vector (called by AP-side LLM policy manager)."""
        self.throughput_weight = throughput_weight
        self.latency_penalty = latency_penalty
        self.npca_switch_bonus = npca_switch_bonus
        self.npca_switch_cost = npca_switch_cost
        if context_vec is not None:
            self.context_vec = context_vec

    def _end_option(self):
        if self._opt_active:
            # Reward based on actual successful transmission, regardless of action
            if hasattr(self, 'tx_success') and self.tx_success:
                attempted_transmission_slots = self.current_tx_duration
            else:
                attempted_transmission_slots = 0

            throughput_reward = self.throughput_weight * attempted_transmission_slots
            latency_penalty = self.latency_penalty * self._opt_tau
            switch_bonus = self.npca_switch_bonus if self._opt_a == 1 else 0.0
            switch_cost = self.npca_switch_cost if self._opt_a == 1 else 0.0
            cumulative_reward = throughput_reward - latency_penalty + switch_bonus - switch_cost
            self.new_episode_reward += cumulative_reward

            if hasattr(self, 'decision_log') and self.decision_log:
                for i in range(len(self.decision_log) - 1, -1, -1):
                    if (self.decision_log[i]['sta_id'] == self.sta_id and 
                        'reward' not in self.decision_log[i]):
                        self.decision_log[i]['reward'] = cumulative_reward
                        self.decision_log[i]['tau'] = self._opt_tau
                        break
            
            self._pending = (self._opt_s, self._opt_a, cumulative_reward, self._opt_tau)
            self._opt_active = False
            self._opt_s = None
            self._opt_a = None
            self._opt_R = 0.0
            self._opt_tau = 0

    def _finalize_pending_with_next_state(self, next_obs_vec, memory, done: bool, normalize: bool = True, device=None):
        if self._pending is None:
            return
        s_dict, a, R, tau = self._pending
        s_vec  = self.obs_to_vec(s_dict, normalize=normalize)
        s_vec  = torch.tensor(s_vec, dtype=torch.float32, device=device)
        s_next = torch.tensor(next_obs_vec, dtype=torch.float32, device=device)
        
        normalized_R = R / self.num_slots_per_episode
        if memory is not None:
            memory.push(s_vec, a, s_next, normalized_R, tau, done)
        self._pending = None

class Simulator:
    def __init__(self, num_slots: int, stas: List['STA'], channels: List['Channel']):
        self.num_slots = num_slots
        self.stas = stas
        self.channels = channels
        self.log = []

    def run(self):
        for slot in range(self.num_slots):
            for ch in self.channels:
                ch.update(slot)

            for sta in self.stas:
                sta.occupy_request = None
                sta.step(slot)

            obss_reqs = []
            for ch in self.channels:
                obss_req = ch.generate_obss(slot)
                if obss_req:
                    obss_reqs.append((None, obss_req))

            sta_reqs = [(sta, sta.occupy_request) for sta in self.stas if sta.occupy_request is not None]

            all_reqs = sta_reqs + obss_reqs

            channel_requests = defaultdict(list)
            for sta, req in all_reqs:
                channel_requests[req.channel_id].append((sta, req))

            for ch_id, reqs in channel_requests.items():
                if len(reqs) == 1:
                    sta, req = reqs[0]
                    if req.is_obss:
                        self.channels[ch_id].add_obss_traffic(req, slot)
                    else:
                        self.channels[ch_id].occupy(slot, req.duration, sta.sta_id)
                    if sta:
                        sta.tx_success = True
                else:
                    for sta, req in reqs:
                        if sta is not None:
                            if req.is_obss:
                                self.channels[ch_id].add_obss_traffic(req, slot)
                            else:
                                self.channels[ch_id].occupy(slot, req.duration, sta.sta_id)
                            sta.tx_success = False

            for sta in self.stas:
                sta.state = sta.next_state

            self.log_slot(slot)

        for sta in self.stas:
            if sta._opt_active:
                sta._end_option()
            if sta._pending:
                final_obs = sta.get_obs()
                final_obs_vec = sta.obs_to_vec(final_obs, normalize=True)
                sta._finalize_pending_with_next_state(
                    next_obs_vec=final_obs_vec,
                    memory=self.memory,
                    done=True,
                    device=self.device
                )

    def log_slot(self, slot: int):
        row = {
            "slot": slot,
            "time": slot * SLOTTIME,
        }

        for ch_id, ch in enumerate(self.channels):
            stas_in_ch = [sta for sta in self.stas if sta.channel_id == ch_id]

            row[f"states_ch_{ch_id}"] = [sta.state.name.lower() for sta in stas_in_ch]
            row[f"backoff_ch_{ch_id}"] = [sta.backoff for sta in stas_in_ch]
            row[f"npca_enabled_ch_{ch_id}"] = [sta.npca_enabled for sta in stas_in_ch]

            row[f"channel_{ch_id}_occupied_remained"] = ch.occupied_remain
            row[f"channel_{ch_id}_obss_occupied_remained"] = ch.obss_remain
            
        self.log.append(row)

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)

    
