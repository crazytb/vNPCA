"""
STA state machine for HARQ-NPCA simulation — Step 4 (NPCA-HARQ policy).

D1.2 §37.18.3  Switching to the NPCA channel
  → can_transition_to_npca(): condition 1 implemented
  → _start_npca_transition(): save primary state, init NPCA state, start switching delay

D1.2 §37.18.4  NPCA transmission rules
  → pt 3:  use same EDCA param set on NPCA channel  (modelled via npca_initial_qsrc)
  → pt 4a: NPCA_TIMER = NPCA_PPDU_REM_DUR − switch_back_delay
  → pt 4b: return to primary within aSlotTime of NPCA_TIMER expiry

Guidelines §5  Primary/NPCA EDCA state separation
  → _save_primary_state() / _restore_primary_state()
  → _init_npca_state() with fresh CW from npca_initial_qsrc

Guidelines §9  HARQ-CC Chase Combining
  → harq_buffer: HARQBuffer — stores accumulated SNR from failed PHY attempts
  → _is_harq_retx_applicable(): buffer active + valid + same packet_id
  → HARQ_RETX uses harq_buffer.original_mcs (MCS constraint §9.4)
  → effective_snr_db = accumulated + current attempt SNR (§9.3)
  → Buffer flushed on delivery or drop; collision does NOT store soft info (§9.2)

Guidelines §10 / §12  Action selection policy (Step 4+)
  → policy: NPCAHARQPolicy — select_action() called on OBSS detection
  → Compares primary_delay vs npca_delay to decide which channel is faster
  → None (default) → backward-compatible: always go NPCA when conditions met

Guidelines §18  Backoff update after success/failure
  → reset_backoff_after_success() / increase_backoff_after_failure()
  → NPCA CW increase does NOT propagate to primary CW

Guidelines §7  AP absence failure
  → handle_tx_result() uses FailureReason.AP_ABSENCE_DUE_TO_NPCA
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from harq_sim.channel import Channel
from harq_sim.configs import (
    NPCA_FAILURE_WINDOW,
    ENERGY_TX_PER_SLOT_UJ, ENERGY_LISTEN_PER_SLOT_UJ, ENERGY_NPCA_TRANSITION_UJ,
)
from harq_sim.enums import (
    Action, ChannelType, FailureReason, NPCA_ACTIONS, NPCA_MODES, TX_MODES,
    PacketStatus, STAMode, TxType,
)
from harq_sim.harq_buffer import HARQBuffer
from harq_sim.packet import Packet, TransmissionAttempt
from harq_sim import phy

# Imported lazily to avoid circular import at module level
# Policy is only used at runtime via the `policy` parameter.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from harq_sim.policy import NPCAHARQPolicy

# ──────────────────────────────────────────────────────────────────────────────
# 802.11 AC_BE EDCA defaults (used for both primary and NPCA channels)
# D1.2 §37.18.4 pt 3: same EDCA parameter set on NPCA as on BSS primary
# ──────────────────────────────────────────────────────────────────────────────
CW_MIN: int = 15
CW_MAX: int = 1023


@dataclass
class TxRequest:
    """Pending transmission request — consumed by Simulator each slot."""
    sta_id:       int
    channel_type: ChannelType
    duration:     int
    packet:       Optional[Packet]
    tx_type:      TxType
    mcs:          int   = 0
    snr_db:       float = 0.0   # SNR sampled at TX start — logged in CSV


class STA:
    # ──────────────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        sta_id:                    int,
        primary_channel:           Channel,
        npca_channel:              Optional[Channel] = None,
        npca_enabled:              bool = False,
        ppdu_duration:             int = 33,       # slots
        switching_delay:           int = 1,        # radio transition (slots) — D1.2 NPCA switching delay
        switch_back_delay:         int = 1,        # D1.2 NPCA switch back delay (slots)
        npca_min_duration_threshold: int = 0,      # D1.2 §37.18.3.1.c.i minimum duration threshold
        npca_initial_qsrc:         int = 0,        # initial NPCA CW exponent (research variable)
        retry_limit:               int = 7,
        ap_on_primary:             bool = True,    # False when AP is on NPCA (single-radio model)
        infinite_queue:            bool = True,    # auto-generate new packets when queue empties
        snr_db_mean:               float = 25.0,  # mean link SNR to AP (dB) — Step 2+
        snr_db_std:                float = 0.0,   # SNR std dev; 0 = deterministic
        harq_enabled:              bool = False,   # Step 3+: HARQ-CC Chase Combining
        harq_validity_horizon:     int  = 200,     # buffer lifetime in slots (~1.8 ms coherence time)
        policy:                    Optional["NPCAHARQPolicy"] = None,  # Step 4+: action selection policy
        adaptive_cw:               bool = False,   # Step 5+: adaptive CW_npca_init via select_npca_qsrc
    ):
        self.sta_id                     = sta_id
        self.primary_channel            = primary_channel
        self.npca_channel               = npca_channel
        self.npca_enabled               = npca_enabled
        self.ppdu_duration              = ppdu_duration
        self.switching_delay            = switching_delay
        self.switch_back_delay          = switch_back_delay
        self.npca_min_duration_threshold = npca_min_duration_threshold
        self.npca_initial_qsrc          = npca_initial_qsrc
        self.retry_limit                = retry_limit
        self.ap_on_primary              = ap_on_primary
        self.infinite_queue             = infinite_queue
        self.snr_db_mean                = snr_db_mean
        self.snr_db_std                 = snr_db_std
        self.harq_enabled               = harq_enabled
        self.harq_validity_horizon      = harq_validity_horizon
        self.policy: Optional["NPCAHARQPolicy"] = policy  # Step 4+: rule-based or RL policy
        self.adaptive_cw                = adaptive_cw      # Step 5+: adaptive qsrc selection

        # ── Primary EDCA state (guidelines §5.1) ──────────────────────────────
        self.primary_cw:              int = CW_MIN
        self.primary_backoff_counter: int = random.randint(0, CW_MIN)
        self.primary_backoff_stage:   int = 0
        self.primary_retry_counter:   int = 0

        # ── NPCA EDCA state (guidelines §5.2) ─────────────────────────────────
        self.npca_cw:              int = CW_MIN
        self.npca_backoff_counter: int = 0
        self.npca_backoff_stage:   int = 0
        self.npca_retry_counter:   int = 0

        # ── Saved primary state for restore after NPCA (guidelines §5.3/5.4) ──
        self.saved_primary_state: Optional[dict] = None

        # ── intra-BSS NAV (D1.2 §37.18.3.1.e) — simplified: always 0 ────────
        self.intra_bss_nav: int = 0

        # ── Mode state machine ────────────────────────────────────────────────
        self.mode:      STAMode = STAMode.PRIMARY_BACKOFF
        self.next_mode: STAMode = self.mode

        # ── Radio switching countdown ─────────────────────────────────────────
        self.switching_remain: int = 0

        # ── NPCA_TIMER (D1.2 §37.18.4 pt 4a) ────────────────────────────────
        # Set at switch time to: obss_remain − switch_back_delay
        # Decremented each slot while in any NPCA mode.
        self.npca_timer: int = 0

        # ── PHY state (Step 2+) ───────────────────────────────────────────────
        # SNR sampled at TX start; carried to TX end for PHY success evaluation.
        self._current_tx_snr_db: float = 0.0
        # Set in handle_tx_result(False, PHY_ERROR) so logger can capture it.
        self._phy_failure_tx: Optional[dict] = None

        # ── HARQ buffer (Step 3+) ─────────────────────────────────────────────
        # Per-STA Chase Combining buffer — stores accumulated SNR from failed
        # PHY attempts.  Always allocated; only active when harq_enabled=True.
        self.harq_buffer: HARQBuffer = HARQBuffer(validity_horizon=harq_validity_horizon)

        # ── TX state ──────────────────────────────────────────────────────────
        self.tx_remaining:   int             = 0
        self.tx_request:     Optional[TxRequest] = None
        self.current_packet: Optional[Packet]    = None

        # ── Packet queue ──────────────────────────────────────────────────────
        self.packet_queue: Deque[Packet] = deque()
        self._pkt_arrival_counter: int   = 0

        # ── Episode statistics ────────────────────────────────────────────────
        self.stats: dict = {
            "primary_tx_success":  0,
            "primary_tx_fail":     0,
            "npca_tx_success":     0,
            "npca_tx_fail":        0,
            "npca_transitions":    0,
            "switch_backs":        0,
            "packets_delivered":   0,
            "packets_dropped":     0,
            "ap_absence_failures": 0,
            "phy_error_failures":  0,   # Step 2+: PHY decoding failures
            "harq_tx_success":     0,   # Step 3+: HARQ_RETX attempts that succeeded
            "harq_tx_fail":        0,   # Step 3+: HARQ_RETX attempts that failed
            "policy_npca_chosen":    0, # Step 4+: policy chose NPCA (transition happened)
            "policy_primary_chosen": 0, # Step 4+: policy chose to stay primary (override)
            "primary_collision_count": 0,  # MAC collisions on primary channel
            "npca_collision_count":    0,  # MAC collisions on NPCA channel
        }

        # ── Trace log (optional, filled by simulator) ─────────────────────────
        self.trace: list = []

        # ── TX completion event (read by simulator each slot, then cleared) ──
        # Set inside handle_tx_result(success=True) so the logger can capture it.
        self._completed_tx: Optional[dict] = None

        # ── Policy decision event (Step 4+, cleared each slot) ───────────────
        # Set when policy.select_action() is called; logger reads it for CSV.
        self._last_action: Optional[Action] = None

        # ── Step 5+: NPCA failure rate tracking ───────────────────────────────
        # Sliding window of recent NPCA TX outcomes (True=success, False=fail).
        self._npca_tx_window: deque = deque(maxlen=NPCA_FAILURE_WINDOW)
        # Injected by Simulator each slot: global NPCA transition count in recent window.
        self.num_recent_npca_transitions: int = 0
        # History of qsrc values used at each NPCA transition (for summary stats).
        self._npca_qsrc_history: list = []

        # ── Adaptive qsrc counters (active when adaptive_cw=True) ─────────────
        # Observation window of K visits; update qsrc at window boundary.
        # "visit" = one NPCA entry (one per OBSS trigger). Each visit can have multiple TXs.
        self._adap_K:              int = 5    # window size (visits); small for sparse N
        self._adap_trans:          int = 0    # visits in current window
        self._adap_col:            int = 0    # total collisions across all TXs in window
        self._adap_tx:             int = 0    # total TX attempts in window
        self._adap_visits_with_tx: int = 0    # visits where ≥1 TX was attempted
        self._adap_cur_tx:         int = 0    # TX attempts in the current ongoing visit
        self._theta_col:   float = 0.70  # per-TX collision rate threshold → increase qsrc
        self._theta_waste: float = 0.30  # per-visit waste rate threshold  → decrease qsrc

        # ── Step 6+: energy and delivery-delay tracking ───────────────────────
        self.total_energy_uj:   float = 0.0   # cumulative energy this episode (μJ)
        self._delivered_delays: list  = []    # delivery delay (slots) per delivered packet

    # ──────────────────────────────────────────────────────────────────────────
    # NPCA initial CW   (guidelines §5.3)
    # npca_cw = 2^qsrc × (CW_MIN + 1) − 1
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_npca_cw_init(self, qsrc: Optional[int] = None) -> int:
        q = self.npca_initial_qsrc if qsrc is None else qsrc
        return 2 ** q * (CW_MIN + 1) - 1

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5+: NPCA failure rate  (guidelines §13.2)
    # ──────────────────────────────────────────────────────────────────────────
    @property
    def npca_failure_rate(self) -> float:
        """Fraction of recent NPCA TX attempts that failed (0.0 if no history)."""
        if not self._npca_tx_window:
            return 0.0
        return sum(1 for ok in self._npca_tx_window if not ok) / len(self._npca_tx_window)

    # ──────────────────────────────────────────────────────────────────────────
    # PHY helpers (Step 2+)
    # ──────────────────────────────────────────────────────────────────────────
    def _sample_snr(self) -> float:
        """Sample instantaneous SNR (guidelines §15 channel model)."""
        if self.snr_db_std == 0.0:
            return self.snr_db_mean
        return random.gauss(self.snr_db_mean, self.snr_db_std)

    # ──────────────────────────────────────────────────────────────────────────
    # HARQ helpers (Step 3+)
    # ──────────────────────────────────────────────────────────────────────────
    def _is_harq_retx_applicable(self, pkt: Packet, slot: int) -> bool:
        """True if HARQ Chase Combining should be used for this TX attempt.

        Checks: harq_enabled AND buffer active AND same packet_id AND valid.
        Flushes the buffer if it has expired (falls back to ARQ next attempt).
        """
        if not self.harq_enabled:
            return False
        if not self.harq_buffer.active:
            return False
        if self.harq_buffer.packet_id != pkt.packet_id:
            return False
        if not self.harq_buffer.is_valid(slot):
            self.harq_buffer.flush()   # validity horizon expired → ARQ fallback
            return False
        return True

    def _compute_effective_snr(self, pkt: Optional[Packet]) -> float:
        """Effective SNR for PHY judgment at TX end (guidelines §9.3).

        When a HARQ buffer is active for this packet, combines accumulated SNR
        with the current attempt's SNR.  Otherwise returns raw current SNR.
        """
        if (pkt is not None
                and self.harq_enabled
                and self.harq_buffer.active
                and self.harq_buffer.packet_id == pkt.packet_id):
            new_snr_linear = phy.snr_db_to_linear(self._current_tx_snr_db)
            return self.harq_buffer.effective_snr_db(new_snr_linear)
        return self._current_tx_snr_db

    # ──────────────────────────────────────────────────────────────────────────
    # NPCA transition condition  (D1.2 §37.18.3 Condition 1)
    # ──────────────────────────────────────────────────────────────────────────
    def can_transition_to_npca(self, slot: int) -> bool:
        if not self.npca_enabled or self.npca_channel is None:
            return False
        # b. Primary channel busy due to inter-BSS PPDU
        if not self.primary_channel.is_busy_by_obss(slot):
            return False
        # c.i. NPCA_PPDU_REM_DUR ≥ NPCA Minimum Duration Threshold
        if self.primary_channel.obss_remain < self.npca_min_duration_threshold:
            return False
        # d. NPCA channel does not overlap with OBSS PPDU channel (always true)
        if self.npca_channel.overlaps_with_obss_ppdu:
            return False
        # e. Intra-BSS NAV = 0
        if self.intra_bss_nav > 0:
            return False
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Primary state save / restore  (guidelines §5.3 / §5.4)
    # ──────────────────────────────────────────────────────────────────────────
    def _save_primary_state(self) -> None:
        self.saved_primary_state = {
            "cw":              self.primary_cw,
            "backoff_counter": self.primary_backoff_counter,
            "backoff_stage":   self.primary_backoff_stage,
            "retry_counter":   self.primary_retry_counter,
        }

    def _restore_primary_state(self) -> None:
        if self.saved_primary_state is not None:
            self.primary_cw              = self.saved_primary_state["cw"]
            self.primary_backoff_counter = self.saved_primary_state["backoff_counter"]
            self.primary_backoff_stage   = self.saved_primary_state["backoff_stage"]
            self.primary_retry_counter   = self.saved_primary_state["retry_counter"]
            self.saved_primary_state     = None

    # ──────────────────────────────────────────────────────────────────────────
    # NPCA EDCA state initialization  (guidelines §5.3)
    # D1.2 §37.18.4 pt 3: same EDCA param set — here modelled as fresh CW from qsrc
    # ──────────────────────────────────────────────────────────────────────────
    def _init_npca_state(self, qsrc: Optional[int] = None) -> None:
        self.npca_cw              = self._compute_npca_cw_init(qsrc)
        self.npca_backoff_counter = random.randint(0, self.npca_cw)
        self.npca_backoff_stage   = 0
        self.npca_retry_counter   = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Backoff management after success / failure  (guidelines §18)
    # Key invariant: NPCA CW changes do NOT affect primary CW.
    # ──────────────────────────────────────────────────────────────────────────
    def reset_backoff_after_success(self, channel_type: ChannelType) -> None:
        if channel_type == ChannelType.PRIMARY:
            self.primary_backoff_stage   = 0
            self.primary_cw              = CW_MIN
            self.primary_backoff_counter = random.randint(0, self.primary_cw)
        else:  # NPCA
            self.npca_backoff_stage   = 0
            self.npca_cw              = self._compute_npca_cw_init()
            self.npca_backoff_counter = random.randint(0, self.npca_cw)

    def increase_backoff_after_failure(self, channel_type: ChannelType) -> None:
        if channel_type == ChannelType.PRIMARY:
            self.primary_backoff_stage   += 1
            self.primary_cw              = min(2 * (self.primary_cw + 1) - 1, CW_MAX)
            self.primary_backoff_counter = random.randint(0, self.primary_cw)
            self.primary_retry_counter   += 1
        else:  # NPCA — increase NPCA CW only, primary CW unchanged
            self.npca_backoff_stage   += 1
            self.npca_cw              = min(2 * (self.npca_cw + 1) - 1, CW_MAX)
            self.npca_backoff_counter = random.randint(0, self.npca_cw)
            self.npca_retry_counter   += 1

    # ──────────────────────────────────────────────────────────────────────────
    # NPCA switch-back decision
    # D1.2 §37.18.4 pt 4b: return within aSlotTime of NPCA_TIMER expiry
    # ──────────────────────────────────────────────────────────────────────────
    def _should_switch_back(self) -> bool:
        return (
            self.npca_timer <= 0
            or self.primary_channel.obss_remain <= self.switch_back_delay
        )

    # ──────────────────────────────────────────────────────────────────────────
    # NPCA transition helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _decide_npca_or_stay(self, slot: int) -> bool:
        """Consult policy (if set) to decide whether to transition to NPCA.

        Called when ``can_transition_to_npca()`` is already True.

        With policy (Step 4+):
          Calls ``policy.select_action()`` → records ``_last_action`` and stats.
          If the action is in NPCA_ACTIONS → transition; else stay frozen.

        Without policy (Step 3 backward compat):
          Always transitions (original eager-NPCA behavior).

        Returns True if a NPCA transition was started this slot, else False.
        """
        if self.policy is None:
            # Step 3 behavior: always go NPCA when conditions are met
            self._start_npca_transition(slot)
            return True

        action = self.policy.select_action(self, slot)
        self._last_action = action

        if action in NPCA_ACTIONS:
            self.stats["policy_npca_chosen"] += 1
            self._start_npca_transition(slot)
            return True
        else:
            # Policy chose to stay on primary (OBSS will clear soon enough)
            self.stats["policy_primary_chosen"] += 1
            self.next_mode = STAMode.PRIMARY_FROZEN
            return False

    def _start_npca_transition(self, slot: int) -> None:
        """Save primary state and start NPCA switching delay."""
        self._save_primary_state()
        self._init_npca_state()
        # D1.2 §37.18.4 pt 4a: NPCA_TIMER = NPCA_PPDU_REM_DUR − switch_back_delay
        self.npca_timer       = max(0, self.primary_channel.obss_remain - self.switch_back_delay)
        self.switching_remain = self.switching_delay
        self.next_mode        = STAMode.NPCA_SWITCHING
        self.stats["npca_transitions"] += 1
        # Step 5+: record qsrc used at this transition (for avg_npca_qsrc stats)
        self._npca_qsrc_history.append(self.npca_initial_qsrc)
        # Adaptive qsrc: start a new visit
        if self.adaptive_cw:
            self._adap_trans  += 1
            self._adap_cur_tx  = 0
        # Step 6+: radio switching energy event
        self.total_energy_uj += ENERGY_NPCA_TRANSITION_UJ

    def _start_switch_back(self) -> None:
        """Begin radio switching back to BSS primary channel."""
        self.switching_remain = self.switching_delay
        self.next_mode        = STAMode.SWITCH_BACK
        self.stats["switch_backs"] += 1
        # Adaptive qsrc: record whether this visit had any TX, then check for update
        if self.adaptive_cw:
            if self._adap_cur_tx > 0:
                self._adap_visits_with_tx += 1
            self._maybe_update_qsrc()

    # ──────────────────────────────────────────────────────────────────────────
    # Packet queue helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _peek_head(self, slot: int = 0) -> Optional[Packet]:
        if self.infinite_queue and not self.packet_queue:
            self._pkt_arrival_counter += 1
            self.packet_queue.append(Packet(
                arrival_time=slot,
            ))
        return self.packet_queue[0] if self.packet_queue else None

    def _dequeue_current(self) -> None:
        if self.current_packet and self.packet_queue:
            if self.packet_queue[0] is self.current_packet:
                self.packet_queue.popleft()
        self.current_packet = None

    # ──────────────────────────────────────────────────────────────────────────
    # Main step  (called once per slot by Simulator)
    # ──────────────────────────────────────────────────────────────────────────
    def step(self, slot: int) -> None:
        self.tx_request      = None
        self._completed_tx   = None   # clear last slot's completion event
        self._phy_failure_tx = None   # clear PHY-failure event
        self._last_action    = None   # clear policy decision from previous slot

        # Step 6+: per-slot energy based on mode at start of this slot
        if self.mode in TX_MODES:
            self.total_energy_uj += ENERGY_TX_PER_SLOT_UJ
        else:
            self.total_energy_uj += ENERGY_LISTEN_PER_SLOT_UJ

        # Decrement NPCA_TIMER every slot while in any NPCA mode
        if self.mode in NPCA_MODES and self.npca_timer > 0:
            self.npca_timer -= 1

        if   self.mode == STAMode.PRIMARY_BACKOFF: self._handle_primary_backoff(slot)
        elif self.mode == STAMode.PRIMARY_FROZEN:  self._handle_primary_frozen(slot)
        elif self.mode == STAMode.PRIMARY_TX:      self._handle_primary_tx(slot)
        elif self.mode == STAMode.NPCA_SWITCHING:  self._handle_npca_switching(slot)
        elif self.mode == STAMode.NPCA_BACKOFF:    self._handle_npca_backoff(slot)
        elif self.mode == STAMode.NPCA_FROZEN:     self._handle_npca_frozen(slot)
        elif self.mode == STAMode.NPCA_TX:         self._handle_npca_tx(slot)
        elif self.mode == STAMode.SWITCH_BACK:     self._handle_switch_back(slot)

    # ──────────────────────────────────────────────────────────────────────────
    # Per-mode handlers
    # ──────────────────────────────────────────────────────────────────────────
    def _handle_primary_backoff(self, slot: int) -> None:
        # 1. Intra-BSS busy → freeze (no NPCA opportunity — NAV set implicitly)
        if self.primary_channel.is_busy_by_intra_bss(slot):
            self.next_mode = STAMode.PRIMARY_FROZEN
            return

        # 2. OBSS detected → consider NPCA transition (D1.2 §37.18.3 Condition 1)
        if self.primary_channel.is_busy_by_obss(slot):
            if self.can_transition_to_npca(slot):
                self._decide_npca_or_stay(slot)   # Step 4+: policy-driven
            else:
                self.next_mode = STAMode.PRIMARY_FROZEN
            return

        # 3. Channel idle: count down backoff or transmit
        if self.primary_backoff_counter == 0:
            pkt = self._peek_head(slot)
            if pkt is not None:
                self.current_packet = pkt
                snr = self._sample_snr()
                self._current_tx_snr_db = snr
                # HARQ_RETX: use original_mcs constraint (§9.4); else fresh MCS
                if self._is_harq_retx_applicable(pkt, slot):
                    mcs     = self.harq_buffer.original_mcs
                    tx_type = TxType.HARQ_RETX
                else:
                    mcs     = phy.select_mcs(snr)
                    tx_type = TxType.NEW if pkt.retry_count == 0 else TxType.ARQ_RETX
                pkt.current_mcs = mcs
                self.tx_request = TxRequest(
                    sta_id=self.sta_id,
                    channel_type=ChannelType.PRIMARY,
                    duration=self.ppdu_duration,
                    packet=pkt,
                    tx_type=tx_type,
                    mcs=mcs,
                    snr_db=snr,
                )
                self.tx_remaining = self.ppdu_duration
                self.next_mode    = STAMode.PRIMARY_TX
        else:
            self.primary_backoff_counter -= 1

    def _handle_primary_frozen(self, slot: int) -> None:
        # OBSS present → check NPCA opportunity (policy may choose to stay)
        if self.primary_channel.is_busy_by_obss(slot):
            if self.can_transition_to_npca(slot):
                transitioned = self._decide_npca_or_stay(slot)  # Step 4+
                if transitioned:
                    return

        if not self.primary_channel.is_busy(slot):
            self.next_mode = STAMode.PRIMARY_BACKOFF

    def _handle_primary_tx(self, slot: int) -> None:
        self.tx_remaining -= 1
        if self.tx_remaining == 0:
            # TX complete — apply PHY model; collision was already dispatched by simulator
            pkt = self.current_packet
            mcs = pkt.current_mcs if pkt else 0
            # HARQ-CC: effective SNR = accumulated + current attempt (§9.3)
            eff_snr_db = self._compute_effective_snr(pkt)
            if phy.attempt_success(eff_snr_db, mcs):
                self.handle_tx_result(
                    True, FailureReason.NONE, ChannelType.PRIMARY, slot,
                    snr_db=self._current_tx_snr_db, effective_snr_db=eff_snr_db,
                )
            else:
                self.handle_tx_result(
                    False, FailureReason.PHY_ERROR, ChannelType.PRIMARY, slot,
                    snr_db=self._current_tx_snr_db, effective_snr_db=eff_snr_db,
                )

    def _handle_npca_switching(self, slot: int) -> None:
        """Count down radio switching delay before entering NPCA backoff."""
        self.switching_remain -= 1
        if self.switching_remain == 0:
            # D1.2 §37.18.4 pt 1: STA shall be ready to TX no later than switching delay
            if self.npca_channel.is_busy(slot):
                self.next_mode = STAMode.NPCA_FROZEN
            else:
                self.next_mode = STAMode.NPCA_BACKOFF

    def _handle_npca_backoff(self, slot: int) -> None:
        # Check NPCA_TIMER / primary OBSS expiry → switch back
        if self._should_switch_back():
            self._start_switch_back()
            return

        if self.npca_channel.is_busy(slot):
            self.next_mode = STAMode.NPCA_FROZEN
            return

        if self.npca_backoff_counter == 0:
            pkt = self.current_packet or self._peek_head(slot)
            if pkt is not None:
                self.current_packet = pkt
                snr = self._sample_snr()
                self._current_tx_snr_db = snr
                # HARQ_RETX: use original_mcs constraint (§9.4); else fresh MCS
                if self._is_harq_retx_applicable(pkt, slot):
                    mcs     = self.harq_buffer.original_mcs
                    tx_type = TxType.HARQ_RETX
                else:
                    mcs     = phy.select_mcs(snr)
                    tx_type = TxType.NEW if pkt.retry_count == 0 else TxType.ARQ_RETX
                pkt.current_mcs = mcs
                # TX duration bounded by remaining OBSS time (D1.2 §37.18.4 pt 4a)
                tx_dur = min(self.ppdu_duration, self.primary_channel.obss_remain)
                self.tx_request = TxRequest(
                    sta_id=self.sta_id,
                    channel_type=ChannelType.NPCA,
                    duration=tx_dur,
                    packet=pkt,
                    tx_type=tx_type,
                    mcs=mcs,
                    snr_db=snr,
                )
                self.tx_remaining = tx_dur
                self.next_mode    = STAMode.NPCA_TX
        else:
            self.npca_backoff_counter -= 1

    def _handle_npca_frozen(self, slot: int) -> None:
        # Primary OBSS ended or NPCA_TIMER expired → switch back
        if self._should_switch_back():
            self._start_switch_back()
            return

        if not self.npca_channel.is_busy(slot):
            self.next_mode = STAMode.NPCA_BACKOFF

    def _handle_npca_tx(self, slot: int) -> None:
        self.tx_remaining -= 1
        if self.tx_remaining == 0:
            pkt = self.current_packet
            mcs = pkt.current_mcs if pkt else 0
            eff_snr_db = self._compute_effective_snr(pkt)
            if phy.attempt_success(eff_snr_db, mcs):
                self.handle_tx_result(
                    True, FailureReason.NONE, ChannelType.NPCA, slot,
                    snr_db=self._current_tx_snr_db, effective_snr_db=eff_snr_db,
                )
            else:
                self.handle_tx_result(
                    False, FailureReason.PHY_ERROR, ChannelType.NPCA, slot,
                    snr_db=self._current_tx_snr_db, effective_snr_db=eff_snr_db,
                )

    def _handle_switch_back(self, slot: int) -> None:
        """Count down radio switching delay back to primary channel."""
        self.switching_remain -= 1
        if self.switching_remain == 0:
            # D1.2 §37.18.4: restore primary EDCA state (guidelines §5.4)
            self._restore_primary_state()
            self.next_mode = STAMode.PRIMARY_BACKOFF

    # ──────────────────────────────────────────────────────────────────────────
    # TX result handler  (called by Simulator after collision resolution)
    # Guidelines §17
    # ──────────────────────────────────────────────────────────────────────────
    def handle_tx_result(
        self,
        success: bool,
        failure_reason: FailureReason,
        channel_type: ChannelType,
        slot: int,
        snr_db: float = 0.0,           # sampled SNR for this attempt (Step 2+)
        effective_snr_db: float = 0.0, # combined SNR after HARQ-CC (Step 3+)
    ) -> None:
        pkt = self.current_packet
        if pkt is None:
            return

        # Capture tx_type / retry_count BEFORE any mutations (for logging)
        _is_harq = (
            self.harq_enabled
            and self.harq_buffer.active
            and self.harq_buffer.packet_id == pkt.packet_id
        )
        _tx_type_at_attempt = (
            TxType.HARQ_RETX if _is_harq
            else (TxType.NEW if pkt.retry_count == 0 else TxType.ARQ_RETX)
        )
        _retry_at_attempt        = pkt.retry_count
        _harq_count_at_attempt   = self.harq_buffer.combining_count if _is_harq else 0
        # When harq_enabled but buffer not yet active, effective_snr_db == snr_db (raw)
        _eff_snr = effective_snr_db if effective_snr_db != 0.0 else snr_db

        attempt = TransmissionAttempt(
            packet_id=pkt.packet_id,
            sta_id=self.sta_id,
            channel_type=channel_type,
            tx_type=_tx_type_at_attempt,
            mcs=pkt.current_mcs,
            start_time=slot - self.ppdu_duration + 1,
            duration=self.ppdu_duration,
            success=success,
            failure_reason=failure_reason,
            collision=(failure_reason == FailureReason.COLLISION),
            snr_db=snr_db,
            effective_snr_db=_eff_snr,
            harq_combining_count=_harq_count_at_attempt,
        )
        pkt.transmission_history.append(attempt)

        # Step 5+: record NPCA TX outcome in sliding window for failure rate tracking
        if channel_type == ChannelType.NPCA:
            self._npca_tx_window.append(success)

        if success:
            # Capture completion event for CSV logger BEFORE flushing buffer/dequeueing
            self._completed_tx = {
                "channel_type":        channel_type,
                "tx_type":             _tx_type_at_attempt,
                "packet_id":           pkt.packet_id,
                "retry_count":         _retry_at_attempt,
                "snr_db":              snr_db,
                "effective_snr_db":    _eff_snr,
                "harq_combining_count": _harq_count_at_attempt,
            }
            # Step 6+: record delivery delay (slots from packet creation to delivery)
            self._delivered_delays.append(slot - pkt.arrival_time)
            if _is_harq:
                self.stats["harq_tx_success"] += 1
            if self.harq_enabled:
                self.harq_buffer.flush()
            pkt.status = PacketStatus.DELIVERED
            self._dequeue_current()
            self.reset_backoff_after_success(channel_type)
            if channel_type == ChannelType.PRIMARY:
                self.stats["primary_tx_success"] += 1
                self.next_mode = STAMode.PRIMARY_BACKOFF
            else:
                self.stats["npca_tx_success"] += 1
                if self.adaptive_cw:
                    self._adap_tx    += 1
                    self._adap_cur_tx += 1
                if self._should_switch_back():
                    self._start_switch_back()
                else:
                    self.next_mode = STAMode.NPCA_BACKOFF
            self.stats["packets_delivered"] += 1
            return

        # ── Failure path ──────────────────────────────────────────────────────
        if _is_harq:
            self.stats["harq_tx_fail"] += 1
        pkt.retry_count += 1
        if channel_type == ChannelType.PRIMARY:
            self.stats["primary_tx_fail"] += 1
        else:
            self.stats["npca_tx_fail"] += 1
            if self.adaptive_cw:
                self._adap_tx    += 1
                self._adap_cur_tx += 1

        if failure_reason == FailureReason.COLLISION:
            if channel_type == ChannelType.PRIMARY:
                self.stats["primary_collision_count"] += 1
            else:
                self.stats["npca_collision_count"] += 1
                if self.adaptive_cw:
                    self._adap_col += 1

        if failure_reason == FailureReason.AP_ABSENCE_DUE_TO_NPCA:
            self.stats["ap_absence_failures"] += 1

        # PHY decoding failure — record for CSV logger (STA self-reported at TX end)
        if failure_reason == FailureReason.PHY_ERROR:
            self._phy_failure_tx = {
                "channel_type":        channel_type,
                "tx_type":             _tx_type_at_attempt,
                "packet_id":           pkt.packet_id,
                "retry_count":         _retry_at_attempt,
                "snr_db":              snr_db,
                "effective_snr_db":    _eff_snr,
                "harq_combining_count": _harq_count_at_attempt,
            }
            self.stats["phy_error_failures"] += 1
            # Store soft information in HARQ buffer (§9.2: only for PHY_ERROR)
            if self.harq_enabled:
                self.harq_buffer.store(pkt, phy.snr_db_to_linear(snr_db), slot)
                pkt.harq_count += 1

        # Drop if retry limit exceeded or deadline passed
        if pkt.retry_count > self.retry_limit:
            pkt.status = PacketStatus.DROPPED
            if self.harq_enabled:
                self.harq_buffer.flush()
            self._dequeue_current()
            self.stats["packets_dropped"] += 1
            self._post_drop_backoff(channel_type)
            return

        if pkt.is_deadline_expired(slot):
            pkt.status = PacketStatus.DROPPED
            if self.harq_enabled:
                self.harq_buffer.flush()
            self._dequeue_current()
            self.stats["packets_dropped"] += 1
            self._post_drop_backoff(channel_type)
            return

        # Retry: increase CW (NPCA CW ↑ does NOT affect primary CW)
        self.increase_backoff_after_failure(channel_type)

        # After failed NPCA TX: check switch-back
        if channel_type == ChannelType.NPCA:
            if self._should_switch_back():
                self._start_switch_back()
            else:
                self.next_mode = STAMode.NPCA_BACKOFF
        else:
            self.next_mode = STAMode.PRIMARY_BACKOFF

    def _post_drop_backoff(self, channel_type: ChannelType) -> None:
        """Reset backoff after packet drop so STA can contend again."""
        self.reset_backoff_after_success(channel_type)
        if channel_type == ChannelType.NPCA:
            if self._should_switch_back():
                self._start_switch_back()
            else:
                self.next_mode = STAMode.NPCA_BACKOFF
        else:
            self.next_mode = STAMode.PRIMARY_BACKOFF

    # ──────────────────────────────────────────────────────────────────────────
    # Commit next_mode at end of each slot (called by Simulator)
    # ──────────────────────────────────────────────────────────────────────────
    # Adaptive qsrc update  (active when adaptive_cw=True)
    # Called at each switch_back. Updates npca_initial_qsrc every K transitions.
    # ──────────────────────────────────────────────────────────────────────────
    def _maybe_update_qsrc(self) -> None:
        if self._adap_trans < self._adap_K:
            return
        # col_rate: P(collision | TX attempted) — one visit can have multiple TX attempts
        col_rate   = self._adap_col / self._adap_tx if self._adap_tx > 0 else 0.0
        # waste_rate: P(no TX in visit) — backoff exceeded NPCA timer before first attempt
        waste_rate = 1.0 - self._adap_visits_with_tx / self._adap_trans
        if col_rate > self._theta_col:
            self.npca_initial_qsrc = min(self.npca_initial_qsrc + 1, 5)
        elif waste_rate > self._theta_waste:
            self.npca_initial_qsrc = max(self.npca_initial_qsrc - 1, 0)
        self._adap_trans          = 0
        self._adap_col            = 0
        self._adap_tx             = 0
        self._adap_visits_with_tx = 0

    # ──────────────────────────────────────────────────────────────────────────
    def commit_mode(self) -> None:
        self.mode = self.next_mode

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    def reset_episode_stats(self) -> None:
        for k in self.stats:
            self.stats[k] = 0
        self.trace.clear()
        self._npca_tx_window.clear()
        self._npca_qsrc_history.clear()
        self.num_recent_npca_transitions = 0
        # Step 6+
        self.total_energy_uj = 0.0
        self._delivered_delays.clear()
        # Adaptive qsrc: reset observation window counters (qsrc itself persists)
        self._adap_trans          = 0
        self._adap_col            = 0
        self._adap_tx             = 0
        self._adap_visits_with_tx = 0
        self._adap_cur_tx         = 0

    def __repr__(self) -> str:
        return (
            f"STA(id={self.sta_id}, mode={self.mode.name}, "
            f"p_cw={self.primary_cw}, n_cw={self.npca_cw}, "
            f"npca_timer={self.npca_timer})"
        )
