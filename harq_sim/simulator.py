"""
Slot-based HARQ-NPCA simulator.

TX 흐름 설계:
  - 충돌(Collision) / AP-absence → simulator가 즉시 handle_tx_result(False) 호출
  - 충돌 없는 정상 TX 시작    → simulator가 occupy_intra() 만 호출
                               STA는 PRIMARY_TX / NPCA_TX 모드로 진입
  - TX 완료(tx_remaining==0) → STA가 self-report handle_tx_result(True) 호출

이 구조로 multi-slot PPDU 동안 채널 occupation이 slot 전반에 걸쳐 유지됨.

Event loop (guidelines §16.1):
  1. update_channels()     — expire old OBSS, refresh obss_remain
  2. generate_obss()       — stochastic OBSS arrival
  3. snapshot_state()      — CSV용 슬롯 시작 상태 기록
  4. sta.step()            — each STA advances its state machine
  5. collect_tx_requests() — 이번 슬롯에 TX를 시작하는 STA 목록
  6. resolve_collisions()  — per-channel: 단독→occupy, 다수→collision
  7. dispatch_failures()   — collision/AP-absence만 handle_tx_result(False)
  8. commit_modes()        — sta.mode = sta.next_mode
  9. log_slot()            — snapshot + 이번 슬롯 TX 결과 합쳐 기록
"""

from __future__ import annotations

import csv
import os
import statistics as _stats
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from harq_sim.channel import Channel
from harq_sim.configs import NPCA_TRANSITION_WINDOW
from harq_sim.enums import ChannelType, FailureReason, STAMode
from harq_sim.sta import STA, TxRequest

SLOT_US = 9.0   # μs per slot


# ─────────────────────────────────────────────────────────────────────────────
# Log entry: one row per (slot, sta)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SlotLog:
    # Time
    slot:        int
    time_us:     float

    # STA identity
    sta_id:      int

    # Current mode (BEFORE commit — i.e., mode at start of step)
    mode:        str

    # Primary EDCA state
    primary_cw:              int
    primary_backoff_counter: int
    primary_backoff_stage:   int
    primary_retry_counter:   int

    # NPCA EDCA state
    npca_cw:              int
    npca_backoff_counter: int
    npca_backoff_stage:   int
    npca_retry_counter:   int

    # NPCA timer
    npca_timer:  int

    # Saved primary state (non-None while STA is in NPCA mode)
    saved_primary_cw:              Optional[int]
    saved_primary_backoff_counter: Optional[int]

    # Channel occupancy
    primary_obss_remain: int
    npca_obss_remain:    int

    # TX outcome this slot (None if no TX attempt resolved this slot)
    tx_channel:     Optional[str]    # PRIMARY / NPCA
    tx_type:        Optional[str]    # NEW / ARQ_RETX / HARQ_RETX
    tx_success:     Optional[bool]
    failure_reason: Optional[str]
    packet_id:      Optional[int]
    retry_count:    Optional[int]
    snr_db:         Optional[float]  # SNR at TX attempt (None for collision/AP-absence)

    # HARQ fields (Step 3+) — None when HARQ is not involved in this event
    effective_snr_db:     Optional[float] = None  # combined SNR after Chase Combining
    harq_combining_count: Optional[int]   = None  # number of previous failed attempts combined

    # Policy field (Step 4+) — None when no policy is attached to this STA
    action_taken: Optional[str] = None  # Action.value chosen by policy this slot

    # Mode-change events (for quick filtering in the CSV)
    npca_transition_start: bool = False  # STA just decided to switch to NPCA this slot
    switch_back_start:     bool = False  # STA just decided to return to primary this slot


class Simulator:
    def __init__(
        self,
        num_slots:    int,
        stas:         List[STA],
        channels:     List[Channel],
        enable_trace: bool = True,
    ):
        self.num_slots    = num_slots
        self.stas         = stas
        self.channels     = channels
        self.enable_trace = enable_trace
        self.log: List[SlotLog] = []

        self._ch: Dict[int, Channel] = {ch.channel_id: ch for ch in channels}
        self._primary_ch_id = 0
        self._npca_ch_id    = 1

        # Step 5+: sliding deque of slots in which NPCA transitions occurred (any STA)
        # Used to compute num_recent_npca_transitions for adaptive CW policy.
        _max_deque = NPCA_TRANSITION_WINDOW * max(len(stas), 1) * 4
        self._npca_transition_deque: deque = deque(maxlen=_max_deque)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5+: NPCA transition count in recent window  (guidelines §13.2)
    # ─────────────────────────────────────────────────────────────────────────
    def _count_recent_transitions(self, current_slot: int) -> int:
        """Number of NPCA transition events (any STA) within the last NPCA_TRANSITION_WINDOW slots."""
        cutoff = current_slot - NPCA_TRANSITION_WINDOW
        return sum(1 for s in self._npca_transition_deque if s > cutoff)

    # ─────────────────────────────────────────────────────────────────────────
    # Main run
    # ─────────────────────────────────────────────────────────────────────────
    def run(self) -> None:
        for slot in range(self.num_slots):
            self._step_slot(slot)

    def _step_slot(self, slot: int) -> None:
        # Step 5+: push last slot's transition count into each STA before step()
        # (one-slot lag: this slot's transitions appear in next slot's count)
        recent_n = self._count_recent_transitions(slot)
        for sta in self.stas:
            sta.num_recent_npca_transitions = recent_n

        # 1 & 2: Channel update + OBSS generation
        for ch in self.channels:
            ch.update(slot)
            ch.generate_obss(slot)

        # 3: Snapshot state BEFORE step() (for CSV — reflects "what the STA sees this slot")
        pre_snap: Dict[int, dict] = {}
        if self.enable_trace:
            for sta in self.stas:
                pre_snap[sta.sta_id] = _snap_sta(sta)

        # 4: STA state machines advance
        # Track mode before step to detect transition events
        modes_before = {sta.sta_id: sta.mode for sta in self.stas}
        npca_cw_before = {sta.sta_id: sta.npca_cw for sta in self.stas}

        for sta in self.stas:
            sta.step(slot)

        # 5: Collect TX requests issued this slot
        tx_reqs: List[Tuple[STA, TxRequest]] = [
            (sta, sta.tx_request)
            for sta in self.stas
            if sta.tx_request is not None
        ]

        # 6 & 7: Per-channel collision resolution
        per_channel: Dict[int, List[Tuple[STA, TxRequest]]] = defaultdict(list)
        for sta, req in tx_reqs:
            ch_id = (
                sta.primary_channel.channel_id  # use STA's actual primary (supports native NPCA STAs)
                if req.channel_type == ChannelType.PRIMARY
                else self._npca_ch_id
            )
            per_channel[ch_id].append((sta, req))

        # Results: only FAILURE events (collision / AP-absence)
        # Successful TX → STA self-reports when tx_remaining==0
        failure_results: List[Tuple[STA, FailureReason, ChannelType]] = []

        for ch_id, reqs in per_channel.items():
            ch = self._ch.get(ch_id)
            if ch is None:
                continue

            if len(reqs) == 1:
                sta, req = reqs[0]
                if req.channel_type == ChannelType.PRIMARY and not sta.ap_on_primary:
                    # AP is on NPCA — uplink from primary fails (guidelines §7)
                    failure_results.append((sta, FailureReason.AP_ABSENCE_DUE_TO_NPCA, req.channel_type))
                else:
                    # Success start: occupy channel; STA will self-report when done
                    ch.occupy_intra(slot, req.duration)
            else:
                # Collision: immediate failure for all contestants
                for sta, req in reqs:
                    failure_results.append((sta, FailureReason.COLLISION, req.channel_type))

        # Dispatch failures immediately
        for sta, reason, ch_type in failure_results:
            sta.handle_tx_result(False, reason, ch_type, slot)

        # 8: Commit mode transitions
        for sta in self.stas:
            sta.commit_mode()

        # Step 5+: detect NPCA transitions just committed and record for sliding window
        for sta in self.stas:
            if (sta.mode == STAMode.NPCA_SWITCHING
                    and modes_before[sta.sta_id] != STAMode.NPCA_SWITCHING):
                self._npca_transition_deque.append(slot)

        # 9: Log
        if self.enable_trace:
            # Build per-STA TX result map for this slot (only failures resolved here)
            fail_map = {sta.sta_id: (reason, ch_type) for sta, reason, ch_type in failure_results}
            # For successful TX completions: check if STA just finished (tx_remaining==0 AND was in TX mode)
            # These were self-reported inside sta.step() via _handle_primary_tx / _handle_npca_tx
            self._log_slot(slot, pre_snap, modes_before, fail_map, tx_reqs)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-slot logging
    # ─────────────────────────────────────────────────────────────────────────
    def _log_slot(
        self,
        slot: int,
        pre_snap: Dict[int, dict],
        modes_before: Dict[int, object],
        fail_map: Dict[int, Tuple[FailureReason, ChannelType]],
        tx_reqs: List[Tuple[STA, TxRequest]],
    ) -> None:
        tx_req_map = {sta.sta_id: req for sta, req in tx_reqs}

        for sta in self.stas:
            snap = pre_snap[sta.sta_id]
            req  = tx_req_map.get(sta.sta_id)

            # TX outcome fields
            tx_channel     = None
            tx_type_str    = None
            tx_success     = None
            fail_reason    = None
            pkt_id         = None
            retry_cnt      = None
            snr_val        = None
            eff_snr_val    = None   # Step 3+: effective SNR after HARQ combining
            harq_cnt       = None   # Step 3+: number of combined failed attempts

            if sta.sta_id in fail_map:
                # Collision or AP-absence — immediate failure dispatched by simulator
                reason, ch_type = fail_map[sta.sta_id]
                tx_channel  = ch_type.value
                tx_type_str = req.tx_type.value if req else None
                tx_success  = False
                fail_reason = reason.value
                pkt_id      = req.packet.packet_id if req and req.packet else None
                retry_cnt   = req.packet.retry_count if req and req.packet else None
                snr_val     = None  # no PHY measurement for collision/AP-absence
                eff_snr_val = None
                harq_cnt    = None
            elif sta._completed_tx is not None:
                # TX completion — STA self-reported success (tx_remaining == 0, PHY ok)
                c = sta._completed_tx
                tx_channel  = c["channel_type"].value
                tx_type_str = c["tx_type"].value
                tx_success  = True
                fail_reason = FailureReason.NONE.value
                pkt_id      = c["packet_id"]
                retry_cnt   = c["retry_count"]
                snr_val     = c.get("snr_db")
                eff_snr_val = c.get("effective_snr_db")
                harq_cnt    = c.get("harq_combining_count")
            elif sta._phy_failure_tx is not None:
                # PHY decoding failure — STA self-reported (tx_remaining == 0, PHY fail)
                c = sta._phy_failure_tx
                tx_channel  = c["channel_type"].value
                tx_type_str = c["tx_type"].value
                tx_success  = False
                fail_reason = FailureReason.PHY_ERROR.value
                pkt_id      = c["packet_id"]
                retry_cnt   = c["retry_count"]
                snr_val     = c.get("snr_db")
                eff_snr_val = c.get("effective_snr_db")
                harq_cnt    = c.get("harq_combining_count")
            elif req is not None:
                # TX start — STA entering TX mode (multi-slot, outcome TBD)
                tx_channel  = req.channel_type.value
                tx_type_str = req.tx_type.value
                tx_success  = None
                fail_reason = None
                pkt_id      = req.packet.packet_id if req.packet else None
                retry_cnt   = req.packet.retry_count if req.packet else None
                snr_val     = req.snr_db if req.snr_db != 0.0 else None
                eff_snr_val = None   # effective SNR not yet known (outcome TBD)
                harq_cnt    = None

            # Detect transition events
            from harq_sim.enums import NPCA_MODES
            mode_now    = sta.mode   # post-commit
            mode_pre    = modes_before[sta.sta_id]
            npca_trans  = (mode_now == STAMode.NPCA_SWITCHING and mode_pre != STAMode.NPCA_SWITCHING)
            swbk_start  = (mode_now == STAMode.SWITCH_BACK   and mode_pre != STAMode.SWITCH_BACK)
            action_val  = sta._last_action.value if sta._last_action is not None else None

            saved = snap.get("saved_primary_state")
            self.log.append(SlotLog(
                slot=slot,
                time_us=round(slot * SLOT_US, 1),
                sta_id=sta.sta_id,
                mode=snap["mode"],                      # mode at START of this slot
                primary_cw=snap["primary_cw"],
                primary_backoff_counter=snap["primary_backoff_counter"],
                primary_backoff_stage=snap["primary_backoff_stage"],
                primary_retry_counter=snap["primary_retry_counter"],
                npca_cw=snap["npca_cw"],
                npca_backoff_counter=snap["npca_backoff_counter"],
                npca_backoff_stage=snap["npca_backoff_stage"],
                npca_retry_counter=snap["npca_retry_counter"],
                npca_timer=snap["npca_timer"],
                saved_primary_cw=saved["cw"] if saved else None,
                saved_primary_backoff_counter=saved["backoff_counter"] if saved else None,
                primary_obss_remain=snap["primary_obss_remain"],
                npca_obss_remain=snap["npca_obss_remain"],
                tx_channel=tx_channel,
                tx_type=tx_type_str,
                tx_success=tx_success,
                failure_reason=fail_reason,
                packet_id=pkt_id,
                retry_count=retry_cnt,
                snr_db=snr_val,
                effective_snr_db=eff_snr_val,
                harq_combining_count=harq_cnt,
                action_taken=action_val,
                npca_transition_start=npca_trans,
                switch_back_start=swbk_start,
            ))

    # ─────────────────────────────────────────────────────────────────────────
    # CSV export
    # ─────────────────────────────────────────────────────────────────────────
    def to_csv(self, path: str) -> None:
        """Write per-slot log to CSV. Creates parent directories if needed."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not self.log:
            print("Warning: log is empty (enable_trace=True required).")
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.log[0]).keys()))
            writer.writeheader()
            for row in self.log:
                writer.writerow(asdict(row))
        print(f"CSV saved → {path}  ({len(self.log)} rows)")

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregate metrics  (guidelines §19, §26)
    # ─────────────────────────────────────────────────────────────────────────
    def compute_metrics(self) -> dict:
        metrics: dict = {}
        for sta in self.stas:
            s = sta.stats
            total_tx = (
                s["primary_tx_success"] + s["primary_tx_fail"]
                + s["npca_tx_success"]  + s["npca_tx_fail"]
            )
            delivered  = s["packets_delivered"]
            dropped    = s["packets_dropped"]
            total_pkts = delivered + dropped
            collisions = (
                s.get("primary_collision_count", 0)
                + s.get("npca_collision_count", 0)
            )
            qsrc_hist = sta._npca_qsrc_history
            avg_qsrc  = sum(qsrc_hist) / len(qsrc_hist) if qsrc_hist else None
            metrics[sta.sta_id] = {
                "npca_transitions":    s["npca_transitions"],
                "switch_backs":        s["switch_backs"],
                "primary_tx_success":  s["primary_tx_success"],
                "primary_tx_fail":     s["primary_tx_fail"],
                "npca_tx_success":     s["npca_tx_success"],
                "npca_tx_fail":        s["npca_tx_fail"],
                "ap_absence_failures": s["ap_absence_failures"],
                "phy_error_failures":  s.get("phy_error_failures", 0),
                "harq_tx_success":       s.get("harq_tx_success", 0),
                "harq_tx_fail":          s.get("harq_tx_fail", 0),
                "policy_npca_chosen":    s.get("policy_npca_chosen", 0),
                "policy_primary_chosen": s.get("policy_primary_chosen", 0),
                "avg_npca_qsrc":         avg_qsrc,
                "npca_tx_truncated":     s.get("npca_tx_truncated", 0),
                "packets_delivered":   delivered,
                "packets_dropped":     dropped,
                "pdr":                 delivered / total_pkts if total_pkts else 0.0,
                "collision_count":     collisions,
                "collision_prob":      collisions / total_tx if total_tx else 0.0,
                # Step 6+
                "total_energy_uj":   sta.total_energy_uj,
            }

        # Step 6+: aggregate across all STAs (guidelines §19.1)
        all_delays = []
        for sta in self.stas:
            all_delays.extend(sta._delivered_delays)

        total_delivered = sum(sta.stats["packets_delivered"] for sta in self.stas)
        total_dropped   = sum(sta.stats["packets_dropped"]   for sta in self.stas)
        total_pkts_all  = total_delivered + total_dropped

        total_transitions = sum(sta.stats["npca_transitions"] for sta in self.stas)

        # Channel-split collision counts from STA stats (enable_trace-independent)
        primary_col = sum(sta.stats.get("primary_collision_count", 0) for sta in self.stas)
        npca_col    = sum(sta.stats.get("npca_collision_count",    0) for sta in self.stas)

        primary_tx_total = sum(
            sta.stats["primary_tx_success"] + sta.stats["primary_tx_fail"]
            for sta in self.stas
        )
        npca_tx_total = sum(
            sta.stats["npca_tx_success"] + sta.stats["npca_tx_fail"]
            for sta in self.stas
        )
        total_tx_all = primary_tx_total + npca_tx_total

        # Jain's fairness index over per-STA packets_delivered
        per_sta_tp = [sta.stats["packets_delivered"] for sta in self.stas]
        n = len(per_sta_tp)
        sum_x  = sum(per_sta_tp)
        sum_x2 = sum(xi * xi for xi in per_sta_tp)
        jain = (sum_x ** 2) / (n * sum_x2) if (n > 0 and sum_x2 > 0) else 1.0

        metrics["aggregate"] = {
            "aggregate_throughput":          total_delivered,
            "mean_access_delay":             _stats.mean(all_delays) if all_delays else 0.0,
            "p95_access_delay":              _percentile(all_delays, 95),
            "p99_access_delay":              _percentile(all_delays, 99),
            "packet_delivery_ratio":         total_delivered / total_pkts_all if total_pkts_all else 0.0,
            "packet_loss_probability":       total_dropped / total_pkts_all if total_pkts_all else 0.0,
            "collision_probability":         (primary_col + npca_col) / total_tx_all if total_tx_all else 0.0,
            "collision_probability_primary": primary_col / primary_tx_total if primary_tx_total else 0.0,
            "collision_probability_npca":    npca_col / npca_tx_total if npca_tx_total else 0.0,
            "jain_fairness_index":           jain,
            "legacy_throughput_degradation": 0.0,   # set externally if baseline run available
            "total_energy_uj":               sum(sta.total_energy_uj for sta in self.stas),
            "npca_transition_count":         total_transitions,
            "npca_transition_rate":          total_transitions / self.num_slots if self.num_slots > 0 else 0.0,
        }

        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Helper: p-th percentile of a numeric list  (Step 6+)
# ─────────────────────────────────────────────────────────────────────────────
def _percentile(data: list, pct: float) -> float:
    """Return the p-th percentile of data. Returns 0.0 for empty list."""
    if not data:
        return 0.0
    s   = sorted(data)
    idx = int(len(s) * pct / 100)
    return float(s[min(idx, len(s) - 1)])


# ─────────────────────────────────────────────────────────────────────────────
# Helper: snapshot STA state (called BEFORE step())
# ─────────────────────────────────────────────────────────────────────────────
def _snap_sta(sta: STA) -> dict:
    npca_ch = sta.npca_channel
    return {
        "mode":                      sta.mode.name,
        "primary_cw":                sta.primary_cw,
        "primary_backoff_counter":   sta.primary_backoff_counter,
        "primary_backoff_stage":     sta.primary_backoff_stage,
        "primary_retry_counter":     sta.primary_retry_counter,
        "npca_cw":                   sta.npca_cw,
        "npca_backoff_counter":      sta.npca_backoff_counter,
        "npca_backoff_stage":        sta.npca_backoff_stage,
        "npca_retry_counter":        sta.npca_retry_counter,
        "npca_timer":                sta.npca_timer,
        "saved_primary_state":       sta.saved_primary_state,   # reference (read-only)
        "primary_obss_remain":       sta.primary_channel.obss_remain,
        "npca_obss_remain":          npca_ch.obss_remain if npca_ch else 0,
    }
