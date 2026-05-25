"""
NPCA-HARQ rule-based action selection policy.

Guidelines §10  Action space definition
Guidelines §12  select_action() decision logic
Guidelines §13  Adaptive CW_npca_init (placeholder — Step 5)

Step 4 adds an explicit action-selection layer consulted whenever a STA
detects OBSS on the primary channel and considers switching to NPCA.

Decision principle (guidelines §12):
  Compare estimated primary access delay vs estimated NPCA access delay.
  Choose whichever channel offers faster access, then pick the TX type
  (HARQ_RETX if buffer is valid, else ARQ_RETX or fresh TX).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from harq_sim.configs import (
    DEFAULT_NPCA_QSRC, NPCA_QSRC_MIN, NPCA_QSRC_MAX,
    NPCA_TRANSITION_THRESHOLD, URGENT_DEADLINE_THRESHOLD,
)
from harq_sim.enums import Action, NPCA_ACTIONS
from harq_sim.sta import CW_MIN

if TYPE_CHECKING:
    from harq_sim.sta import STA

# Re-export for convenient import by tests / run scripts
__all__ = [
    "NPCAHARQPolicy",
    "estimate_primary_access_delay",
    "estimate_npca_access_delay",
    "select_npca_qsrc",
    "NPCA_ACTIONS",
]


# ─────────────────────────────────────────────────────────────────────────────
# Delay estimators  (guidelines §12)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_primary_access_delay(sta: STA) -> int:
    """Estimated slots until STA can TX on primary channel.

    = remaining OBSS duration + remaining backoff counter.

    After OBSS clears the STA needs ``primary_backoff_counter`` more idle
    slots before it can transmit.  This is a lower-bound estimate (ignores
    possible new OBSS events or intra-BSS collisions after OBSS ends).
    """
    return sta.primary_channel.obss_remain + sta.primary_backoff_counter


def estimate_npca_access_delay(sta: STA) -> int:
    """Estimated slots until STA can TX on NPCA channel.

    = switching_delay + expected_backoff_slots

    Expected backoff ≈ npca_cw_init // 2  (uniform distribution mean).
    Does not account for NPCA channel busy periods (hidden channel problem).
    """
    npca_cw_init     = sta._compute_npca_cw_init()
    expected_backoff = npca_cw_init // 2
    return sta.switching_delay + expected_backoff


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive CW_npca_init selector  (guidelines §13, Step 5)
# ─────────────────────────────────────────────────────────────────────────────

def select_npca_qsrc(sta: STA, slot: int, default_qsrc: int = DEFAULT_NPCA_QSRC) -> int:
    """Compute adaptive initial NPCA QSRC (guidelines §13.3).

    Increases qsrc (wider CW) when congestion is high, decreases it when the
    packet deadline is urgent.  Result is clamped to [NPCA_QSRC_MIN, NPCA_QSRC_MAX].

    Rules (all additive before clamping):
      +1  primary_cw ≥ 4 × CW_MIN       → primary contention is high, NPCA may be crowded
      +1  num_recent_npca_transitions > NPCA_TRANSITION_THRESHOLD  → NPCA is congested
      +1  npca_failure_rate > 0.3        → recent NPCA TX failures are high
      −1  deadline_remaining < URGENT_DEADLINE_THRESHOLD            → be aggressive
    """
    q = default_qsrc

    if sta.primary_cw >= 4 * CW_MIN:
        q += 1

    if sta.num_recent_npca_transitions > NPCA_TRANSITION_THRESHOLD:
        q += 1

    if sta.npca_failure_rate > 0.3:
        q += 1

    pkt = sta.current_packet or sta._peek_head(slot)
    if pkt is not None:
        dr = pkt.deadline_remaining(slot)
        if dr is not None and dr < URGENT_DEADLINE_THRESHOLD:
            q -= 1

    return max(NPCA_QSRC_MIN, min(q, NPCA_QSRC_MAX))


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based policy  (guidelines §12)
# ─────────────────────────────────────────────────────────────────────────────

class NPCAHARQPolicy:
    """Rule-based NPCA switching and HARQ retransmission policy.

    Implements the ``select_action()`` logic from guidelines §12/§13.

    Called by STA when primary is FROZEN (or BACKOFF interrupted by OBSS)
    and ``can_transition_to_npca()`` might be True.  The policy decides:
      1. Whether NPCA is worth using (delay comparison).
      2. Which TX type to use (HARQ_RETX > ARQ_RETX > TX_NEW).
      3. (Step 5, adaptive_cw=True) Which qsrc to use for NPCA CW initialisation.

    Integration with STA:
      - Attach via ``STA(... policy=NPCAHARQPolicy())``
      - Or pass ``policy=None`` for the Step 3 default (always go NPCA).
      - For Step 5: ``NPCAHARQPolicy(adaptive_cw=True)``
    """

    def __init__(self, adaptive_cw: bool = False) -> None:
        self.adaptive_cw = adaptive_cw

    def select_action(self, sta: STA, slot: int) -> Action:
        """Return the recommended action for this decision point.

        Parameters
        ----------
        sta  : STA whose state is used for the decision.
        slot : current simulation slot (for HARQ validity check).

        Returns
        -------
        Action
            One of the Action enum values.  NPCA_ACTIONS → transition to NPCA.
            All others → stay on primary channel.
        """
        pkt = sta.current_packet or sta._peek_head(slot)
        if pkt is None:
            return Action.STAY_PRIMARY

        can_npca   = sta.can_transition_to_npca(slot)
        harq_valid = sta._is_harq_retx_applicable(pkt, slot)
        has_retry  = pkt.retry_count > 0

        if not can_npca:
            # NPCA transition conditions not satisfied — commit to primary
            if harq_valid:
                return Action.HARQ_RETX_PRIMARY
            elif has_retry:
                return Action.ARQ_RETX_PRIMARY
            else:
                return Action.TX_NEW_PRIMARY

        # NPCA is available — compare estimated access delays (guidelines §12)
        primary_delay = estimate_primary_access_delay(sta)
        npca_delay    = estimate_npca_access_delay(sta)

        if npca_delay < primary_delay:
            # NPCA channel offers faster access → switch
            # Step 5: if adaptive CW enabled, update sta.npca_initial_qsrc before transition
            if self.adaptive_cw:
                sta.npca_initial_qsrc = select_npca_qsrc(sta, slot, sta.npca_initial_qsrc)
            if harq_valid:
                return Action.HARQ_RETX_NPCA
            elif has_retry:
                return Action.ARQ_RETX_NPCA
            else:
                return Action.TX_NEW_NPCA
        else:
            # Primary will clear soon enough → wait
            if harq_valid:
                return Action.HARQ_RETX_PRIMARY
            elif has_retry:
                return Action.ARQ_RETX_PRIMARY
            else:
                return Action.TX_NEW_PRIMARY
