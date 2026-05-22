"""
LLM-based Reward Designer for vNPCA DRL

AP 측 제어 평면(NPCA Policy Manager)에서 LLM이 네트워크 통계를 분석하여
DRL 에이전트의 보상 함수 파라미터를 적응적으로 설계한다.

vNPCA 아키텍처 매핑:
  - LLMRewardDesigner  ↔  AP의 NPCA Policy Manager (제어 평면)
  - 설계된 reward params  ↔  beacon/management 프레임으로 전파되는 정책 파라미터
  - STA의 DRL 에이전트  ↔  vNPCA 모듈 (데이터 평면)
"""

import json
import os
import time
import anthropic

DEFAULT_PARAMS = {
    "throughput_weight": 1.0,
    "latency_penalty": 0.05,
    "npca_switch_bonus": 0.0,
    "npca_switch_cost": 0.0,
}

PARAM_BOUNDS = {
    "throughput_weight": (0.5, 3.0),
    "latency_penalty": (0.01, 0.2),
    "npca_switch_bonus": (0.0, 0.5),
    "npca_switch_cost": (0.0, 0.3),
}

QOS_PRIORITIES = ("throughput", "latency", "energy", "balanced")

# Normalization bounds for context vector (same as PARAM_BOUNDS)
_CTX_NORM = {
    "throughput_weight":  (0.5, 3.0),
    "latency_penalty":    (0.01, 0.2),
    "npca_switch_bonus":  (0.0, 0.5),
    "npca_switch_cost":   (0.0, 0.3),
}

_SYSTEM_PROMPT = """You are an expert in IEEE 802.11bn Non-Primary Channel Access (NPCA) optimization acting as an AP-side QoS policy manager.

Your task is to design reward function parameters for a DRL agent that decides whether to stay on the primary channel (action=0) or switch to the NPCA secondary channel (action=1) when OBSS interference is detected.

## Reward Formula
  reward = throughput_weight  * successful_tx_slots
         - latency_penalty    * option_duration
         + npca_switch_bonus  * (1 if action==NPCA else 0)
         - npca_switch_cost   * (1 if action==NPCA else 0)

npca_switch_bonus encourages switching; npca_switch_cost penalizes switching energy overhead.
Net switching incentive = npca_switch_bonus - npca_switch_cost.

## CRITICAL: User Intent comes first
When a user QoS intent is provided, it is your PRIMARY directive. Translate the intent into parameter choices BEFORE considering network statistics. Network stats are secondary — use them only to fine-tune the magnitude of adjustments.

### Intent → Parameter mapping
- Latency-sensitive intent (e.g. "video call", "real-time", "minimize delay"):
  → HIGH latency_penalty (0.1–0.2), LOW npca_switch_bonus (0–0.05), LOW npca_switch_cost (0–0.05), qos_priority="latency"
  → NPCA switching should only happen when OBSS duration clearly justifies the overhead
- Throughput-hungry intent (e.g. "file download", "maximize speed", "high bandwidth"):
  → HIGH throughput_weight (1.5–3.0), HIGH npca_switch_bonus (0.1–0.3), LOW npca_switch_cost (0–0.05), qos_priority="throughput"
  → Aggressively exploit NPCA to increase successful TX slots
- Energy-saving intent (e.g. "battery saving", "low power", "energy efficient"):
  → MODERATE throughput_weight (0.8–1.2), MODERATE latency_penalty (0.05–0.1), HIGH npca_switch_cost (0.1–0.25), LOW npca_switch_bonus (0–0.05), qos_priority="energy"
  → Switch to NPCA only when long OBSS clearly saves more energy than the radio transition cost
- Balanced / no intent:
  → Use network statistics as the primary guide (see below), qos_priority="balanced"

## Network-stat guidelines (when no dominant intent, or for fine-tuning magnitude)
- HIGH obss_occupancy_rate (>0.05) + LOW npca_switch_ratio (<0.3): NPCA underutilized → increase throughput_weight/npca_switch_bonus
- LOW obss_occupancy_rate (<0.02) + HIGH npca_switch_ratio (>0.7): over-switching → decrease throughput_weight/npca_switch_bonus, increase npca_switch_cost
- HIGH avg_option_duration: switching overhead large → increase latency_penalty
- Early training (high epsilon, low progress): be conservative, make smaller adjustments

## Constraints
throughput_weight ∈ [0.5, 3.0], latency_penalty ∈ [0.01, 0.2], npca_switch_bonus ∈ [0.0, 0.5], npca_switch_cost ∈ [0.0, 0.3]
Make incremental adjustments (~10–30% per update).

## Output Format
Respond with ONLY a valid JSON object (no markdown, no extra text):
{"throughput_weight": <float>, "latency_penalty": <float>, "npca_switch_bonus": <float>, "npca_switch_cost": <float>, "qos_priority": "<throughput|latency|energy|balanced>", "reasoning": "<one concise sentence explaining how intent was honored>"}"""


class LLMRewardDesigner:
    """
    LLM-based adaptive reward function designer.

    Operates at the slow timescale (every update_interval episodes) to analyze
    network statistics and update the reward parameters used by STA DRL agents.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        update_interval: int = 50,
        use_mock: bool = False,
    ):
        self.model = model
        self.update_interval = update_interval
        self.use_mock = use_mock
        self.current_params = dict(DEFAULT_PARAMS)
        self.call_count = 0
        self.call_history: list[dict] = []

        self.current_qos_priority: str = "balanced"
        self.user_intent: str = ""

        if not use_mock:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY environment variable not set. "
                    "Use use_mock=True for testing without API."
                )
            self.client = anthropic.Anthropic(api_key=api_key)

    def set_intent(self, intent: str):
        """Set user QoS intent in natural language. Empty string = no intent (stat-driven only)."""
        self.user_intent = intent.strip()

    def build_user_prompt(self, network_stats: dict) -> str:
        window = network_stats.get("window_episodes", self.update_interval)
        intent_section = (
            f"## User QoS Intent\n\"{self.user_intent}\"\n\n"
            f"Honor this intent as your primary directive.\n\n"
            if self.user_intent else
            "## User QoS Intent\n(none — use network statistics as the primary guide)\n\n"
        )
        return (
            intent_section +
            f"## Network Statistics (last {window} episodes)\n"
            f"- OBSS occupancy rate: {network_stats['obss_occupancy_rate']:.4f}\n"
            f"- Average episode reward: {network_stats['avg_episode_reward']:.3f}\n"
            f"- NPCA switch ratio: {network_stats['npca_switch_ratio']:.3f} "
            f"(fraction of OBSS-triggered decisions that chose NPCA)\n"
            f"- Avg successful tx slots/episode: {network_stats['avg_throughput_slots']:.1f}\n"
            f"- Avg option duration (slots): {network_stats['avg_option_duration']:.1f}\n"
            f"- Current epsilon: {network_stats['current_epsilon']:.4f}\n"
            f"- Training progress: {network_stats['episode_progress']:.1%}\n"
            f"\n## Current reward parameters\n"
            f"- throughput_weight:  {self.current_params['throughput_weight']:.4f}\n"
            f"- latency_penalty:    {self.current_params['latency_penalty']:.4f}\n"
            f"- npca_switch_bonus:  {self.current_params['npca_switch_bonus']:.4f}\n"
            f"- npca_switch_cost:   {self.current_params['npca_switch_cost']:.4f}\n"
            f"- qos_priority:       {self.current_qos_priority}\n"
            f"\nDesign updated parameters that best satisfy the user intent and network conditions."
        )

    def design_reward_params(self, network_stats: dict) -> dict:
        """
        Call LLM to design reward parameters based on current network statistics.
        Returns dict with throughput_weight, latency_penalty, and reasoning.
        Fallbacks to current params on any error.
        """
        if self.use_mock:
            return self._mock_design(network_stats)

        user_prompt = self.build_user_prompt(network_stats)

        # Retry on transient server overload (HTTP 529) with exponential backoff.
        _retry_delays = [5, 15, 30]
        last_error: Exception | None = None

        for attempt, delay in enumerate([0] + _retry_delays):
            if delay:
                print(f"  [LLM] OverloadedError, retrying in {delay}s (attempt {attempt+1}/{len(_retry_delays)+1})…")
                time.sleep(delay)
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    system=[
                        {
                            "type": "text",
                            "text": _SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_prompt}],
                )
                last_error = None
                break  # success
            except anthropic.OverloadedError as e:
                last_error = e
                continue
            except (anthropic.APIError, anthropic.APIConnectionError) as e:
                last_error = e
                break  # non-retryable API error

        if last_error is not None:
            tw = self.current_params["throughput_weight"]
            lp = self.current_params["latency_penalty"]
            sb = self.current_params["npca_switch_bonus"]
            sc = self.current_params["npca_switch_cost"]
            qp = self.current_qos_priority
            reasoning = f"Error ({type(last_error).__name__}), kept previous params."
        else:
            try:
                raw = response.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()
                result = json.loads(raw)

                tw = float(result.get("throughput_weight", self.current_params["throughput_weight"]))
                lp = float(result.get("latency_penalty", self.current_params["latency_penalty"]))
                sb = float(result.get("npca_switch_bonus", self.current_params["npca_switch_bonus"]))
                sc = float(result.get("npca_switch_cost", self.current_params["npca_switch_cost"]))
                qp = result.get("qos_priority", self.current_qos_priority)
                if qp not in QOS_PRIORITIES:
                    qp = "balanced"
                tw = max(PARAM_BOUNDS["throughput_weight"][0], min(PARAM_BOUNDS["throughput_weight"][1], tw))
                lp = max(PARAM_BOUNDS["latency_penalty"][0], min(PARAM_BOUNDS["latency_penalty"][1], lp))
                sb = max(PARAM_BOUNDS["npca_switch_bonus"][0], min(PARAM_BOUNDS["npca_switch_bonus"][1], sb))
                sc = max(PARAM_BOUNDS["npca_switch_cost"][0], min(PARAM_BOUNDS["npca_switch_cost"][1], sc))
                reasoning = result.get("reasoning", "")

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                tw = self.current_params["throughput_weight"]
                lp = self.current_params["latency_penalty"]
                sb = self.current_params["npca_switch_bonus"]
                sc = self.current_params["npca_switch_cost"]
                qp = self.current_qos_priority
                reasoning = f"Error ({type(e).__name__}), kept previous params."

        self.current_params = {"throughput_weight": tw, "latency_penalty": lp,
                               "npca_switch_bonus": sb, "npca_switch_cost": sc}
        self.current_qos_priority = qp
        self.call_count += 1

        entry = {
            "call_count": self.call_count,
            "user_intent": self.user_intent,
            "network_stats": dict(network_stats),
            "throughput_weight": tw,
            "latency_penalty": lp,
            "npca_switch_bonus": sb,
            "npca_switch_cost": sc,
            "qos_priority": qp,
            "reasoning": reasoning,
        }
        self.call_history.append(entry)

        return {"throughput_weight": tw, "latency_penalty": lp, "npca_switch_bonus": sb,
                "npca_switch_cost": sc, "qos_priority": qp, "reasoning": reasoning}

    def _mock_design(self, network_stats: dict) -> dict:
        """Deterministic mock for testing without API calls. Intent takes priority."""
        obss = network_stats.get("obss_occupancy_rate", 0.03)
        switch_ratio = network_stats.get("npca_switch_ratio", 0.5)
        avg_tau = network_stats.get("avg_option_duration", 10.0)
        progress = network_stats.get("episode_progress", 0.5)

        tw = self.current_params["throughput_weight"]
        lp = self.current_params["latency_penalty"]
        sb = self.current_params["npca_switch_bonus"]
        sc = self.current_params["npca_switch_cost"]

        intent = self.user_intent.lower()
        _latency_kw = ("latency", "delay", "지연", "video", "화상", "voip", "real-time", "realtime", "gaming")
        _throughput_kw = ("throughput", "speed", "속도", "download", "다운로드", "bandwidth", "빠르게", "파일")
        _energy_kw = ("energy", "battery", "power", "에너지", "배터리", "전력", "low power", "절전", "efficient")

        if intent and any(kw in intent for kw in _latency_kw):
            lp = min(PARAM_BOUNDS["latency_penalty"][1], max(lp, 0.12))
            sb = max(PARAM_BOUNDS["npca_switch_bonus"][0], min(sb, 0.03))
            sc = max(PARAM_BOUNDS["npca_switch_cost"][0], min(sc, 0.03))
            tw = max(PARAM_BOUNDS["throughput_weight"][0], min(tw, 1.2))
            qp = "latency"
            reasoning = f"Mock: latency-sensitive intent detected → high latency_penalty, suppressed npca_switch_bonus"
        elif intent and any(kw in intent for kw in _throughput_kw):
            tw = min(PARAM_BOUNDS["throughput_weight"][1], max(tw, 1.8))
            sb = min(PARAM_BOUNDS["npca_switch_bonus"][1], max(sb, 0.15))
            sc = max(PARAM_BOUNDS["npca_switch_cost"][0], min(sc, 0.03))
            lp = max(PARAM_BOUNDS["latency_penalty"][0], min(lp, 0.04))
            qp = "throughput"
            reasoning = f"Mock: throughput-hungry intent detected → high throughput_weight and npca_switch_bonus"
        elif intent and any(kw in intent for kw in _energy_kw):
            sc = min(PARAM_BOUNDS["npca_switch_cost"][1], max(sc, 0.15))
            sb = max(PARAM_BOUNDS["npca_switch_bonus"][0], min(sb, 0.03))
            tw = max(PARAM_BOUNDS["throughput_weight"][0], min(tw, 1.2))
            lp = max(PARAM_BOUNDS["latency_penalty"][0], min(lp, 0.08))
            qp = "energy"
            reasoning = f"Mock: energy-saving intent detected → high npca_switch_cost to discourage unnecessary NPCA transitions"
        elif obss > 0.05 and switch_ratio < 0.3 and progress > 0.1:
            tw = min(PARAM_BOUNDS["throughput_weight"][1], tw * 1.1)
            sb = min(PARAM_BOUNDS["npca_switch_bonus"][1], sb + 0.05)
            qp = "throughput"
            reasoning = (f"Mock: high OBSS ({obss:.3f}), low switch ratio ({switch_ratio:.2f}) "
                         f"→ increased throughput_weight and npca_switch_bonus")
        elif obss < 0.02 and switch_ratio > 0.7:
            tw = max(PARAM_BOUNDS["throughput_weight"][0], tw * 0.9)
            sb = max(PARAM_BOUNDS["npca_switch_bonus"][0], sb - 0.05)
            qp = "balanced"
            reasoning = (f"Mock: low OBSS ({obss:.3f}), high switch ratio ({switch_ratio:.2f}) "
                         f"→ decreased throughput_weight and npca_switch_bonus")
        elif avg_tau > 50:
            lp = min(PARAM_BOUNDS["latency_penalty"][1], lp * 1.1)
            qp = "latency"
            reasoning = f"Mock: high avg_option_duration ({avg_tau:.1f}) → increased latency_penalty"
        else:
            qp = "balanced"
            reasoning = "Mock: stable conditions, no change"

        tw = round(tw, 4)
        lp = round(lp, 4)
        sb = round(sb, 4)
        sc = round(sc, 4)
        self.current_params = {"throughput_weight": tw, "latency_penalty": lp,
                               "npca_switch_bonus": sb, "npca_switch_cost": sc}
        self.current_qos_priority = qp
        self.call_count += 1

        entry = {
            "call_count": self.call_count,
            "user_intent": self.user_intent,
            "network_stats": dict(network_stats),
            "throughput_weight": tw,
            "latency_penalty": lp,
            "npca_switch_bonus": sb,
            "npca_switch_cost": sc,
            "qos_priority": qp,
            "reasoning": reasoning,
        }
        self.call_history.append(entry)

        return {"throughput_weight": tw, "latency_penalty": lp, "npca_switch_bonus": sb,
                "npca_switch_cost": sc, "qos_priority": qp, "reasoning": reasoning}

    def get_current_params(self) -> dict:
        return dict(self.current_params)

    def get_context_vec(self) -> list[float]:
        """Return normalized reward params as context vector for DQN state input.

        Maps each parameter to [0, 1] using its design bounds so that different
        LLM outputs for the same intent are distinguishable by the agent.
        Order: [tw_norm, lp_norm, sb_norm, sc_norm]
        """
        def _norm(val, key):
            lo, hi = _CTX_NORM[key]
            return (val - lo) / (hi - lo)

        return [
            _norm(self.current_params["throughput_weight"],  "throughput_weight"),
            _norm(self.current_params["latency_penalty"],    "latency_penalty"),
            _norm(self.current_params["npca_switch_bonus"],  "npca_switch_bonus"),
            _norm(self.current_params["npca_switch_cost"],   "npca_switch_cost"),
        ]

    def save_history(self, path: str):
        """Save LLM call history to JSON file."""
        import json as _json
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            _json.dump(self.call_history, f, indent=2)
