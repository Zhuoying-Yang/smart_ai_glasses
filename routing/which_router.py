from dataclasses import dataclass
from typing import Optional, List

from .action_space import Action
from .task_classifier import classify_task
from .empirical_task_risk import get_task_risk


@dataclass
class WhichSignals:
    question: str = ""
    image_path: Optional[str] = None

    # System availability
    network_available: bool = True
    large_available: bool = False
    api_budget_ok: bool = True

    # Optional network measurement
    network_rtt_ms: Optional[float] = None

    # Device status
    phone_busy: bool = False
    memory_pressure_high: bool = False

    # Optional future learned LARGE-benefit score.
    learned_score: Optional[float] = None

    # Debug
    force_small: bool = False
    force_large: bool = False


@dataclass
class WhichDecision:
    action: Action
    score: float
    task_type: str
    reason: str
    factors: List[str]


class WhichRouter:
    """
    WHICH Router v1.

    Current decision signal:
        empirical E2B failure risk on WearVQA

    Hard constraints:
        network availability
        LARGE availability
        API budget
        excessive cloud RTT

    Future:
        replace SMALL-failure risk with learned
        SMALL-vs-LARGE advantage / rescue score.
    """

    def __init__(
        self,
        large_threshold: float = 0.40,
        max_cloud_rtt_ms: float = 800.0,
    ):
        self.large_threshold = large_threshold
        self.max_cloud_rtt_ms = max_cloud_rtt_ms


    def route(self, s: WhichSignals) -> WhichDecision:

        # --------------------------------------------------
        # Explicit debug overrides
        # --------------------------------------------------

        if s.force_small:
            return WhichDecision(
                action=Action.SMALL_1F,
                score=0.0,
                task_type="override",
                reason="Forced SMALL.",
                factors=["force_small"],
            )

        if (
            s.force_large
            and s.network_available
            and s.large_available
        ):
            return WhichDecision(
                action=Action.LARGE_1F,
                score=1.0,
                task_type="override",
                reason="Forced LARGE.",
                factors=["force_large"],
            )

        # --------------------------------------------------
        # Hard feasibility constraints
        # --------------------------------------------------

        if not s.network_available:
            return WhichDecision(
                action=Action.SMALL_1F,
                score=0.0,
                task_type="system_constraint",
                reason="No network: LARGE cloud path unavailable.",
                factors=["offline"],
            )

        if not s.large_available:
            return WhichDecision(
                action=Action.SMALL_1F,
                score=0.0,
                task_type="system_constraint",
                reason="LARGE backend unavailable.",
                factors=["large_unavailable"],
            )

        if not s.api_budget_ok:
            return WhichDecision(
                action=Action.SMALL_1F,
                score=0.0,
                task_type="system_constraint",
                reason="Cloud/API budget unavailable.",
                factors=["api_budget_unavailable"],
            )

        if (
            s.network_rtt_ms is not None
            and s.network_rtt_ms > self.max_cloud_rtt_ms
        ):
            return WhichDecision(
                action=Action.SMALL_1F,
                score=0.0,
                task_type="system_constraint",
                reason=(
                    f"Cloud RTT {s.network_rtt_ms:.0f} ms "
                    f"exceeds {self.max_cloud_rtt_ms:.0f} ms."
                ),
                factors=["network_too_slow"],
            )

        # --------------------------------------------------
        # Estimate request difficulty for SMALL
        # --------------------------------------------------

        task_type, classifier_reason = classify_task(
            s.question
        )

        empirical_risk = get_task_risk(
            task_type
        )

        factors = [
            classifier_reason,
            f"empirical_small_failure_risk={empirical_risk:.2f}",
        ]

        # --------------------------------------------------
        # Future learned router overrides empirical proxy
        # --------------------------------------------------

        if s.learned_score is not None:
            score = max(
                0.0,
                min(1.0, s.learned_score),
            )
            factors.append(
                f"learned_large_benefit_score={score:.2f}"
            )
        else:
            score = empirical_risk

        # --------------------------------------------------
        # Phone resource pressure:
        # offload if cloud is feasible.
        # --------------------------------------------------

        if s.phone_busy:
            score = min(1.0, score + 0.10)
            factors.append("phone_busy:+0.10")

        if s.memory_pressure_high:
            score = min(1.0, score + 0.15)
            factors.append("memory_pressure:+0.15")

        # --------------------------------------------------
        # Final WHICH decision
        # --------------------------------------------------

        if score >= self.large_threshold:
            return WhichDecision(
                action=Action.LARGE_1F,
                score=score,
                task_type=task_type,
                reason=(
                    f"SMALL-risk score {score:.2f} >= "
                    f"threshold {self.large_threshold:.2f}."
                ),
                factors=factors,
            )

        return WhichDecision(
            action=Action.SMALL_1F,
            score=score,
            task_type=task_type,
            reason=(
                f"SMALL-risk score {score:.2f} < "
                f"threshold {self.large_threshold:.2f}."
            ),
            factors=factors,
        )
