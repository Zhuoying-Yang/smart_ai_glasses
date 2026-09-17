from dataclasses import dataclass
from typing import Optional, List

from .action_space import Action
from .learned_failure_router import E2BFailureRouter
from .temporal_router import TemporalRouter


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

    # Debug
    force_small: bool = False
    force_large: bool = False


@dataclass
class WhichDecision:
    action: Action

    # calibrated P(E2B failure)
    score: float

    # static / temporal
    task_type: str

    reason: str
    factors: List[str]


class WhichRouter:
    """
    Combined WHICH router.

    Two independent decisions:

    1. Model routing
       question -> calibrated P(E2B failure)
       -> SMALL / LARGE

    2. Visual context routing
       question -> temporal need
       -> 1F / MULTI

    Combined actions:
       SMALL_1F
       SMALL_MULTI
       LARGE_1F
       LARGE_MULTI
    """

    def __init__(
        self,
        model_path=None,
        max_cloud_rtt_ms: float = 800.0,
    ):
        # SMALL vs LARGE
        if model_path is None:
            self.failure_router = E2BFailureRouter()
        else:
            self.failure_router = E2BFailureRouter(
                model_path=model_path
            )

        self.large_threshold = (
            self.failure_router.threshold
        )

        # 1F vs MULTI
        self.temporal_router = TemporalRouter()

        self.max_cloud_rtt_ms = max_cloud_rtt_ms

    # ========================================================
    # Helper
    # ========================================================

    @staticmethod
    def _action(
        use_large: bool,
        use_multi: bool,
    ) -> Action:

        if use_large and use_multi:
            return Action.LARGE_MULTI

        if use_large:
            return Action.LARGE_1F

        if use_multi:
            return Action.SMALL_MULTI

        return Action.SMALL_1F

    # ========================================================
    # Main routing
    # ========================================================

    def route(
        self,
        s: WhichSignals,
    ) -> WhichDecision:

        # ----------------------------------------------------
        # First decide whether temporal evidence is needed.
        # ----------------------------------------------------

        temporal = self.temporal_router.route(
            s.question
        )

        use_multi = temporal.multi_frame

        temporal_label = (
            "temporal"
            if use_multi
            else "static"
        )

        temporal_factor = (
            "visual_context=MULTI"
            if use_multi
            else "visual_context=1F"
        )

        # ----------------------------------------------------
        # Explicit overrides
        # ----------------------------------------------------

        if s.force_small:
            return WhichDecision(
                action=self._action(
                    use_large=False,
                    use_multi=use_multi,
                ),
                score=0.0,
                task_type=temporal_label,
                reason="Forced SMALL.",
                factors=[
                    "force_small",
                    temporal_factor,
                ],
            )

        if (
            s.force_large
            and s.network_available
            and s.large_available
            and s.api_budget_ok
        ):
            return WhichDecision(
                action=self._action(
                    use_large=True,
                    use_multi=use_multi,
                ),
                score=1.0,
                task_type=temporal_label,
                reason="Forced LARGE.",
                factors=[
                    "force_large",
                    temporal_factor,
                ],
            )

        # ----------------------------------------------------
        # Hard feasibility constraints
        # ----------------------------------------------------

        cloud_feasible = True
        cloud_reasons = []

        if not s.network_available:
            cloud_feasible = False
            cloud_reasons.append("offline")

        if not s.large_available:
            cloud_feasible = False
            cloud_reasons.append(
                "large_unavailable"
            )

        if not s.api_budget_ok:
            cloud_feasible = False
            cloud_reasons.append(
                "api_budget_unavailable"
            )

        if (
            s.network_rtt_ms is not None
            and s.network_rtt_ms
            > self.max_cloud_rtt_ms
        ):
            cloud_feasible = False
            cloud_reasons.append(
                "network_too_slow"
            )

        # ----------------------------------------------------
        # Predict calibrated E2B failure probability
        # ----------------------------------------------------

        score = (
            self.failure_router
            .predict_failure_risk(
                s.question
            )
        )

        factors = [
            "router=calibrated_text_failure_predictor",
            f"predicted_e2b_failure_risk={score:.3f}",
            f"threshold={self.large_threshold:.3f}",
            temporal_factor,
        ]

        if temporal.matched_patterns:
            factors.append(
                "temporal_pattern_match"
            )

        # ----------------------------------------------------
        # Device pressure
        # ----------------------------------------------------

        if s.phone_busy:
            score = min(
                1.0,
                score + 0.10,
            )
            factors.append(
                "phone_busy:+0.10"
            )

        if s.memory_pressure_high:
            score = min(
                1.0,
                score + 0.15,
            )
            factors.append(
                "memory_pressure:+0.15"
            )

        # ----------------------------------------------------
        # SMALL vs LARGE
        # ----------------------------------------------------

        wants_large = (
            score >= self.large_threshold
        )

        use_large = (
            wants_large
            and cloud_feasible
        )

        # ----------------------------------------------------
        # Final action
        # ----------------------------------------------------

        action = self._action(
            use_large=use_large,
            use_multi=use_multi,
        )

        if wants_large and not cloud_feasible:
            factors.extend(cloud_reasons)

            reason = (
                f"E2B failure risk {score:.2f} "
                f">= threshold "
                f"{self.large_threshold:.2f}, "
                f"but LARGE is unavailable; "
                f"falling back to SMALL."
            )

        elif use_large:
            reason = (
                f"E2B failure risk {score:.2f} "
                f">= threshold "
                f"{self.large_threshold:.2f}; "
                f"route LARGE."
            )

        else:
            reason = (
                f"E2B failure risk {score:.2f} "
                f"< threshold "
                f"{self.large_threshold:.2f}; "
                f"route SMALL."
            )

        if use_multi:
            reason += (
                " Temporal evidence detected; "
                "use multiple frames."
            )
        else:
            reason += (
                " Single-frame evidence is sufficient."
            )

        return WhichDecision(
            action=action,
            score=score,
            task_type=temporal_label,
            reason=reason,
            factors=factors,
        )
