import re
from dataclasses import dataclass
from typing import List


@dataclass
class TemporalDecision:
    multi_frame: bool
    score: float
    reason: str
    matched_patterns: List[str]


class TemporalRouter:
    """
    Decide whether a question requires temporal / multi-frame evidence.

    This router is intentionally separate from SMALL-vs-LARGE routing.

    Outputs:
        1F    -> one current/relevant frame is sufficient
        MULTI -> multiple frames across time are useful
    """

    def __init__(self):
        # Strong temporal cues.
        self.strong_patterns = [
            r"\bwhat happened\b",
            r"\bwhat happens\b",
            r"\bwhat will happen\b",
            r"\bwhat happens next\b",
            r"\bwhat happened before\b",
            r"\bwhat happened after\b",

            r"\bbefore\b",
            r"\bafter\b",
            r"\bnext\b",
            r"\bpreviously\b",
            r"\bearlier\b",
            r"\blater\b",

            r"\bwhat did\b",
            r"\bwhat was .* doing\b",
            r"\bwhat were .* doing\b",

            r"\bwhat is .* doing\b",
            r"\bwhat are .* doing\b",
            r"\bwhy is .* doing\b",
            r"\bwhy are .* doing\b",

            r"\bhow did\b",
            r"\bhow does .* move\b",
            r"\bhow is .* moving\b",

            r"\bwhole action\b",
            r"\bentire action\b",
            r"\bwhole process\b",
            r"\bentire process\b",

            r"\bsequence\b",
            r"\bmovement\b",
            r"\bmotion\b",

            r"\bwhere did\b",
            r"\bwhere is .* going\b",
            r"\bwhich direction\b",

            r"\bpicked up\b",
            r"\bput down\b",
            r"\bmoved\b",
            r"\bwalking\b",
            r"\brunning\b",
        ]

        # Questions that are normally answerable from one frame.
        self.static_patterns = [
            r"\bwhat color\b",
            r"\bwhat colour\b",
            r"\bhow many\b",
            r"\bis there\b",
            r"\bare there\b",
            r"\bwhat object\b",
            r"\bwhat type\b",
            r"\bwhat kind\b",
            r"\bwhere is\b",
            r"\bwhere are\b",
            r"\bwhat is in front\b",
            r"\bwhat's in front\b",
            r"\bwho is\b",
            r"\bcan you see\b",
            r"\bdo you see\b",
        ]

    def route(self, question: str) -> TemporalDecision:
        q = str(question).strip().lower()

        if not q:
            return TemporalDecision(
                multi_frame=False,
                score=0.0,
                reason="Empty question; default to one frame.",
                matched_patterns=[],
            )

        temporal_matches = [
            p for p in self.strong_patterns
            if re.search(p, q)
        ]

        static_matches = [
            p for p in self.static_patterns
            if re.search(p, q)
        ]

        # Strong temporal evidence wins.
        if temporal_matches:
            return TemporalDecision(
                multi_frame=True,
                score=1.0,
                reason="Question contains temporal/action cues.",
                matched_patterns=temporal_matches,
            )

        # Explicitly static query.
        if static_matches:
            return TemporalDecision(
                multi_frame=False,
                score=0.0,
                reason="Question appears answerable from one frame.",
                matched_patterns=static_matches,
            )

        # Conservative default:
        # one frame unless temporal evidence is present.
        return TemporalDecision(
            multi_frame=False,
            score=0.25,
            reason="No strong temporal cue; default to one frame.",
            matched_patterns=[],
        )
