from dataclasses import dataclass
from pathlib import Path

import joblib


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "visual_rescue_router_final.joblib"
)


@dataclass(frozen=True)
class VisualRescueDecision:
    backend: str
    rescue_probability: float
    threshold: float


class VisualRescueRouter:
    def __init__(self, model_path=MODEL_PATH):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Visual rescue router not found: {model_path}"
            )

        bundle = joblib.load(model_path)

        self.vectorizer = bundle["vectorizer"]
        self.model = bundle["model"]
        self.threshold = float(
            bundle["threshold"]
        )

        self.version = bundle.get(
            "version",
            "unknown",
        )

    def route(
        self,
        question: str,
    ) -> VisualRescueDecision:

        question = str(question).strip()

        X = self.vectorizer.transform(
            [question]
        )

        rescue_probability = float(
            self.model.predict_proba(
                X
            )[0, 1]
        )

        backend = (
            "LARGE"
            if rescue_probability
            >= self.threshold
            else "SMALL"
        )

        return VisualRescueDecision(
            backend=backend,
            rescue_probability=rescue_probability,
            threshold=self.threshold,
        )
