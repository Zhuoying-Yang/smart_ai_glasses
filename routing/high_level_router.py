from dataclasses import dataclass
from pathlib import Path

import joblib


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "question_type_router_sol.joblib"
)

VALID_ROUTES = {
    "DIRECT_VISUAL",
    "TEMPORAL",
    "MEMORY",
    "KNOWLEDGE",
}


@dataclass(frozen=True)
class HighLevelDecision:
    route: str
    confidence: float
    probabilities: dict
    reason: str


class HighLevelRouter:
    def __init__(self, model_path=MODEL_PATH):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"High-level router not found: {model_path}"
            )

        bundle = joblib.load(model_path)

        self.model = bundle["model"]

        self.classes = list(
            getattr(
                self.model,
                "classes_",
                bundle.get("classes", []),
            )
        )

        self.test_accuracy = bundle.get(
            "test_accuracy"
        )

        self.test_macro_f1 = bundle.get(
            "test_macro_f1"
        )

    def route(self, question: str) -> HighLevelDecision:
        question = str(question).strip()

        if not question:
            raise ValueError(
                "Question is empty."
            )

        prediction = str(
            self.model.predict(
                [question]
            )[0]
        )

        if prediction not in VALID_ROUTES:
            raise ValueError(
                f"Unexpected route: {prediction}"
            )

        probabilities = {}

        try:
            values = self.model.predict_proba(
                [question]
            )[0]

            probabilities = {
                str(label): float(prob)
                for label, prob in zip(
                    self.classes,
                    values,
                )
            }

            confidence = float(
                probabilities.get(
                    prediction,
                    max(values),
                )
            )

        except Exception:
            confidence = float("nan")

        return HighLevelDecision(
            route=prediction,
            confidence=confidence,
            probabilities=probabilities,
            reason=(
                "TF-IDF unigram/bigram + "
                "class-balanced logistic regression"
            ),
        )
