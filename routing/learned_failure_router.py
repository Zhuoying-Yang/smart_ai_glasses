from pathlib import Path

import joblib


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "e2b_calibrated_failure_router_30k.joblib"
)


class E2BFailureRouter:
    """
    Calibrated question-only predictor for Galaxy E2B failure risk.

    Input:
        question text

    Output:
        calibrated P(E2B failure)

    Current routing threshold:
        loaded from trained model bundle.
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Router model not found: {self.model_path}"
            )

        bundle = joblib.load(self.model_path)

        self.vectorizer = bundle["vectorizer"]
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])

        self.global_failure_rate = bundle.get(
            "global_failure_rate"
        )

        self.failure_definition = bundle.get(
            "failure_definition",
            "vqa_score < 0.5",
        )

    def predict_failure_risk(self, question: str) -> float:
        question = str(question).strip()

        if not question:
            return 1.0

        X = self.vectorizer.transform(
            [question]
        )

        risk = self.model.predict_proba(X)[0, 1]

        return float(risk)
