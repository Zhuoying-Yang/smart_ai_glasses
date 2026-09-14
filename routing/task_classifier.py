import pickle
from pathlib import Path


MODEL_PATH = Path(
    __file__
).parent / "task_classifier.pkl"


_model = None


def _load_model():

    global _model

    if _model is None:

        with open(
            MODEL_PATH,
            "rb",
        ) as f:
            _model = pickle.load(f)

    return _model


def classify_task(
    question: str,
) -> tuple[str, str]:

    model = _load_model()

    prediction = model.predict(
        [question]
    )[0]

    # Get confidence if supported.
    try:

        probabilities = model.predict_proba(
            [question]
        )[0]

        confidence = max(
            probabilities
        )

        reason = (
            f"WearVQA text classifier "
            f"confidence={confidence:.2f}"
        )

    except Exception:

        reason = (
            "WearVQA text classifier"
        )

    return prediction, reason
