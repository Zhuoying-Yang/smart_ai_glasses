"""
Empirical SMALL-model failure risk estimated from the
50-sample balanced WearVQA-mini pilot.

Ambiguous samples are excluded.

Laplace smoothing:
    risk = (wrong + 1) / (judgeable + 2)

IMPORTANT:
This estimates P(SMALL fails | task type).
It does NOT yet estimate P(LARGE helps | task type).
Once LARGE results are available, replace this table with
SMALL-vs-LARGE advantage / rescue statistics.
"""

TASK_RISK = {
    "activity_recognition": 2 / 6,                  # 1 wrong / 4
    "how_to_purpose": 1 / 6,                       # 0 / 4
    "image_attribute_simple_recognition": 2 / 6,   # 1 / 4
    "image_general_reasoning": 2 / 7,              # 1 / 5
    "math": 6 / 7,                                 # 5 / 5
    "next_state_prediction": 3 / 7,                # 2 / 5
    "object_counting": 4 / 7,                      # 3 / 5
    "spatial_reasoning": 3 / 5,                    # 2 / 3
    "text_general_reasoning": 3 / 7,               # 2 / 5
    "text_simple_recognition": 4 / 7,              # 3 / 5
}


def get_task_risk(task_type: str) -> float:
    """
    Return empirical SMALL-failure risk.

    Unknown tasks receive a neutral prior of 0.5.
    """
    return TASK_RISK.get(task_type, 0.50)
