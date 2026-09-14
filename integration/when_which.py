from routing.which_router import WhichRouter, WhichSignals
from when.types import Route


class WhenWhichAdapter:
    def __init__(self):
        self.router = WhichRouter(
            large_threshold=0.40
        )

    def route(self, event):
        if event.route is not Route.TRIGGER:
            return None

        # IMPORTANT:
        # LARGE is marked available so WHICH is allowed
        # to actually choose SMALL vs LARGE.
        #
        # The LARGE executor itself is intentionally blank.
        signals = WhichSignals(
            question=event.query.text,
            network_available=True,
            large_available=True,
            api_budget_ok=True,
        )

        return self.router.route(signals)


def print_decision(event, decision):
    print()
    print("=" * 65)
    print("WHEN -> WHICH")
    print("=" * 65)
    print("Trigger :", event.trigger_type.value)
    print("Question:", event.query.text)
    print("Action  :", decision.action.value)
    print("Task    :", decision.task_type)
    print("Score   :", f"{decision.score:.3f}")
    print("Reason  :", decision.reason)
    print("Factors :", ", ".join(decision.factors))
    print("=" * 65)
