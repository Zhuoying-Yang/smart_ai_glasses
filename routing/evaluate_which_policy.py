import csv
from collections import Counter, defaultdict

from routing.which_router import (
    WhichRouter,
    WhichSignals,
)


GRADED_FILE = (
    "routing/benchmarks/results/"
    "wearvqa_e2b_50_graded.csv"
)

THRESHOLD = 0.40


router = WhichRouter(
    large_threshold=THRESHOLD
)


with open(
    GRADED_FILE,
    newline="",
    encoding="utf-8",
) as f:
    rows = list(csv.DictReader(f))


judgeable = [
    r for r in rows
    if r["manual_correct"] in {"0", "1"}
]

ambiguous = [
    r for r in rows
    if r["manual_correct"] == "ambiguous"
]


results = []


for r in judgeable:

    decision = router.route(
        WhichSignals(
            question=r["question"],

            # Simulate the future complete system:
            network_available=True,
            large_available=True,
            api_budget_ok=True,
        )
    )

    small_correct = (
        r["manual_correct"] == "1"
    )

    route_large = (
        decision.action.value == "LARGE_1F"
    )

    results.append({
        "sample_id": r["sample_id"],
        "category": r["category"],
        "question": r["question"],
        "small_correct": small_correct,
        "route_large": route_large,
        "score": decision.score,
        "task_type": decision.task_type,
    })


# =========================================================
# COUNTS
# =========================================================

n = len(results)

small_failures = [
    r for r in results
    if not r["small_correct"]
]

small_successes = [
    r for r in results
    if r["small_correct"]
]

routed_large = [
    r for r in results
    if r["route_large"]
]

routed_small = [
    r for r in results
    if not r["route_large"]
]


# E2B failure that router caught and escalated
caught_failures = [
    r for r in small_failures
    if r["route_large"]
]

# E2B failure mistakenly kept local
missed_failures = [
    r for r in small_failures
    if not r["route_large"]
]

# E2B was correct, but router would pay for LARGE anyway
unnecessary_large = [
    r for r in small_successes
    if r["route_large"]
]

# E2B correct and router correctly stays local
good_small = [
    r for r in small_successes
    if not r["route_large"]
]


print()
print("=" * 78)
print("WHICH ROUTER POLICY EVALUATION")
print("=" * 78)

print(f"Threshold                    : {THRESHOLD:.2f}")
print(f"Total samples                : {len(rows)}")
print(f"Judgeable samples            : {n}")
print(f"Ambiguous excluded           : {len(ambiguous)}")

print()
print("ROUTING USAGE")
print("-" * 78)

print(
    f"Would route SMALL            : "
    f"{len(routed_small)}/{n} "
    f"({100*len(routed_small)/n:.1f}%)"
)

print(
    f"Would route LARGE            : "
    f"{len(routed_large)}/{n} "
    f"({100*len(routed_large)/n:.1f}%)"
)


print()
print("FAILURE DETECTION")
print("-" * 78)

print(
    f"Actual E2B failures          : "
    f"{len(small_failures)}"
)

if small_failures:
    print(
        f"Failures routed LARGE        : "
        f"{len(caught_failures)}/{len(small_failures)} "
        f"({100*len(caught_failures)/len(small_failures):.1f}%)"
    )

    print(
        f"Failures missed / kept SMALL : "
        f"{len(missed_failures)}/{len(small_failures)} "
        f"({100*len(missed_failures)/len(small_failures):.1f}%)"
    )


print()
print("OVER-ESCALATION")
print("-" * 78)

print(
    f"Actual E2B successes         : "
    f"{len(small_successes)}"
)

if small_successes:
    print(
        f"Correct E2B sent to LARGE    : "
        f"{len(unnecessary_large)}/{len(small_successes)} "
        f"({100*len(unnecessary_large)/len(small_successes):.1f}%)"
    )

    print(
        f"Correct E2B kept SMALL       : "
        f"{len(good_small)}/{len(small_successes)} "
        f"({100*len(good_small)/len(small_successes):.1f}%)"
    )


print()
print("=" * 78)
print("MISSED E2B FAILURES")
print("=" * 78)

for r in missed_failures:
    print()
    print("ID      :", r["sample_id"])
    print("Dataset :", r["category"])
    print("Detected:", r["task_type"])
    print("Score   :", f'{r["score"]:.2f}')
    print("Question:", r["question"])


print()
print("=" * 78)
print("UNNECESSARY LARGE ROUTES")
print("=" * 78)

for r in unnecessary_large:
    print()
    print("ID      :", r["sample_id"])
    print("Dataset :", r["category"])
    print("Detected:", r["task_type"])
    print("Score   :", f'{r["score"]:.2f}')
    print("Question:", r["question"])

print()
