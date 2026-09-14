import csv

from routing.which_router import (
    WhichRouter,
    WhichSignals,
)


PATH = (
    "routing/benchmarks/results/"
    "wearvqa_e2b_50_graded.csv"
)

THRESHOLDS = [
    0.20,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.80,
]


with open(PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))


rows = [
    r for r in rows
    if r["manual_correct"] in {"0", "1"}
]


print()
print("=" * 94)
print("WHICH ROUTER THRESHOLD SWEEP")
print("=" * 94)

print(
    f"{'Threshold':>10} "
    f"{'Cloud use':>12} "
    f"{'Failure caught':>16} "
    f"{'Miss rate':>12} "
    f"{'Over-escalation':>18}"
)

print("-" * 94)


for threshold in THRESHOLDS:

    router = WhichRouter(
        large_threshold=threshold
    )

    failures = 0
    successes = 0

    caught = 0
    unnecessary = 0
    large_count = 0

    for r in rows:

        d = router.route(
            WhichSignals(
                question=r["question"],
                network_available=True,
                large_available=True,
                api_budget_ok=True,
            )
        )

        route_large = (
            d.action.value == "LARGE_1F"
        )

        if route_large:
            large_count += 1

        if r["manual_correct"] == "0":

            failures += 1

            if route_large:
                caught += 1

        else:

            successes += 1

            if route_large:
                unnecessary += 1

    cloud_usage = large_count / len(rows)

    catch_rate = (
        caught / failures
        if failures
        else 0
    )

    miss_rate = 1 - catch_rate

    over_rate = (
        unnecessary / successes
        if successes
        else 0
    )

    print(
        f"{threshold:10.2f} "
        f"{100*cloud_usage:11.1f}% "
        f"{100*catch_rate:15.1f}% "
        f"{100*miss_rate:11.1f}% "
        f"{100*over_rate:17.1f}%"
    )

print()
