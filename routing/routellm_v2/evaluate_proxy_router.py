import csv
from pathlib import Path


DATA = Path(
    "routing/routellm_v2/results/"
    "proxy_router_loo_scores.csv"
)

THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]


with open(DATA, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))


print()
print("=" * 100)
print("ROUTELLM-INSPIRED WHICH V2 THRESHOLD SWEEP")
print("=" * 100)

print(
    f"{'Threshold':>10} "
    f"{'Cloud use':>12} "
    f"{'Failure caught':>16} "
    f"{'Miss rate':>12} "
    f"{'Over-escalation':>18}"
)

print("-" * 100)


for threshold in THRESHOLDS:

    large = 0
    failures = 0
    successes = 0
    caught = 0
    unnecessary = 0

    for r in rows:

        score = float(
            r["router_score"]
        )

        small_failed = (
            r["small_failed"] == "1"
        )

        route_large = (
            score >= threshold
        )

        if route_large:
            large += 1

        if small_failed:

            failures += 1

            if route_large:
                caught += 1

        else:

            successes += 1

            if route_large:
                unnecessary += 1


    cloud_usage = (
        large / len(rows)
    )

    catch_rate = (
        caught / failures
        if failures
        else 0
    )

    miss_rate = (
        1 - catch_rate
    )

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
