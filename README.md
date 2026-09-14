# Smart AI Glasses

AI glasses edge-cloud routing prototype.

## Current routing pipeline

WHEN -> WHICH -> SMALL / LARGE

### SMALL
- Gemma E2B
- Runs locally on Galaxy phone

### LARGE
- Not connected yet

## WHICH baseline v1

The current WHICH router is a task-aware empirical baseline.

Pipeline:

question
-> lightweight WearVQA task classifier
-> empirical E2B failure risk
-> threshold
-> SMALL / LARGE

The task classifier uses TF-IDF + Logistic Regression.

The empirical risk is calibrated from a manually graded
50-sample WearVQA-mini pilot.

Current threshold:

    tau = 0.40

Pilot/calibration result on 45 judgeable samples:

- Route SMALL: 33.3%
- Route LARGE: 66.7%
- E2B failures caught for escalation: 85.0%
- E2B failures missed: 15.0%

These are calibration results, not independent test results.

The next version will explore learned SMALL-vs-LARGE routing,
inspired by strong/weak model routing approaches such as RouteLLM.
