 # Acceptance policy (v1, written before any retraining experiment)
- Primary metric: F1 of the churn class (label 1).
- Candidate F1 must beat the champion by at least 0.01, OR be within 0.005 while improving recall.
- Safety floor: candidate recall must be at least 0.60.
- Minimum new labeled rows: 200, with at least 30 per class.
- Evidence: the 95% bootstrap interval of (candidate F1 minus champion F1) must not be entirely below 0.
- Segment check: no monitored segment (Contract type) loses more than 0.05 F1.
- Cooldown: at most one retraining run per 24 hours.
- Insufficient or ambiguous evidence means reject.