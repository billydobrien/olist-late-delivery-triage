# nonML_distanceRule: longest distance rule (business rule 1)

Each day, investigate the 10 approved orders with the longest straight-line distance between the seller's and
the customer's zip-code areas (`distance_km`). No training. Orders with no distance are ranked last.

```bash
python nonML_distanceRule/run_distance_rule.py
```

Reads `../dataPrep/output/train.csv` and `validation.csv` (not test). `daily_eval.py` holds the shared daily
top-10 selection and scoring code used by every method.

- `results/`: `daily_picks_<period>.csv` (the 10 picks per day and their outcome), `daily_scores_<period>.csv`
  (hits, wasted, unknown, missed, best possible and random expected per day), `summary.csv` (period totals and
  PR-AUC), `budget_sensitivity_validation.csv` (1 to 50 investigations per day).
- `figures/`: summary table, daily picks, running total of late orders caught, weekly hit rate, budget
  sensitivity, and one example day's list.

Validation: 218 hits out of 1,045 investigations (22.0% hit rate, 1.85 times random).
