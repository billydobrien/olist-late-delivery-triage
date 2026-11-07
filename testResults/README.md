# testResults: the single final test run

Scores the test period (orders approved 2018-04-16 to 2018-07-31: 103 days with orders, 22,855 candidates,
840 late) once, with all four methods.

```bash
python testResults/run_test.py
```

- The two models are loaded from `../baseline_logisticRegression/results/model.joblib` and
  `../model_gradientBoosting/results/model.joblib` without retraining. Their settings were chosen on
  validation before this run, and nothing was changed afterwards.
- The two rules use `distance_km` and `seller_prev_late_rate` directly.
- The same `daily_eval.py` scoring as validation: top 10 per day, hits, wasted, unknown and missed.

Outputs in `results/`:

| File | Contents |
|---|---|
| `daily_list_test_<method>.csv` | **The saved daily prioritization lists.** One row per investigated order: approval date, rank, order ID, approval time, risk score, customer and seller state, outcome (hit, wasted or unknown). 1,019 rows per method. |
| `predictions_test_<method>.csv` | The risk score of every test order |
| `daily_scores_test_<method>.csv` | Per-day candidates, late orders, hits, wasted, unknown, missed, best possible and random expected |
| `summary_test.csv` | Totals for all four methods, the random list and the perfect list |
| `budget_sensitivity_test.csv` | Hits and hit rate at 1 to 50 investigations per day |
| `test_run_log.txt` | When the test was scored, and a note that nothing was tuned on it |

`<method>` is `logistic_regression`, `gradient_boosting`, `distance_rule` or `seller_rule`.

`figures/`: test results table, validation compared with test, the proposed model's daily picks, running total
of late orders caught, weekly hit rate, budget sensitivity.

Result: logistic regression 105 hits, distance rule 87, gradient boosting 82, seller rule 76, random 35,
perfect 560.
