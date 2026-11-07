# diagnosticAnalysis: where the methods work and fail (validation only)

```bash
python diagnosticAnalysis/run_diagnostics.py      # about 2 minutes
```

Reads the saved validation daily picks of all four methods (nothing is re-ranked) and refits the gradient
boosting settings on train for the number-of-trees check. The test set is not used.

| Check | Outputs | Finding |
|---|---|---|
| Customer region | `results/region_validation.csv`, figures 1-2 | Gradient boosting sends 72% of picks to the Southeast, logistic regression about half to the Northeast, the distance rule to the North and Northeast. All three nearly miss the South and Center-West (573 late orders). |
| Number of trees (gradient boosting, 1 to 1,000) | `results/trees_sensitivity_validation.csv`, figures 3-4 | The chosen setting peaks at 200-300 trees (335 hits) and falls to 289 by 1,000 trees, so 300 was a good stopping point. |
| Weekends | `results/late_rate_by_promised_weekday.csv`, `promised_weekday_validation.csv`, `approved_weekend_validation.csv`, figures 5-7 | Olist never promises weekend delivery. The late rate is flat across promised weekdays and for weekend approvals, and the methods rank the same way, so weekends do not matter here. |

The investigation-budget sensitivity (1 to 50 per day) is in each method folder for validation and in
`testResults/` for the test period.
