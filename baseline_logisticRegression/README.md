# baseline_logisticRegression: logistic regression (learned model 1, recommended)

Predicts the chance that an order will be late, from the 43 features the availability audit allows
(`../dataPrep/output/feature_availability.csv`). The 10 highest-scoring orders each day are investigated.

```bash
python baseline_logisticRegression/train_logistic.py      # about 1 minute
```

Steps (all preprocessing is learned on train only):

1. Counts, prices, weights and time gaps: fill missing with the train median plus a missing flag, log(1 + x),
   standardize. Other numbers: fill missing plus flag, standardize. States, payment type, category and weekday:
   one 0/1 column per value, with values seen fewer than 100 times in train grouped together.
2. Fit on the 43,345 labeled 2017 orders.
3. Try C in {0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10} with and without balanced class weights (16 settings),
   and keep the one with the most validation hits at 10 per day (ties by validation PR-AUC).
   Chosen: **C = 0.003, no class weighting**.

Outputs:

- `results/`: `model.joblib` (the fitted pipeline), `config.json` (settings and feature lists),
  `tuning_validation.csv`, `coefficients.csv`, `predictions_validation.csv`, `daily_picks_<period>.csv`,
  `daily_scores_<period>.csv`, `summary.csv`, `budget_sensitivity_validation.csv`, `convergence.csv`
  (loss and validation hits after each solver iteration; it converges in 22).
- `figures/`: comparison with the two rules, tuning table, daily picks, running total of late orders caught,
  largest coefficients, budget sensitivity, weekly hit rate, convergence.

Validation: 311 hits (31.2% hit rate, 2.64 times random). Test (from `testResults/`): 105 hits (10.6%,
2.96 times random), the best of all methods.

To load the saved model in another script, define `log1p_nonneg` first (see `testResults/run_test.py`),
because the pipeline was saved with it.
