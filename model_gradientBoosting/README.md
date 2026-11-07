# model_gradientBoosting: gradient boosted trees (learned model 2)

scikit-learn's `HistGradientBoostingClassifier` with the same 43 features and the same daily top-10 scoring
as the logistic regression.

```bash
python model_gradientBoosting/train_gradient_boosting.py      # about 5 minutes
```

Steps:

1. Numbers go in unchanged. The trees handle missing values themselves. States, payment type, category and
   weekday are coded as integers learned on train and treated as categories, with values seen fewer than 100
   times grouped together.
2. Fit on the 43,345 labeled 2017 orders (log-loss, L2 regularization 1.0, no early stopping, `random_state=0`).
3. Try learning rate {0.01, 0.03, 0.1} x leaves per tree {2, 3, 7, 15, 31} x minimum orders per leaf {50, 200}
   x trees {100, 300, 600} (90 settings), and keep the one with the most validation hits at 10 per day
   (ties by validation PR-AUC). Chosen: **learning rate 0.1, 3 leaves, 200 orders per leaf, 300 trees**.

Outputs:

- `results/`: `model.joblib`, `config.json`, `tuning_validation.csv`, `training_curve.csv` (loss and validation
  hits as trees are added), `feature_importance_validation.csv` (drop in validation PR-AUC when each input is
  shuffled), `predictions_validation.csv`, `daily_picks_<period>.csv`, `daily_scores_<period>.csv`,
  `summary.csv`, `budget_sensitivity_validation.csv`.
- `figures/`: comparison with logistic regression and both rules, the 12 best settings, daily picks, running
  total, training curve, input importance, budget sensitivity, weekly hit rate.

Validation: 335 hits (33.5% hit rate, 2.84 times random), the best on validation, so it was the model chosen
before the test run. Test (from `testResults/`): 82 hits (8.3%, 2.31 times random), behind logistic regression
and about the same as the distance rule.
