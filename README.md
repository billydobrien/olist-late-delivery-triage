# Which orders should be checked? Prioritizing late-delivery investigations

Each day, rank newly approved Olist marketplace orders by their risk of arriving after the promised delivery
date, and send the 10 riskiest to a logistics team that can investigate at most 10 orders per day.

**Task.** One prediction per order, made at payment approval, using only information known at that moment.
An order is **late** if it was delivered on a later calendar day than the promised (estimated) delivery date.
Each method is judged by its daily top-10 list: **hits** (picked and late), **wasted** investigations (picked
and on time), **unknown** (picked, outcome never recorded) and **missed** late orders, compared with two simple
business rules, a random list and a perfect list under the same 10-per-day limit.

## Folder map

| Path | What it is |
|---|---|
| `dataset/` | The nine Olist CSV files (not included in this repository; see "Data") |
| `exploration/` | Data audit: `explore_data.py` makes figures 1-36, `explore_time_split.py` makes figures 37-44 (time split choice, end-of-record check, planned columns) |
| `dataPrep/` | `build_dataset.py`: one row per order, the late label, 44 approval-time features, the train/validation/test split, and the feature-availability audit |
| `nonML_distanceRule/` | Business rule 1: investigate the 10 orders with the longest seller-to-customer distance |
| `nonML_riskiestSellers/` | Business rule 2: investigate the 10 orders whose seller has the worst past late rate |
| `baseline_logisticRegression/` | Learned model 1 (simple baseline, **recommended**): logistic regression |
| `model_gradientBoosting/` | Learned model 2: histogram gradient boosted trees |
| `diagnosticAnalysis/` | Region breakdown, number-of-trees sensitivity and weekend checks (validation only) |
| `testResults/` | The single final test run of all four methods, and the saved test-period daily lists |
| `requirements.txt` | Python packages and versions |

Every method folder holds its own copy of `daily_eval.py`, the shared daily top-10 selection and scoring code.
The copies are identical, so each folder can be read and run on its own. Each folder's `README.md` lists its
inputs, outputs and how to run it.

### Files not included

- `dataset/`: the raw Olist data (see "Data" below)
- `dataPrep/output/train.csv`, `validation.csv`, `test.csv` (55 MB): the order-level tables built from the raw
  data. Step 2 below recreates them in about 2 minutes, and every later step needs them.
- The scripts and Overleaf sources that draw the report figures and build the report PDFs

Everything else is included: all code, the split summary, feature-availability audit and build log, every
saved model, configuration, tuning table, prediction file, daily list and figure.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use  source .venv/bin/activate  on Linux/macOS
python -m pip install -r requirements.txt
```

Everything runs on a CPU. No GPU is needed.

## Data

Put the nine CSV files of the Brazilian E-Commerce Public Dataset by Olist in `dataset/`:
`olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_order_payments_dataset.csv`,
`olist_order_reviews_dataset.csv`, `olist_customers_dataset.csv`, `olist_sellers_dataset.csv`,
`olist_products_dataset.csv`, `olist_geolocation_dataset.csv` and `product_category_name_translation.csv`.
Source: [Olist on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), CC BY-NC-SA 4.0.
The raw files are not modified.

## Method in brief

- **Rows:** orders approved from 2017-01-01 to 2018-07-31 (92,338). Earlier months are too sparse; after July
  2018, many promised dates fall after deliveries stop being recorded (2018-09-03), so late orders could not
  be observed.
- **Label:** late = delivered on a later calendar day than promised. Only delivered orders with consistent
  dates have a label (89,632). Canceled, unavailable, never-delivered and impossible-date orders stay in the
  daily candidate lists, and if picked they count as "unknown".
- **Features:** 44 built, 43 used (approval month is left out). Items and payments are summarized to one row per
  order before joining. History features (for example a seller's past late rate) only count orders delivered
  before the scored order was approved. See `dataPrep/output/feature_availability.csv`.
- **Split by approval date:** train 2017, validation 2018-01-01 to 04-15, test 2018-04-16 to 07-31.
  Train fits the models, validation chooses the settings, and the test period was scored once at the end.
- **Daily policy:** every approved order on a day is a candidate. Each method ranks them and the top 10 are
  investigated (all of them on days with fewer than 10 orders). Ties are broken by order ID.
- **Model settings:** chosen by the most validation hits at 10 per day (ties by validation PR-AUC).
  Logistic regression: C = 0.003, no class weighting (16 settings tried). Gradient boosting: learning rate 0.1,
  3 leaves, at least 200 orders per leaf, 300 trees (90 settings tried).

## Reproducing the results

Run from the project folder, in this order. Each step reads the outputs of the steps before it.

**1. Data exploration** (optional; not needed for the later steps)
```bash
python exploration/explore_data.py
python exploration/explore_time_split.py
```

**2. Build the modeling table** (about 2 minutes)
```bash
python dataPrep/build_dataset.py
```
Writes `dataPrep/output/train.csv`, `validation.csv`, `test.csv`, `split_summary.csv`,
`feature_availability.csv`, `feature_summary.csv` and `build_log.txt` (row counts and a leakage spot-check).

**3. The two business rules** (seconds each)
```bash
python nonML_distanceRule/run_distance_rule.py
python nonML_riskiestSellers/run_seller_rule.py
```

**4. The two learned models** (about 1 minute and 5 minutes)
```bash
python baseline_logisticRegression/train_logistic.py
python model_gradientBoosting/train_gradient_boosting.py
```
Each tunes on validation, saves `results/model.joblib` and `config.json`, and scores the daily top-10 on
train and validation. Steps 3 and 4 never read `test.csv`.

**5. Diagnostics** (validation only, about 2 minutes)
```bash
python diagnosticAnalysis/run_diagnostics.py
```

**6. Test period, once, after everything above is frozen**
```bash
python testResults/run_test.py
```
Loads the saved models without retraining and scores all four methods on the test period. The daily lists
are saved as `testResults/results/daily_list_test_<method>.csv`, one row per investigated order with its
order ID, approval date, rank, risk score and outcome.

## Results (10 investigations per day)

| Method | Validation hits | Test hits | Test wasted | Test unknown | Test missed | Test hit rate | Test vs random | Test PR-AUC |
|---|---|---|---|---|---|---|---|---|
| Perfect list | 956 | 560 | 459 | 0 | 280 | 55.0% | 15.8x | 1.000 |
| **Logistic regression** | 311 | **105** | 885 | 29 | 735 | **10.6%** | **2.96x** | **0.080** |
| Gradient boosting | 335 | 82 | 909 | 28 | 758 | 8.3% | 2.31x | 0.077 |
| Longest distance rule | 218 | 87 | 916 | 16 | 753 | 8.7% | 2.45x | 0.060 |
| Riskiest seller rule | 149 | 76 | 915 | 28 | 764 | 7.7% | 2.14x | 0.049 |
| Random list (expected) | 118 | 35 | 965 | 18 | 805 | 3.5% | 1.00x | 0.037 |

Validation: 105 days, 1,045 investigations, 2,831 late orders (11.9% late). Test: 103 days, 1,019
investigations, 840 late orders (3.7% late). Gradient boosting was the model chosen before the test run;
logistic regression did best on test and is the recommended model. Full tables: `testResults/results/summary_test.csv`
and each folder's `results/summary.csv`.

## Hardware and runtime

Windows laptop CPU, Python 3.14.3. The whole pipeline (steps 2 to 6) runs in about 10 minutes. Logistic
regression and gradient boosting are deterministic with the saved settings and `random_state=0`, so reruns
give the same results.

## Reused code and assistance

- Models: scikit-learn 1.9.1 (`LogisticRegression`, `HistGradientBoostingClassifier`, `ColumnTransformer`,
  `OneHotEncoder`, `OrdinalEncoder`, `SimpleImputer`, `StandardScaler`, `permutation_importance`). Data handling:
  pandas and NumPy. Figures: matplotlib.
- Everything else (label construction, leakage-safe history features, the daily top-10 policy and scoring,
  business rules, tuning, diagnostics, figures) was written for this project.
- AI assistance: Claude (Anthropic), through Claude Code, was used to help write and debug the code.
