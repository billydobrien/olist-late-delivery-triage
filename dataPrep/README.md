# dataPrep: the one-row-per-order modeling table

`build_dataset.py` turns the nine raw tables in `../dataset/` into one row per approved order, with the late
label, 44 approval-time features and the train/validation/test split.

```bash
python dataPrep/build_dataset.py      # about 2 minutes
```

Outputs in `output/`:

| File | Contents |
|---|---|
| `train.csv`, `validation.csv`, `test.csv` | One row per order approved in each period (44,973 / 24,510 / 22,855 rows, 57 columns): IDs, approval time, outcome columns, `eligible`, `late`, and the 44 features |
| `split_summary.csv` | Rows, labeled orders, late, on time and late rate per period |
| `feature_availability.csv` | The feature-availability audit: every feature's source, why it is known at approval, and whether models use it |
| `feature_summary.csv` | Missing-value share per period and value ranges for every feature |
| `build_log.txt` | Row counts, join checks, orders without a label by reason, and the leakage spot-check result |

Key rules (see the comments at the top of `build_dataset.py`):

- Rows are orders approved 2017-01-01 to 2018-07-31. The order items and payments tables are summarized to one
  row per order before joining, and the script checks that every order still appears exactly once.
- `late` = 1 if delivered on a later calendar day than the promised date. It is only set when `eligible` = 1
  (delivered, both dates present, no impossible dates). Other orders keep `late` empty but stay in the files,
  because they are still daily candidates.
- History features count only earlier approvals, and late rates count only orders delivered strictly before
  this order's approval. Sparse histories are pulled toward the overall late rate known at that time
  (20 pseudo-orders). The log reports a brute-force recheck on 300 random orders.
- `approval_month` is built but marked "do not use" in the audit, because training covers only one year.
- Training orders from 2017 that were delivered in 2018 (2,439) are kept.
