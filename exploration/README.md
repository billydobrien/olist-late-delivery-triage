# exploration: data audit and the choice of time split

Reads the raw CSV files in `../dataset/` and saves figures and tables (as PNG images) to `figuresAndTables/`.
`FIGURE_INDEX.md` lists every figure with one line on what it shows.

```bash
python exploration/explore_data.py         # figures 1-36: tables, joins, missing values, late rates over time and by feature
python exploration/explore_time_split.py   # figures 37-44: split options, end-of-record check, planned columns
```

- `explore_data.py`: table sizes, rows per order in each child table, missing values, data-quality checks,
  order status, late-delivery summary, orders and late rate over time, daily capacity, and the late rate by
  every candidate feature. Figures 34-35 show two post-approval signals (review score, carrier pickup) that
  are linked to lateness but cannot be used as inputs.
- `explore_time_split.py`: compares six candidate train/validation/test splits (option F, the rounded
  50/25/25 split, was chosen), shows why the data ends on 2018-07-31, and lists the planned model columns
  (also saved as `planned_order_columns.csv`).

These figures were made before the final data build removed 56 orders with impossible dates, so a few counts
differ slightly from the final ones. The official counts are in `dataPrep/output/split_summary.csv`.
