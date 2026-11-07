# Exploration figures and tables

Made by `explore_data.py`. Working definition: **late** = delivered on a later calendar day than the promised date; rates use delivered orders with approval and delivery dates.

| # | File | What it shows |
|---|---|---|
| 1 | [fig01_table_datasets.png](figuresAndTables/fig01_table_datasets.png) | What each raw table is, its size, key, and when its information becomes known. |
| 2 | [fig02_table_join_multiplicity.png](figuresAndTables/fig02_table_join_multiplicity.png) | How many rows each order has in the child tables; why tables must be summarized to one row per order before joining. |
| 3 | [fig03_table_missing_values.png](figuresAndTables/fig03_table_missing_values.png) | Every column that has missing values, by table. |
| 4 | [fig04_table_quality_checks.png](figuresAndTables/fig04_table_quality_checks.png) | Inconsistent or impossible records found in the raw data. |
| 5 | [fig05_table_order_status.png](figuresAndTables/fig05_table_order_status.png) | Order statuses; only delivered orders with dates have a known late/on-time outcome. |
| 6 | [fig06_table_late_summary.png](figuresAndTables/fig06_table_late_summary.png) | How many orders are late and by how much. |
| 7 | [fig07_days_late_hist.png](figuresAndTables/fig07_days_late_hist.png) | Distribution of actual minus promised delivery day. |
| 8 | [fig08_orders_per_day.png](figuresAndTables/fig08_orders_per_day.png) | Number of orders approved each day, compared with the 10-per-day limit. |
| 9 | [fig09_late_over_time_weekly.png](figuresAndTables/fig09_late_over_time_weekly.png) | Weekly number of late orders and weekly late rate. |
| 10 | [fig10_table_monthly.png](figuresAndTables/fig10_table_monthly.png) | Month-by-month counts, share without an outcome, and late rate. |
| 11 | [fig11_late_rate_monthly.png](figuresAndTables/fig11_late_rate_monthly.png) | Monthly late rate (months with at least 100 approved orders). |
| 12 | [fig12_no_outcome_monthly.png](figuresAndTables/fig12_no_outcome_monthly.png) | Share of approved orders that were never delivered in the data (canceled, still shipping, etc.). |
| 13 | [fig13_table_daily_capacity.png](figuresAndTables/fig13_table_daily_capacity.png) | How 10 investigations per day compares with daily order and late-order counts. |
| 14 | [fig14_late_per_day_hist.png](figuresAndTables/fig14_late_per_day_hist.png) | Distribution of the number of eventually-late orders per approval day. |
| 15 | [fig15_by_weekday.png](figuresAndTables/fig15_by_weekday.png) | Orders and late rate by weekday of approval. |
| 16 | [fig16_by_hour.png](figuresAndTables/fig16_by_hour.png) | Orders and late rate by hour of day of approval. |
| 17 | [fig17_by_promised_window.png](figuresAndTables/fig17_by_promised_window.png) | Orders and late rate by how many days the customer was promised. |
| 18 | [fig18_by_approval_delay.png](figuresAndTables/fig18_by_approval_delay.png) | Orders and late rate by how long payment approval took. |
| 19 | [fig19_by_distance.png](figuresAndTables/fig19_by_distance.png) | Orders and late rate by straight-line distance between seller and customer zip areas. |
| 20 | [fig20_by_same_state.png](figuresAndTables/fig20_by_same_state.png) | Orders and late rate for same-state versus cross-state shipments. |
| 21 | [fig21_by_customer_state.png](figuresAndTables/fig21_by_customer_state.png) | Orders and late rate by the customer's state. |
| 22 | [fig22_by_seller_state.png](figuresAndTables/fig22_by_seller_state.png) | Orders and late rate by the main seller's state. |
| 23 | [fig23_by_weight.png](figuresAndTables/fig23_by_weight.png) | Orders and late rate by total weight of the products in the order. |
| 24 | [fig24_by_n_items.png](figuresAndTables/fig24_by_n_items.png) | Orders and late rate by item count. |
| 25 | [fig25_by_n_sellers.png](figuresAndTables/fig25_by_n_sellers.png) | Orders and late rate for single- versus multi-seller orders. |
| 26 | [fig26_by_price.png](figuresAndTables/fig26_by_price.png) | Orders and late rate by total item price. |
| 27 | [fig27_by_freight.png](figuresAndTables/fig27_by_freight.png) | Orders and late rate by total freight charged. |
| 28 | [fig28_by_payment_type.png](figuresAndTables/fig28_by_payment_type.png) | Orders and late rate by payment method. |
| 29 | [fig29_by_installments.png](figuresAndTables/fig29_by_installments.png) | Orders and late rate by installment count. |
| 30 | [fig30_by_category.png](figuresAndTables/fig30_by_category.png) | Orders and late rate by main product category (20 largest). |
| 31 | [fig31_by_seller_history.png](figuresAndTables/fig31_by_seller_history.png) | Orders and late rate by the seller's number of previous orders (counted only from earlier approvals). |
| 32 | [fig32_by_customer_history.png](figuresAndTables/fig32_by_customer_history.png) | Orders and late rate for new versus repeat customers (earlier orders only). |
| 33 | [fig33_seller_late_rate_hist.png](figuresAndTables/fig33_seller_late_rate_hist.png) | Spread of late rates across sellers (whole period, exploration only). |
| 34 | [fig34_after_review_score.png](figuresAndTables/fig34_after_review_score.png) | Late rate by review score (outcome-side information; leakage example). |
| 35 | [fig35_after_carrier_pickup.png](figuresAndTables/fig35_after_carrier_pickup.png) | Late rate by time to carrier pickup (after-approval information; leakage example). |
| 36 | [fig36_table_feature_availability.png](figuresAndTables/fig36_table_feature_availability.png) | Which information is known at approval (usable) versus only later (label or leakage). |
| 37 | [fig37_table_split_edges.png](figuresAndTables/fig37_table_split_edges.png) | Why the split covers approvals from 2017-01-01 to 2018-07-31 only. |
| 38 | [fig38_table_split_options.png](figuresAndTables/fig38_table_split_options.png) | Size, late count and late rate of train/validation/test under four candidate split dates. |
| 39 | [fig39_split_candidates_monthly.png](figuresAndTables/fig39_split_candidates_monthly.png) | Monthly on-time/late counts and late rate with the A/B/C split dates drawn in. |
| 40 | [fig40_table_split_recommended.png](figuresAndTables/fig40_table_split_recommended.png) | Per-period counts, late/on-time breakdown and daily late counts for the recommended split. |
| 41 | [fig41_table_label_delay.png](figuresAndTables/fig41_table_label_delay.png) | How many orders in each period were delivered (label known) only after the period ended. |
| 42 | [fig42_table_end_censoring.png](figuresAndTables/fig42_table_end_censoring.png) | Weekly counts near the end of the data, and the share of orders whose promised date is after deliveries stop being recorded (2018-09-03). |
| 43 | [fig43_split_recommended_timeline.png](figuresAndTables/fig43_split_recommended_timeline.png) | Weekly late rate with the train/validation/test periods shaded. |
| 44 | [fig44_table_order_columns.png](figuresAndTables/fig44_table_order_columns.png) | Every planned column of the modeling table with its source and role (ID, outcome, label, feature). |
