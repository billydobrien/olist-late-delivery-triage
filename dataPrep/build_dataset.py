"""Build the modeling table: one row per approved order, with approval-time features,
the late/on-time label, and the train/validation/test split.

Run:   python dataPrep/build_dataset.py
Out:   dataPrep/output/train.csv, validation.csv, test.csv   (one row per order, by split)
       dataPrep/output/split_summary.csv         (rows, labeled, late, on time per split)
       dataPrep/output/feature_availability.csv  (feature-availability audit)
       dataPrep/output/feature_summary.csv       (missing values and ranges per column)
       dataPrep/output/build_log.txt             (row counts and checks)

Decisions (see exploration/ figures 37-44):
  * Decision time = order_approved_at. Every feature uses only information known then.
  * Rows = orders approved 2017-01-01 .. 2018-07-31. Earlier data is sparse; after July 2018
    late deliveries cannot be observed (deliveries stop being recorded on 2018-09-03).
  * Label: late = 1 if delivered on a later calendar day than the promised (estimated) date.
    Only "eligible" orders (status delivered, delivery date present) have a label;
    canceled / unavailable / never-delivered orders stay in the table with late = NaN.
  * Split by approval date: train 2017, validation 2018-01-01..04-15, test 2018-04-16..07-31.
  * Child tables (items, payments, products) are summarized to one row per order BEFORE joining.
  * Train orders whose label arrives after the train period ends (2,439 delivered on/after
    2018-01-01) are KEPT (decided 2026-11-06); noted as a limitation in the report.
  * History features count only earlier orders; late-rate history counts only orders that
    were DELIVERED before this order's approval time (strictly earlier).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "dataset"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2018-08-01")            # exclusive
VAL_START = pd.Timestamp("2018-01-01")
TEST_START = pd.Timestamp("2018-04-16")
SMOOTH_M = 20                                # pseudo-orders used to smooth history late rates
RECENT_DAYS = 14

LOG = []


def log(msg):
    print(msg)
    LOG.append(str(msg))


def read(name):
    return pd.read_csv(DATA / f"olist_{name}_dataset.csv")


# ============================================================ load
orders = read("orders")
items = read("order_items")
customers = read("customers")
sellers = read("sellers")
products = read("products")
payments = read("order_payments")
geo = read("geolocation")
cat_en = pd.read_csv(DATA / "product_category_name_translation.csv")

for c in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
          "order_delivered_customer_date", "order_estimated_delivery_date"]:
    orders[c] = pd.to_datetime(orders[c])
items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"])
log(f"raw orders: {len(orders):,}")

# ============================================================ order-level outcome (all orders; history uses them)
o = orders.merge(customers, on="customer_id", how="left", validate="one_to_one")
# impossible records: delivered before approval, or promised before approval -> label not trusted
o["outcome_issue"] = ((o.order_delivered_customer_date < o.order_approved_at)
                      | (o.order_estimated_delivery_date < o.order_approved_at)).astype(int)
o["eligible"] = (o.order_status == "delivered") & o.order_delivered_customer_date.notna() \
    & o.order_approved_at.notna() & (o.outcome_issue == 0)
o["days_late"] = (o.order_delivered_customer_date.dt.normalize()
                  - o.order_estimated_delivery_date.dt.normalize()).dt.days
o["late"] = np.where(o.eligible, (o.days_late > 0).astype(float), np.nan)

# ============================================================ items + products -> one row per order
prod = products.merge(cat_en, on="product_category_name", how="left")
prod["category"] = prod.product_category_name_english.fillna(prod.product_category_name).fillna("unknown")
prod["volume_cm3"] = prod.product_length_cm * prod.product_height_cm * prod.product_width_cm
it = items.merge(prod[["product_id", "product_weight_g", "volume_cm3", "category"]], on="product_id",
                 how="left", validate="many_to_one")
it = it.merge(sellers, on="seller_id", how="left", validate="many_to_one")
assert len(it) == len(items), "item join changed row count"

main_item = (it.sort_values(["order_id", "price", "order_item_id"], ascending=[True, False, True])
               .drop_duplicates("order_id").set_index("order_id"))
item_agg = it.groupby("order_id").agg(
    n_items=("order_item_id", "size"),
    n_products=("product_id", "nunique"),
    n_sellers=("seller_id", "nunique"),
    n_categories=("category", "nunique"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
    total_weight_g=("product_weight_g", lambda s: s.sum(min_count=len(s))),   # NaN if any item weight missing
    total_volume_cm3=("volume_cm3", lambda s: s.sum(min_count=len(s))),
    ship_limit_max=("shipping_limit_date", "max"),
)
item_agg["main_seller_id"] = main_item.seller_id
item_agg["main_category"] = main_item.category
item_agg["seller_zip_code_prefix"] = main_item.seller_zip_code_prefix
item_agg["seller_state"] = main_item.seller_state
item_agg["seller_city"] = main_item.seller_city

# ============================================================ payments -> one row per order
pay_main = (payments.sort_values(["order_id", "payment_value"], ascending=[True, False])
                    .drop_duplicates("order_id").set_index("order_id"))
pay_agg = payments.groupby("order_id").agg(n_payments=("payment_sequential", "size"),
                                           max_installments=("payment_installments", "max"),
                                           payment_value=("payment_value", "sum"))
pay_agg["payment_type"] = pay_main.payment_type

o = o.merge(item_agg, left_on="order_id", right_index=True, how="left", validate="one_to_one")
o = o.merge(pay_agg, left_on="order_id", right_index=True, how="left", validate="one_to_one")
assert o.order_id.is_unique and len(o) == len(orders), "order joins duplicated orders"
log("joins: one row per order confirmed (items and payments summarized first)")

# ============================================================ geolocation (zip-prefix centroids inside Brazil)
g = geo[geo.geolocation_lat.between(-34, 6) & geo.geolocation_lng.between(-74, -34)]
cent = g.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
o["customer_lat"] = o.customer_zip_code_prefix.map(cent.geolocation_lat)
o["customer_lng"] = o.customer_zip_code_prefix.map(cent.geolocation_lng)
o["seller_lat"] = o.seller_zip_code_prefix.map(cent.geolocation_lat)
o["seller_lng"] = o.seller_zip_code_prefix.map(cent.geolocation_lng)
la1, lo1, la2, lo2 = (np.radians(o[c]) for c in ["customer_lat", "customer_lng", "seller_lat", "seller_lng"])
hav = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
o["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(hav))
o["same_state"] = np.where(o.seller_state.isna(), np.nan, (o.seller_state == o.customer_state).astype(float))
o["same_city"] = np.where(o.seller_city.isna(), np.nan, (o.seller_city == o.customer_city).astype(float))

# ============================================================ simple approval-time features
t = o.order_approved_at
o["approval_date"] = t.dt.normalize()
o["approval_hour"] = t.dt.hour
o["approval_weekday"] = t.dt.day_of_week         # 0 = Monday
o["approval_month"] = t.dt.month
o["purchase_to_approval_h"] = (t - o.order_purchase_timestamp).dt.total_seconds() / 3600
o["promised_days"] = (o.order_estimated_delivery_date - t).dt.total_seconds() / 86400
o["seller_ship_limit_days"] = (o.ship_limit_max - t).dt.total_seconds() / 86400
o["freight_ratio"] = o.total_freight / o.total_price
o["has_items"] = o.n_items.notna().astype(int)

# ============================================================ history features (strictly earlier information)
appr = o[o.order_approved_at.notna()].copy()
appr = appr.sort_values(["order_approved_at", "order_id"])

# counts of earlier APPROVALS (known at approval time, no outcome needed)
appr["seller_prev_orders"] = appr.groupby("main_seller_id").cumcount()
appr.loc[appr.main_seller_id.isna(), "seller_prev_orders"] = np.nan
appr["customer_prev_orders"] = appr.groupby("customer_unique_id").cumcount()
at = appr.order_approved_at.values
appr["orders_last_24h"] = (np.searchsorted(at, at, side="left")
                           - np.searchsorted(at, at - np.timedelta64(24, "h"), side="left"))

# deliveries known before approval: event = (delivery time, late)
ev = o[o.eligible].copy()
ev["route"] = ev.seller_state + ">" + ev.customer_state
appr["route"] = appr.seller_state + ">" + appr.customer_state


def delivered_before(query_times, by=None, query_keys=None):
    """For each query, count deliveries strictly before the query time (optionally within the
    same key, e.g. same seller). Returns (n_delivered, n_late) arrays aligned with the query."""
    q = pd.DataFrame({"qt": query_times.values, "pos": np.arange(len(query_times))})
    e = ev[["order_delivered_customer_date", "late"]].rename(columns={"order_delivered_customer_date": "et"})
    if by is not None:
        q["key"] = query_keys.values
        e["key"] = ev[by].values
        e = e[e.key.notna()]
        e = e.sort_values(["et"])
        e["n_cum"] = e.groupby("key").cumcount() + 1
        e["late_cum"] = e.groupby("key").late.cumsum()
        qk = q[q.key.notna()].sort_values("qt")
        m = pd.merge_asof(qk, e[["et", "key", "n_cum", "late_cum"]], left_on="qt", right_on="et",
                          by="key", allow_exact_matches=False, direction="backward")
    else:
        e = e.sort_values("et")
        e["n_cum"] = np.arange(1, len(e) + 1)
        e["late_cum"] = e.late.cumsum()
        m = pd.merge_asof(q.sort_values("qt"), e[["et", "n_cum", "late_cum"]], left_on="qt", right_on="et",
                          allow_exact_matches=False, direction="backward")
    n_del = np.full(len(q), np.nan)
    n_late = np.full(len(q), np.nan)
    n_del[m.pos.values] = m.n_cum.fillna(0).values
    n_late[m.pos.values] = m.late_cum.fillna(0).values
    return n_del, n_late


T = appr.order_approved_at
g_n, g_late = delivered_before(T)
g_n14, g_late14 = delivered_before(T - pd.Timedelta(days=RECENT_DAYS))
prior = np.where(g_n > 0, g_late / np.maximum(g_n, 1), np.nan)        # overall late rate known at approval
appr["overall_prev_late_rate"] = prior
recent_n = g_n - g_n14
appr["recent_late_rate_14d"] = np.where(recent_n > 0, (g_late - g_late14) / np.maximum(recent_n, 1), np.nan)
appr["recent_deliveries_14d"] = recent_n

for name, by, key in [("seller", "main_seller_id", appr.main_seller_id),
                      ("route", "route", appr.route),
                      ("customer_state", "customer_state", appr.customer_state),
                      ("category", "main_category", appr.main_category)]:
    n_d, n_l = delivered_before(T, by=by, query_keys=key)
    appr[f"{name}_prev_delivered"] = n_d
    # smoothed late rate: shrinks toward the overall rate known at that time when there is little history
    appr[f"{name}_prev_late_rate"] = (n_l + SMOOTH_M * prior) / (n_d + SMOOTH_M)

hist_cols = ["seller_prev_orders", "customer_prev_orders", "orders_last_24h", "overall_prev_late_rate",
             "recent_late_rate_14d", "recent_deliveries_14d",
             "seller_prev_delivered", "seller_prev_late_rate", "route_prev_delivered", "route_prev_late_rate",
             "customer_state_prev_delivered", "customer_state_prev_late_rate",
             "category_prev_delivered", "category_prev_late_rate"]
o = o.merge(appr[["order_id"] + hist_cols], on="order_id", how="left", validate="one_to_one")

# ============================================================ keep the study window, assign split
before = len(o)
o_all = o.copy()          # every order, kept for the leakage spot-check below
o = o[o.order_approved_at.notna() & (o.order_approved_at >= START) & (o.order_approved_at < END)].copy()
log(f"rows kept (approved {START.date()} .. {(END - pd.Timedelta(days=1)).date()}): {len(o):,} "
    f"(dropped {before - len(o):,}: no approval date, before 2017, or Aug 2018 and later)")
o["split"] = np.select([o.order_approved_at < VAL_START, o.order_approved_at < TEST_START],
                       ["train", "validation"], "test")

# ============================================================ final column order
id_cols = ["order_id", "order_approved_at", "approval_date", "split", "customer_unique_id", "main_seller_id"]
outcome_cols = ["order_status", "order_delivered_customer_date", "order_estimated_delivery_date",
                "days_late", "outcome_issue", "eligible", "late"]
feature_cols = [
    # time
    "approval_hour", "approval_weekday", "approval_month", "purchase_to_approval_h", "promised_days",
    "seller_ship_limit_days",
    # order
    "has_items", "n_items", "n_products", "n_sellers", "total_price", "total_freight", "freight_ratio",
    # product
    "total_weight_g", "total_volume_cm3", "main_category", "n_categories",
    # payment
    "payment_type", "n_payments", "max_installments", "payment_value",
    # location
    "customer_state", "seller_state", "same_state", "same_city", "distance_km",
    "customer_lat", "customer_lng", "seller_lat", "seller_lng",
] + hist_cols
table = o[id_cols + outcome_cols + feature_cols].rename(columns={
    "order_delivered_customer_date": "delivered_date", "order_estimated_delivery_date": "estimated_date"})
table["eligible"] = table.eligible.astype(int)
table = table.sort_values(["order_approved_at", "order_id"]).reset_index(drop=True)
assert table.order_id.is_unique
old = OUT / "order_table.csv"
if old.exists():
    old.unlink()          # replaced by the three split files
for sp in ["train", "validation", "test"]:
    part = table[table.split == sp]
    part.to_csv(OUT / f"{sp}.csv", index=False)
    log(f"saved {sp}.csv: {len(part):,} rows x {part.shape[1]} columns")
log(f"({len(feature_cols)} feature columns; every approved order is kept so each day's full candidate "
    f"pool is available; only eligible == 1 rows have a late label)")

# ============================================================ split summary
summ = table.groupby("split", sort=False).agg(rows=("order_id", "size"), eligible=("eligible", "sum"),
                                              late=("late", "sum"))
summ["on_time"] = summ.eligible - summ.late
summ["late_rate"] = (summ.late / summ.eligible).round(4)
summ["pct_of_eligible"] = (summ.eligible / summ.eligible.sum()).round(4)
summ["first_day"] = table.groupby("split", sort=False).approval_date.min().dt.date
summ["last_day"] = table.groupby("split", sort=False).approval_date.max().dt.date
summ.to_csv(OUT / "split_summary.csv")
log("\nsplit summary:\n" + summ.to_string())
ne = table[table.eligible == 0]
log("\nrows without a usable label (kept as daily candidates, outcome unknown):\n"
    + ne.assign(reason=np.where(ne.outcome_issue == 1, "impossible dates", ne.order_status))
        .reason.value_counts().to_string())

# ============================================================ feature summary (missing values per split)
rows = []
for c in feature_cols:
    s = table[c]
    r = {"column": c, "dtype": str(s.dtype),
         **{f"missing_{sp}": round(table.loc[table.split == sp, c].isna().mean(), 4)
            for sp in ["train", "validation", "test"]}}
    if pd.api.types.is_numeric_dtype(s):
        r.update(min=s.min(), median=s.median(), max=s.max())
    else:
        r.update(n_levels=s.nunique())
    rows.append(r)
fs = pd.DataFrame(rows)
fs.to_csv(OUT / "feature_summary.csv", index=False)
log("\nfeature columns with any missing values:\n" +
    fs[(fs.filter(like="missing_") > 0).any(axis=1)][["column", "missing_train", "missing_validation",
                                                      "missing_test"]].to_string(index=False))

# ============================================================ feature-availability audit (required deliverable)
A = "known at approval"
audit = {
    "approval_hour": ("time", "orders.order_approved_at", A, "yes"),
    "approval_weekday": ("time", "orders.order_approved_at", A, "yes"),
    "approval_month": ("time", "orders.order_approved_at", A,
                       "no: training covers one year, so month would only memorize 2017 events"),
    "purchase_to_approval_h": ("time", "orders purchase + approval timestamps", A, "yes"),
    "promised_days": ("time", "orders.order_estimated_delivery_date",
                      "promised date is shown to the customer at purchase (assumption)", "yes"),
    "seller_ship_limit_days": ("time", "order_items.shipping_limit_date",
                               "set at approval: 79% equal approval time-of-day + whole days", "yes"),
    "has_items": ("order", "order_items", A, "yes"),
    "n_items": ("order", "order_items", A, "yes"),
    "n_products": ("order", "order_items", A, "yes"),
    "n_sellers": ("order", "order_items", A, "yes"),
    "total_price": ("order", "order_items.price", A, "yes"),
    "total_freight": ("order", "order_items.freight_value", A, "yes"),
    "freight_ratio": ("order", "derived", A, "yes"),
    "total_weight_g": ("product", "products", "catalog attribute", "yes"),
    "total_volume_cm3": ("product", "products", "catalog attribute", "yes"),
    "main_category": ("product", "products + translation", "catalog attribute", "yes"),
    "n_categories": ("product", "products", "catalog attribute", "yes"),
    "payment_type": ("payment", "order_payments", "payment is approved at this time", "yes"),
    "n_payments": ("payment", "order_payments", "payment is approved at this time", "yes"),
    "max_installments": ("payment", "order_payments", "payment is approved at this time", "yes"),
    "payment_value": ("payment", "order_payments", "payment is approved at this time", "yes"),
    "customer_state": ("location", "customers", "address given at purchase", "yes"),
    "seller_state": ("location", "sellers", "fixed seller attribute", "yes"),
    "same_state": ("location", "derived", A, "yes"),
    "same_city": ("location", "derived", A, "yes"),
    "distance_km": ("location", "geolocation (zip centroids)", "fixed lookup", "yes"),
    "customer_lat": ("location", "geolocation", "fixed lookup", "yes"),
    "customer_lng": ("location", "geolocation", "fixed lookup", "yes"),
    "seller_lat": ("location", "geolocation", "fixed lookup", "yes"),
    "seller_lng": ("location", "geolocation", "fixed lookup", "yes"),
}
hist_desc = "earlier records only (approvals strictly before; late rates from deliveries strictly before)"
for c in hist_cols:
    same_for_whole_day = c in ["orders_last_24h", "overall_prev_late_rate", "recent_late_rate_14d",
                               "recent_deliveries_14d"]
    audit[c] = ("history", "derived from all orders", hist_desc,
                "yes (nearly the same for every order on a day: cannot change the daily ranking alone)"
                if same_for_whole_day else "yes")
excluded = [
    ("order_delivered_carrier_date", "orders", "after approval (carrier pickup)", "never"),
    ("order_delivered_customer_date", "orders", "after approval; used only to build the label", "label only"),
    ("order_status", "orders", "final status known only later; used only for eligibility", "never"),
    ("review_score / comments / dates", "order_reviews", "written after delivery", "never"),
]
fa = pd.DataFrame([[c, *audit[c]] for c in feature_cols],
                  columns=["column", "group", "source", "why available at approval", "use as model input"])
fa = pd.concat([fa, pd.DataFrame([[c, "excluded", s, w, u] for c, s, w, u in excluded], columns=fa.columns)])
fa.to_csv(OUT / "feature_availability.csv", index=False)
assert set(feature_cols) <= set(audit), "a feature is missing from the availability audit"
log(f"\nsaved feature_availability.csv ({len(fa)} rows: every feature + excluded post-approval fields)")

# ============================================================ leakage spot-check (brute force on a sample)
sample = table[table.main_seller_id.notna()].sample(300, random_state=0)
elig = o_all[o_all.eligible]
bad = 0
for _, r in sample.iterrows():
    # late-rate history: only deliveries strictly before this approval
    prev = elig[(elig.main_seller_id == r.main_seller_id)
                & (elig.order_delivered_customer_date < r.order_approved_at)]
    allprev = elig[elig.order_delivered_customer_date < r.order_approved_at]
    prior_bf = allprev.late.mean()
    expect = (prev.late.sum() + SMOOTH_M * prior_bf) / (len(prev) + SMOOTH_M)
    if len(prev) != r.seller_prev_delivered or not np.isclose(expect, r.seller_prev_late_rate):
        bad += 1
    # order-count history: only approvals strictly before this one
    earlier = o_all[(o_all.main_seller_id == r.main_seller_id) & (o_all.order_approved_at < r.order_approved_at)]
    if r.seller_prev_orders != len(earlier):
        bad += 1
log(f"\nleakage spot-check on 300 random orders (recomputed seller history by brute force): "
    f"{'PASS' if bad == 0 else f'FAIL ({bad} mismatches)'}")

(OUT / "build_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
