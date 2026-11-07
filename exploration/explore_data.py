"""Explore the Olist tables and save simple figures and tables (as PNGs).

Run from anywhere:
    python exploration/explore_data.py
Outputs go to exploration/figuresAndTables/figNN_name.png

Working definition used only for exploring (not final):
    late = delivered to the customer on a later calendar day than the promised
           (estimated) delivery date.
    Rates use orders with status "delivered" that have both an approval date
    and a customer delivery date.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "dataset"
OUT = HERE / "figuresAndTables"
OUT.mkdir(exist_ok=True)

# Colors used everywhere
ONTIME = "#2a78d6"   # blue  = on time / counts
LATE = "#eb6834"     # orange = late
GRAY = "#6b6b6b"
AFTER = "#9b9b9b"    # information only known after approval

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "-", "axes.axisbelow": True, "figure.dpi": 150,
})

FIG_NUM = [0]
INDEX = []


def next_name(slug):
    FIG_NUM[0] += 1
    return FIG_NUM[0], f"fig{FIG_NUM[0]:02d}_{slug}.png"


def save(fig, slug, what, bbox="tight"):
    n, name = next_name(slug)
    fig.savefig(OUT / name, bbox_inches=bbox, facecolor="white")
    plt.close(fig)
    INDEX.append((n, name, what))
    print("saved", name)


def table_fig(df, title, slug, what, note=None, col_align=None):
    """Draw a DataFrame as a clean table image."""
    nrows, ncols = df.shape
    fig_h = 0.9 + 0.32 * (nrows + 1) + (0.3 if note else 0)
    fig, ax = plt.subplots(figsize=(min(2.0 + 1.6 * ncols, 13), fig_h))
    ax.axis("off")
    cells = df.astype(str).values.tolist()
    tbl = ax.table(cellText=cells, colLabels=list(df.columns), loc="upper center",
                   cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.35)
    tbl.auto_set_column_width(list(range(ncols)))
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        elif r % 2 == 0:
            cell.set_facecolor("#f8f8f8")
    # place title above and note below the table's own left edge, then crop to them
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tb = tbl.get_window_extent(r).transformed(ax.transAxes.inverted())
    up = mtrans.offset_copy(ax.transAxes, fig=fig, y=6, units="points")
    down = mtrans.offset_copy(ax.transAxes, fig=fig, y=-6, units="points")
    arts = [tbl, ax.text(tb.x0, tb.y1, f"Figure {FIG_NUM[0] + 1}. {title}", transform=up,
                         ha="left", va="bottom", fontweight="bold", fontsize=12)]
    if note:
        arts.append(ax.text(tb.x0, tb.y0, note, transform=down, ha="left", va="top", fontsize=9, color=GRAY))
    fig.canvas.draw()
    bb = mtrans.Bbox.union([a.get_window_extent(r) for a in arts])
    save(fig, slug, what, bbox=bb.transformed(fig.dpi_scale_trans.inverted()).padded(0.12))


def title(ax, text):
    ax.set_title(f"Figure {FIG_NUM[0] + 1}. {text}", loc="left", fontweight="bold")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def fmt_int(x):
    return f"{int(x):,}"


# ---------------------------------------------------------------- load
def load(name):
    return pd.read_csv(DATA / f"olist_{name}_dataset.csv")


orders = load("orders")
items = load("order_items")
customers = load("customers")
sellers = load("sellers")
products = load("products")
payments = load("order_payments")
reviews = load("order_reviews")
geo = load("geolocation")
cat_en = pd.read_csv(DATA / "product_category_name_translation.csv")

date_cols = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
             "order_delivered_customer_date", "order_estimated_delivery_date"]
for c in date_cols:
    orders[c] = pd.to_datetime(orders[c])
items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"])
reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"])

# ---------------------------------------------------------------- order-level frame
o = orders.copy()
o["has_outcome"] = (o.order_status == "delivered") & o.order_delivered_customer_date.notna() \
    & o.order_approved_at.notna()
o["days_late"] = (o.order_delivered_customer_date.dt.normalize()
                  - o.order_estimated_delivery_date.dt.normalize()).dt.days
o["late"] = np.where(o.has_outcome, (o.days_late > 0).astype(float), np.nan)
o["approve_day"] = o.order_approved_at.dt.normalize()
o["promised_days"] = (o.order_estimated_delivery_date - o.order_approved_at).dt.total_seconds() / 86400
o["approval_delay_h"] = (o.order_approved_at - o.order_purchase_timestamp).dt.total_seconds() / 3600
o["carrier_days"] = (o.order_delivered_carrier_date - o.order_approved_at).dt.total_seconds() / 86400

# items + products (weight, category)
prod = products.merge(cat_en, on="product_category_name", how="left")
prod["category"] = prod.product_category_name_english.fillna(prod.product_category_name).fillna("unknown")
it = items.merge(prod[["product_id", "product_weight_g", "category"]], on="product_id", how="left")
it = it.merge(sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]], on="seller_id", how="left")
# main item = most expensive item in the order
main = it.sort_values(["order_id", "price"], ascending=[True, False]).drop_duplicates("order_id")
agg = it.groupby("order_id").agg(
    n_items=("order_item_id", "size"), n_sellers=("seller_id", "nunique"),
    n_products=("product_id", "nunique"), total_price=("price", "sum"),
    total_freight=("freight_value", "sum"), total_weight_g=("product_weight_g", "sum"))
agg = agg.join(main.set_index("order_id")[["seller_id", "seller_zip_code_prefix", "seller_state", "category"]])
o = o.merge(agg, left_on="order_id", right_index=True, how="left")
o = o.merge(customers, on="customer_id", how="left")

# distance between seller and customer (zip-prefix centroids, Brazil bounds only)
g = geo[geo.geolocation_lat.between(-34, 6) & geo.geolocation_lng.between(-74, -34)]
cent = g.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
cl = cent.reindex(o.customer_zip_code_prefix).values
sl = cent.reindex(o.seller_zip_code_prefix).values
lat1, lon1, lat2, lon2 = map(np.radians, [cl[:, 0], cl[:, 1], sl[:, 0], sl[:, 1]])
a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
o["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))
o["same_state"] = np.where(o.seller_state.isna(), np.nan, (o.seller_state == o.customer_state).astype(float))

# payments
pmain = payments.sort_values(["order_id", "payment_value"], ascending=[True, False]).drop_duplicates("order_id")
pagg = payments.groupby("order_id").agg(n_payments=("payment_sequential", "size"),
                                         installments=("payment_installments", "max"))
pagg = pagg.join(pmain.set_index("order_id")[["payment_type"]])
o = o.merge(pagg, left_on="order_id", right_index=True, how="left")

# reviews (after delivery; outcome side only)
ragg = reviews.groupby("order_id").review_score.mean()
o["review_score"] = o.order_id.map(ragg)

# seller history: number of EARLIER approved orders by the same seller (no future info)
oh = o[o.order_approved_at.notna() & o.seller_id.notna()].sort_values("order_approved_at")
o.loc[oh.index, "seller_prev_orders"] = oh.groupby("seller_id").cumcount()
oc = o[o.order_approved_at.notna()].sort_values("order_approved_at")
o.loc[oc.index, "customer_prev_orders"] = oc.groupby("customer_unique_id").cumcount()

d = o[o.has_outcome].copy()          # orders with a known outcome
LATE_ALL = d.late.mean()

# ================================================================ TABLES: data audit
when = {"orders": "both (dates up to approval = input; delivery dates = outcome)",
        "order_items": "approval", "customers": "approval", "sellers": "approval",
        "products": "approval", "order_payments": "approval",
        "order_reviews": "after delivery (outcome side)", "geolocation": "approval (fixed lookup)",
        "category_translation": "approval (fixed lookup)"}
keys = {"orders": "order_id", "order_items": "order_id + order_item_id", "customers": "customer_id",
        "sellers": "seller_id", "products": "product_id", "order_payments": "order_id + payment_sequential",
        "order_reviews": "review_id / order_id", "geolocation": "zip prefix (many rows each)",
        "category_translation": "category name"}
tabs = {"orders": orders, "order_items": items, "customers": customers, "sellers": sellers,
        "products": products, "order_payments": payments, "order_reviews": reviews,
        "geolocation": geo, "category_translation": cat_en}
rows = []
for k, df in tabs.items():
    rows.append([k, fmt_int(len(df)), df.shape[1], keys[k], pct(df.isna().mean().mean()), when[k]])
table_fig(pd.DataFrame(rows, columns=["Table", "Rows", "Columns", "Key", "Missing cells", "When known"]),
          "The nine provided tables", "table_datasets",
          "What each raw table is, its size, key, and when its information becomes known.")

# join multiplicity
rows = []
for k, df in [("order_items", items), ("order_payments", payments), ("order_reviews", reviews)]:
    c = df.groupby("order_id").size()
    rows.append([k, fmt_int(c.size), f"{c.mean():.2f}", c.max(), pct((c > 1).mean()),
                 fmt_int(len(orders) - orders.order_id.isin(df.order_id).sum())])
gc = geo.groupby("geolocation_zip_code_prefix").size()
rows.append(["geolocation (per zip)", fmt_int(gc.size), f"{gc.mean():.1f}", gc.max(), pct((gc > 1).mean()), "-"])
table_fig(pd.DataFrame(rows, columns=["Table", "Orders (keys) present", "Rows per order (mean)",
                                      "Max rows", "Orders with >1 row", "Orders missing here"]),
          "Rows per order: joining these directly would duplicate orders", "table_join_multiplicity",
          "How many rows each order has in the child tables; why tables must be summarized to one row per order before joining.",
          note="Each child table is first summarized to one row per order (sum, count, or main item) and then joined.")

# missing values (non-zero only)
rows = []
for k in ["orders", "order_items", "products", "order_payments", "order_reviews", "customers", "sellers"]:
    df = tabs[k]
    for c in df.columns:
        m = df[c].isna().sum()
        if m:
            rows.append([k, c, fmt_int(m), pct(m / len(df), 2)])
table_fig(pd.DataFrame(rows, columns=["Table", "Column", "Missing", "Missing %"]),
          "Columns with missing values", "table_missing_values",
          "Every column that has missing values, by table.")

# data quality checks
cz = customers.customer_zip_code_prefix.isin(cent.index)
sz = sellers.seller_zip_code_prefix.isin(cent.index)
dl = orders[orders.order_status == "delivered"]
checks = [
    ["Delivered orders without a customer delivery date", fmt_int(dl.order_delivered_customer_date.isna().sum())],
    ["Delivered orders without an approval date", fmt_int(dl.order_approved_at.isna().sum())],
    ["Orders approved before purchase", fmt_int((orders.order_approved_at < orders.order_purchase_timestamp).sum())],
    ["Handed to carrier before approval", fmt_int((orders.order_delivered_carrier_date < orders.order_approved_at).sum())],
    ["Delivered to customer before carrier pickup", fmt_int((orders.order_delivered_customer_date < orders.order_delivered_carrier_date).sum())],
    ["Delivered to customer before approval", fmt_int((orders.order_delivered_customer_date < orders.order_approved_at).sum())],
    ["Orders with no items", fmt_int((~orders.order_id.isin(items.order_id)).sum())],
    ["Orders with no payment record", fmt_int((~orders.order_id.isin(payments.order_id)).sum())],
    ["Products with no category", fmt_int(products.product_category_name.isna().sum())],
    ["Products with weight 0 g", fmt_int((products.product_weight_g == 0).sum())],
    ["Customer zip prefixes not in geolocation", f"{fmt_int((~cz).sum())} ({pct((~cz).mean())})"],
    ["Seller zip prefixes not in geolocation", f"{fmt_int((~sz).sum())} ({pct((~sz).mean())})"],
    ["Geolocation rows outside Brazil (dropped)", fmt_int(len(geo) - len(g))],
    ["Customers (customer_id) per real person (unique_id)", f"{len(customers) / customers.customer_unique_id.nunique():.2f}"],
]
table_fig(pd.DataFrame(checks, columns=["Check", "Count"]), "Data quality checks", "table_quality_checks",
          "Inconsistent or impossible records found in the raw data.")

# status & outcome availability
st = o.groupby("order_status").agg(orders=("order_id", "size"),
                                   approved=("order_approved_at", lambda s: s.notna().sum()),
                                   delivered_date=("order_delivered_customer_date", lambda s: s.notna().sum()),
                                   outcome=("has_outcome", "sum")).sort_values("orders", ascending=False)
st.loc["total"] = st.sum()
st_t = pd.DataFrame({"Status": st.index, "Orders": st.orders.map(fmt_int),
                     "% of all": (st.orders / len(o)).map(pct),
                     "Have approval date": st.approved.map(fmt_int),
                     "Have delivery date": st.delivered_date.map(fmt_int),
                     "Outcome known (late / on time)": st.outcome.map(fmt_int)})
table_fig(st_t, "Order status and whether the delivery outcome is known", "table_order_status",
          "Order statuses; only delivered orders with dates have a known late/on-time outcome.",
          note="Canceled, unavailable, shipped, etc. orders never reached the customer in the data, so late/on time cannot be decided for them.")

# late summary
dlt = d[d.late == 1].days_late
early = -d[d.late == 0].days_late
summ = [
    ["Orders with known outcome", fmt_int(len(d))],
    ["On time (delivered on or before promised day)", f"{fmt_int((d.late == 0).sum())} ({pct(1 - LATE_ALL)})"],
    ["Late (delivered after promised day)", f"{fmt_int((d.late == 1).sum())} ({pct(LATE_ALL)})"],
    ["Late orders: median days late", f"{dlt.median():.0f}"],
    ["Late orders: mean days late", f"{dlt.mean():.1f}"],
    ["Late orders: 90th percentile days late", f"{dlt.quantile(0.9):.0f}"],
    ["Late orders: maximum days late", f"{dlt.max():.0f}"],
    ["On-time orders: median days early", f"{early.median():.0f}"],
    ["Late by 1-3 days", pct(dlt.between(1, 3).mean()) + " of late orders"],
    ["Late by more than 7 days", pct((dlt > 7).mean()) + " of late orders"],
]
table_fig(pd.DataFrame(summ, columns=["Measure", "Value"]), "Late delivery summary", "table_late_summary",
          "How many orders are late and by how much.")

# ================================================================ FIGURES: outcome
fig, ax = plt.subplots(figsize=(8, 4))
x = d.days_late.clip(-40, 40)
bins = np.arange(-40.5, 41.5, 1)
ax.hist(x[x <= 0], bins=bins, color=ONTIME, label="on time")
ax.hist(x[x > 0], bins=bins, color=LATE, label="late")
ax.axvline(0.5, color="black", lw=1)
ax.set_xlabel("Delivered date minus promised date (days, clipped at +/-40)")
ax.set_ylabel("Orders")
ax.legend(frameon=False)
title(ax, "Most orders arrive well before the promised date")
save(fig, "days_late_hist", "Distribution of actual minus promised delivery day.")

# ================================================================ TIME
daily = o[o.order_approved_at.notna()].groupby("approve_day").agg(
    orders=("order_id", "size"), outcome=("has_outcome", "sum"), late=("late", "sum"))
daily = daily.asfreq("D", fill_value=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(daily.index, daily.orders, color=ONTIME, lw=0.7, alpha=0.5, label="per day")
ax.plot(daily.index, daily.orders.rolling(7, center=True).mean(), color=ONTIME, lw=2, label="7-day average")
ax.axhline(10, color=LATE, ls="--", lw=1.2, label="10 per day (investigation limit)")
ax.set_ylabel("Approved orders")
ax.legend(frameon=False, loc="upper left")
title(ax, "Approved orders per day over time")
save(fig, "orders_per_day", "Number of orders approved each day, compared with the 10-per-day limit.")

wk = o[o.has_outcome].set_index("order_approved_at").resample("W").agg({"late": ["sum", "mean", "size"]})
wk.columns = ["late", "rate", "n"]
wk.loc[wk.n < 50, ["late", "rate"]] = np.nan   # gaps stay gaps instead of being joined by a line
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].bar(wk.index, wk.late, width=6, color=LATE)
axs[0].set_ylabel("Late orders per week")
axs[1].plot(wk.index, 100 * wk.rate, color=LATE, lw=2, marker="o", ms=3)
axs[1].axhline(100 * LATE_ALL, color=GRAY, ls="--", lw=1, label=f"overall {pct(LATE_ALL)}")
axs[1].set_ylabel("Late rate (%)")
axs[1].legend(frameon=False)
title(axs[0], "Late orders over time (by week of approval)")
fig.text(0.01, -0.01, "Weeks with fewer than 50 orders with known outcome are not shown.", fontsize=9, color=GRAY)
save(fig, "late_over_time_weekly", "Weekly number of late orders and weekly late rate.")

# monthly table
o["month"] = o.order_approved_at.dt.to_period("M")
mo = o[o.month.notna()].groupby("month").agg(approved=("order_id", "size"), outcome=("has_outcome", "sum"),
                                               late=("late", "sum"))
mo["rate"] = mo.late / mo.outcome
mt = pd.DataFrame({"Month": mo.index.astype(str), "Approved": mo.approved.map(fmt_int),
                   "Outcome known": mo.outcome.map(fmt_int),
                   "No outcome": ((mo.approved - mo.outcome) / mo.approved).map(pct),
                   "Late": mo.late.map(fmt_int),
                   "Late rate": mo.rate.map(lambda v: pct(v) if np.isfinite(v) else "-")})
table_fig(mt, "Orders and late rate by month of approval", "table_monthly",
          "Month-by-month counts, share without an outcome, and late rate.")

fig, ax = plt.subplots(figsize=(10, 4))
mm = mo[mo.approved >= 100]
ax.bar(mm.index.astype(str), 100 * mm.rate, color=LATE)
ax.axhline(100 * LATE_ALL, color=GRAY, ls="--", lw=1, label=f"overall {pct(LATE_ALL)}")
ax.set_ylabel("Late rate (%)")
ax.tick_params(axis="x", rotation=60)
ax.legend(frameon=False)
title(ax, "Late rate changes a lot from month to month")
save(fig, "late_rate_monthly", "Monthly late rate (months with at least 100 approved orders).")

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(mm.index.astype(str), 100 * (mm.approved - mm.outcome) / mm.approved, color=GRAY)
ax.set_ylabel("Approved orders with no outcome (%)")
ax.tick_params(axis="x", rotation=60)
title(ax, "Share of approved orders whose delivery outcome is unknown")
save(fig, "no_outcome_monthly", "Share of approved orders that were never delivered in the data (canceled, still shipping, etc.).")

# daily capacity
dd = daily[daily.orders > 0]
cap = [
    ["Days with at least one approved order", fmt_int(len(dd))],
    ["Approved orders per day: median", f"{dd.orders.median():.0f}"],
    ["Approved orders per day: 10th to 90th percentile", f"{dd.orders.quantile(0.1):.0f} to {dd.orders.quantile(0.9):.0f}"],
    ["Days with fewer than 10 approved orders", f"{fmt_int((dd.orders < 10).sum())} ({pct((dd.orders < 10).mean())})"],
    ["Late orders per day: median", f"{dd.late.median():.0f}"],
    ["Days with 10 or more late orders", f"{fmt_int((dd.late >= 10).sum())} ({pct((dd.late >= 10).mean())})"],
    ["Days with zero late orders", f"{fmt_int((dd.late == 0).sum())} ({pct((dd.late == 0).mean())})"],
    ["Share of approved orders that 10/day could cover", pct(10 * len(dd) / dd.orders.sum())],
]
table_fig(pd.DataFrame(cap, columns=["Measure", "Value"]), "Daily investigation capacity versus daily volume",
          "table_daily_capacity", "How 10 investigations per day compares with daily order and late-order counts.")

fig, ax = plt.subplots(figsize=(8, 4))
ld = dd.late.clip(upper=60)
ax.hist(ld, bins=np.arange(-0.5, 61.5, 1), color=LATE)
ax.axvline(10, color="black", ls="--", lw=1.2, label="10 per day")
ax.set_xlabel("Late orders among one day's approvals (clipped at 60)")
ax.set_ylabel("Days")
ax.legend(frameon=False)
title(ax, "How many late orders each day has")
save(fig, "late_per_day_hist", "Distribution of the number of eventually-late orders per approval day.")


# ================================================================ BINNED FEATURES
def count_and_rate(series, labels_order, xlabel, ttl, slug, what, horizontal=False, min_n=30, note=None,
                   color_rate=LATE):
    df = pd.DataFrame({"g": series, "late": d.late}).dropna(subset=["g"])
    grp = df.groupby("g", observed=True).late.agg(["size", "mean"])
    if labels_order is not None:
        grp = grp.reindex([l for l in labels_order if l in grp.index])
    grp = grp[grp["size"] >= min_n]
    lab = [str(i) for i in grp.index]
    if horizontal:
        fig, axs = plt.subplots(1, 2, figsize=(11, 0.33 * len(grp) + 1.5), sharey=True)
        y = np.arange(len(grp))
        axs[0].barh(y, grp["size"], color=ONTIME)
        axs[1].barh(y, 100 * grp["mean"], color=color_rate)
        axs[1].axvline(100 * LATE_ALL, color=GRAY, ls="--", lw=1)
        axs[0].set_yticks(y, lab)
        axs[0].invert_yaxis()
        axs[0].set_xlabel("Orders")
        axs[1].set_xlabel("Late rate (%)")
    else:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        axs[0].bar(lab, grp["size"], color=ONTIME)
        axs[1].bar(lab, 100 * grp["mean"], color=color_rate)
        axs[1].axhline(100 * LATE_ALL, color=GRAY, ls="--", lw=1, label=f"overall {pct(LATE_ALL)}")
        axs[1].set_ylim(0, 1.25 * max(100 * grp["mean"].max(), 100 * LATE_ALL))
        axs[1].legend(frameon=False, loc="upper right")
        for ax in axs:
            ax.set_xlabel(xlabel)
            if len(lab) > 12:   # many bars (e.g. hours): label every third one
                ax.set_xticks(range(0, len(lab), 3), lab[::3])
            elif sum(len(s) for s in lab) > 45:
                ax.tick_params(axis="x", rotation=35)
                plt.setp(ax.get_xticklabels(), ha="right")
        axs[0].set_ylabel("Orders")
        axs[1].set_ylabel("Late rate (%)")
    axs[0].set_title("Number of orders", loc="left")
    axs[1].set_title("Share that arrived late", loc="left")
    fig.suptitle(f"Figure {FIG_NUM[0] + 1}. {ttl}", x=0.01, ha="left", fontweight="bold")
    if note:
        fig.text(0.01, -0.02, note, fontsize=9, color=GRAY)
    fig.tight_layout()
    save(fig, slug, what)


def binned(col, edges, labels):
    return pd.cut(d[col], edges, labels=labels, right=False)


count_and_rate(d.order_approved_at.dt.day_name().str[:3], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
               "Day of approval", "Late rate by day of the week", "by_weekday",
               "Orders and late rate by weekday of approval.")
count_and_rate(d.order_approved_at.dt.hour, list(range(24)), "Hour of approval", "Late rate by hour of approval",
               "by_hour", "Orders and late rate by hour of day of approval.")
count_and_rate(binned("promised_days", [0, 10, 15, 20, 25, 30, 40, 1e9], ["<10", "10-15", "15-20", "20-25", "25-30", "30-40", "40+"]),
               None, "Days from approval to promised date", "Late rate by length of the promised delivery window",
               "by_promised_window", "Orders and late rate by how many days the customer was promised.")
count_and_rate(binned("approval_delay_h", [0, 1, 6, 24, 48, 1e9], ["<1 h", "1-6 h", "6-24 h", "1-2 days", ">2 days"]),
               None, "Time from purchase to approval", "Late rate by time from purchase to approval",
               "by_approval_delay", "Orders and late rate by how long payment approval took.")
count_and_rate(binned("distance_km", [0, 50, 200, 500, 1000, 2000, 1e9], ["<50", "50-200", "200-500", "500-1000", "1000-2000", "2000+"]),
               None, "Seller to customer distance (km)", "Late rate by seller-to-customer distance",
               "by_distance", "Orders and late rate by straight-line distance between seller and customer zip areas.",
               note="Distance uses the average coordinates of each zip prefix; orders with an unknown zip are left out.")
count_and_rate(d.same_state.map({1.0: "same state", 0.0: "different state"}), ["same state", "different state"],
               "", "Late rate when seller and customer are in the same state or not", "by_same_state",
               "Orders and late rate for same-state versus cross-state shipments.")
cs = d.groupby("customer_state").late.mean().sort_values(ascending=False).index
count_and_rate(d.customer_state, list(cs), "", "Late rate by customer state (sorted by late rate)", "by_customer_state",
               "Orders and late rate by the customer's state.", horizontal=True)
ss = d.groupby("seller_state").late.mean().sort_values(ascending=False).index
count_and_rate(d.seller_state, list(ss), "", "Late rate by seller state (states with 30+ orders)", "by_seller_state",
               "Orders and late rate by the main seller's state.", horizontal=True)
count_and_rate(binned("total_weight_g", [0, 500, 1000, 2000, 5000, 10000, 20000, 1e9],
                      ["<0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20", "20+"]),
               None, "Total order weight (kg)", "Late rate by total product weight", "by_weight",
               "Orders and late rate by total weight of the products in the order.")
count_and_rate(d.n_items.clip(upper=5).map(lambda v: "5+" if v >= 5 else str(int(v))), ["1", "2", "3", "4", "5+"],
               "Items in the order", "Late rate by number of items in the order", "by_n_items",
               "Orders and late rate by item count.")
count_and_rate(d.n_sellers.map(lambda v: "1" if v == 1 else "2+"), ["1", "2+"], "Sellers in the order",
               "Late rate by number of different sellers in the order", "by_n_sellers",
               "Orders and late rate for single- versus multi-seller orders.")
count_and_rate(binned("total_price", [0, 50, 100, 200, 500, 1e9], ["<50", "50-100", "100-200", "200-500", "500+"]),
               None, "Order price (BRL)", "Late rate by order price", "by_price", "Orders and late rate by total item price.")
count_and_rate(binned("total_freight", [0, 10, 15, 20, 30, 50, 1e9], ["<10", "10-15", "15-20", "20-30", "30-50", "50+"]),
               None, "Freight cost (BRL)", "Late rate by freight (shipping) cost", "by_freight",
               "Orders and late rate by total freight charged.")
count_and_rate(d.payment_type, None, "Main payment type", "Late rate by payment type", "by_payment_type",
               "Orders and late rate by payment method.")
count_and_rate(pd.cut(d.installments, [0, 1, 3, 6, 10, 1e9], labels=["1", "2-3", "4-6", "7-10", "11+"]),
               None, "Credit installments", "Late rate by number of payment installments", "by_installments",
               "Orders and late rate by installment count.")
top = d.category.value_counts().head(20).index
cat_order = d[d.category.isin(top)].groupby("category").late.mean().sort_values(ascending=False).index
count_and_rate(d.category.where(d.category.isin(top)), list(cat_order), "",
               "Late rate for the 20 most common product categories", "by_category",
               "Orders and late rate by main product category (20 largest).", horizontal=True)
count_and_rate(pd.cut(d.seller_prev_orders, [-1, 0, 9, 49, 199, 999, 1e9],
                      labels=["0 (new)", "1-9", "10-49", "50-199", "200-999", "1000+"]),
               None, "Seller's earlier orders", "Late rate by how many earlier orders the seller had",
               "by_seller_history", "Orders and late rate by the seller's number of previous orders (counted only from earlier approvals).")
count_and_rate(d.customer_prev_orders.map(lambda v: "first order" if v == 0 else "returning"),
               ["first order", "returning"], "", "Late rate for first-time versus returning customers",
               "by_customer_history", "Orders and late rate for new versus repeat customers (earlier orders only).")

# seller-level spread
sv = d.groupby("seller_id").late.agg(["size", "mean"])
sv = sv[sv["size"] >= 30]
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(100 * sv["mean"], bins=np.arange(0, 52, 2), color=LATE)
ax.axvline(100 * LATE_ALL, color=GRAY, ls="--", lw=1, label=f"overall {pct(LATE_ALL)}")
ax.set_xlabel("Seller's late rate (%)")
ax.set_ylabel("Sellers")
ax.legend(frameon=False)
title(ax, f"Late rate differs across sellers ({len(sv)} sellers with 30+ orders)")
save(fig, "seller_late_rate_hist", "Spread of late rates across sellers (whole period, exploration only).")

# ================================================================ AFTER-APPROVAL INFO (cannot be inputs)
count_and_rate(d.review_score.round().map(lambda v: str(int(v)) if np.isfinite(v) else np.nan),
               ["1", "2", "3", "4", "5"], "Review score (stars)",
               "Reviews are written after delivery: strongly tied to lateness, NOT usable as an input",
               "after_review_score", "Late rate by review score (outcome-side information; leakage example).",
               color_rate=AFTER)
count_and_rate(binned("carrier_days", [-1e9, 1, 2, 3, 5, 7, 1e9], ["<1", "1-2", "2-3", "3-5", "5-7", "7+"]),
               None, "Days from approval to carrier pickup",
               "Carrier pickup happens after approval: NOT usable as an input", "after_carrier_pickup",
               "Late rate by time to carrier pickup (after-approval information; leakage example).",
               color_rate=AFTER)

# ================================================================ feature availability (draft)
fa = [
    ["Purchase and approval time, weekday, hour", "orders", "Yes", "known at approval"],
    ["Promised (estimated) delivery date / window", "orders", "Yes", "shown to customer at purchase"],
    ["Carrier pickup date", "orders", "No", "happens after approval"],
    ["Customer delivery date", "orders", "No (label only)", "used to define late"],
    ["Order status", "orders", "No", "final status is known only later"],
    ["Items, prices, freight, sellers per order", "order_items", "Yes", "fixed when ordered"],
    ["Shipping limit date for seller", "order_items", "Probably", "set at order time; check"],
    ["Product category, weight, size, photos", "products", "Yes", "catalog data"],
    ["Customer and seller zip, city, state", "customers / sellers", "Yes", "fixed"],
    ["Seller-customer distance", "geolocation", "Yes", "computed from zips"],
    ["Payment type, installments, value", "order_payments", "Yes", "paid before approval"],
    ["Review score and comments", "order_reviews", "No", "written after delivery"],
    ["Seller / customer / route past late rate", "derived", "Yes, if only earlier orders", "must use orders delivered before this approval"],
]
table_fig(pd.DataFrame(fa, columns=["Information", "Source table", "Usable as input?", "Why"]),
          "First draft of the feature-availability audit", "table_feature_availability",
          "Which information is known at approval (usable) versus only later (label or leakage).")

# ================================================================ index
lines = ["# Exploration figures and tables", "",
         "Made by `explore_data.py`. Working definition: **late** = delivered on a later calendar day than the promised date; "
         "rates use delivered orders with approval and delivery dates.", "",
         "| # | File | What it shows |", "|---|---|---|"]
lines += [f"| {n} | [{name}](figuresAndTables/{name}) | {what} |" for n, name, what in INDEX]
(HERE / "FIGURE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("done:", len(INDEX), "figures")
