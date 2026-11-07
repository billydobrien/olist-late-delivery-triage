"""Investigate how to split the orders by time into train / validation / test.

Run:  python exploration/explore_time_split.py
Figures continue the numbering of explore_data.py (start at 37) and are saved
to exploration/figuresAndTables. Working label as before: late = delivered on a
later calendar day than the promised date; eligible = delivered with dates.
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
FIRST_FIG = 37

ONTIME, LATE, GRAY = "#2a78d6", "#eb6834", "#6b6b6b"
PERIOD_COLORS = {"train": "#d9d9d9", "validation": "#bcd6f5", "test": "#f7c9b3"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True, "figure.dpi": 150})

FIG_NUM = [FIRST_FIG - 1]
INDEX = []


def save(fig, slug, what, bbox="tight"):
    FIG_NUM[0] += 1
    name = f"fig{FIG_NUM[0]:02d}_{slug}.png"
    fig.savefig(OUT / name, bbox_inches=bbox, facecolor="white")
    plt.close(fig)
    INDEX.append((FIG_NUM[0], name, what))
    print("saved", name)


def table_fig(df, title, slug, what, note=None):
    nrows, ncols = df.shape
    fig, ax = plt.subplots(figsize=(min(2.0 + 1.6 * ncols, 13), 0.9 + 0.32 * (nrows + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=df.astype(str).values.tolist(), colLabels=list(df.columns),
                   loc="upper center", cellLoc="left", colLoc="left")
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


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def n(x):
    return f"{int(x):,}"


# ---------------------------------------------------------------- data
o = pd.read_csv(DATA / "olist_orders_dataset.csv")
for c in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
          "order_delivered_customer_date", "order_estimated_delivery_date"]:
    o[c] = pd.to_datetime(o[c])
o = o[o.order_approved_at.notna()].copy()
o["eligible"] = (o.order_status == "delivered") & o.order_delivered_customer_date.notna()
o["late"] = (o.order_delivered_customer_date.dt.normalize()
             > o.order_estimated_delivery_date.dt.normalize()).astype(float)
o.loc[~o.eligible, "late"] = np.nan
o["day"] = o.order_approved_at.dt.normalize()
LAST_RECORD = o[["order_approved_at", "order_delivered_customer_date"]].max().max()

START = pd.Timestamp("2017-01-01")      # before this: a few hundred orders in Oct 2016, then a 2-month gap
# Deliveries in the data essentially stop on 2018-09-03 (1,607 in the last week of August, 28 the
# week after). Orders approved in August often have promised dates after that, so a late delivery
# could never be observed for them. Keep only approvals up to 2018-07-31.
DATA_STOP = pd.Timestamp("2018-09-03")
END = pd.Timestamp("2018-08-01")


def summarize(periods):
    """periods: list of (name, start, end_exclusive). Returns one row per period."""
    rows = []
    elig_total = o[o.eligible & o.day.between(periods[0][1], periods[-1][2] - pd.Timedelta(days=1))].shape[0]
    for name, s, e in periods:
        p = o[(o.day >= s) & (o.day < e)]
        el = p[p.eligible]
        daily = el.groupby("day").late.sum()
        rows.append({
            "Period": name,
            "Approval dates": f"{s:%Y-%m-%d} to {(e - pd.Timedelta(days=1)):%Y-%m-%d}",
            "Days": (e - s).days,
            "Approved": n(len(p)),
            "Eligible": n(len(el)),
            "% of eligible": pct(len(el) / elig_total),
            "On time": n((el.late == 0).sum()),
            "Late": n((el.late == 1).sum()),
            "Late rate": pct(el.late.mean()),
            "Late per day": f"{daily.mean():.1f}",
            "Days with 10+ late": pct((daily >= 10).mean(), 0),
        })
    return pd.DataFrame(rows)


def period_bounds(cuts):
    edges = [START] + [pd.Timestamp(c) for c in cuts] + [END]
    names = ["train", "validation", "test"]
    return [(names[i], edges[i], edges[i + 1]) for i in range(3)]


# Candidate splits
options = {
    "A: train 2017, val Jan-Apr 2018, test May-Jul 2018": ["2018-01-01", "2018-05-01"],
    "B: train 2017, val Jan-Mar 2018, test Apr-Jul 2018": ["2018-01-01", "2018-04-01"],
    "C: train to Feb 2018, val Mar-Apr 2018, test May-Jul 2018": ["2018-03-01", "2018-05-01"],
    "D: 70/15/15 of eligible orders by approval date": None,
}
el_days = o[o.eligible & (o.day >= START) & (o.day < END)].sort_values("order_approved_at").day.reset_index(drop=True)
q70, q85 = el_days.iloc[int(0.70 * len(el_days))], el_days.iloc[int(0.85 * len(el_days))]
options["D: 70/15/15 of eligible orders by approval date"] = [f"{q70:%Y-%m-%d}", f"{q85:%Y-%m-%d}"]
q50, q75 = el_days.iloc[int(0.50 * len(el_days))], el_days.iloc[int(0.75 * len(el_days))]
options["E: 50/25/25 of eligible orders by approval date"] = [f"{q50:%Y-%m-%d}", f"{q75:%Y-%m-%d}"]
# CHOSEN (2026-11-06): E rounded to clean dates
options["F: 50/25/25 rounded - train 2017, val Jan 1-Apr 15 2018, test Apr 16-Jul 31 2018"] = ["2018-01-01", "2018-04-16"]

# ---------------------------------------------------------------- figure: why start in 2017
pre = o[o.day < START]
post = o[o.day >= END]
pdays = sorted(pre.day.unique())
gaps = np.diff(pdays)
gi = int(np.argmax(gaps))
dlv = o.order_delivered_customer_date.dropna()
table_fig(pd.DataFrame([
    ["Approved before 2017-01-01", f"{n(len(pre))} ({n(pre.eligible.sum())} eligible, {n(pre.late.sum())} late)"],
    ["Days with approvals before 2017", n(len(pdays))],
    ["Longest gap with no approvals", f"{pd.Timestamp(pdays[gi]):%Y-%m-%d} to {pd.Timestamp(pdays[gi + 1]):%Y-%m-%d}"],
    ["Deliveries in week of 2018-08-27", n(((dlv >= "2018-08-27") & (dlv < "2018-09-03")).sum())],
    ["Deliveries after 2018-09-03 (whole rest of data)", n((dlv >= DATA_STOP).sum())],
    ["Approved Aug 2018 with promised date after 2018-09-03",
     pct((o[(o.day >= "2018-08-01") & (o.day < "2018-09-01")].order_estimated_delivery_date > DATA_STOP).mean())],
    ["Approved 2018-08-01 or later (left out)", f"{n(len(post))} ({n(post.eligible.sum())} eligible, {n(post.late.sum())} late)"],
], columns=["Item", "Value"]), "Orders left out of every split (edges of the data)", "table_split_edges",
    "Why the split covers approvals from 2017-01-01 to 2018-07-31 only.",
    note="Before 2017 the data is sparse. After July 2018 many promised dates fall after deliveries stop being recorded, so late orders would be missing.")

# ---------------------------------------------------------------- comparison of options
rows = []
for label, cuts in options.items():
    s = summarize(period_bounds(cuts))
    for _, r in s.iterrows():
        rows.append([label.split(":")[0], r.Period, r["Approval dates"], r["Eligible"], r["% of eligible"],
                     r["On time"], r["Late"], r["Late rate"]])
table_fig(pd.DataFrame(rows, columns=["Option", "Period", "Approval dates", "Eligible", "% of eligible",
                                      "On time", "Late", "Late rate"]),
          "Six candidate time splits (F = chosen)", "table_split_options",
          "Size, late count and late rate of train/validation/test under four candidate split dates.",
          note=" / ".join(k for k in options))

# ---------------------------------------------------------------- monthly bars with candidate cuts
o["month"] = o.day.dt.to_period("M")
m = o[(o.day >= START) & (o.day < END) & o.eligible].groupby("month").late.agg(["sum", "size"])
m["ontime"] = m["size"] - m["sum"]
fig, axs = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
xs = np.arange(len(m))
axs[0].bar(xs, m.ontime, color=ONTIME, label="on time")
axs[0].bar(xs, m["sum"], bottom=m.ontime, color=LATE, label="late")
axs[0].set_ylabel("Eligible orders")
axs[0].legend(frameon=False, loc="upper left")
axs[1].bar(xs, 100 * m["sum"] / m["size"], color=LATE)
axs[1].set_ylabel("Late rate (%)")
axs[1].set_xticks(xs, [str(p) for p in m.index], rotation=60)
months = [str(p) for p in m.index]
handles = []
for k, (lab, ls) in enumerate([("A", "-"), ("B", "--"), ("C", ":")]):
    cuts = next(v for kk, v in options.items() if kk.startswith(lab))
    for c in cuts:
        x = months.index(c[:7]) - 0.5 + (k - 1) * 0.12   # small offset so shared dates stay visible
        for ax in axs:
            ax.axvline(x, color="black", ls=ls, lw=1.2)
    handles.append(plt.Line2D([], [], color="black", ls=ls, label=f"option {lab}"))
axs[1].legend(handles=handles, frameon=False, loc="upper left")
axs[0].set_title(f"Figure {FIG_NUM[0] + 1}. On-time and late orders per month, with candidate split dates",
                 loc="left", fontweight="bold")
fig.text(0.01, -0.02, "Each option has two lines: where validation starts and where test starts.",
         fontsize=9, color=GRAY)
save(fig, "split_candidates_monthly", "Monthly on-time/late counts and late rate with the A/B/C split dates drawn in.")

# ---------------------------------------------------------------- recommended split detail
REC = "F"
rec_periods = period_bounds(options[[k for k in options if k.startswith(REC)][0]])
rs = summarize(rec_periods)
table_fig(rs, "Chosen split (option F, rounded 50/25/25) in detail", "table_split_recommended",
          "Per-period counts, late/on-time breakdown and daily late counts for the recommended split.",
          note="Late per day and days with 10+ late are over days with eligible orders; they show how full a 10-per-day list could be.")

# ---------------------------------------------------------------- label-delay check
rows = []
for name, s, e in rec_periods:
    p = o[(o.day >= s) & (o.day < e) & o.eligible]
    after = p.order_delivered_customer_date >= e
    rows.append([name, n(len(p)), n(after.sum()), pct(after.mean(), 2),
                 n((after & (p.late == 1)).sum())])
t = (o.order_delivered_customer_date - o.order_approved_at).dt.days[o.eligible]
rows.append(["(all eligible) approval to delivery, days", f"median {t.median():.0f}",
             f"95th pct {t.quantile(0.95):.0f}", f"99th pct {t.quantile(0.99):.0f}", ""])
table_fig(pd.DataFrame(rows, columns=["Period", "Eligible", "Delivered after period ends",
                                      "Share", "Of those, late"]),
          "Labels that are only known after the period ends", "table_label_delay",
          "How many orders in each period were delivered (label known) only after the period ended.",
          note="Late orders take longest, so they are the ones whose label arrives after the cut. History features must only use orders delivered before the scored order's approval.")

# ---------------------------------------------------------------- end-of-data check (censoring)
tail = o[(o.day >= "2018-06-25") & (o.day < "2018-09-03")].copy()
tail["week"] = tail.day.dt.to_period("W").dt.start_time
tail["promise_after_stop"] = tail.order_estimated_delivery_date > DATA_STOP
w = tail.groupby("week").agg(approved=("order_id", "size"), eligible=("eligible", "sum"),
                             late=("late", "sum"), after=("promise_after_stop", "mean"))
w["late_rate"] = w.late / w.eligible
table_fig(pd.DataFrame({"Week starting": [f"{d:%Y-%m-%d}" for d in w.index], "Approved": w.approved.map(n),
                        "Eligible": w.eligible.map(n), "Late": w.late.map(n), "Late rate": w.late_rate.map(pct),
                        "Promised after deliveries stop": w.after.map(pct),
                        "In split?": ["yes" if d < END else "no (left out)" for d in w.index]}),
          "End of the data: late orders from August cannot be observed", "table_end_censoring",
          "Weekly counts near the end of the data, and the share of orders whose promised date is after deliveries stop being recorded (2018-09-03).",
          note="An order promised after 2018-09-03 could only be recorded as late if delivered after that date, which almost never appears in the data.")

# ---------------------------------------------------------------- timeline of recommended split
wk = o[o.eligible & (o.day >= START) & (o.day < END)].set_index("day").late.resample("W").agg(["mean", "size"])
fig, ax = plt.subplots(figsize=(11, 4))
for name, s, e in rec_periods:
    ax.axvspan(s, e, color=PERIOD_COLORS[name], alpha=0.8, lw=0)
    ax.text(s + (e - s) / 2, 27, name, ha="center", fontweight="bold")
ax.plot(wk.index, 100 * wk["mean"], color=LATE, lw=2)
ax.set_ylim(0, 30)
ax.set_ylabel("Weekly late rate (%)")
ax.set_title(f"Figure {FIG_NUM[0] + 1}. Chosen split (option F) over the weekly late rate",
             loc="left", fontweight="bold")
save(fig, "split_recommended_timeline", "Weekly late rate with the train/validation/test periods shaded.")

# ---------------------------------------------------------------- one-row-per-order column plan
cols = [
    # name, source, role, description
    ["order_id", "orders", "ID", "anonymized order ID (traceability)"],
    ["order_approved_at", "orders", "ID / time", "approval timestamp = decision time"],
    ["approval_date", "orders", "ID / time", "calendar day used for the daily top-10 list"],
    ["split", "derived", "ID / time", "train / validation / test by approval date"],
    ["customer_unique_id", "customers", "ID (history only)", "used to count earlier orders; not a feature"],
    ["main_seller_id", "order_items", "ID (history only)", "seller of the most expensive item; not a feature"],
    ["order_status", "orders", "Outcome / audit", "final status (eligibility only)"],
    ["delivered_date", "orders", "Outcome / audit", "actual customer delivery date"],
    ["estimated_date", "orders", "Outcome / audit", "promised delivery date"],
    ["days_late", "derived", "Outcome / audit", "delivered day minus promised day"],
    ["eligible", "derived", "Outcome / audit", "delivered with both dates present"],
    ["late", "derived", "LABEL", "1 if delivered on a later day than promised"],
    ["approval_hour", "orders", "Feature: time", "hour of approval"],
    ["approval_weekday", "orders", "Feature: time", "day of week of approval"],
    ["purchase_to_approval_h", "orders", "Feature: time", "hours from purchase to approval"],
    ["promised_days", "orders", "Feature: time", "days from approval to promised date"],
    ["seller_ship_limit_days", "order_items", "Feature: time", "days from approval to seller shipping deadline (check)"],
    ["n_items", "order_items", "Feature: order", "number of items"],
    ["n_products", "order_items", "Feature: order", "number of different products"],
    ["n_sellers", "order_items", "Feature: order", "number of different sellers"],
    ["total_price", "order_items", "Feature: order", "sum of item prices"],
    ["total_freight", "order_items", "Feature: order", "sum of freight charges"],
    ["freight_ratio", "derived", "Feature: order", "freight / price"],
    ["total_weight_g", "products", "Feature: product", "sum of item weights"],
    ["total_volume_cm3", "products", "Feature: product", "sum of length x width x height"],
    ["main_category", "products", "Feature: product", "category of the most expensive item (English)"],
    ["n_categories", "products", "Feature: product", "number of different categories"],
    ["payment_type", "order_payments", "Feature: payment", "main payment method"],
    ["n_payments", "order_payments", "Feature: payment", "number of payment records"],
    ["max_installments", "order_payments", "Feature: payment", "largest installment count"],
    ["payment_value", "order_payments", "Feature: payment", "total paid"],
    ["customer_state", "customers", "Feature: location", "customer state"],
    ["seller_state", "sellers", "Feature: location", "main seller state"],
    ["same_state", "derived", "Feature: location", "seller and customer in the same state"],
    ["distance_km", "geolocation", "Feature: location", "seller-to-customer straight-line distance"],
    ["customer_lat / customer_lng", "geolocation", "Feature: location", "customer zip-area coordinates"],
    ["seller_lat / seller_lng", "geolocation", "Feature: location", "seller zip-area coordinates"],
    ["seller_prev_orders", "derived (history)", "Feature: history", "seller's orders approved earlier"],
    ["seller_prev_late_rate", "derived (history)", "Feature: history", "seller's late rate over orders DELIVERED before this approval"],
    ["customer_prev_orders", "derived (history)", "Feature: history", "customer's orders approved earlier"],
    ["route_prev_late_rate", "derived (history)", "Feature: history", "late rate for seller state to customer state, delivered before approval"],
    ["recent_late_rate_14d", "derived (history)", "Feature: history", "late rate of orders delivered in the 14 days before approval"],
    ["orders_last_24h", "derived (history)", "Feature: history", "orders approved in the previous 24 hours (workload)"],
]
cdf = pd.DataFrame(cols, columns=["Column", "Source", "Role", "Meaning"])
table_fig(cdf, "Planned columns of the one-row-per-order modeling table", "table_order_columns",
          "Every planned column of the modeling table with its source and role (ID, outcome, label, feature).",
          note="Not included at all: carrier pickup date, review score/comments (known only after approval).")
cdf.to_csv(HERE / "planned_order_columns.csv", index=False)

# shipping-limit check (is it set at order time?)
it = pd.read_csv(DATA / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
sl = it.groupby("order_id").shipping_limit_date.max()
gap = (o.set_index("order_id").order_approved_at.reindex(sl.index)).rsub(sl).dt.total_seconds() / 86400
print("shipping_limit - approval (days): ", gap.describe().round(2).to_dict())

# ---------------------------------------------------------------- index append
idx = HERE / "FIGURE_INDEX.md"
text = idx.read_text(encoding="utf-8") if idx.exists() else ""
def row_num(line):
    try:
        return int(line.split("|")[1])
    except (IndexError, ValueError):
        return None


keep = [l for l in text.splitlines() if (row_num(l) or 0) < FIRST_FIG]
keep += [f"| {k} | [{nm}](figuresAndTables/{nm}) | {w} |" for k, nm, w in INDEX]
idx.write_text("\n".join(keep) + "\n", encoding="utf-8")
print("done:", len(INDEX), "figures")
