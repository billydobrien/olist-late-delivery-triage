"""Non-ML business rule: each day, investigate the 10 orders whose seller has the worst
past late-delivery record.

Seller record = seller_prev_late_rate from data prep: the seller's late rate over orders
DELIVERED before this order's approval, smoothed toward the overall late rate known at that
time (20 pseudo-orders), so a seller with 1 late order out of 1 is not ranked above a seller
with 30 out of 100. Orders with no seller (no items) rank last.

Identical to nonML_distanceRule except for SCORE and RULE below.

Run:  python nonML_riskiestSellers/run_seller_rule.py
Uses: dataPrep/output/train.csv and validation.csv  (the test set is NOT touched here;
      it is scored once at the end, together with the models)
Out:  nonML_riskiestSellers/results/   daily picks, per-day scores, period totals
      nonML_riskiestSellers/figures/   simple figures of how the rule performs
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd

from daily_eval import select_daily, score_days, totals, average_precision, outcome_label

HERE = Path(__file__).resolve().parent
PREP = HERE.parent / "dataPrep" / "output"
RES = HERE / "results"
FIG = HERE / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

K = 10
SCORE = "seller_prev_late_rate"
RULE = "Riskiest seller"

HIT, WASTED, UNKNOWN, GRAY = "#eb6834", "#2a78d6", "#b5b5b5", "#6b6b6b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True, "figure.dpi": 150})
FIG_NUM = [0]


def save(fig, slug, bbox="tight"):
    FIG_NUM[0] += 1
    fig.savefig(FIG / f"fig{FIG_NUM[0]:02d}_{slug}.png", bbox_inches=bbox, facecolor="white")
    plt.close(fig)
    print("saved", f"fig{FIG_NUM[0]:02d}_{slug}.png")


def table_fig(df, title, slug, note=None):
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
    save(fig, slug, bbox=bb.transformed(fig.dpi_scale_trans.inverted()).padded(0.12))


def title(ax, text):
    ax.set_title(f"Figure {FIG_NUM[0] + 1}. {text}", loc="left", fontweight="bold")


# ---------------------------------------------------------------- run the rule
summary, days, picks = {}, {}, {}
for split in ["train", "validation"]:
    df = pd.read_csv(PREP / f"{split}.csv", parse_dates=["approval_date"])
    p = select_daily(df, SCORE, K)
    d = score_days(df, p, K)
    s = totals(d)
    lab = df[df.eligible == 1]
    s["pr_auc_all_orders"] = round(average_precision(lab[SCORE], lab.late), 4)
    s["pr_auc_random"] = round(lab.late.mean(), 4)
    summary[split], days[split], picks[split] = s, d, p
    p.assign(outcome=outcome_label(p))[["approval_date", "rank", "order_id", "order_approved_at", SCORE,
                                        "customer_state", "seller_state", "outcome"]] \
        .to_csv(RES / f"daily_picks_{split}.csv", index=False)
    d.to_csv(RES / f"daily_scores_{split}.csv")
pd.DataFrame(summary).to_csv(RES / "summary.csv")
print(pd.DataFrame(summary).to_string())

# ---------------------------------------------------------------- fig: summary table
def pct(x):
    return f"{100 * x:.1f}%"


rows = [
    ("Days", "days", str),
    ("Candidate orders (all approved)", "candidates", lambda v: f"{v:,}"),
    ("Late orders", "late_orders", lambda v: f"{v:,}"),
    ("Orders investigated (10 per day)", "picked", lambda v: f"{v:,}"),
    ("Hits: investigated and late", "hits", lambda v: f"{v:,}"),
    ("Wasted: investigated but on time", "wasted", lambda v: f"{v:,}"),
    ("Unknown outcome (canceled etc.)", "unknown", lambda v: f"{v:,}"),
    ("Missed: late but not investigated", "missed", lambda v: f"{v:,}"),
    ("Hit rate of investigations (precision)", "hit_rate_of_picks", pct),
    ("Share of all late orders caught (recall)", "share_of_late_caught", pct),
    ("Best possible hits (perfect list)", "best_possible_hits", lambda v: f"{v:,}"),
    ("Share of best possible reached", "share_of_best_possible", pct),
    ("Expected hits picking at random", "random_expected_hits", lambda v: f"{v:,.0f}"),
    ("Hits compared with random", "lift_over_random", lambda v: f"{v:.2f}x"),
    ("PR-AUC of seller late rate (random = late rate)", "pr_auc_all_orders", lambda v: f"{v:.3f}"),
    ("PR-AUC of random ranking", "pr_auc_random", lambda v: f"{v:.3f}"),
]
tab = pd.DataFrame([[name, f(summary["train"][key]), f(summary["validation"][key])] for name, key, f in rows],
                   columns=["Measure", "Train (2017)", "Validation (Jan 1 - Apr 15, 2018)"])
table_fig(tab, f"{RULE} rule: 10 investigations per day", "table_rule_summary",
          note="The rule needs no training, so train is shown only as a second period. Test is not used yet.")

# ---------------------------------------------------------------- fig: per-day picks (validation)
d = days["validation"]
fig, ax = plt.subplots(figsize=(12, 4.2))
x = d.index
ax.bar(x, d.hit, color=HIT, label="hit (late)", width=0.85)
ax.bar(x, d.wasted, bottom=d.hit, color=WASTED, label="wasted (on time)", width=0.85)
ax.bar(x, d.unknown, bottom=d.hit + d.wasted, color=UNKNOWN, label="unknown outcome", width=0.85)
ax.step(x, d.best_possible, where="mid", color="black", lw=1.3, label="best possible hits")
ax.set_ylabel("Orders investigated")
ax.set_ylim(0, 11.5)
ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
title(ax, f"{RULE} rule: what the 10 daily investigations found (validation)")
save(fig, "validation_daily_picks")

# ---------------------------------------------------------------- fig: cumulative hits (validation)
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(x, d.best_possible.cumsum(), color="black", lw=2, label="best possible (perfect list)")
ax.plot(x, d.hit.cumsum(), color=HIT, lw=2.5, label=f"{RULE.lower()} rule")
ax.plot(x, d.random_expected.cumsum(), color=GRAY, lw=2, ls="--", label="random 10 per day")
ax.set_ylabel("Late orders caught (running total)")
ax.legend(frameon=False, loc="upper left")
title(ax, "Running total of late orders caught (validation)")
save(fig, "validation_cumulative_hits")

# ---------------------------------------------------------------- fig: weekly hit rate vs base late rate
fig, ax = plt.subplots(figsize=(10, 4.2))
for split, ls in [("train", "-"), ("validation", "-")]:
    w = days[split].resample("W").sum()
    w = w[w.picked >= 30]
    ax.plot(w.index, 100 * w.hit / (w.hit + w.wasted), color=HIT, lw=2,
            label="hit rate of the 10 picks" if split == "train" else None)
    ax.plot(w.index, 100 * w.late / w.labeled, color=GRAY, lw=1.5, ls="--",
            label="late rate of all orders (random picks)" if split == "train" else None)
ax.axvline(pd.Timestamp("2018-01-01"), color="black", lw=1)
ax.text(pd.Timestamp("2018-01-05"), ax.get_ylim()[1] * 0.92, "validation", fontsize=10)
ax.text(pd.Timestamp("2017-01-10"), ax.get_ylim()[1] * 0.92, "train", fontsize=10)
ax.set_ylabel("Percent late (%)")
ax.legend(frameon=False, loc="upper center")
title(ax, f"{RULE} rule: weekly hit rate compared with picking at random")
save(fig, "weekly_hit_rate")

# ---------------------------------------------------------------- fig: budget sensitivity (validation)
val = pd.read_csv(PREP / "validation.csv", parse_dates=["approval_date"])
ks = [1, 2, 3, 5, 10, 15, 20, 30, 50]
rows = []
for k in ks:
    t = totals(score_days(val, select_daily(val, SCORE, k), k))
    rows.append((k, t["hits"], t["hit_rate_of_picks"], t["share_of_late_caught"], t["share_of_best_possible"],
                 t["random_expected_hits"]))
bs = pd.DataFrame(rows, columns=["k", "hits", "precision", "recall", "of_best", "random_hits"])
bs.to_csv(RES / "budget_sensitivity_validation.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(bs.k, bs.hits, color=HIT, marker="o", lw=2, label=f"{RULE.lower()} rule")
axs[0].plot(bs.k, bs.random_hits, color=GRAY, marker="o", lw=1.5, ls="--", label="random")
axs[0].set_xlabel("Investigations per day")
axs[0].set_ylabel("Late orders caught")
axs[0].set_title("Late orders caught", loc="left")
axs[0].legend(frameon=False)
axs[1].plot(bs.k, 100 * bs.precision, color=HIT, marker="o", lw=2)
axs[1].axhline(100 * val[val.eligible == 1].late.mean(), color=GRAY, ls="--", lw=1.5)
axs[1].set_xlabel("Investigations per day")
axs[1].set_ylabel("Hit rate of investigations (%)")
axs[1].set_title("Share of investigations that were late", loc="left")
for ax in axs:
    ax.axvline(10, color="black", lw=0.8, ls=":")
fig.suptitle(f"Figure {FIG_NUM[0] + 1}. What if the team could investigate more or fewer orders per day? (validation)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "budget_sensitivity")

# ---------------------------------------------------------------- fig: one example day
# a typical day: the median number of hits, and among those the day closest to the median late count
typical = d[d.hit == d.hit.median()]
ex_day = (typical.late - d.late.median()).abs().idxmin()
ex = picks["validation"]
ex = ex[ex.approval_date == ex_day].assign(outcome=lambda t: outcome_label(t))
ex_tab = pd.DataFrame({"Rank": ex["rank"], "Order ID (first 8)": ex.order_id.str[:8],
                       "Seller past late rate": (100 * ex.seller_prev_late_rate).map("{:.1f}%".format),
                       "Seller past deliveries": ex.seller_prev_delivered.astype("Int64"),
                       "Seller state": ex.seller_state, "Customer state": ex.customer_state,
                       "Outcome": ex.outcome})
dd = d.loc[ex_day]
table_fig(ex_tab, f"Example day {ex_day:%Y-%m-%d}: the 10 orders the rule would investigate", "example_day",
          note=f"That day: {int(dd.candidates)} approved orders, {int(dd.late)} eventually late; "
               f"the rule caught {int(dd.hit)} of them.")
