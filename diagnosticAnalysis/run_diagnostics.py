"""Diagnostic and sensitivity analyses on the VALIDATION period (test is not touched).

Run:  python diagnosticAnalysis/run_diagnostics.py
Reads the saved validation daily picks of all four methods (nothing is re-ranked), the
validation table from data prep, and refits the chosen gradient boosting settings on train
for the number-of-trees check.

  A. Region     : which Brazilian regions the 10 daily picks go to, and how many of each
                  region's late orders each method catches.
  B. Trees      : gradient boosting with 1..1000 trees (chosen settings + the runner-up),
                  validation hits and loss as trees are added.
  C. Weekend    : Olist never promises weekend delivery (every promised date is Mon-Fri), so
                  the weekend question is answered two ways:
                    - late rate by weekday of the PROMISED date (e.g. Monday, right after a weekend)
                    - orders APPROVED on Saturday/Sunday versus weekdays, and how each method does.
Out:  figures/ and results/ in this folder.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from daily_eval import select_daily, score_days, totals, average_precision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREP = ROOT / "dataPrep" / "output"
RES = HERE / "results"
FIG = HERE / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
K = 10

METHODS = {  # name -> (results folder, color)
    "Gradient boosting": (ROOT / "model_gradientBoosting" / "results", "#c2185b"),
    "Logistic regression": (ROOT / "baseline_logisticRegression" / "results", "#7b3fbf"),
    "Longest distance rule": (ROOT / "nonML_distanceRule" / "results", "#1baf7a"),
    "Riskiest seller rule": (ROOT / "nonML_riskiestSellers" / "results", "#d4a017"),
}
REGION = {**{s: "North" for s in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
          **{s: "Northeast" for s in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
          **{s: "Center-West" for s in ["DF", "GO", "MT", "MS"]},
          **{s: "Southeast" for s in ["ES", "MG", "RJ", "SP"]},
          **{s: "South" for s in ["PR", "RS", "SC"]}}
REGION_ORDER = ["North", "Northeast", "Center-West", "Southeast", "South"]
GRAY, BASE = "#6b6b6b", "#9b9b9b"

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


def pct(x):
    return f"{100 * x:.1f}%" if np.isfinite(x) else "-"


# ---------------------------------------------------------------- data
val = pd.read_csv(PREP / "validation.csv", parse_dates=["approval_date", "estimated_date", "order_approved_at"])
val["region"] = val.customer_state.map(REGION)
val["promised_weekday"] = val.estimated_date.dt.day_name().str[:3]
val["approved_weekend"] = np.where(val.order_approved_at.dt.day_of_week >= 5, "Sat/Sun approval", "Mon-Fri approval")
lab = val[val.eligible == 1]
info = val.set_index("order_id")[["region", "promised_weekday", "approved_weekend"]]
picks = {}
for name, (folder, _) in METHODS.items():
    p = pd.read_csv(folder / "daily_picks_validation.csv")
    picks[name] = p.join(info, on="order_id")


def by_group(col, order):
    """For each group: orders, late orders, late rate, and per method: picks, hits, hit rate,
    share of the group's late orders caught."""
    base = lab.groupby(col).agg(orders=("order_id", "size"), late=("late", "sum"))
    base["late_rate"] = base.late / base.orders
    rows = []
    for g in order:
        r = {"group": g, "orders": int(base.loc[g, "orders"]), "late": int(base.loc[g, "late"]),
             "late_rate": base.loc[g, "late_rate"]}
        for name, p in picks.items():
            pg = p[p[col] == g]
            hits, wasted = (pg.outcome == "hit").sum(), (pg.outcome == "wasted").sum()
            r[f"{name}|picks"] = len(pg)
            r[f"{name}|hits"] = hits
            r[f"{name}|hit_rate"] = hits / (hits + wasted) if hits + wasted else np.nan
            r[f"{name}|caught"] = hits / base.loc[g, "late"]
        rows.append(r)
    return pd.DataFrame(rows)


# ================================================================ A. REGION
reg = by_group("region", REGION_ORDER)
reg.to_csv(RES / "region_validation.csv", index=False)
tab = pd.DataFrame({"Region": reg.group, "Orders": reg.orders.map("{:,}".format),
                    "Late": reg.late.map("{:,}".format), "Late rate": reg.late_rate.map(pct)})
for name in METHODS:
    tab[f"{name.split()[0]} hits"] = reg[f"{name}|hits"].astype(int)
table_fig(tab, "Late orders by customer region and hits of each method (validation)", "table_region",
          note="Regions: North, Northeast, Center-West, Southeast (incl. SP, RJ, MG), South. "
               "Hits = picked and late. Methods: Gradient boosting, Logistic regression, Longest distance, Riskiest seller.")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
xs = np.arange(len(REGION_ORDER))
w = 0.17
# left: where the late orders are vs where each method's picks go
late_share = reg.late / reg.late.sum()
axs[0].bar(xs - 2 * w, 100 * late_share, w, color=BASE, label="share of all late orders")
for i, (name, (_, col)) in enumerate(METHODS.items()):
    share = reg[f"{name}|picks"] / reg[f"{name}|picks"].sum()
    axs[0].bar(xs + (i - 1) * w, 100 * share, w, color=col, label=f"{name.lower()} picks")
axs[0].set_xticks(xs, REGION_ORDER)
axs[0].set_ylabel("Share (%)")
axs[0].set_title("Where late orders are vs where the picks go", loc="left")
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, ncol=5, fontsize=9,
           loc="lower center", bbox_to_anchor=(0.5, -0.08))
# right: share of each region's late orders caught
for i, (name, (_, col)) in enumerate(METHODS.items()):
    axs[1].bar(xs + (i - 1.5) * w, 100 * reg[f"{name}|caught"], w, color=col, label=name.lower())
axs[1].set_xticks(xs, REGION_ORDER)
axs[1].set_ylabel("Late orders caught (%)")
axs[1].set_title("Share of each region's late orders caught", loc="left")
for ax in axs:
    plt.setp(ax.get_xticklabels(), rotation=15)
fig.suptitle(f"Figure {FIG_NUM[0] + 1}. Which regions each method covers (validation)", x=0.01, ha="left",
             fontweight="bold")
fig.tight_layout()
save(fig, "region_coverage")

# ================================================================ B. NUMBER OF TREES
CAT = ["customer_state", "seller_state", "payment_type", "main_category", "approval_weekday"]
audit = pd.read_csv(PREP / "feature_availability.csv")
FEATURES = audit[(audit.group != "excluded") & audit["use as model input"].str.startswith("yes")].column.tolist()
NUM = [c for c in FEATURES if c not in CAT]


def load(split):
    df = pd.read_csv(PREP / f"{split}.csv", parse_dates=["approval_date"])
    for c in CAT:
        df[c] = df[c].astype("string").astype(object).where(df[c].notna(), np.nan)
    return df


def make_model(lr, leaves, msl, trees):   # identical to model_gradientBoosting
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan, min_frequency=100), CAT),
        ("num", "passthrough", NUM)])
    gb = HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl,
                                        max_iter=trees, l2_regularization=1.0, early_stopping=False,
                                        categorical_features=[True] * len(CAT) + [False] * len(NUM),
                                        random_state=0)
    return Pipeline([("prep", pre), ("gb", gb)])


train, vtab = load("train"), load("validation")
tr = train[train.eligible == 1]
vmask = (vtab.eligible == 1).to_numpy()
MAX_TREES = 1000
configs = {"chosen: learning rate 0.1, 3 leaves": (0.1, 3, 200),
           "runner-up: learning rate 0.03, 7 leaves": (0.03, 7, 200)}
curves = []
for label, (lr, leaves, msl) in configs.items():
    m = make_model(lr, leaves, msl, MAX_TREES).fit(tr[FEATURES], tr.late)
    Xtr = m.named_steps["prep"].transform(tr[FEATURES])
    Xva = m.named_steps["prep"].transform(vtab[FEATURES])
    gb = m.named_steps["gb"]
    for n, (ptr, pva) in enumerate(zip(gb.staged_predict_proba(Xtr), gb.staged_predict_proba(Xva)), start=1):
        if n % 10 and n not in (1, 5):
            continue
        vtab["s"] = pva[:, 1]
        t = totals(score_days(vtab, select_daily(vtab, "s", K), K))
        curves.append({"setting": label, "n_trees": n, "train_log_loss": log_loss(tr.late, ptr[:, 1]),
                       "val_log_loss": log_loss(vtab.late[vmask], pva[vmask, 1]),
                       "val_hits": t["hits"],
                       "val_pr_auc": average_precision(vtab.s[vmask], vtab.late[vmask])})
    print("trees curve done:", label)
curves = pd.DataFrame(curves)
curves.to_csv(RES / "trees_sensitivity_validation.csv", index=False)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
for (label, cdf), col in zip(curves.groupby("setting", sort=False), ["#c2185b", "#2a78d6"]):
    axs[0].plot(cdf.n_trees, cdf.val_hits, color=col, lw=2, label=label)
    axs[1].plot(cdf.n_trees, cdf.train_log_loss, color=col, lw=1.5, ls="--")
    axs[1].plot(cdf.n_trees, cdf.val_log_loss, color=col, lw=2)
lr_hits = float(pd.read_csv(METHODS["Logistic regression"][0] / "summary.csv", index_col=0).loc["hits", "validation"])
axs[0].axhline(lr_hits, color="#7b3fbf", lw=1.2, ls=":", label=f"logistic regression ({lr_hits:.0f})")
axs[0].set_ylabel("Validation hits (10 per day)")
axs[0].set_title("Late orders caught on validation", loc="left")
axs[0].legend(frameon=False, fontsize=9, loc="lower right")
axs[1].set_ylabel("Log-loss (solid = validation, dashed = train)")
axs[1].set_title("Loss", loc="left")
for ax in axs:
    ax.set_xlabel("Number of trees")
    ax.axvline(300, color="black", lw=0.8, ls=":")
fig.suptitle(f"Figure {FIG_NUM[0] + 1}. Sensitivity to the number of trees (validation, up to {MAX_TREES})",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "trees_sensitivity")

pick_n = [10, 50, 100, 200, 300, 500, 700, 1000]
tt = curves[curves.n_trees.isin(pick_n)].copy()
tt = pd.DataFrame({"Setting": tt.setting.str.split(":").str[0], "Trees": tt.n_trees,
                   "Validation hits": tt.val_hits.astype(int), "PR-AUC": tt.val_pr_auc.map("{:.3f}".format),
                   "Val log-loss": tt.val_log_loss.map("{:.4f}".format),
                   "Train log-loss": tt.train_log_loss.map("{:.4f}".format)})
table_fig(tt, "Validation results at selected numbers of trees", "table_trees",
          note="Chosen model = 300 trees, learning rate 0.1, 3 leaves. Runner-up = learning rate 0.03, 7 leaves.")

# ================================================================ C. WEEKEND
# C1: late rate by promised weekday (train + validation labeled orders; test untouched)
both = pd.concat([train.assign(split="train (2017)"), val.assign(split="validation")], ignore_index=True)
both = both[both.eligible == 1].copy()
both["estimated_date"] = pd.to_datetime(both.estimated_date)
both["promised_weekday"] = both.estimated_date.dt.day_name().str[:3]
wd_order = ["Mon", "Tue", "Wed", "Thu", "Fri"]
pw = both.groupby(["split", "promised_weekday"]).late.agg(["mean", "size"]).reset_index()
pw.to_csv(RES / "late_rate_by_promised_weekday.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 4))
xs = np.arange(len(wd_order))
for i, (sp, col) in enumerate([("train (2017)", "#2a78d6"), ("validation", "#eb6834")]):
    s = pw[pw.split == sp].set_index("promised_weekday").reindex(wd_order)
    ax.bar(xs + (i - 0.5) * 0.38, 100 * s["mean"], 0.38, color=col, label=sp)
ax.set_xticks(xs, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
ax.set_ylabel("Late rate (%)")
ax.set_xlabel("Weekday of the promised delivery date")
ax.legend(frameon=False)
title(ax, "Late rate by weekday of the promised date (no promises fall on weekends)")
fig.text(0.01, -0.03, "Olist never promises Saturday or Sunday delivery; about 7% of actual deliveries "
         "still happen on weekends.", fontsize=9, color=GRAY)
save(fig, "late_rate_promised_weekday")

# C2: approved on weekend vs weekday -- base late rate and each method
aw = by_group("approved_weekend", ["Mon-Fri approval", "Sat/Sun approval"])
pwk = by_group("promised_weekday", wd_order)
aw.to_csv(RES / "approved_weekend_validation.csv", index=False)
pwk.to_csv(RES / "promised_weekday_validation.csv", index=False)

rows = []
for df_, kind in [(aw, ""), (pwk, "Promised for ")]:
    for _, r in df_.iterrows():
        row = {"Group": f"{kind}{r.group}", "Orders": f"{r.orders:,}", "Late rate": pct(r.late_rate)}
        for name in METHODS:
            row[f"{name.split()[0]} hit rate"] = pct(r[f"{name}|hit_rate"])
            row[f"{name.split()[0]} picks"] = int(r[f"{name}|picks"])
        rows.append(row)
table_fig(pd.DataFrame(rows), "Weekend and weekday groups: late rate and each method's picks (validation)",
          "table_weekend",
          note="Hit rate = share of that method's picks in the group that were late. "
               "Picks = how many of the 1,045 investigations fell in the group.")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
for ax, df_, grp_labels, ttl in [(axs[0], aw, ["Mon-Fri approval", "Sat/Sun approval"], "By approval day"),
                                 (axs[1], pwk, wd_order, "By weekday of promised date")]:
    xs = np.arange(len(grp_labels))
    ax.bar(xs - 2 * w, 100 * df_.late_rate, w, color=BASE, label="late rate (random picks)")
    for i, (name, (_, col)) in enumerate(METHODS.items()):
        ax.bar(xs + (i - 1) * w, 100 * df_[f"{name}|hit_rate"], w, color=col, label=name.lower())
    ax.set_xticks(xs, grp_labels)
    ax.set_ylabel("Hit rate of picks (%)")
    ax.set_title(ttl, loc="left")
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, ncol=5, fontsize=9,
           loc="lower center", bbox_to_anchor=(0.5, -0.08))
fig.suptitle(f"Figure {FIG_NUM[0] + 1}. Hit rate of each method's picks, weekend vs weekday (validation)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "weekend_hit_rate")
