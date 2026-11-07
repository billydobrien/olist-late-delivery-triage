"""Second learned model: gradient boosted trees predicting P(late) at order approval.

Run:  python model_gradientBoosting/train_gradient_boosting.py
Uses: dataPrep/output/train.csv (fit), validation.csv (choose settings, score the daily lists),
      dataPrep/output/feature_availability.csv (which columns may be model inputs).
      The test set is NOT touched here.
Out:  results/  tuning table, chosen config, feature importance, training curve,
                validation predictions, daily picks and scores, summary, saved model
      figures/  simple figures, compared with logistic regression and the two non-ML rules

Same inputs and same daily scoring as the logistic regression, so the comparison is fair.
Model = scikit-learn HistGradientBoostingClassifier (log-loss, trees added one at a time,
each new tree correcting the errors of the trees before it).
  * Numbers go in as they are: trees do not need scaling, and missing values are handled
    by the model itself (each split learns which side missing values go to).
  * Text columns (states, payment type, category, weekday) are coded as integers learned on
    TRAIN only and treated as categories by the trees; values seen fewer than 100 times in
    train are grouped as "infrequent", unseen values are treated as missing.
  * Fit on train orders with a label (eligible == 1). No internal early stopping: the number
    of trees is one of the tuned settings.
  * Grid: learning rate x tree size (leaves) x minimum orders per leaf x number of trees.
    The setting with the most validation hits in the daily top-10 is kept
    (ties -> higher validation PR-AUC), the same rule used for logistic regression.
"""
import json
import time
from itertools import product
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from daily_eval import select_daily, score_days, totals, average_precision, outcome_label

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREP = ROOT / "dataPrep" / "output"
RES = HERE / "results"
FIG = HERE / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
K = 10
SEED = 0

# ---------------------------------------------------------------- features (same as logistic regression)
audit = pd.read_csv(PREP / "feature_availability.csv")
FEATURES = audit[(audit.group != "excluded") & audit["use as model input"].str.startswith("yes")].column.tolist()
CAT = ["customer_state", "seller_state", "payment_type", "main_category", "approval_weekday"]
NUM = [c for c in FEATURES if c not in CAT]


def make_model(learning_rate, max_leaf_nodes, min_samples_leaf, max_iter):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan, min_frequency=100), CAT),
        ("num", "passthrough", NUM),
    ])
    gb = HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        max_iter=max_iter, l2_regularization=1.0, early_stopping=False,
        categorical_features=[True] * len(CAT) + [False] * len(NUM), random_state=SEED)
    return Pipeline([("prep", pre), ("gb", gb)])


def load(split):
    df = pd.read_csv(PREP / f"{split}.csv", parse_dates=["approval_date"])
    for c in CAT:
        df[c] = df[c].astype("string").astype(object).where(df[c].notna(), np.nan)
    return df


train, val = load("train"), load("validation")
tr = train[train.eligible == 1]
val_lab = val[val.eligible == 1]
print(f"train labeled: {len(tr):,} ({int(tr.late.sum()):,} late); features: {len(FEATURES)} "
      f"({len(NUM)} numeric, {len(CAT)} categorical)")

# ---------------------------------------------------------------- tuning grid
GRID = dict(learning_rate=[0.01, 0.03, 0.1], max_leaf_nodes=[2, 3, 7, 15, 31], min_samples_leaf=[50, 200],
            max_iter=[100, 300, 600])
grid = []
for lr, leaves, msl, iters in product(*GRID.values()):
    t0 = time.time()
    m = make_model(lr, leaves, msl, iters).fit(tr[FEATURES], tr.late)
    val["score"] = m.predict_proba(val[FEATURES])[:, 1]
    t = totals(score_days(val, select_daily(val, "score", K), K))
    grid.append({"learning_rate": lr, "max_leaf_nodes": leaves, "min_samples_leaf": msl, "n_trees": iters,
                 "val_hits": t["hits"], "val_wasted": t["wasted"],
                 "val_hit_rate": round(t["hit_rate_of_picks"], 4),
                 "val_pr_auc": round(average_precision(val.loc[val.eligible == 1, "score"], val_lab.late), 4),
                 "fit_seconds": round(time.time() - t0, 1)})
    print(grid[-1])
grid = pd.DataFrame(grid)
grid.to_csv(RES / "tuning_validation.csv", index=False)
best = grid.sort_values(["val_hits", "val_pr_auc"], ascending=False).iloc[0]
cfg = {"model": "HistGradientBoostingClassifier", "learning_rate": float(best.learning_rate),
       "max_leaf_nodes": int(best.max_leaf_nodes), "min_samples_leaf": int(best.min_samples_leaf),
       "n_trees": int(best.n_trees), "l2_regularization": 1.0, "loss": "log_loss",
       "selection": "most validation daily top-10 hits, ties -> validation PR-AUC",
       "grid": GRID, "features": FEATURES, "categorical": CAT, "category_min_frequency": 100,
       "k_per_day": K, "random_state": SEED, "train_rows_labeled": int(len(tr))}
(RES / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("chosen:", {k: cfg[k] for k in ["learning_rate", "max_leaf_nodes", "min_samples_leaf", "n_trees"]})

# ---------------------------------------------------------------- final fit with the chosen setting
model = make_model(cfg["learning_rate"], cfg["max_leaf_nodes"], cfg["min_samples_leaf"],
                   cfg["n_trees"]).fit(tr[FEATURES], tr.late)
joblib.dump(model, RES / "model.joblib")
summary, days = {}, {}
for split, df in [("train", train), ("validation", val)]:
    df["score"] = model.predict_proba(df[FEATURES])[:, 1]
    p = select_daily(df, "score", K)
    d = score_days(df, p, K)
    s = totals(d)
    lab = df[df.eligible == 1]
    s["pr_auc_all_orders"] = round(average_precision(lab.score, lab.late), 4)
    s["pr_auc_random"] = round(lab.late.mean(), 4)
    summary[split], days[split] = s, d
    p.assign(outcome=outcome_label(p))[["approval_date", "rank", "order_id", "order_approved_at", "score",
                                        "customer_state", "seller_state", "outcome"]] \
        .to_csv(RES / f"daily_picks_{split}.csv", index=False)
    d.to_csv(RES / f"daily_scores_{split}.csv")
val[["order_id", "approval_date", "eligible", "late", "score"]].to_csv(RES / "predictions_validation.csv", index=False)
pd.DataFrame(summary).to_csv(RES / "summary.csv")
print(pd.DataFrame(summary).to_string())

# ---------------------------------------------------------------- training curve (tree by tree)
gb = model.named_steps["gb"]
Xtr = model.named_steps["prep"].transform(tr[FEATURES])
Xva = model.named_steps["prep"].transform(val[FEATURES])
lab_mask = (val.eligible == 1).to_numpy()
curve = []
for n_trees, (ptr, pva) in enumerate(zip(gb.staged_predict_proba(Xtr), gb.staged_predict_proba(Xva)), start=1):
    if n_trees % 5 and n_trees != 1:
        continue
    val["score_k"] = pva[:, 1]
    curve.append({"n_trees": n_trees, "train_log_loss": log_loss(tr.late, ptr[:, 1]),
                  "val_log_loss": log_loss(val_lab.late, pva[lab_mask, 1]),
                  "val_hits": totals(score_days(val, select_daily(val, "score_k", K), K))["hits"]})
curve = pd.DataFrame(curve)
curve.to_csv(RES / "training_curve.csv", index=False)

# ---------------------------------------------------------------- feature importance (validation)
# Permutation importance: shuffle one input column at a time and measure how much the
# validation PR-AUC drops. A big drop = the model relies on that input.
pi = permutation_importance(model, val_lab[FEATURES], val_lab.late, scoring="average_precision",
                            n_repeats=5, random_state=SEED, n_jobs=-1)
imp = pd.DataFrame({"feature": FEATURES, "pr_auc_drop": pi.importances_mean, "std": pi.importances_std})
imp = imp.sort_values("pr_auc_drop", ascending=False)
imp.to_csv(RES / "feature_importance_validation.csv", index=False)

# ================================================================ figures
HIT, WASTED, UNKNOWN, GRAY = "#eb6834", "#2a78d6", "#b5b5b5", "#6b6b6b"
GB_C, LR_C, DIST_C, SELL_C = "#c2185b", "#7b3fbf", "#1baf7a", "#d4a017"
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
    return f"{100 * float(x):.1f}%"


# other methods' validation results (read from their own folders; nothing is recomputed here)
others = {"Logistic regression": ROOT / "baseline_logisticRegression" / "results",
          "Longest distance rule": ROOT / "nonML_distanceRule" / "results",
          "Riskiest seller rule": ROOT / "nonML_riskiestSellers" / "results"}
o_sum = {k: pd.read_csv(v / "summary.csv", index_col=0)["validation"] for k, v in others.items()}
o_days = {k: pd.read_csv(v / "daily_scores_validation.csv", index_col=0, parse_dates=True) for k, v in others.items()}
o_budget = {k: pd.read_csv(others[k] / "budget_sensitivity_validation.csv")
            for k in ["Logistic regression", "Longest distance rule"]}

vs = summary["validation"]
d = days["validation"]
x = d.index
n_days, n_picked = len(d), int(vs["picked"])

# --- fig 1: comparison table
cols = {"Gradient boosting": vs, **o_sum}
rows = [("Hits: investigated and late", "hits", lambda v: f"{int(v):,}"),
        ("Wasted: investigated but on time", "wasted", lambda v: f"{int(v):,}"),
        ("Unknown outcome", "unknown", lambda v: f"{int(v):,}"),
        ("Missed: late but not investigated", "missed", lambda v: f"{int(v):,}"),
        ("Hit rate of investigations", "hit_rate_of_picks", pct),
        ("Share of all late orders caught", "share_of_late_caught", pct),
        (f"Share of best possible ({int(vs['best_possible_hits'])})", "share_of_best_possible", pct),
        (f"Hits compared with random ({vs['random_expected_hits']:.0f})", "lift_over_random",
         lambda v: f"{float(v):.2f}x"),
        (f"PR-AUC (random = {vs['pr_auc_random']:.3f})", "pr_auc_all_orders", lambda v: f"{float(v):.3f}")]
tab = pd.DataFrame([[n] + [f(cols[c][k]) for c in cols] for n, k, f in rows], columns=["Validation"] + list(cols))
table_fig(tab, "Gradient boosting compared with logistic regression and the two rules (validation)",
          "table_validation_comparison",
          note=f"Validation = Jan 1 - Apr 15, 2018: {n_days} days, {n_picked:,} investigations (10 per day). "
               f"Models trained on 2017 only.")

# --- fig 2: tuning table (top 12 settings + chosen)
gt = grid.sort_values(["val_hits", "val_pr_auc"], ascending=False).head(12).copy()
gt["chosen"] = np.where((gt.learning_rate == best.learning_rate) & (gt.max_leaf_nodes == best.max_leaf_nodes)
                        & (gt.min_samples_leaf == best.min_samples_leaf) & (gt.n_trees == best.n_trees),
                        "<- chosen", "")
gt = gt[["learning_rate", "max_leaf_nodes", "min_samples_leaf", "n_trees", "val_hits", "val_wasted",
         "val_hit_rate", "val_pr_auc", "chosen"]]
gt.columns = ["Learning rate", "Leaves per tree", "Min orders per leaf", "Trees", "Hits", "Wasted",
              "Hit rate", "PR-AUC", ""]
gt["Hit rate"] = gt["Hit rate"].map(pct)
table_fig(gt, f"The 12 best of {len(grid)} settings tried, scored on validation", "table_tuning",
          note="Hits spread: best {} / worst {} across all settings. Full list in results/tuning_validation.csv."
               .format(int(grid.val_hits.max()), int(grid.val_hits.min())))

# --- fig 3: daily picks
fig, ax = plt.subplots(figsize=(12, 4.2))
ax.bar(x, d.hit, color=HIT, label="hit (late)", width=0.85)
ax.bar(x, d.wasted, bottom=d.hit, color=WASTED, label="wasted (on time)", width=0.85)
ax.bar(x, d.unknown, bottom=d.hit + d.wasted, color=UNKNOWN, label="unknown outcome", width=0.85)
ax.step(x, d.best_possible, where="mid", color="black", lw=1.3, label="best possible hits")
ax.set_ylabel("Orders investigated")
ax.set_ylim(0, 11.5)
ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
title(ax, "Gradient boosting: what the 10 daily investigations found (validation)")
save(fig, "validation_daily_picks")

# --- fig 4: cumulative hits vs everything
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(x, d.best_possible.cumsum(), color="black", lw=2, label="best possible (perfect list)")
ax.plot(x, d.hit.cumsum(), color=GB_C, lw=2.5, label="gradient boosting")
for name, col in [("Logistic regression", LR_C), ("Longest distance rule", DIST_C), ("Riskiest seller rule", SELL_C)]:
    ax.plot(x, o_days[name].hit.cumsum(), color=col, lw=2, label=name.lower())
ax.plot(x, d.random_expected.cumsum(), color=GRAY, lw=1.5, ls="--", label="random 10 per day")
ax.set_ylabel("Late orders caught (running total)")
ax.legend(frameon=False, loc="upper left")
title(ax, "Running total of late orders caught (validation)")
save(fig, "validation_cumulative_hits")

# --- fig 5: training curve (tree by tree)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(curve.n_trees, curve.train_log_loss, color=GB_C, lw=2, label="train (2017)")
axs[0].plot(curve.n_trees, curve.val_log_loss, color=GRAY, lw=2, ls="--", label="validation")
axs[0].set_ylabel("Log-loss (lower is better)")
axs[0].set_title("Loss", loc="left")
axs[0].legend(frameon=False)
axs[1].plot(curve.n_trees, curve.val_hits, color=GB_C, lw=2)
axs[1].axhline(float(o_sum["Logistic regression"]["hits"]), color=LR_C, lw=1.5, ls=":",
               label=f"logistic regression ({int(o_sum['Logistic regression']['hits'])})")
axs[1].set_ylabel("Validation hits (10 per day)")
axs[1].set_title("Late orders caught on validation", loc="left")
axs[1].legend(frameon=False, loc="lower right")
for ax in axs:
    ax.set_xlabel("Number of trees")
fig.suptitle(f"Figure {FIG_NUM[0] + 1}. How the model improves as trees are added "
             f"(final model: {cfg['n_trees']} trees)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "training_curve")

# --- fig 6: feature importance
top = imp.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top.feature, top.pr_auc_drop, xerr=top["std"], color=GB_C, alpha=0.85)
ax.set_xlabel("Drop in validation PR-AUC when the input is shuffled")
title(ax, "The 15 inputs the gradient boosting relies on most")
fig.text(0.01, -0.02, "Bigger drop = the model depends more on that input. Error bars: spread over 5 shuffles.",
         fontsize=9, color=GRAY)
save(fig, "feature_importance")

# --- fig 7: budget sensitivity vs logistic regression and distance rule
ks = [1, 2, 3, 5, 10, 15, 20, 30, 50]
bs = []
for k in ks:
    t = totals(score_days(val, select_daily(val, "score", k), k))
    bs.append((k, t["hits"], t["hit_rate_of_picks"], t["random_expected_hits"]))
bs = pd.DataFrame(bs, columns=["k", "hits", "precision", "random_hits"])
bs.to_csv(RES / "budget_sensitivity_validation.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(bs.k, bs.hits, color=GB_C, marker="o", lw=2, label="gradient boosting")
axs[0].plot(o_budget["Logistic regression"].k, o_budget["Logistic regression"].hits, color=LR_C, marker="o",
            lw=2, label="logistic regression")
axs[0].plot(o_budget["Longest distance rule"].k, o_budget["Longest distance rule"].hits, color=DIST_C,
            marker="o", lw=2, label="longest distance rule")
axs[0].plot(bs.k, bs.random_hits, color=GRAY, marker="o", lw=1.5, ls="--", label="random")
axs[0].set_ylabel("Late orders caught")
axs[0].set_title("Late orders caught", loc="left")
axs[0].legend(frameon=False)
axs[1].plot(bs.k, 100 * bs.precision, color=GB_C, marker="o", lw=2)
axs[1].plot(o_budget["Logistic regression"].k, 100 * o_budget["Logistic regression"].precision, color=LR_C,
            marker="o", lw=2)
axs[1].plot(o_budget["Longest distance rule"].k, 100 * o_budget["Longest distance rule"].precision,
            color=DIST_C, marker="o", lw=2)
axs[1].axhline(100 * val_lab.late.mean(), color=GRAY, ls="--", lw=1.5)
axs[1].set_ylabel("Hit rate of investigations (%)")
axs[1].set_title("Share of investigations that were late", loc="left")
for ax in axs:
    ax.set_xlabel("Investigations per day")
    ax.axvline(10, color="black", lw=0.8, ls=":")
fig.suptitle(f"Figure {FIG_NUM[0] + 1}. More or fewer investigations per day (validation)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "budget_sensitivity")

# --- fig 8: weekly hit rate
fig, ax = plt.subplots(figsize=(10, 4.4))
series = [("gradient boosting", d, GB_C)] + [(n.lower(), o_days[n], c) for n, c in
                                             [("Logistic regression", LR_C), ("Longest distance rule", DIST_C),
                                              ("Riskiest seller rule", SELL_C)]]
for name, dd, col in series:
    w = dd.resample("W").sum()
    ax.plot(w.index, 100 * w.hit / (w.hit + w.wasted), color=col, lw=2, marker="o", ms=3, label=name)
w = d.resample("W").sum()
ax.plot(w.index, 100 * w.late / w.labeled, color=GRAY, lw=1.5, ls="--", label="random (late rate)")
ax.set_ylabel("Hit rate of investigations (%)")
ax.legend(frameon=False, loc="upper left")
title(ax, "Weekly hit rate of the 10 daily picks (validation)")
save(fig, "weekly_hit_rate")
