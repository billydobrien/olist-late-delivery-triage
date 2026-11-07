"""Baseline ML model: logistic regression predicting P(late) at order approval.

Run:  python baseline_logisticRegression/train_logistic.py
Uses: dataPrep/output/train.csv (fit), validation.csv (choose settings, score the daily lists),
      dataPrep/output/feature_availability.csv (which columns may be model inputs).
      The test set is NOT touched here.
Out:  results/  tuning table, chosen config, coefficients, validation predictions,
                daily picks and scores, summary, saved model
      figures/  simple figures, compared with the two non-ML rules

Steps
  1. Inputs = every feature the availability audit marks "yes" (approval_month is excluded).
  2. Preprocessing is learned on TRAIN only, then applied unchanged to validation:
       - count / money / size / time-gap columns: fill missing with the train median
         (plus a "was missing" flag), log(1 + x), then standardize
       - other numeric columns: fill missing with the train median (+ flag), standardize
       - text columns (states, payment type, category, weekday): one 0/1 column per value;
         values seen fewer than 100 times in train are grouped as "infrequent"
  3. Fit on train orders that have a label (eligible == 1).
  4. Small grid: regularization strength C x class weighting. The setting with the most
     validation hits in the daily top-10 is kept (ties -> higher validation PR-AUC).
  5. Score every validation candidate, pick 10 per day, count hits / wasted / unknown / missed.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

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

# ---------------------------------------------------------------- features
audit = pd.read_csv(PREP / "feature_availability.csv")
FEATURES = audit[(audit.group != "excluded") & audit["use as model input"].str.startswith("yes")].column.tolist()
CAT = ["customer_state", "seller_state", "payment_type", "main_category", "approval_weekday"]
LOG = ["purchase_to_approval_h", "promised_days", "seller_ship_limit_days", "n_items", "n_products",
       "n_sellers", "total_price", "total_freight", "freight_ratio", "total_weight_g", "total_volume_cm3",
       "n_categories", "n_payments", "max_installments", "payment_value", "distance_km",
       "seller_prev_orders", "customer_prev_orders", "orders_last_24h", "recent_deliveries_14d",
       "seller_prev_delivered", "route_prev_delivered", "customer_state_prev_delivered",
       "category_prev_delivered"]
PLAIN = [c for c in FEATURES if c not in CAT + LOG]
assert set(CAT + LOG + PLAIN) == set(FEATURES)


def log1p_nonneg(x):
    return np.log1p(np.clip(x, 0, None))


def make_model(C, class_weight):
    pre = ColumnTransformer([
        ("log", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              FunctionTransformer(log1p_nonneg, feature_names_out="one-to-one"),
                              StandardScaler()), LOG),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), PLAIN),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100)), CAT),
    ])
    return Pipeline([("prep", pre),
                     ("lr", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000,
                                               random_state=SEED))])


def load(split):
    df = pd.read_csv(PREP / f"{split}.csv", parse_dates=["approval_date"])
    for c in CAT:
        df[c] = df[c].astype("string").astype(object).where(df[c].notna(), np.nan)
    return df


train, val = load("train"), load("validation")
tr = train[train.eligible == 1]
val_lab = val[val.eligible == 1]
print(f"train labeled: {len(tr):,} ({int(tr.late.sum()):,} late); features: {len(FEATURES)} "
      f"({len(LOG)} log-scaled, {len(PLAIN)} plain, {len(CAT)} categorical)")

# ---------------------------------------------------------------- tuning grid
grid = []
for C in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0]:
    for cw in [None, "balanced"]:
        m = make_model(C, cw).fit(tr[FEATURES], tr.late)
        val["score"] = m.predict_proba(val[FEATURES])[:, 1]
        t = totals(score_days(val, select_daily(val, "score", K), K))
        grid.append({"C": C, "class_weight": cw or "none", "val_hits": t["hits"], "val_wasted": t["wasted"],
                     "val_hit_rate": round(t["hit_rate_of_picks"], 4),
                     "val_pr_auc": round(average_precision(val.loc[val.eligible == 1, "score"], val_lab.late), 4)})
        print(grid[-1])
grid = pd.DataFrame(grid)
grid.to_csv(RES / "tuning_validation.csv", index=False)
best = grid.sort_values(["val_hits", "val_pr_auc"], ascending=False).iloc[0]
cfg = {"model": "logistic regression", "C": float(best.C),
       "class_weight": None if best.class_weight == "none" else best.class_weight,
       "selection": "most validation daily top-10 hits, ties -> validation PR-AUC",
       "features": FEATURES, "log_scaled": LOG, "categorical": CAT, "plain": PLAIN,
       "one_hot_min_frequency": 100, "k_per_day": K, "train_rows_labeled": int(len(tr))}
(RES / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("chosen:", cfg["C"], cfg["class_weight"])

# ---------------------------------------------------------------- final fit with the chosen setting
model = make_model(cfg["C"], cfg["class_weight"]).fit(tr[FEATURES], tr.late)
joblib.dump(model, RES / "model.joblib")
summary, days, picks = {}, {}, {}
for split, df in [("train", train), ("validation", val)]:
    df["score"] = model.predict_proba(df[FEATURES])[:, 1]
    p = select_daily(df, "score", K)
    d = score_days(df, p, K)
    s = totals(d)
    lab = df[df.eligible == 1]
    s["pr_auc_all_orders"] = round(average_precision(lab.score, lab.late), 4)
    s["pr_auc_random"] = round(lab.late.mean(), 4)
    summary[split], days[split], picks[split] = s, d, p
    p.assign(outcome=outcome_label(p))[["approval_date", "rank", "order_id", "order_approved_at", "score",
                                        "customer_state", "seller_state", "outcome"]] \
        .to_csv(RES / f"daily_picks_{split}.csv", index=False)
    d.to_csv(RES / f"daily_scores_{split}.csv")
val[["order_id", "approval_date", "eligible", "late", "score"]].to_csv(RES / "predictions_validation.csv", index=False)
pd.DataFrame(summary).to_csv(RES / "summary.csv")
print(pd.DataFrame(summary).to_string())

names = model.named_steps["prep"].get_feature_names_out()
coef = pd.DataFrame({"feature": names, "coefficient": model.named_steps["lr"].coef_[0]})
coef["abs"] = coef.coefficient.abs()
coef.sort_values("abs", ascending=False).drop(columns="abs").to_csv(RES / "coefficients.csv", index=False)

# ================================================================ figures
HIT, WASTED, UNKNOWN, GRAY = "#eb6834", "#2a78d6", "#b5b5b5", "#6b6b6b"
MODEL_C, DIST_C, SELL_C = "#7b3fbf", "#1baf7a", "#d4a017"     # model / distance rule / seller rule
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
    return f"{100 * x:.1f}%"


# rule results (read from their own folders; nothing is recomputed here)
rules = {"Longest distance rule": ROOT / "nonML_distanceRule" / "results",
         "Riskiest seller rule": ROOT / "nonML_riskiestSellers" / "results"}
rule_sum = {k: pd.read_csv(v / "summary.csv", index_col=0)["validation"] for k, v in rules.items()}
rule_days = {k: pd.read_csv(v / "daily_scores_validation.csv", index_col=0, parse_dates=True) for k, v in rules.items()}

# --- fig 1: validation comparison table
vs = summary["validation"]
cols = {"Logistic regression": vs, **rule_sum}
rows = [("Hits: investigated and late", "hits", lambda v: f"{int(v):,}"),
        ("Wasted: investigated but on time", "wasted", lambda v: f"{int(v):,}"),
        ("Unknown outcome", "unknown", lambda v: f"{int(v):,}"),
        ("Missed: late but not investigated", "missed", lambda v: f"{int(v):,}"),
        ("Hit rate of investigations", "hit_rate_of_picks", pct),
        ("Share of all late orders caught", "share_of_late_caught", pct),
        ("Share of best possible (956)", "share_of_best_possible", pct),
        ("Hits compared with random (118)", "lift_over_random", lambda v: f"{float(v):.2f}x"),
        ("PR-AUC (random = 0.119)", "pr_auc_all_orders", lambda v: f"{float(v):.3f}")]
tab = pd.DataFrame([[n] + [f(cols[c][k]) for c in cols] for n, k, f in rows], columns=["Validation"] + list(cols))
table_fig(tab, "Logistic regression compared with the two non-ML rules (validation, 10 per day)",
          "table_validation_comparison",
          note=f"Validation = Jan 1 - Apr 15, 2018: 105 days, 1,045 investigations. Model trained on 2017 only.")

# --- fig 2: tuning table
gt = grid.copy()
gt["chosen"] = np.where((gt.C == best.C) & (gt.class_weight == best.class_weight), "<- chosen", "")
gt.columns = ["C (regularization)", "Class weight", "Hits", "Wasted", "Hit rate", "PR-AUC", ""]
gt["Hit rate"] = gt["Hit rate"].map(pct)
table_fig(gt, "Settings tried, scored on validation", "table_tuning",
          note="Smaller C = simpler model. 'balanced' gives late orders more weight during training. "
               "Chosen by most validation hits.")

# --- fig 3: daily picks
d = days["validation"]
x = d.index
fig, ax = plt.subplots(figsize=(12, 4.2))
ax.bar(x, d.hit, color=HIT, label="hit (late)", width=0.85)
ax.bar(x, d.wasted, bottom=d.hit, color=WASTED, label="wasted (on time)", width=0.85)
ax.bar(x, d.unknown, bottom=d.hit + d.wasted, color=UNKNOWN, label="unknown outcome", width=0.85)
ax.step(x, d.best_possible, where="mid", color="black", lw=1.3, label="best possible hits")
ax.set_ylabel("Orders investigated")
ax.set_ylim(0, 11.5)
ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
title(ax, "Logistic regression: what the 10 daily investigations found (validation)")
save(fig, "validation_daily_picks")

# --- fig 4: cumulative hits vs rules
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(x, d.best_possible.cumsum(), color="black", lw=2, label="best possible (perfect list)")
ax.plot(x, d.hit.cumsum(), color=MODEL_C, lw=2.5, label="logistic regression")
ax.plot(x, rule_days["Longest distance rule"].hit.cumsum(), color=DIST_C, lw=2, label="longest distance rule")
ax.plot(x, rule_days["Riskiest seller rule"].hit.cumsum(), color=SELL_C, lw=2, label="riskiest seller rule")
ax.plot(x, d.random_expected.cumsum(), color=GRAY, lw=1.5, ls="--", label="random 10 per day")
ax.set_ylabel("Late orders caught (running total)")
ax.legend(frameon=False, loc="upper left")
title(ax, "Running total of late orders caught (validation)")
save(fig, "validation_cumulative_hits")

# --- fig 5: largest coefficients
top = coef.sort_values("abs", ascending=False).head(20).iloc[::-1]
lbl = (top.feature.str.replace(r"^(log|num|cat)__", "", regex=True)
       .str.replace("missingindicator_", "missing: ", regex=False))
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(lbl, top.coefficient, color=np.where(top.coefficient > 0, HIT, WASTED))
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Coefficient (positive = more likely late)")
title(ax, "The 20 strongest inputs of the logistic regression")
fig.text(0.01, -0.02, "Numeric inputs are standardized, so sizes are comparable. "
         "State/category names are 0/1 columns compared with the other values.", fontsize=9, color=GRAY)
save(fig, "top_coefficients")

# --- fig 6: budget sensitivity vs distance rule
ks = [1, 2, 3, 5, 10, 15, 20, 30, 50]
bs = []
for k in ks:
    t = totals(score_days(val, select_daily(val, "score", k), k))
    bs.append((k, t["hits"], t["hit_rate_of_picks"], t["random_expected_hits"]))
bs = pd.DataFrame(bs, columns=["k", "hits", "precision", "random_hits"])
bs.to_csv(RES / "budget_sensitivity_validation.csv", index=False)
db = pd.read_csv(rules["Longest distance rule"] / "budget_sensitivity_validation.csv")
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(bs.k, bs.hits, color=MODEL_C, marker="o", lw=2, label="logistic regression")
axs[0].plot(db.k, db.hits, color=DIST_C, marker="o", lw=2, label="longest distance rule")
axs[0].plot(bs.k, bs.random_hits, color=GRAY, marker="o", lw=1.5, ls="--", label="random")
axs[0].set_ylabel("Late orders caught")
axs[0].set_title("Late orders caught", loc="left")
axs[0].legend(frameon=False)
axs[1].plot(bs.k, 100 * bs.precision, color=MODEL_C, marker="o", lw=2)
axs[1].plot(db.k, 100 * db.precision, color=DIST_C, marker="o", lw=2)
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

# --- fig 7: weekly hit rate vs rules
fig, ax = plt.subplots(figsize=(10, 4.2))
for name, dd, col in [("logistic regression", d, MODEL_C),
                      ("longest distance rule", rule_days["Longest distance rule"], DIST_C),
                      ("riskiest seller rule", rule_days["Riskiest seller rule"], SELL_C)]:
    w = dd.resample("W").sum()
    ax.plot(w.index, 100 * w.hit / (w.hit + w.wasted), color=col, lw=2, marker="o", ms=3, label=name)
w = d.resample("W").sum()
ax.plot(w.index, 100 * w.late / w.labeled, color=GRAY, lw=1.5, ls="--", label="random (late rate)")
ax.set_ylabel("Hit rate of investigations (%)")
ax.legend(frameon=False, loc="upper left")
title(ax, "Weekly hit rate of the 10 daily picks (validation)")
save(fig, "weekly_hit_rate")

# --- fig 8: convergence of the solver
# The L-BFGS solver is deterministic and starts from all-zero weights, so stopping it after
# k steps reproduces exactly the k-th point of the final run's path. Refit for k = 1..n.
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss

prep = model.named_steps["prep"]
Xtr, Xva = prep.transform(tr[FEATURES]), prep.transform(val[FEATURES])
lab_mask = (val.eligible == 1).to_numpy()
n_final = int(model.named_steps["lr"].n_iter_[0])
steps = sorted(set(list(range(1, min(n_final, 40) + 1)) + list(np.linspace(1, n_final, 40).astype(int))))
conv = []
for k in steps:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lr_k = LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"], max_iter=k,
                                  random_state=SEED).fit(Xtr, tr.late)
    val["score_k"] = lr_k.predict_proba(Xva)[:, 1]
    hits_k = totals(score_days(val, select_daily(val, "score_k", K), K))["hits"]
    conv.append({"iteration": k,
                 "train_log_loss": log_loss(tr.late, lr_k.predict_proba(Xtr)[:, 1]),
                 "val_log_loss": log_loss(val_lab.late, val.loc[lab_mask, "score_k"]),
                 "val_hits": hits_k})
conv = pd.DataFrame(conv)
conv.to_csv(RES / "convergence.csv", index=False)
print(f"solver converged after {n_final} iterations")

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(conv.iteration, conv.train_log_loss, color=MODEL_C, lw=2, label="train (2017)")
axs[0].plot(conv.iteration, conv.val_log_loss, color=GRAY, lw=2, ls="--", label="validation")
axs[0].set_ylabel("Log-loss (lower is better)")
axs[0].set_title("Loss", loc="left")
axs[0].legend(frameon=False)
axs[1].plot(conv.iteration, conv.val_hits, color=MODEL_C, lw=2)
axs[1].axhline(218, color=DIST_C, lw=1.5, ls=":", label="longest distance rule (218)")
axs[1].set_ylabel("Validation hits (10 per day)")
axs[1].set_title("Late orders caught on validation", loc="left")
axs[1].legend(frameon=False, loc="lower right")
for ax in axs:
    ax.set_xlabel("Solver iteration")
    ax.axvline(n_final, color="black", lw=0.8, ls=":")
fig.suptitle(f"Figure {FIG_NUM[0] + 1}. How fast the logistic regression converges "
             f"(stopped after {n_final} iterations)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "convergence")
