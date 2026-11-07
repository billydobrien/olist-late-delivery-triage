"""FINAL TEST: score the held-out test period (approvals 2018-04-16 .. 2018-07-31) once.

Decisions fixed BEFORE this run (2026-11-06):
  * Proposed model: gradient boosting. Logistic regression and the two non-ML rules are
    scored the same way for comparison.
  * No retraining: the models are the saved files trained on 2017 only, with the settings
    chosen on validation. Nothing here is tuned or chosen from test results.
  * Same daily policy and scoring as validation: every approved order on a day is a
    candidate, top 10 per day, hits / wasted / unknown / missed (daily_eval.py).

Run:  python testResults/run_test.py
Out:  results/daily_list_test_<method>.csv   saved daily prioritization lists (required deliverable)
      results/predictions_test_<method>.csv  score for every test order
      results/daily_scores_test_<method>.csv per-day counts
      results/summary_test.csv               totals for all methods + random + perfect
      results/budget_sensitivity_test.csv
      figures/                               simple figures
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd

from daily_eval import select_daily, score_days, totals, average_precision, outcome_label

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREP = ROOT / "dataPrep" / "output"
RES = HERE / "results"
FIG = HERE / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
K = 10
CAT = ["customer_state", "seller_state", "payment_type", "main_category", "approval_weekday"]


def log1p_nonneg(x):   # needed to load the saved logistic regression pipeline (same function it was saved with)
    return np.log1p(np.clip(x, 0, None))


# ---------------------------------------------------------------- data
test = pd.read_csv(PREP / "test.csv", parse_dates=["approval_date"])
for c in CAT:
    test[c] = test[c].astype("string").astype(object).where(test[c].notna(), np.nan)
lab = test[test.eligible == 1]
print(f"test: {len(test):,} candidate orders, {len(lab):,} with an outcome, {int(lab.late.sum()):,} late, "
      f"{test.approval_date.nunique()} days")

# ---------------------------------------------------------------- scores for each method
gb_dir, lr_dir = ROOT / "model_gradientBoosting" / "results", ROOT / "baseline_logisticRegression" / "results"
gb_cfg = json.loads((gb_dir / "config.json").read_text())
lr_cfg = json.loads((lr_dir / "config.json").read_text())
test["score_gradient_boosting"] = joblib.load(gb_dir / "model.joblib").predict_proba(test[gb_cfg["features"]])[:, 1]
test["score_logistic_regression"] = joblib.load(lr_dir / "model.joblib").predict_proba(test[lr_cfg["features"]])[:, 1]
test["score_distance_rule"] = test.distance_km
test["score_seller_rule"] = test.seller_prev_late_rate
lab = test[test.eligible == 1]          # labeled orders, now with every method's score

METHODS = {"Gradient boosting (proposed)": ("score_gradient_boosting", "gradient_boosting", "#c2185b"),
           "Logistic regression": ("score_logistic_regression", "logistic_regression", "#7b3fbf"),
           "Longest distance rule": ("score_distance_rule", "distance_rule", "#1baf7a"),
           "Riskiest seller rule": ("score_seller_rule", "seller_rule", "#d4a017")}

summary, days = {}, {}
for name, (col, slug, _) in METHODS.items():
    p = select_daily(test, col, K)
    d = score_days(test, p, K)
    s = totals(d)
    s["pr_auc"] = round(average_precision(lab[col], lab.late), 4)
    summary[name], days[name] = s, d
    p.assign(outcome=outcome_label(p)).rename(columns={col: "risk_score"})[
        ["approval_date", "rank", "order_id", "order_approved_at", "risk_score", "customer_state",
         "seller_state", "outcome"]].to_csv(RES / f"daily_list_test_{slug}.csv", index=False)
    test[["order_id", "approval_date", "eligible", "late", col]].rename(columns={col: "risk_score"}) \
        .to_csv(RES / f"predictions_test_{slug}.csv", index=False)
    d.to_csv(RES / f"daily_scores_test_{slug}.csv")

# random and perfect references (same per-day arithmetic as validation)
d0 = days["Gradient boosting (proposed)"]
k = np.minimum(K, d0.candidates)
r_hit, r_wasted = (k * d0.late / d0.candidates).sum(), (k * (d0.labeled - d0.late) / d0.candidates).sum()
r_unk = (k * (d0.candidates - d0.labeled) / d0.candidates).sum()
p_hit = d0.best_possible.sum()
L = d0.late.sum()
ref = {
    "Random 10 per day (expected)": {"days": len(d0), "picked": int(k.sum()), "hits": r_hit, "wasted": r_wasted,
                                     "unknown": r_unk, "missed": L - r_hit,
                                     "hit_rate_of_picks": r_hit / (r_hit + r_wasted), "share_of_late_caught": r_hit / L,
                                     "share_of_best_possible": r_hit / p_hit, "lift_over_random": 1.0,
                                     "pr_auc": round(lab.late.mean(), 4)},
    "Perfect list (knows the future)": {"days": len(d0), "picked": int(k.sum()), "hits": p_hit,
                                        "wasted": k.sum() - p_hit, "unknown": 0, "missed": L - p_hit,
                                        "hit_rate_of_picks": p_hit / k.sum(), "share_of_late_caught": p_hit / L,
                                        "share_of_best_possible": 1.0, "lift_over_random": p_hit / r_hit,
                                        "pr_auc": 1.0},
}
full = {**summary, **ref}
cols_keep = ["days", "picked", "hits", "wasted", "unknown", "missed", "hit_rate_of_picks", "share_of_late_caught",
             "share_of_best_possible", "lift_over_random", "pr_auc"]
pd.DataFrame({k_: {c: v[c] for c in cols_keep} for k_, v in full.items()}).to_csv(RES / "summary_test.csv")
print(pd.DataFrame({k_: {c: v[c] for c in cols_keep} for k_, v in full.items()}).round(3).to_string())

# budget sensitivity on test
bs = []
for name, (col, slug, _) in METHODS.items():
    for kk in [1, 2, 3, 5, 10, 15, 20, 30, 50]:
        t = totals(score_days(test, select_daily(test, col, kk), kk))
        bs.append({"method": name, "k": kk, "hits": t["hits"], "hit_rate": t["hit_rate_of_picks"],
                   "random_hits": t["random_expected_hits"]})
bs = pd.DataFrame(bs)
bs.to_csv(RES / "budget_sensitivity_test.csv", index=False)

(RES / "test_run_log.txt").write_text(
    f"Test scored on {datetime.now():%Y-%m-%d %H:%M}. Models loaded from saved files (trained on 2017, "
    f"settings chosen on validation); no retraining or tuning on test.\n", encoding="utf-8")

# ================================================================ figures
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


def pct(x):
    return f"{100 * float(x):.1f}%"


# --- fig 1: main test table (all methods + random + perfect)
short = {"Gradient boosting (proposed)": "Gradient boosting", "Logistic regression": "Logistic regr.",
         "Longest distance rule": "Distance rule", "Riskiest seller rule": "Seller rule",
         "Random 10 per day (expected)": "Random", "Perfect list (knows the future)": "Perfect"}
order = ["Perfect list (knows the future)", "Gradient boosting (proposed)", "Logistic regression",
         "Longest distance rule", "Riskiest seller rule", "Random 10 per day (expected)"]
rows = [("Hits: investigated and late", "hits", lambda v: f"{float(v):,.0f}"),
        ("Wasted: investigated but on time", "wasted", lambda v: f"{float(v):,.0f}"),
        ("Unknown outcome", "unknown", lambda v: f"{float(v):,.0f}"),
        ("Missed: late but not investigated", "missed", lambda v: f"{float(v):,.0f}"),
        ("Hit rate of investigations", "hit_rate_of_picks", pct),
        ("Share of all late orders caught", "share_of_late_caught", pct),
        ("Share of best possible", "share_of_best_possible", pct),
        ("Hits compared with random", "lift_over_random", lambda v: f"{float(v):.2f}x"),
        ("PR-AUC", "pr_auc", lambda v: f"{float(v):.3f}")]
tab = pd.DataFrame([[n] + [f(full[m][key]) for m in order] for n, key, f in rows],
                   columns=["Test (Apr 16 - Jul 31, 2018)"] + [short[m] for m in order])
table_fig(tab, "Final test results: 10 investigations per day", "table_test_results",
          note=f"{len(d0)} days, {int(k.sum()):,} investigations, {int(L):,} late orders. Models trained on 2017, "
               f"settings chosen on validation, scored on test once. Random = expected value.")

# --- fig 2: validation vs test for each method
val_sum = {"Gradient boosting (proposed)": gb_dir, "Logistic regression": lr_dir,
           "Longest distance rule": ROOT / "nonML_distanceRule" / "results",
           "Riskiest seller rule": ROOT / "nonML_riskiestSellers" / "results"}
rows = []
for m, folder in val_sum.items():
    v = pd.read_csv(folder / "summary.csv", index_col=0)["validation"]
    t = full[m]
    rows.append([short[m], f"{int(v['hits'])}", pct(v["hit_rate_of_picks"]), f"{float(v['lift_over_random']):.2f}x",
                 f"{int(t['hits'])}", pct(t["hit_rate_of_picks"]), f"{float(t['lift_over_random']):.2f}x"])
table_fig(pd.DataFrame(rows, columns=["Method", "Val hits", "Val hit rate", "Val vs random",
                                      "Test hits", "Test hit rate", "Test vs random"]),
          "Validation compared with test", "table_validation_vs_test",
          note=f"Validation: 105 days, 11.9% of orders late. Test: {len(d0)} days, "
               f"{100 * lab.late.mean():.1f}% late, so fewer hits are possible on test.")

# --- fig 3: daily picks of the proposed model
d = days["Gradient boosting (proposed)"]
x = d.index
fig, ax = plt.subplots(figsize=(12, 4.2))
ax.bar(x, d.hit, color=HIT, label="hit (late)", width=0.85)
ax.bar(x, d.wasted, bottom=d.hit, color=WASTED, label="wasted (on time)", width=0.85)
ax.bar(x, d.unknown, bottom=d.hit + d.wasted, color=UNKNOWN, label="unknown outcome", width=0.85)
ax.step(x, d.best_possible, where="mid", color="black", lw=1.3, label="best possible hits")
ax.set_ylabel("Orders investigated")
ax.set_ylim(0, 11.5)
ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
title(ax, "Gradient boosting: what the 10 daily investigations found (test)")
save(fig, "test_daily_picks")

# --- fig 4: cumulative hits
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(x, d.best_possible.cumsum(), color="black", lw=2, label="best possible (perfect list)")
for name, (col, slug, c) in METHODS.items():
    ax.plot(x, days[name].hit.cumsum(), color=c, lw=2.5 if "proposed" in name else 2, label=name.lower())
ax.plot(x, d.random_expected.cumsum(), color=GRAY, lw=1.5, ls="--", label="random 10 per day")
ax.set_ylabel("Late orders caught (running total)")
ax.legend(frameon=False, loc="upper left")
title(ax, "Running total of late orders caught (test)")
save(fig, "test_cumulative_hits")

# --- fig 5: weekly hit rate
fig, ax = plt.subplots(figsize=(10, 4.4))
for name, (col, slug, c) in METHODS.items():
    w = days[name].resample("W").sum()
    ax.plot(w.index, 100 * w.hit / (w.hit + w.wasted), color=c, lw=2, marker="o", ms=3, label=name.lower())
w = d.resample("W").sum()
ax.plot(w.index, 100 * w.late / w.labeled, color=GRAY, lw=1.5, ls="--", label="random (late rate)")
ax.set_ylabel("Hit rate of investigations (%)")
ax.legend(frameon=False, loc="upper right")
title(ax, "Weekly hit rate of the 10 daily picks (test)")
save(fig, "test_weekly_hit_rate")

# --- fig 6: budget sensitivity
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for name, (col, slug, c) in METHODS.items():
    b = bs[bs.method == name]
    axs[0].plot(b.k, b.hits, color=c, marker="o", lw=2, label=name.lower())
    axs[1].plot(b.k, 100 * b.hit_rate, color=c, marker="o", lw=2)
b = bs[bs.method == "Gradient boosting (proposed)"]
axs[0].plot(b.k, b.random_hits, color=GRAY, marker="o", lw=1.5, ls="--", label="random")
axs[1].axhline(100 * lab.late.mean(), color=GRAY, ls="--", lw=1.5)
axs[0].set_ylabel("Late orders caught")
axs[0].set_title("Late orders caught", loc="left")
axs[0].legend(frameon=False, fontsize=9)
axs[1].set_ylabel("Hit rate of investigations (%)")
axs[1].set_title("Share of investigations that were late", loc="left")
for ax in axs:
    ax.set_xlabel("Investigations per day")
    ax.axvline(10, color="black", lw=0.8, ls=":")
fig.suptitle(f"Figure {FIG_NUM[0] + 1}. More or fewer investigations per day (test)", x=0.01, ha="left",
             fontweight="bold")
fig.tight_layout()
save(fig, "test_budget_sensitivity")
