"""Daily top-k selection and scoring, shared by the non-ML rule and (later) the ML models.

Every approved order on a day is a candidate. The policy ranks that day's candidates by a
score (higher = riskier) and picks at most k. Picked orders are then compared with their
eventual outcome:
    hit      = picked and late
    wasted   = picked and on time
    unknown  = picked but no usable outcome (canceled, never delivered, bad dates)
    missed   = late but not picked
    best     = min(k, number of late orders that day)   -> what a perfect list could catch
    random   = expected hits if k orders were picked at random from that day's candidates
"""
import numpy as np
import pandas as pd


def select_daily(df, score_col, k=10):
    """Return the picked rows: top-k per approval_date by score (missing scores rank last,
    ties broken by order_id so the result is reproducible)."""
    d = df.copy()
    d["_s"] = d[score_col].fillna(-np.inf)
    d = d.sort_values(["approval_date", "_s", "order_id"], ascending=[True, False, True])
    d["rank"] = d.groupby("approval_date").cumcount() + 1
    return d[d["rank"] <= k].drop(columns="_s")


def outcome_label(df):
    return np.where(df.eligible == 0, "unknown", np.where(df.late == 1, "hit", "wasted"))


def score_days(df, picked, k=10):
    """One row per day with candidate counts and hits / wasted / unknown / missed."""
    day = df.groupby("approval_date").agg(candidates=("order_id", "size"),
                                          labeled=("eligible", "sum"),
                                          late=("late", "sum"))
    p = picked.assign(outcome=outcome_label(picked))
    counts = p.pivot_table(index="approval_date", columns="outcome", values="order_id",
                           aggfunc="size", fill_value=0)
    for c in ["hit", "wasted", "unknown"]:
        if c not in counts:
            counts[c] = 0
    day = day.join(counts[["hit", "wasted", "unknown"]]).fillna(0)
    day["picked"] = day[["hit", "wasted", "unknown"]].sum(axis=1)
    day["missed"] = day.late - day.hit
    day["best_possible"] = np.minimum(k, day.late)
    day["random_expected"] = np.minimum(k, day.candidates) * day.late / day.candidates
    return day.astype(float)


def totals(day):
    """Period totals and the main rates."""
    t = day[["candidates", "labeled", "late", "picked", "hit", "wasted", "unknown", "missed",
             "best_possible", "random_expected"]].sum()
    out = {
        "days": len(day),
        "candidates": int(t.candidates),
        "late_orders": int(t.late),
        "picked": int(t.picked),
        "hits": int(t.hit),
        "wasted": int(t.wasted),
        "unknown": int(t.unknown),
        "missed": int(t.missed),
        "hit_rate_of_picks": t.hit / max(t.hit + t.wasted, 1),          # precision of the list
        "share_of_late_caught": t.hit / max(t.late, 1),                  # recall of the list
        "share_of_best_possible": t.hit / max(t.best_possible, 1),
        "best_possible_hits": int(t.best_possible),
        "random_expected_hits": round(float(t.random_expected), 1),
        "lift_over_random": t.hit / max(t.random_expected, 1e-9),
    }
    return out


def average_precision(score, late):
    """Average precision (PR-AUC) over labeled orders; missing scores rank last."""
    s = pd.Series(score).fillna(-np.inf).to_numpy()
    y = pd.Series(late).to_numpy()
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    precision_at_k = tp / np.arange(1, len(y) + 1)
    return float((precision_at_k * y).sum() / max(y.sum(), 1))
