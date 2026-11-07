# nonML_riskiestSellers: riskiest seller rule (business rule 2)

Each day, investigate the 10 approved orders whose seller has the worst past late rate
(`seller_prev_late_rate`: late orders among the seller's orders delivered before this approval, pulled toward
the overall late rate when the seller has little history). No training. Orders with no seller are ranked last.

Identical to `nonML_distanceRule/` except for `SCORE` and `RULE` in `run_seller_rule.py` and the columns of the
example-day table.

```bash
python nonML_riskiestSellers/run_seller_rule.py
```

Outputs have the same names and layout as `nonML_distanceRule/` (`results/` and `figures/`).

Validation: 149 hits out of 1,045 investigations (14.6% hit rate, 1.26 times random). Many picks are short
trips within Sao Paulo, which are rarely late.
