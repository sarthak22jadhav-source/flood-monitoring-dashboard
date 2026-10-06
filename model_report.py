"""Print trained-model metrics for documentation."""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import data as D
import model as M

raw_rows, df = D.load_dataset()
print(f"raw rows: {raw_rows:,} -> cleaned: {len(df):,}")
print(f"rule_check (Is_Flood == WL>=Danger): {M.rule_check(df)}")
bundle = M.train_model(df)
print(f"train={bundle.n_train:,} test={bundle.n_test:,} "
      f"prevalence={bundle.positive_rate:.3f}")
for k, v in bundle.metrics.items():
    print(f"  {k}: {v:.4f}")
print("confusion:\n", bundle.confusion)
print("feature importances:")
for f, imp in bundle.importances.items():
    print(f"  {f}: {imp:.4f}")

# Example prediction: latest row of a station at danger threshold
sample = df[df[D.CWC_FLOOD] == 1].iloc[-1]
row = {c: float(sample[c]) for c in D.FEATURE_COLUMNS}
print(f"example flood-row probability: {bundle.predict_proba_one(row):.3f}")
safe = df[df[D.CWC_FLOOD] == 0].iloc[-1]
row0 = {c: float(safe[c]) for c in D.FEATURE_COLUMNS}
print(f"example safe-row probability:  {bundle.predict_proba_one(row0):.3f}")
print("contributions (flood row):")
print(bundle.contributions(row).to_string())
