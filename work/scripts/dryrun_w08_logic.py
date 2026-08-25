"""Execute w08's pure-python half (rule, matrix, hybrid, evaluate) on a synthetic frame
that mimics the warehouse schema. Proves the modelling logic runs before it ever touches
the gated data."""
import json, numpy as np, pandas as pd, sklearn

NB = r"C:\Users\real time\Desktop\Rayan_flyrank\.claude\worktrees\ml11-paper\work\notebooks\w08_forward_window_validation.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

# the cell that defines build_rule / build_matrix / make_pipe / hybrid_key / evaluate
model_cell = next(s for s in code_cells if "def evaluate(" in s)
boot_cell = next(s for s in code_cells if "def bootstrap_ci(" in s)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
N, NCLIENT = 4000, 12

# synthetic frame with EXACTLY the columns build_frame produces
imp90 = rng.lognormal(4.0, 1.6, N)
frame = pd.DataFrame({
    "client_hash_id": rng.integers(0, NCLIENT, N).astype(str),
    "content_hash_id": [f"c{i}" for i in range(N)],
    "impressions_90d": imp90,
    "clicks_90d": imp90 * rng.uniform(0, 0.08, N),
    "days_with_impressions": rng.integers(0, 89, N),
    "avg_position": rng.uniform(1, 60, N),
    "position_sd": rng.uniform(0, 8, N),
    "imp_prior30": imp90 / 3 * rng.uniform(0.5, 1.6, N),
    "imp_prev30": imp90 / 3 * rng.uniform(0.5, 1.6, N),
    "imp_fwd30": imp90 / 3 * rng.uniform(0.4, 1.7, N),
    "content_age_days": rng.integers(90, 900, N).astype(float),
    "days_since_last_update": rng.integers(0, 400, N).astype(float),
})
# realistic missingness, which is what usually breaks a pipeline
frame.loc[frame.sample(frac=0.04, random_state=1).index, "avg_position"] = np.nan
frame.loc[frame.sample(frac=0.10, random_state=2).index, "position_sd"] = np.nan
frame.loc[frame.sample(frac=0.07, random_state=3).index, "content_age_days"] = np.nan
frame.loc[frame.sample(frac=0.07, random_state=4).index, "days_since_last_update"] = np.nan

DECLINE_THRESHOLD = 0.8
frame["is_declining_fwd"] = (frame["imp_fwd30"] < DECLINE_THRESHOLD * frame["imp_prior30"]).astype(int)
frame["is_declining_conc"] = (frame["imp_prior30"] < DECLINE_THRESHOLD * frame["imp_prev30"]).astype(int)

g = {"np": np, "pd": pd, "sklearn": sklearn, "json": json,
     "RANDOM_STATE": RANDOM_STATE, "HAS_STALE": True, "HAS_AGE": True}
import ast


def defs_only(src):
    """Extract just the function definitions and simple assignments - drop the driver lines
    that reference notebook-scope frames like `dev`."""
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.Assign)
                and all(isinstance(t, ast.Name) for t in n.targets)
                and isinstance(n.value, (ast.Constant, ast.List, ast.Tuple, ast.Dict)))]
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


exec(defs_only(model_cell), g)
exec(defs_only(boot_cell), g)

print("=" * 72)
print("DRY RUN on synthetic data (schema-identical, values random)")
print("=" * 72)

# 1. build_rule
band, codes = g["build_rule"](frame)
print(f"\nbuild_rule: band dtype={band.dtype} range {band.min()}..{band.max()} (max possible 9)")
print(f"  reason codes sample: {codes[:3].tolist()}")
print(f"  rows with no_signal: {(codes == 'no_signal').sum()}")
assert band.max() <= 9 and band.min() >= 0

# 2. build_matrix, both variants
X_full = g["build_matrix"](frame)
X_excl = g["build_matrix"](frame, exclude=g["RECONSTRUCTS_CONCURRENT"])
print(f"\nbuild_matrix full:    {X_full.shape[1]} cols {list(X_full.columns)}")
print(f"build_matrix excluded:{X_excl.shape[1]} cols {list(X_excl.columns)}")
assert "imp_prior30" in X_full.columns and "imp_prior30" not in X_excl.columns

# 3. hybrid_key ordering property: band must dominate the model probability
hk = g["hybrid_key"]
assert hk(9, 0.0) > hk(8, 1.0), "band does not dominate - hybrid key is wrong"
print(f"\nhybrid_key: band dominates (9@p=0.0 -> {hk(9,0.0)} > 8@p=1.0 -> {hk(8,1.0)})")

# 4. full evaluate, both labels
res_fwd = g["evaluate"](frame, "is_declining_fwd", "SYNTHETIC forward")
res_conc = g["evaluate"](frame, "is_declining_conc", "SYNTHETIC concurrent",
                         exclude=g["RECONSTRUCTS_CONCURRENT"])

# 5. the leakage guard actually matters: prove the pair reconstructs the concurrent label
res_leak = g["evaluate"](frame, "is_declining_conc", "SYNTHETIC concurrent WITH the pair (leak demo)")
print(f"\n>>> leak demo: keeping the pair lifts concurrent AUC to "
      f"{res_leak['systems']['logistic']['roc_auc']:.4f} vs "
      f"{res_conc['systems']['logistic']['roc_auc']:.4f} with it dropped.")
print("    That gap is why RECONSTRUCTS_CONCURRENT exists.")

# 6. bootstrap
lo, hi = g["bootstrap_ci"](res_fwd["scores"]["y"], res_fwd["scores"]["hybrid"])
print(f"\nbootstrap_ci -> [{lo:.3f}, {hi:.3f}]")
assert 0 <= lo <= hi <= 1

# 7. receipt serialises (the classic numpy-int64 JSON crash)
payload = {k: v for k, v in res_fwd.items() if k != "scores"}
json.dumps(payload, default=float)
print("\nreceipt JSON-serialises OK")

print("\n" + "=" * 72)
print("DRY RUN PASSED - modelling half is executable; only the DuckDB half needs the token")
print("=" * 72)
