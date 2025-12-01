import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt, seaborn as sns

from scipy.stats import spearmanr
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance

from statsmodels.stats.multitest import multipletests

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei","SimHei","Songti SC","PingFang SC","Arial Unicode MS","DejaVu Sans"],
    "axes.unicode_minus": False, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
})

def seal_axes(ax):
    for s in ["top","right","bottom","left"]:
        ax.spines[s].set_visible(True); ax.spines[s].set_linewidth(1.1)
    ax.tick_params(axis='both', which='both', length=3, width=0.8)
    return ax

CSV = Path("Q1_clean_男胎检测数据.csv")
read_ok = False
for enc in (None, "utf-8-sig", "gbk"):
    try:
        df = pd.read_csv(CSV, dtype=str, encoding=enc) if enc else pd.read_csv(CSV, dtype=str)
        read_ok = True; break
    except Exception: pass
if not read_ok: raise RuntimeError("CSV 读取失败")

df.columns = [c.strip() for c in df.columns]

candidates = [
    "年龄","检测孕周","孕妇BMI",
    "原始读段数","在参考基因组上比对的比例","重复读段的比例","唯一比对的读段数","GC含量",
    "13号染色体的Z值","18号染色体的Z值","21号染色体的Z值",
    "X染色体浓度",
    "13号染色体的GC含量","18号染色体的GC含量","21号染色体的GC含量",
    "被过滤掉读段数的比例"
]
target_col = "Y染色体浓度"

for c in [*candidates, target_col]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

feat_cols = [c for c in candidates if c in df.columns]

data = df[feat_cols + [target_col]].dropna().copy()


y = data[target_col].astype(float).clip(1e-6, 1-1e-6)
X = data[feat_cols].astype(float).values


sp_rows = []
for i, v in enumerate(feat_cols):
    rho, p = spearmanr(X[:, i], y)
    sp_rows.append([v, int(len(y)), float(rho), float(p)])
sp_df = pd.DataFrame(sp_rows, columns=["变量","N","Spearman ρ","p值"]).sort_values(
    by="Spearman ρ", key=lambda s: s.abs(), ascending=False
)
sp_df.to_csv("Q1_Spearman_vs_Y.csv", index=False, encoding="utf-8-sig")
top3 = sp_df["变量"].head(3).tolist()


def logit(p): return np.log(p/(1-p))
def inv_logit(z): return 1/(1+np.exp(-z))

prep = Pipeline(steps=[
    ("qt", QuantileTransformer(output_distribution="normal",
                               n_quantiles=min(200, max(10, len(y)-1)),
                               subsample=int(1e9), random_state=42)),
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False))
])

reg = GradientBoostingRegressor(
    loss="squared_error",
    n_estimators=int(1000),
    learning_rate=0.03,
    max_depth=int(3),
    subsample=0.9,
    max_features=0.8,
    random_state=88
)


base_pipe = Pipeline([("prep", prep), ("reg", reg)])
tt = TransformedTargetRegressor(regressor=base_pipe, func=logit, inverse_func=inv_logit)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
spearman_scorer = make_scorer(lambda yt, yp: spearmanr(yt, yp).correlation)

oof_pred = np.zeros_like(y.values, dtype=float)
fold_metrics = []
for fold, (tr, te) in enumerate(kf.split(X), 1):
    est = TransformedTargetRegressor(regressor=Pipeline([
        ("prep", prep),
        ("reg", GradientBoostingRegressor(
            loss="squared_error", n_estimators=int(900), learning_rate=0.03,
            max_depth=int(3), subsample=0.9, max_features=0.8, random_state=100+fold
        ))
    ]), func=logit, inverse_func=inv_logit)
    est.fit(X[tr], y.iloc[tr].values)
    pred = est.predict(X[te])
    try: rmse = mean_squared_error(y.iloc[te].values, pred, squared=False)
    except TypeError: rmse = np.sqrt(mean_squared_error(y.iloc[te].values, pred))
    m = {
        "Spearman": spearmanr(y.iloc[te].values, pred).correlation,
        "MAE": mean_absolute_error(y.iloc[te].values, pred),
        "RMSE": rmse,
        "R2": r2_score(y.iloc[te].values, pred)
    }
    fold_metrics.append(m); oof_pred[te] = pred
    print(f"[Fold {fold}] Spearman={m['Spearman']:.3f}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.3f}")

try: rmse_oof = mean_squared_error(y.values, oof_pred, squared=False)
except TypeError: rmse_oof = np.sqrt(mean_squared_error(y.values, oof_pred))
m_oof = {
    "Spearman": spearmanr(y.values, oof_pred).correlation,
    "MAE": mean_absolute_error(y.values, oof_pred),
    "RMSE": rmse_oof,
    "R2": r2_score(y.values, oof_pred)
}
print(f"[OOF] Spearman={m_oof['Spearman']:.3f}  MAE={m_oof['MAE']:.4f}  RMSE={m_oof['RMSE']:.4f}  R2={m_oof['R2']:.3f}")
pd.DataFrame(fold_metrics + [m_oof]).to_csv("Q1_CV_metrics.csv", index=False, encoding="utf-8-sig")

final_tt = tt.fit(X, y.values)
y_fit = final_tt.predict(X)
plt.figure(figsize=(15, 10), dpi=220)
plt.scatter(y.values, y_fit, s=20, alpha=0.6, c="tab:blue", label="预测值")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r-", lw=3, label="真实值=预测值")
plt.xlabel("真实值"); plt.ylabel("预测值")
plt.legend(frameon=False)
plt.grid(False)  
seal_axes(plt.gca()); plt.tight_layout(); plt.savefig("Q1_model_fit.png"); plt.close()

order_idx = np.argsort(y.values)
y_true_sorted = y.values[order_idx]
y_pred_sorted = y_fit[order_idx]

plt.figure(figsize=(12, 4.6), dpi=220)
plt.plot(y_true_sorted, lw=1.4, color="tab:orange", label="真实值")
plt.plot(y_pred_sorted, lw=1.4, color="tab:blue", label="预测值")
plt.xlabel("样本（按真实值排序）"); plt.ylabel("胎儿 Y 浓度（比例）")
plt.legend(frameon=False)
plt.grid(False)  
seal_axes(plt.gca()); plt.tight_layout(); plt.savefig("Q1_true_pred_series.png"); plt.close()


from sklearn.base import clone
from time import time

N_PERM_Y = 199         
N_PERM_FEAT = 30        
N_BOOT_PDP = 40        
SAMPLE_IMPORTANCE = min(2500, len(y))  

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_splits = list(kf.split(X))

def cv_r2(estimator, Xmat, yvec, splits):
    scores = []
    for tr, te in splits:
        est = clone(estimator)
        est.fit(Xmat[tr], yvec[tr])
        scores.append(r2_score(yvec[te], est.predict(Xmat[te])))
    return float(np.mean(scores))

t0 = time()
base_r2 = cv_r2(tt, X, y.values, cv_splits)
perm_r2 = np.empty(N_PERM_Y, dtype=float)
rng = np.random.RandomState(2025)
for b in range(N_PERM_Y):
    y_perm = rng.permutation(y.values)
    perm_r2[b] = cv_r2(tt, X, y_perm, cv_splits)
    if (b + 1) % 20 == 0:
        cur_p = (1 + np.sum(perm_r2[:b+1] >= base_r2)) / (b + 1 + 1)
        print(f"... overall perm {b+1}/{N_PERM_Y}, approx p≈{cur_p:.4f}")
p_overall = (1 + np.sum(perm_r2 >= base_r2)) / (N_PERM_Y + 1)

pd.DataFrame({"stat":["R2"], "observed":[base_r2], "p_value":[p_overall]}).to_csv(
    "Q1_permtest_overall.csv", index=False, encoding="utf-8-sig"
)

plt.figure(figsize=(6.8,4.2), dpi=220)
plt.hist(perm_r2, bins=30, density=True, alpha=0.8)
plt.axvline(base_r2, lw=2.0)
plt.xlabel("R²（置换标签的分布）"); plt.ylabel("密度")
seal_axes(plt.gca()); plt.tight_layout(); plt.savefig("Q1_permtest_overall_hist.png"); plt.close()
print(f"[Overall PermTest] R2={base_r2:.3f}, p={p_overall:.4g}, time={time()-t0:.1f}s")

if SAMPLE_IMPORTANCE < len(y):
    idx_imp = np.random.RandomState(7).choice(len(y), SAMPLE_IMPORTANCE, replace=False)
    X_imp, y_imp = X[idx_imp], y.values[idx_imp]
else:
    X_imp, y_imp = X, y.values

pi_r = permutation_importance(
    final_tt, X_imp, y_imp,
    scoring="r2", n_repeats=N_PERM_FEAT, random_state=42, n_jobs=1
)
imp_mean = pi_r.importances_mean
imp_std  = pi_r.importances_std

p_raw = (1 + np.sum(pi_r.importances <= 0, axis=1)) / (N_PERM_FEAT + 1)
reject, p_bh, _, _ = multipletests(p_raw, method="fdr_bh")

feat_sig = pd.DataFrame({
    "变量": feat_cols,
    "importance_R2_mean": imp_mean,
    "importance_R2_std": imp_std,
    "p_value": p_raw,
    "p_value_FDR_BH": p_bh,
    "显著(5%)": reject
}).sort_values("importance_R2_mean", ascending=False)
feat_sig.to_csv("Q1_permutation_importance_R2.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(7.6, 5.0), dpi=220)
order = feat_sig.sort_values("importance_R2_mean")["变量"]
vals  = feat_sig.set_index("变量").loc[order]["importance_R2_mean"].values
errs  = feat_sig.set_index("变量").loc[order]["importance_R2_std"].values
plt.barh(order, vals, xerr=errs, alpha=0.9)
plt.xlabel("置换重要性（R² 提升的均值 ±1SD）"); plt.ylabel("")
seal_axes(plt.gca()); plt.tight_layout(); plt.savefig("Q1_permutation_importance_R2_bar.png"); plt.close()

def pdp_boot(name, grid_points=120, outpng="pdp.png"):
    if name not in feat_cols: return
    j = feat_cols.index(name)
    xs = np.linspace(X[:, j].min(), X[:, j].max(), grid_points)

    base_curve = []
    for v in xs:
        X_mod = X.copy(); X_mod[:, j] = v
        base_curve.append(final_tt.predict(X_mod).mean())
    base_curve = np.array(base_curve)

    rng = np.random.RandomState(2024)
    boots = []
    for b in range(N_BOOT_PDP):
        idx = rng.randint(0, len(y), size=len(y))
        tt_b = TransformedTargetRegressor(regressor=Pipeline([("prep", prep), ("reg", reg)]),
                                          func=logit, inverse_func=inv_logit)
        tt_b.fit(X[idx], y.iloc[idx].values)
        cur = []
        for v in xs:
            X_mod = X.copy(); X_mod[:, j] = v
            cur.append(tt_b.predict(X_mod).mean())
        boots.append(cur)
    boots = np.array(boots)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)

    plt.figure(figsize=(6.8, 4.2), dpi=220)
    plt.fill_between(xs, lo, hi, alpha=0.25)
    plt.plot(xs, base_curve, lw=2.0)
    plt.xlabel(name); plt.ylabel("预测胎儿分数")
    seal_axes(plt.gca()); plt.tight_layout(); plt.savefig(outpng); plt.close()

for name in top3:
    pdp_boot(name, outpng=f"Q1_PDP_{name}_boot95.png")

pd.DataFrame({
    "指标": ["(OOF) Spearman","(OOF) MAE","(OOF) RMSE","(OOF) R2","整体显著性 p(置换)"],
    "数值": [m_oof['Spearman'], m_oof['MAE'], m_oof['RMSE'], m_oof['R2'], p_overall]
}).to_csv("Q1_summary_with_significance.csv", index=False, encoding="utf-8-sig")





