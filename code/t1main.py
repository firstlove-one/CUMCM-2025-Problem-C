def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def inv_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def build_estimator(n_estimators: int = 1000,
                    learning_rate: float = 0.03,
                    max_depth: int = 3,
                    subsample: float = 0.9,
                    max_features: float = 0.8,
                    random_state: int = 88) -> TransformedTargetRegressor:
    prep = Pipeline(steps=[
        ("qt", QuantileTransformer(output_distribution="normal",
                                   n_quantiles=200, subsample=int(1e9),
                                   random_state=42)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("poly", PolynomialFeatures(degree=2, include_bias=False))
    ])
    gbr = GradientBoostingRegressor(
        loss="squared_error",
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        subsample=subsample,
        max_features=max_features,
        random_state=random_state
    )
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", prep), ("reg", gbr)]),
        func=logit, inverse_func=inv_logit
    )

def kfold_oof_metrics(estimator: TransformedTargetRegressor,
                      X: np.ndarray, y: np.ndarray,
                      n_splits: int = 5, seed: int = 42
                      ) -> Tuple[Dict[str, float], np.ndarray]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)
    for k, (tr, te) in enumerate(kf.split(X), 1):
        est_k = clone(estimator)
        est_k.regressor.named_steps["reg"].random_state = 100 + k
        est_k.fit(X[tr], y[tr])
        oof[te] = est_k.predict(X[te])
    rmse = mean_squared_error(y, oof, squared=False)
    metrics = {
        "Spearman": float(spearmanr(y, oof).correlation),
        "MAE": float(mean_absolute_error(y, oof)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y, oof))
    }
    return metrics, oof

def permutation_test_r2(estimator: TransformedTargetRegressor,
                        X: np.ndarray, y: np.ndarray,
                        splits: List[Tuple[np.ndarray, np.ndarray]],
                        n_perm: int = 199, seed: int = 2025
                        ) -> Tuple[float, float, np.ndarray]:
    def cv_r2(est, X_, y_, split_):
        scores = []
        for tr, te in split_:
            m = clone(est); m.fit(X_[tr], y_[tr])
            scores.append(r2_score(y_[te], m.predict(X_[te])))
        return float(np.mean(scores))
    observed = cv_r2(estimator, X, y, splits)
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        null[b] = cv_r2(estimator, X, rng.permutation(y), splits)
    p_val = (1.0 + np.sum(null >= observed)) / (n_perm + 1.0)
    return observed, float(p_val), null

def permutation_importance_r2(estimator: TransformedTargetRegressor,
                              X: np.ndarray, y: np.ndarray,
                              n_repeats: int = 30, seed: int = 42,
                              sample_n: int = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
    if sample_n is not None and sample_n < len(y):
        idx = np.random.RandomState(7).choice(len(y), sample_n, replace=False)
        X_, y_ = X[idx], y[idx]
    else:
        X_, y_ = X, y
    pi = permutation_importance(estimator, X_, y_, scoring="r2",
                                n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return pi.importances_mean, pi.importances_std

def partial_dependence_1d(estimator: TransformedTargetRegressor,
                          X: np.ndarray, j: int, grid_points: int = 120
                          ) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(X[:, j].min(), X[:, j].max(), grid_points)
    pdp = []
    for v in xs:
        X_mod = X.copy(); X_mod[:, j] = v
        pdp.append(estimator.predict(X_mod).mean())
    return xs, np.asarray(pdp)

def fit_y_fraction_model(df: pd.DataFrame,
                         feature_cols: List[str],
                         target_col: str,
                         n_splits: int = 5) -> Dict[str, object]:
    work = df[feature_cols + [target_col]].copy()
    for c in work.columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()
    y = work[target_col].astype(float).clip(1e-6, 1 - 1e-6).values
    X = work[feature_cols].astype(float).values
    est = build_estimator()
    oof_metrics, oof_pred = kfold_oof_metrics(est, X, y, n_splits=n_splits, seed=42)
    est.fit(X, y)
    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X))
    observed_r2, p_overall, null_r2 = permutation_test_r2(est, X, y, splits, n_perm=199, seed=2025)
    imp_mean, imp_std = permutation_importance_r2(est, X, y, n_repeats=30, seed=42, sample_n=min(2500, len(y)))
    sp = [(j, abs(spearmanr(X[:, j], y).correlation)) for j in range(X.shape[1])]
    top3_idx = [j for j, _ in sorted(sp, key=lambda z: z[1], reverse=True)[:3]]
    pdp_dict = {feature_cols[j]: partial_dependence_1d(est, X, j, grid_points=120) for j in top3_idx}
    return {
        "X": X, "y": y,
        "features": feature_cols,
        "estimator": est,
        "oof_metrics": oof_metrics,
        "oof_pred": oof_pred,
        "permtest_R2": {"observed": observed_r2, "p_value": p_overall, "null": null_r2},
        "perm_importance_R2": {"mean": imp_mean, "std": imp_std},
        "top3_pdp": pdp_dict
    }
