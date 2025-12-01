def normalize_name(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"[\s\-\_\(\)\[\]{}·•、，,。:：;；/%]+", "", s)

def parse_week_str(s):
    if pd.isna(s): return np.nan
    if isinstance(s, (int, float)): return float(s)
    m = re.search(r"(\d+(\.\d+)?)", str(s).strip())
    return float(m.group(1)) if m else np.nan

def logit(p: float) -> float:
    p = float(np.clip(p, 1e-9, 1-1e-9))
    return math.log(p/(1-p))

@dataclass
class SubjectInterval:
    pid: str
    bmi: float
    age: float
    z13: float
    z18: float
    z21: float
    L: float
    R: float
    right_censored: bool

def fuzzy_pick(df: pd.DataFrame, prefer, rules):
    norm = {normalize_name(c): c for c in df.columns}
    for p in prefer:
        n = normalize_name(p)
        if n in norm: return norm[n]
    for c in df.columns:
        nc = normalize_name(c)
        for r in rules:
            toks = [t for t in r.split("&") if t]
            if all(t in nc for t in toks): return c
    return None

def pick_columns(df: pd.DataFrame,
                 col_id=None, col_week=None, col_y=None, col_bmi=None,
                 col_height=None, col_weight=None, col_age=None,
                 col_z13=None, col_z18=None, col_z21=None):
    id_col    = col_id    or fuzzy_pick(df, ["孕妇代码","编号","样本号","pid","id"], ["孕&妇&代&码","样&本","编&号","id"])
    week_col  = col_week  or fuzzy_pick(df, ["孕周","周数","检测孕周","week","gestational age"], ["孕&周","检&测&孕&周","gestational&age","week"])
    y_col     = col_y     or fuzzy_pick(df, ["Y浓度","Y浓度(%)","Y%","Y比例","Y分数","y","chrY","fetal y"], ["y&浓","y&%","chry","fetal&y","y&分&数","y&比&例"])
    bmi_col   = col_bmi   or fuzzy_pick(df, ["BMI","bmi"], ["bmi"])
    ht_col    = col_height or fuzzy_pick(df, ["身高","身高cm","height","height(cm)"], ["身&高","heigh"])
    wt_col    = col_weight or fuzzy_pick(df, ["体重","体重kg","weight","weight(kg)"], ["体&重","weigh"])
    age_col   = col_age   or fuzzy_pick(df, ["年龄","age","年龄(岁)"], ["年&龄","age"])
    z13_col   = col_z13   or fuzzy_pick(df, ["Z13","z13"], ["z13"])
    z18_col   = col_z18   or fuzzy_pick(df, ["Z18","z18"], ["z18"])
    z21_col   = col_z21   or fuzzy_pick(df, ["Z21","z21"], ["z21"])
    return {"id": id_col,"week": week_col,"y": y_col,"bmi": bmi_col,
            "height": ht_col,"weight": wt_col,"age": age_col,
            "z13": z13_col,"z18": z18_col,"z21": z21_col}

def read_base_df(excel: str, sheet: str, overrides: dict) -> pd.DataFrame:
    df = pd.read_excel(excel, sheet_name=sheet)
    cols = pick_columns(df,
        col_id=overrides.get("col_id"), col_week=overrides.get("col_week"),
        col_y=overrides.get("col_y"), col_bmi=overrides.get("col_bmi"),
        col_height=overrides.get("col_height"), col_weight=overrides.get("col_weight"),
        col_age=overrides.get("col_age"), col_z13=overrides.get("col_z13"),
        col_z18=overrides.get("col_z18"), col_z21=overrides.get("col_z21")
    )
    if cols["id"] is None or cols["week"] is None or cols["y"] is None:
        raise ValueError("缺少 编号/孕周/Y 列")
    df = df.copy()
    if cols["bmi"] is not None:
        df['__BMI__'] = pd.to_numeric(df[cols["bmi"]], errors='coerce')
    else:
        if cols["height"] is None or cols["weight"] is None:
            raise ValueError("缺少 BMI 或 身高/体重")
        h_m = pd.to_numeric(df[cols["height"]], errors='coerce')/100.0
        w_kg = pd.to_numeric(df[cols["weight"]], errors='coerce')
        df['__BMI__'] = w_kg / (h_m**2)
    df['__WEEK__'] = df[cols["week"]].apply(parse_week_str)
    Yraw = pd.to_numeric(df[cols["y"]], errors='coerce')
    if np.nanpercentile(Yraw, 90) > 1.5: Yraw = Yraw/100.0
    df['__Y__'] = Yraw
    def num(c): return pd.to_numeric(df[c], errors='coerce') if c is not None else None
    age = num(cols["age"]); z13 = num(cols["z13"]); z18 = num(cols["z18"]); z21 = num(cols["z21"])
    df['__AGE__'] = (age.fillna(age.median()) if age is not None else 0.0)
    df['__Z13__'] = (z13.fillna(np.nanmedian(z13)) if z13 is not None else 0.0)
    df['__Z18__'] = (z18.fillna(np.nanmedian(z18)) if z18 is not None else 0.0)
    df['__Z21__'] = (z21.fillna(np.nanmedian(z21)) if z21 is not None else 0.0)
    df.attrs['pid_col'] = cols["id"]
    return df

def intervals_from_df(df: pd.DataFrame, thresh: float, h_week: float) -> List[SubjectInterval]:
    pid_col = df.attrs.get('pid_col', df.columns[0])
    out = []
    for pid, sub in df.groupby(df[pid_col].astype(str)):
        sub = sub.sort_values('__WEEK__')
        weeks = sub['__WEEK__'].to_numpy(float)
        ys    = sub['__Y__'].to_numpy(float)
        bmiv  = float(np.nanmedian(sub['__BMI__']))
        agev  = float(np.nanmedian(sub['__AGE__']))
        z13v  = float(np.nanmedian(sub['__Z13__']))
        z18v  = float(np.nanmedian(sub['__Z18__']))
        z21v  = float(np.nanmedian(sub['__Z21__']))
        if not np.isfinite(bmiv): continue
        idx = np.where(ys >= thresh)[0]
        if idx.size>0:
            j = int(idx[0]); R = float(weeks[j])
            left = weeks[(weeks < R) & (ys < thresh)]
            L = float(left.max()) if left.size>0 else 0.0
            out.append(SubjectInterval(pid, bmiv, agev, z13v, z18v, z21v, L, R, False))
        else:
            L = float(np.nanmax(weeks))
            out.append(SubjectInterval(pid, bmiv, agev, z13v, z18v, z21v, L, float(h_week), True))
    return out

def load_intervals_from_excel(excel: str, sheet: str, thresh: float, h_week: float, overrides: dict) -> List[SubjectInterval]:
    df = read_base_df(excel, sheet, overrides)
    return intervals_from_df(df, thresh, h_week)

class AFTLogLogistic:
    def __init__(self, l2=0.003, maxiter=1200, verbose=False,
                 pen_align_full=3.0, pen_align_pos=6.0, pen_extra=0.8, pen_sigma=0.006,
                 norm_by_sigma=True):
        self.l2 = l2; self.maxiter=maxiter; self.verbose=verbose
        self.pen_align_full=pen_align_full
        self.pen_align_pos=pen_align_pos
        self.pen_extra=pen_extra
        self.pen_sigma=pen_sigma
        self.norm_by_sigma = norm_by_sigma
        self.beta_=None; self.s_=None; self.m_q2_=None
        self.align_mean_pos_ = None; self.align_max_pos_ = None

    @staticmethod
    def _features(s: SubjectInterval):
        b = s.bmi
        return np.array([1.0, b, b*b, s.age, s.z13, s.z18, s.z21], float)

    def _fit_q2_baseline(self, b, L, R, right):
        Xb = np.stack([np.ones_like(b), b, b*b], 1)
        def nll(theta):
            m0,m1,m2,s0,s1 = theta
            mu = Xb @ np.array([m0,m1,m2], float)
            sig = np.exp(s0 + s1*b)
            eps=1e-6; Ls=np.maximum(L,eps)
            zL=(np.log(Ls)-mu)/sig; FL=1/(1+np.exp(-zL))
            ll=0.0
            mask=~right
            if np.any(mask):
                Rs=np.maximum(R[mask],eps)
                zR=(np.log(Rs)-mu[mask])/sig[mask]; FR=1/(1+np.exp(-zR))
                diff=np.maximum(FR-FL[mask],1e-12); ll+=np.sum(np.log(diff))
            if np.any(right):
                surv=np.maximum(1-FL[right],1e-12); ll+=np.sum(np.log(surv))
            pen=1e-6*(m0*m0+m1*m1+m2*m2+s0*s0+s1*s1)
            return -(ll-pen)
        init=np.array([0.0,0.0,0.0, math.log(0.6), 0.0])
        bounds=[(-10,10),(-5,5),(-2,2),(math.log(0.2), math.log(2.5)),(-2,2)]
        res=minimize(nll, init, method="L-BFGS-B", bounds=bounds, options={"maxiter":1000})
        return np.array(res.x[:3],float), np.array(res.x[3:],float)

    def _fit_once(self, X, L, R, right, b, Gq2, lam_full, lam_pos, lam_extra, s_q2):
        n,p = X.shape
        beta0 = np.zeros(p); beta0[:3] = self.m_q2_
        scoef0 = s_q2.copy()
        theta0 = np.concatenate([beta0, scoef0], 0)
        idx_extra = np.array([3,4,5,6], int)
        def nll(theta):
            beta=theta[:p]; scoef=theta[p:]
            mu = X @ beta
            sig = np.exp(scoef[0] + scoef[1]*b)
            eps=1e-6; Ls=np.maximum(L,eps)
            zL=(np.log(Ls)-mu)/sig; FL=1/(1+np.exp(-zL))
            ll=0.0
            mask=~right
            if np.any(mask):
                Rs=np.maximum(R[mask],eps)
                zR=(np.log(Rs)-mu[mask])/sig[mask]; FR=1/(1+np.exp(-zR))
                diff=np.maximum(FR-FL[mask],1e-12); ll+=np.sum(np.log(diff))
            if np.any(right):
                surv=np.maximum(1-FL[right],1e-12); ll+=np.sum(np.log(surv))
            delta = (mu - Gq2) / (sig + 1e-6) if self.norm_by_sigma else (mu - Gq2)
            pen_align = lam_full * float(np.mean(delta**2))
            pen_align_pos = lam_pos * float(np.mean(np.maximum(delta, 0.0)**2))
            pen_base = self.l2 * float(np.sum(beta*beta))
            pen_extra = lam_extra * float(np.sum(beta[idx_extra]*beta[idx_extra]))
            pen_sigma = self.pen_sigma * float(np.sum(scoef*scoef))
            return -(ll - pen_base - pen_extra - pen_align - pen_align_pos - pen_sigma)
        bounds = [(-10,10),(-5,5),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(math.log(0.2), math.log(2.5)),(-2,2)]
        res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": self.maxiter})
        return res.x

    def fit(self, intervals, h_week, cont_rounds=6, cont_growth=2.0, target_mean_pos=0.015, target_max_pos=0.05):
        X = np.vstack([self._features(s) for s in intervals])
        L = np.array([s.L for s in intervals], float)
        R = np.array([s.R for s in intervals], float)
        right = np.array([s.right_censored for s in intervals], bool)
        b = X[:,1]
        m_q2, s_q2 = self._fit_q2_baseline(b, L, R, right)
        self.m_q2_ = m_q2
        Xb = np.stack([np.ones_like(b), b, b*b], 1)
        Gq2 = Xb @ m_q2
        lam_full = float(self.pen_align_full)
        lam_pos  = float(self.pen_align_pos)
        lam_extra= float(self.pen_extra)
        best_theta=None; best_metrics=None
        for r in range(int(cont_rounds)):
            theta = self._fit_once(X, L, R, right, b, Gq2, lam_full, lam_pos, lam_extra, s_q2)
            beta = theta[:X.shape[1]]; scoef = theta[X.shape[1]:]
            mu = X @ beta
            sig = np.exp(scoef[0] + scoef[1]*b)
            delta = (mu - Gq2) / (sig + 1e-6) if self.norm_by_sigma else (mu - Gq2)
            mean_pos = float(np.mean(np.maximum(delta,0.0)))
            max_pos  = float(np.max(np.maximum(delta,0.0)))
            best_theta = theta; best_metrics=(mean_pos, max_pos)
            if (mean_pos <= target_mean_pos) and (max_pos <= target_max_pos): break
            lam_full *= cont_growth; lam_pos  *= cont_growth; lam_extra*= cont_growth**0.5
        self.beta_ = best_theta[:X.shape[1]].copy()
        self.s_    = best_theta[X.shape[1]:].copy()
        self.align_mean_pos_, self.align_max_pos_ = best_metrics

    def F(self, t: np.ndarray, subj_or_x) -> np.ndarray:
        if isinstance(subj_or_x, SubjectInterval): x = self._features(subj_or_x)
        else: x = np.asarray(subj_or_x, float)
        b = float(x[1])
        mu = float(np.dot(x, self.beta_))
        sig = float(np.exp(self.s_[0] + self.s_[1]*b))
        t = np.asarray(t, float); t = np.maximum(t, 1e-6)
        z = (np.log(t) - mu) / sig
        return 1/(1+np.exp(-z))

def build_w_time(alpha: float, beta: float, h_week: float):
    def w_scalar(t):
        if t <= 12.0: return 0.0
        elif t <= 27.0: return alpha * (t - 12.0) / 15.0
        else: return alpha + beta * (t - 27.0) / (h_week - 27.0)
    return np.vectorize(w_scalar), w_scalar

def expected_w_after_t(Fhat, x, t, fine_grid, W_FINE, w_scalar):
    mask = fine_grid > t
    if not np.any(mask): return 0.0
    ts = fine_grid[mask]; Ft = Fhat(ts, x)
    p = np.diff(np.concatenate([[0.0], Ft])); p = np.maximum(p, 0.0)
    p = p / (p.sum() + 1e-12)
    return float(np.sum(p * W_FINE[mask]))

def risk_for_subject(model, subj, T_GRID, c_retest, FINE_GRID, W_FINE, w_vec, w_scalar):
    x = AFTLogLogistic._features(subj)
    Ft = model.F(T_GRID, x)
    not_ready = 1.0 - Ft
    ew = np.array([expected_w_after_t(model.F, x, float(t), FINE_GRID, W_FINE, w_scalar) for t in T_GRID])
    return Ft * w_vec(T_GRID) + not_ready * (c_retest + ew)

def dp_segment_with_ready_size(R, F, T, K, tau, nmin):
    n, TT = R.shape; INF=np.inf
    preR = np.vstack([np.zeros(TT), np.cumsum(R,0)])
    preF = np.vstack([np.zeros(TT), np.cumsum(F,0)])
    Cost = np.full((n,n), INF); Arg = np.full((n,n), -1, int)
    for i in range(n):
        for j in range(i, n):
            L = j - i + 1
            if L < nmin: continue
            segR = preR[j+1]-preR[i]
            segF = (preF[j+1]-preF[i]) / L
            feas = (segF >= tau)
            if not np.any(feas): continue
            best=INF; tid=-1
            for k in np.where(feas)[0]:
                c = segR[k]
                if c < best: best=c; tid=int(k)
            Cost[i,j]=float(best); Arg[i,j]=tid
    dp = np.full((K, n), INF); prev = np.full((K, n), -1, int)
    for j in range(n):
        if Arg[0,j]!=-1: dp[0,j]=Cost[0,j]; prev[0,j]=-1
    for g in range(1, K):
        for j in range(n):
            best=INF; bi=-1
            for i in range(g-1, j):
                if dp[g-1,i]==INF or Arg[i+1,j]==-1: continue
                v = dp[g-1,i]+Cost[i+1,j]
                if v<best: best=v; bi=i
            dp[g,j]=best; prev[g,j]=bi
    if not np.isfinite(dp[K-1,n-1]): return None,None,None
    bounds=[]; tids=[]
    g=K-1; j=n-1
    while g>=0:
        i=prev[g,j]; s=0 if i==-1 else i+1; e=j
        bounds.append((s,e)); tids.append(Arg[s,e])
        j=i; g-=1
    bounds.reverse(); tids.reverse()
    return bounds, tids, float(dp[K-1,n-1])

def enforce_monotone_with_gap(ts, bounds, T_GRID, F_sorted, tau, gap):
    out=list(map(float, ts))
    for k in range(len(bounds)):
        if k>0 and out[k] < out[k-1]+gap: out[k]=out[k-1]+gap
        s,e=bounds[k]; tid=int(np.argmin(np.abs(T_GRID-out[k])))
        if np.mean(F_sorted[s:e+1, tid]) < tau:
            ok=False
            for tid2 in range(tid, len(T_GRID)):
                if np.mean(F_sorted[s:e+1, tid2]) >= tau:
                    out[k]=float(T_GRID[tid2]); ok=True; break
            if not ok: out[k]=float(T_GRID[-1])
        if k>0 and out[k] < out[k-1]: out[k]=out[k-1]
    return out

def optimize_groups_from_intervals(intervals: List[SubjectInterval],
                                   h_week: float,
                                   k: int,
                                   tau_ready: float,
                                   alpha: float,
                                   beta: float,
                                   c_retest: float,
                                   tmin: float,
                                   tmax: float,
                                   tstep: float,
                                   fine_step: float,
                                   nmin: int = 1,
                                   delta_min: float = 0.0) -> Dict[str, object]:
    aft = AFTLogLogistic()
    aft.fit(intervals, h_week)
    T_GRID = np.round(np.arange(tmin, tmax + tstep/2, tstep), 5)
    FINE_GRID = np.round(np.arange(8.0, h_week, fine_step), 5)
    w_vec, w_scalar = build_w_time(alpha, beta, h_week)
    W_FINE = w_vec(FINE_GRID)
    order = np.argsort([s.bmi for s in intervals]); ints = [intervals[i] for i in order]
    F_sorted = np.vstack([aft.F(T_GRID, AFTLogLogistic._features(s)) for s in ints])
    R_sorted = np.vstack([risk_for_subject(aft, s, T_GRID, c_retest, FINE_GRID, W_FINE, w_vec, w_scalar) for s in ints])
    bounds, tids, total = dp_segment_with_ready_size(R_sorted, F_sorted, T_GRID, k, tau_ready, nmin)
    if bounds is None: return {"feasible": False}
    t_rec = [float(T_GRID[i]) for i in tids]
    t_adj = enforce_monotone_with_gap(t_rec, bounds, T_GRID, F_sorted, tau_ready, delta_min)
    groups, assign = [], []
    for g, (seg, t0, t1) in enumerate(zip(bounds, t_rec, t_adj), start=1):
        s_idx,e_idx = seg
        bmin = ints[s_idx].bmi; bmax = ints[e_idx].bmi
        ngrp = e_idx - s_idx + 1
        ages_grp = np.array([ints[ii].age for ii in range(s_idx, e_idx+1)], float)
        n35 = int(np.sum(ages_grp >= 35.0))
        p35 = float(n35 / max(ngrp,1))
        tidx = int(np.argmin(np.abs(T_GRID - t1)))
        Fbar = float(F_sorted[s_idx:e_idx+1, tidx].mean())
        groups.append({"组别": g, "BMI下界": round(bmin,2), "BMI上界": round(bmax,2), "建议检测周": float(t1), "原始最优周": float(t0), "人数": int(ngrp), "≥35人数": n35, "≥35占比": round(p35,3), "组内平均F": round(Fbar,3)})
        for ii in range(s_idx, e_idx+1):
            assign.append({"组别":g,"孕妇代码":ints[ii].pid,"BMI":round(ints[ii].bmi,2),"年龄":float(ints[ii].age),"Z13":float(ints[ii].z13),"Z18":float(ints[ii].z18),"Z21":float(ints[ii].z21),"区间L":float(ints[ii].L),"区间R":float(ints[ii].R),"是否右删":int(ints[ii].right_censored),"建议检测周":float(t1)})
    return {"feasible": True, "params": {"beta": aft.beta_, "s": aft.s_}, "T_GRID": T_GRID, "F_sorted": F_sorted, "R_sorted": R_sorted, "bounds": bounds, "t_rec": t_rec, "t_adj": t_adj, "groups": pd.DataFrame(groups), "assignments": pd.DataFrame(assign)}

def optimize_groups_from_excel(excel_path: str,
                               sheet_name: str,
                               thresh: float,
                               h_week: float,
                               k: int,
                               tau_ready: float,
                               alpha: float,
                               beta: float,
                               c_retest: float,
                               tmin: float,
                               tmax: float,
                               tstep: float,
                               fine_step: float,
                               nmin: int = 1,
                               delta_min: float = 0.0,
                               overrides: Optional[dict] = None) -> Dict[str, object]:
    intervals = load_intervals_from_excel(excel_path, sheet_name, thresh, h_week, overrides or {})
    return optimize_groups_from_intervals(intervals, h_week, k, tau_ready, alpha, beta, c_retest, tmin, tmax, tstep, fine_step, nmin, delta_min)

def sensitivity_threshold_scan(excel: str, sheet: str, thresh_list: List[float], h_week: float, k: int, tau_ready: float, alpha: float, beta: float, c_retest: float, tmin: float, tmax: float, tstep: float, fine_step: float, nmin: int, delta_min: float, overrides: Optional[dict] = None) -> pd.DataFrame:
    rows=[]
    for v in thresh_list:
        ints = load_intervals_from_excel(excel, sheet, v, h_week, overrides or {})
        out = optimize_groups_from_intervals(ints, h_week, k, tau_ready, alpha, beta, c_retest, tmin, tmax, tstep, fine_step, nmin, delta_min)
        rows.append({"Y_threshold": v, "feasible": int(out.get("feasible", False)), "avg_cost": (float(np.nan) if not out.get("feasible", False) else float(np.nanmean([out["R_sorted"][s:e+1, int(np.argmin(np.abs(out["T_GRID"]-t)) )].sum() for (s,e), t in zip(out["bounds"], out["t_adj"])]))/out["R_sorted"].shape[0]), "t_mean": (float(np.nan) if not out.get("feasible", False) else float(np.mean(out["t_adj"]))), "Fbar_mean": (float(np.nan) if not out.get("feasible", False) else float(np.mean([out["F_sorted"][s:e+1, int(np.argmin(np.abs(out["T_GRID"]-t)) )].mean() for (s,e), t in zip(out["bounds"], out["t_adj"])])))})
    return pd.DataFrame(rows)

def load_intervals_with_y_noise(excel: str, sheet: str, thresh: float, h_week: float, noise_abs: float, seed: Optional[int], overrides: dict) -> List[SubjectInterval]:
    df = read_base_df(excel, sheet, overrides)
    rng = np.random.default_rng(seed)
    if noise_abs > 0:
        y = df['__Y__'].to_numpy(dtype=float)
        y_pert = y + rng.normal(0.0, noise_abs, size=y.shape)
        df['__Y__'] = np.clip(y_pert, a_min=0.0, a_max=None)
    return intervals_from_df(df, thresh, h_week)

def sensitivity_noise_scan(excel: str, sheet: str, thresh: float, noise_values: List[float], reps: int, seed: int, h_week: float, k: int, tau_ready: float, alpha: float, beta: float, c_retest: float, tmin: float, tmax: float, tstep: float, fine_step: float, nmin: int, delta_min: float, overrides: Optional[dict] = None) -> pd.DataFrame:
    rows=[]
    for nv in noise_values:
        for r in range(int(reps)):
            ints = load_intervals_with_y_noise(excel, sheet, thresh, h_week, nv, int(seed)+r, overrides or {})
            out = optimize_groups_from_intervals(ints, h_week, k, tau_ready, alpha, beta, c_retest, tmin, tmax, tstep, fine_step, nmin, delta_min)
            rows.append({"noise_abs": nv, "rep": r, "feasible": int(out.get("feasible", False)), "avg_cost": (float(np.nan) if not out.get("feasible", False) else float(np.nanmean([out["R_sorted"][s:e+1, int(np.argmin(np.abs(out["T_GRID"]-t)) )].sum() for (s,e), t in zip(out["bounds"], out["t_adj"])]))/out["R_sorted"].shape[0]), "t_mean": (float(np.nan) if not out.get("feasible", False) else float(np.mean(out["t_adj"]))), "Fbar_mean": (float(np.nan) if not out.get("feasible", False) else float(np.mean([out["F_sorted"][s:e+1, int(np.argmin(np.abs(out["T_GRID"]-t)) )].mean() for (s,e), t in zip(out["bounds"], out["t_adj"])])))})
    return pd.DataFrame(rows)
