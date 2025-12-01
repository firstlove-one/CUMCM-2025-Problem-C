def parse_week_str(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    m = re.match(r"^\s*(\d+)\s*[wW周]?\s*\+\s*(\d+)\s*$", s)
    if m: return int(m.group(1)) + int(m.group(2))/7.0
    m2 = re.match(r"^\s*(\d+)\s*[wW周]\s*$", s)
    if m2: return float(m2.group(1))
    try: return float(s)
    except: return np.nan

def safe_sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

@dataclass
class SubjectInterval:
    pid: str
    bmi: float
    L: float
    R: float

def load_male_intervals(excel_path: str, sheet_name: str, thresh: float, h_week: float) -> List[SubjectInterval]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    need = ['身高','体重','检测孕周','Y染色体浓度','孕妇代码']
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"缺少列: {miss}")
    h = pd.to_numeric(df['身高'], errors="coerce")/100.0
    w = pd.to_numeric(df['体重'], errors="coerce")
    df['BMI_calc'] = w/(h**2)
    df['孕周_周'] = df['检测孕周'].apply(parse_week_str)
    df['Y达标'] = pd.to_numeric(df['Y染色体浓度'], errors="coerce") >= float(thresh)
    out: List[SubjectInterval] = []
    for pid, g in df.groupby('孕妇代码'):
        gi = g.sort_values('孕周_周').dropna(subset=['孕周_周'])
        if gi.empty: continue
        bmi = float(gi.iloc[0]['BMI_calc'])
        rch = gi[gi['Y达标']]
        if not rch.empty:
            R = float(rch.iloc[0]['孕周_周'])
            L = 0.0
            pre = gi[(gi['孕周_周']<R) & (~gi['Y达标'])]
            if not pre.empty: L = float(pre['孕周_周'].max())
        else:
            L = float(gi['孕周_周'].max())
            R = float(h_week)
        if np.isfinite(L) and np.isfinite(R) and L<=R:
            out.append(SubjectInterval(str(pid), bmi, L, R))
    return out

class TimeToReadyModel:
    def __init__(self, intervals: List[SubjectInterval], h_week: float, l2=5e-4):
        self.h_week = float(h_week)
        self.b = np.array([x.bmi for x in intervals], float)
        self.L = np.array([x.L for x in intervals], float)
        self.R = np.array([x.R for x in intervals], float)
        self.l2 = float(l2)
        self.params: Optional[np.ndarray] = None
        self.nll_history: List[float] = []

    def _nll(self, params):
        m0, m1, m2, s0, s1 = params
        mu = m0 + m1*self.b + m2*(self.b**2)
        s  = np.exp(s0 + s1*self.b)
        FL = safe_sigmoid((self.L - mu)/s)
        mask_int = self.R < (self.h_week - 1e-9)
        ll_int = 0.0
        if np.any(mask_int):
            FR = safe_sigmoid((self.R[mask_int] - mu[mask_int])/s[mask_int])
            diff = np.maximum(FR - FL[mask_int], 1e-9)
            ll_int = np.log(diff).sum()
        mask_rc = ~mask_int
        ll_rc = 0.0
        if np.any(mask_rc):
            ll_rc = np.log(np.maximum(1.0 - FL[mask_rc], 1e-9)).sum()
        penalty = self.l2 * float(np.sum(params*params))
        return -(ll_int + ll_rc) + penalty

    def fit(self, maxiter=2000):
        init = np.array([12.0, 0.05, 0.0, math.log(2.0), 0.0], float)
        bounds = [(-10, 40), (-1, 1), (-0.2, 0.2), (math.log(0.2), math.log(10.0)), (-0.5, 0.5)]
        self.nll_history = []
        def _cb(xk): self.nll_history.append(self._nll(xk))
        res = minimize(lambda p: self._nll(p), init, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": maxiter}, callback=_cb)
        self.params = res.x
        if len(self.nll_history)==0 or abs(self.nll_history[-1]-res.fun)>1e-10:
            self.nll_history.append(res.fun)
        return res

    def F_hat(self, t, b):
        if self.params is None: raise RuntimeError("not fitted")
        t = np.asarray(t, float); b = np.asarray(b, float)
        m0, m1, m2, s0, s1 = self.params
        mu = m0 + m1*b + m2*(b**2)
        s  = np.exp(s0 + s1*b)
        return safe_sigmoid((t - mu)/s)

def build_w_time(alpha: float, beta: float, h_week: float):
    def w_scalar(t):
        if t<=12.0: return 0.0
        if t<=27.0: return alpha*(t-12.0)/15.0
        return alpha + beta*(t-27.0)/(h_week-27.0)
    def w_vec(t):
        t = np.asarray(t, float)
        z = np.zeros_like(t)
        m1 = (t>12.0) & (t<=27.0)
        m2 = (t>27.0)
        z[m1] = alpha*(t[m1]-12.0)/15.0
        z[m2] = alpha + beta*(t[m2]-27.0)/(h_week-27.0)
        return z
    return np.vectorize(w_scalar), w_scalar

def expected_w_after_t(b: float, t: float, F_hat, fine_grid: np.ndarray, W_FINE: np.ndarray, w_time_scalar):
    mask = fine_grid > t
    if not np.any(mask): return W_FINE[-1]
    u = fine_grid[mask]
    Fu = F_hat(u, b)
    Fu_prev = np.concatenate([[F_hat(t, b)], F_hat(u[:-1], b)])
    dens = np.clip(Fu - Fu_prev, 0.0, 1.0)
    tail = 1.0 - F_hat(t, b)
    if tail < 1e-8: return w_time_scalar(t)
    return float(np.sum(W_FINE[mask] * dens) / tail)

def risk_at_times_for_b(b: float, t_vec: np.ndarray, F_hat, c_retest: float,
                        fine_grid: np.ndarray, W_FINE: np.ndarray, w_time_vec, w_time_scalar):
    Ft = F_hat(t_vec, b)
    not_ready = 1.0 - Ft
    ew = np.array([expected_w_after_t(b, float(t), F_hat, fine_grid, W_FINE, w_time_scalar) for t in t_vec])
    return Ft * w_time_vec(t_vec) + not_ready * (c_retest + ew)

def dp_segment_with_ready_constraint(Rmat_sorted: np.ndarray, F_all_sorted: np.ndarray,
                                     t_grid: np.ndarray, k_groups: int, tau_ready: float):
    n, T = Rmat_sorted.shape
    INF = np.inf
    prefix_R = np.vstack([np.zeros(T), np.cumsum(Rmat_sorted, axis=0)])
    prefix_F = np.vstack([np.zeros(T), np.cumsum(F_all_sorted, axis=0)])
    CostStar = np.full((n, n), INF); TArg = np.full((n, n), -1)
    for i in range(n):
        for j in range(i, n):
            seg_R = prefix_R[j+1]-prefix_R[i]
            meanF = (prefix_F[j+1]-prefix_F[i])/(j-i+1)
            feas = (meanF >= tau_ready)
            if not np.any(feas): continue
            seg_R = np.where(feas, seg_R, INF)
            t_idx = int(np.argmin(seg_R))
            if not np.isfinite(seg_R[t_idx]): continue
            CostStar[i,j] = float(seg_R[t_idx]); TArg[i,j] = t_idx
    dp = np.full((k_groups, n), INF); prev = np.full((k_groups, n), -1)
    for j in range(n): dp[0,j] = CostStar[0,j]
    for g in range(1, k_groups):
        for j in range(g, n):
            best = INF; bi = -1
            for i in range(g-1, j):
                if not (np.isfinite(dp[g-1,i]) and np.isfinite(CostStar[i+1,j])): continue
                val = dp[g-1,i] + CostStar[i+1,j]
                if val < best: best, bi = val, i
            dp[g,j] = best; prev[g,j] = bi
    if not np.isfinite(dp[k_groups-1, n-1]): return None, None
    boundaries, t_idx = [], []
    g, j = k_groups-1, n-1
    while g >= 0:
        i = prev[g,j]; s = 0 if i==-1 else i+1; e = j
        boundaries.append((s,e)); t_idx.append(TArg[s,e]); j = i; g -= 1
    boundaries.reverse(); t_idx.reverse()
    t_vals = [float(t_grid[idx]) for idx in t_idx]
    for k in range(1,len(t_vals)):
        if t_vals[k] < t_vals[k-1]:
            t_vals[k] = t_vals[k-1]
            t_idx[k] = int(np.argmin(np.abs(t_grid - t_vals[k])))
    return boundaries, t_idx

def dp_segment_unconstrained(Rmat_sorted: np.ndarray, t_grid: np.ndarray, k_groups: int):
    n, T = Rmat_sorted.shape
    prefix_R = np.vstack([np.zeros(T), np.cumsum(Rmat_sorted, axis=0)])
    CostStar = np.full((n, n), np.inf); TArg = np.full((n, n), -1)
    for i in range(n):
        for j in range(i, n):
            seg_R = prefix_R[j+1]-prefix_R[i]
            idx = int(np.argmin(seg_R))
            CostStar[i,j] = float(seg_R[idx]); TArg[i,j] = idx
    dp = np.full((k_groups,n), np.inf); prev = np.full((k_groups,n), -1)
    for j in range(n): dp[0,j] = CostStar[0,j]
    for g in range(1,k_groups):
        for j in range(g,n):
            best=np.inf; bi=-1
            for i in range(g-1,j):
                val = dp[g-1,i] + CostStar[i+1,j]
                if val<best: best,bi=val,i
            dp[g,j]=best; prev[g,j]=bi
    boundaries, t_idx = [], []
    g, j = k_groups-1, n-1
    while g>=0:
        i=prev[g,j]; s=0 if i==-1 else i+1; e=j
        boundaries.append((s,e)); t_idx.append(TArg[s,e]); j=i; g-=1
    boundaries.reverse(); t_idx.reverse()
    t_vals = [float(t_grid[idx]) for idx in t_idx]
    for k in range(1,len(t_vals)):
        if t_vals[k] < t_vals[k-1]:
            t_vals[k] = t_vals[k-1]
            t_idx[k] = int(np.argmin(np.abs(t_grid - t_vals[k])))
    return boundaries, t_idx

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
                                   fine_step: float
                                   ) -> Dict[str, object]:
    model = TimeToReadyModel(intervals, h_week=h_week, l2=5e-4)
    model.fit(maxiter=2000)
    T_GRID = np.round(np.arange(tmin, tmax + tstep/2, tstep), 5)
    FINE_GRID = np.round(np.arange(8.0, h_week, fine_step), 5)
    w_vec, w_scalar = build_w_time(alpha, beta, h_week)
    W_FINE = w_vec(FINE_GRID)
    b = np.array([s.bmi for s in intervals], float)
    pids = [s.pid for s in intervals]
    Rmat = np.zeros((len(intervals), len(T_GRID)), float)
    for i, s in enumerate(intervals):
        Rmat[i,:] = risk_at_times_for_b(s.bmi, T_GRID, model.F_hat, c_retest, FINE_GRID, W_FINE, w_vec, w_scalar)
    order = np.argsort(b)
    b_sorted = b[order]
    pid_sorted = [pids[i] for i in order]
    R_sorted = Rmat[order,:]
    F_sorted = np.vstack([model.F_hat(T_GRID, bi) for bi in b_sorted])
    bnd, tidx = dp_segment_with_ready_constraint(R_sorted, F_sorted, T_GRID, k, tau_ready)
    constrained = True
    if bnd is None:
        bnd, tidx = dp_segment_unconstrained(R_sorted, T_GRID, k)
        constrained = False
    groups, assign = [], []
    for g, ((s_idx,e_idx), ti) in enumerate(zip(bnd, tidx), start=1):
        bmin = float(b_sorted[s_idx]); bmax = float(b_sorted[e_idx]); t_rec = float(T_GRID[ti])
        groups.append({"组别": g, "BMI下界": round(bmin,2), "BMI上界": round(bmax,2), "建议检测周": t_rec})
        for ii in range(s_idx, e_idx+1):
            assign.append({"组别": g, "孕妇代码": pid_sorted[ii], "BMI": round(float(b_sorted[ii]),2), "建议检测周": t_rec})
    return {
        "constrained": constrained,
        "params": model.params,
        "groups": pd.DataFrame(groups),
        "assignments": pd.DataFrame(assign),
        "T_GRID": T_GRID,
        "R_sorted": R_sorted,
        "b_sorted": b_sorted
    }

def optimize_groups_from_excel(excel_path: str,
                               sheet_name: str = "男胎检测数据",
                               thresh: float = 0.04,
                               h_week: float = 30.0,
                               k: int = 5,
                               tau_ready: float = 0.95,
                               alpha: float = 1.0,
                               beta: float = 8.0,
                               c_retest: float = 1.0,
                               tmin: float = 10.0,
                               tmax: float = 25.0,
                               tstep: float = 0.5,
                               fine_step: float = 0.25
                               ) -> Dict[str, object]:
    intervals = load_male_intervals(excel_path, sheet_name, thresh, h_week)
    return optimize_groups_from_intervals(intervals, h_week, k, tau_ready, alpha, beta, c_retest, tmin, tmax, tstep, fine_step)
