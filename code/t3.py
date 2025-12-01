import os, re, math, argparse, warnings, glob, unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans", "Microsoft YaHei"]
mpl.rcParams["axes.unicode_minus"] = False

def normalize_name(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[\s\-\_\(\)\[\]{}·•、，,。:：;；/%]+", "", s)
    return s

def parse_week_str(s):
    if pd.isna(s): return np.nan
    if isinstance(s, (int, float)): return float(s)
    m = re.search(r"(\d+(\.\d+)?)", str(s).strip())
    return float(m.group(1)) if m else np.nan

def _mpl_clean(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

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

def choose_excel_default(excel_arg: Optional[str]) -> str:
    if excel_arg and os.path.exists(excel_arg): return excel_arg
    for name in ["附件.xlsx","附件.xls","数据.xlsx","data.xlsx"]:
        if os.path.exists(name): return name
    for pat in ["*.xlsx","*.xls"]:
        lst = glob.glob(pat)
        if lst: return lst[0]
    raise FileNotFoundError("未找到 Excel，请放在当前目录。")

def choose_sheet_default(excel_path: str, sheet_arg: Optional[str]) -> str:
    xls = pd.ExcelFile(excel_path); sheets = xls.sheet_names
    if sheet_arg and sheet_arg in sheets: return sheet_arg
    if "男胎检测数据" in sheets: return "男胎检测数据"
    if sheets: return sheets[0]
    raise RuntimeError("Excel 中没有工作表。")

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
        raise ValueError("无法自动识别 编号/孕周/Y 列，请用 --col_id / --col_week / --col_y 指定。")
    df = df.copy()
    if cols["bmi"] is not None:
        df['__BMI__'] = pd.to_numeric(df[cols["bmi"]], errors='coerce')
    else:
        if cols["height"] is None or cols["weight"] is None:
            raise ValueError("缺少 BMI，且未找到 身高/体重。")
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

def intervals_from_df(df: pd.DataFrame, thresh: float, h_week: float):
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

def load_intervals_from_excel(excel: str, sheet: str, thresh: float, h_week: float,
                              overrides: dict) -> List[SubjectInterval]:
    df = read_base_df(excel, sheet, overrides)
    return intervals_from_df(df, thresh, h_week)

def load_intervals_with_y_noise(excel: str, sheet: str, thresh: float, h_week: float,
                                noise_abs: float, seed: Optional[int],
                                overrides: dict) -> List[SubjectInterval]:
    df = read_base_df(excel, sheet, overrides)
    rng = np.random.default_rng(seed)
    if noise_abs > 0:
        y = df['__Y__'].to_numpy(dtype=float)
        y_pert = y + rng.normal(0.0, noise_abs, size=y.shape)
        df['__Y__'] = np.clip(y_pert, a_min=0.0, a_max=None)
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
        from scipy.optimize import minimize
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
        from scipy.optimize import minimize
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

        bounds = [
            (-10,10),(-5,5),(-2,2),   # β0, βb, βb2
            (-2,2), (-2,2), (-2,2), (-2,2),  # β_age, β13, β18, β21
            (math.log(0.2), math.log(2.5)),  # s0
            (-2,2)                            # s1
        ]
        res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": self.maxiter})
        return res.x

    def fit(self, intervals, h_week, cont_rounds=6, cont_growth=2.0,
            target_mean_pos=0.015, target_max_pos=0.05):
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
            if self.verbose:
                print(f"[align] round {r+1}: mean_pos={mean_pos:.4f}, max_pos={max_pos:.4f}, "
                      f"lam_full={lam_full:.3g}, lam_pos={lam_pos:.3g}, lam_extra={lam_extra:.3g}")
            if (mean_pos <= target_mean_pos) and (max_pos <= target_max_pos):
                break
            lam_full *= cont_growth
            lam_pos  *= cont_growth
            lam_extra*= cont_growth**0.5
        self.beta_ = best_theta[:X.shape[1]].copy()
        self.s_    = best_theta[X.shape[1]:].copy()
        self.align_mean_pos_, self.align_max_pos_ = best_metrics

    def F(self, t: np.ndarray, subj_or_x) -> np.ndarray:
        if isinstance(subj_or_x, SubjectInterval):
            x = self._features(subj_or_x)
        else:
            x = np.asarray(subj_or_x, float)
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

def plot_t_vs_bmi_steps(groups_df: pd.DataFrame, save_path: str, dpi=150):
    xs, ys = [], []
    for _, r in groups_df.iterrows():
        xs += [r["BMI下界"], r["BMI上界"]]
        ys += [r["建议检测周"], r["建议检测周"]]
    fig, ax = plt.subplots(figsize=(9,5))
    ax.step(xs, ys, where='post')
    ax.set_xlabel("BMI 区间边界"); ax.set_ylabel("建议检测周")
    _mpl_clean(ax); fig.tight_layout(); fig.savefig(save_path, dpi=dpi); plt.close(fig)

def plot_group_curves(groups_df: pd.DataFrame,
                      median_x_by_group: List[np.ndarray],
                      F_model: AFTLogLogistic,
                      t_fine: np.ndarray, save_path: str, dpi=150):
    fig, ax = plt.subplots(figsize=(10,6))
    for i, row in groups_df.iterrows():
        Ft = F_model.F(t_fine, median_x_by_group[i])
        ax.plot(t_fine, Ft, label=f'组{i+1} (n={int(row["人数"])})')
        ax.axvline(row["建议检测周"], ls="--", lw=1)
    ax.set_xlabel("孕周"); ax.set_ylabel("就绪比例 F(t)")
    ax.legend(frameon=False); _mpl_clean(ax)
    fig.tight_layout(); fig.savefig(save_path, dpi=dpi); plt.close(fig)

def plot_passrate_bars(groups_df: pd.DataFrame, save_path: str, dpi=150):
    fig, ax = plt.subplots(figsize=(7,5))
    x = np.arange(len(groups_df))
    y = groups_df["组内平均F"].values
    ax.bar(x, y)
    for i, v in enumerate(y):
        ax.text(i, v+0.003, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0.85, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f'组{i+1}' for i in range(len(groups_df))])
    ax.set_xlabel("分组"); ax.set_ylabel("组内平均就绪率 F")
    _mpl_clean(ax); fig.tight_layout(); fig.savefig(save_path, dpi=dpi); plt.close(fig)

GROUP_COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#C00000", "#7F60A8"]

def plot_curve_ci(t, mean, lo, hi, label, ax, color=None, alpha=0.18):
    ax.plot(t, mean, lw=2.0, label=label, color=color)
    ax.fill_between(t, lo, hi, alpha=alpha, color=color)

def compute_ready_curves(intervals: List[SubjectInterval],
                         model: AFTLogLogistic,
                         t_grid: np.ndarray,
                         group_ranges: List[Tuple[float,float]]):
    Fi_all = []
    bmis = np.array([s.bmi for s in intervals], float)
    for s in intervals:
        Fi_all.append(model.F(t_grid, AFTLogLogistic._features(s)))
    Fi_all = np.vstack(Fi_all)  # (n, T)
    overall = Fi_all.mean(axis=0)
    G = len(group_ranges)
    group_means = np.full((G, len(t_grid)), np.nan)
    for gi, (lo, hi) in enumerate(group_ranges):
        idx = np.where((bmis >= lo - 1e-9) & (bmis <= hi + 1e-9))[0]
        if idx.size > 0:
            group_means[gi, :] = Fi_all[idx, :].mean(axis=0)
    return overall, group_means

def bootstrap_ready(intervals: List[SubjectInterval],
                    t_grid: np.ndarray,
                    group_ranges: List[Tuple[float,float]],
                    B: int = 200,
                    seed: Optional[int] = None,
                    l2: float = 1e-4,
                    maxiter: int = 800):
    rng = np.random.default_rng(seed)
    n = len(intervals)
    overall_mat, groups_list = [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sample = [intervals[i] for i in idx]
        model_b = AFTLogLogistic(l2=l2, maxiter=maxiter, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_b.fit(sample, h_week=max([s.R for s in sample]))
        overall_b, groups_b = compute_ready_curves(sample, model_b, t_grid, group_ranges)
        overall_mat.append(overall_b)
        groups_list.append(groups_b)
    overall_mat = np.vstack(overall_mat)             
    groups_arr = np.stack(groups_list, axis=0)        
    return overall_mat, groups_arr

def ci_from_boot(mat, level=0.95, axis=0):
    lo_q = (1-level)/2*100
    hi_q = (1+level)/2*100
    return (np.nanpercentile(mat, lo_q, axis=axis),
            np.nanpercentile(mat, hi_q, axis=axis))

def time_to_level_from_curve(t_grid, curve, level):
    idx = np.where(curve >= level)[0]
    return float(t_grid[idx[0]]) if idx.size>0 else np.nan

def plot_groups_ci_combined(T_GRID, group_means, lo_groups, hi_groups,
                            out_png, dpi=150):
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    G = group_means.shape[0]
    for gi in range(G):
        if np.all(np.isnan(group_means[gi,:])): continue
        plot_curve_ci(T_GRID, group_means[gi,:], lo_groups[gi,:], hi_groups[gi,:],
                      f"组{gi+1}", ax, color=GROUP_COLORS[gi % len(GROUP_COLORS)])
    ax.set_xlabel("孕周"); ax.set_ylabel("达标比例（Y≥阈值）")
    _mpl_clean(ax); ax.legend(frameon=False, ncols=2)
    fig.tight_layout(); fig.savefig(out_png, dpi=dpi); plt.close(fig)

def _solve_once(intervals_sorted, aft_model, T_GRID,
                FINE_GRID, W_FINE, w_vec, w_scalar,
                K, tau_ready, nmin, delta_min, c_retest) -> dict:
    F_all_sorted = np.zeros((len(intervals_sorted), len(T_GRID)))
    for i, s in enumerate(intervals_sorted):
        F_all_sorted[i, :] = aft_model.F(T_GRID, AFTLogLogistic._features(s))
    Rmat_sorted = np.zeros_like(F_all_sorted)
    for i, s in enumerate(intervals_sorted):
        Rmat_sorted[i, :] = risk_for_subject(aft_model, s, T_GRID,
                                             c_retest, FINE_GRID, W_FINE,
                                             w_vec, w_scalar)
    ans = {"feasible": True}
    r = dp_segment_with_ready_size(Rmat_sorted, F_all_sorted, T_GRID, K, tau_ready, nmin)
    boundaries, t_idx, total_cost = r
    if boundaries is None:
        ans["feasible"] = False
        return ans
    t_rec = [float(T_GRID[idx]) for idx in t_idx]
    t_adj = enforce_monotone_with_gap(t_rec, boundaries, T_GRID, F_all_sorted, tau_ready, delta_min)
    if total_cost is None or not np.isfinite(total_cost):
        total_cost = 0.0
        for (s,e), tid in zip(boundaries, t_idx):
            total_cost += float(Rmat_sorted[s:e+1, tid].sum())
    avg_cost = float(total_cost / max(1, len(intervals_sorted)))
    groups_Fbar = []
    for (s,e), t in zip(boundaries, t_adj):
        tidx_new = int(np.argmin(np.abs(T_GRID - t)))
        groups_Fbar.append(float(F_all_sorted[s:e+1, tidx_new].mean()))
    ans.update({
        "avg_cost": avg_cost,
        "t_rec": t_rec,
        "t_adj": t_adj,
        "groups_Fbar": groups_Fbar,
        "boundaries": boundaries
    })
    return ans

def _parse_list_float(s: str):
    if not s: return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def run_sensitivity(args, base_intervals, base_aft, overrides, dpi=150):
    excel_local = args.excel
    sheet_local = args.sheet

    order = np.argsort([s.bmi for s in base_intervals])
    intervals_sorted = [base_intervals[i] for i in order]

    T_GRID = np.round(np.arange(args.tmin, args.tmax + 1e-8, args.tstep), 5)
    FINE_GRID = np.round(np.arange(8.0, args.hweek, args.fine_step), 5)
    w_vec, w_scalar = build_w_time(args.alpha, args.beta, args.hweek)
    W_FINE = w_vec(FINE_GRID)

    _ = _solve_once(intervals_sorted, base_aft, T_GRID, FINE_GRID, W_FINE, w_vec, w_scalar,
                    args.k, args.tau_ready, args.nmin, args.delta_min, args.c_retest)

    th_values = _parse_list_float(args.sens_threshY)
    rows_thresh = []
    for v in th_values:
        intervals_v = load_intervals_from_excel(excel_local, sheet_local, v, args.hweek, overrides)
        aft_v = AFTLogLogistic(l2=args.l2, maxiter=1200, verbose=False,
                               pen_align_full=args.pen_align_full, pen_align_pos=args.pen_align_pos,
                               pen_extra=args.pen_extra, pen_sigma=args.pen_sigma,
                               norm_by_sigma=bool(args.norm_by_sigma))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aft_v.fit(intervals_v, args.hweek,
                      cont_rounds=args.align_rounds, cont_growth=args.align_growth,
                      target_mean_pos=args.align_target_mean, target_max_pos=args.align_target_max)
        order_v = np.argsort([s.bmi for s in intervals_v])
        intervals_sorted_v = [intervals_v[i] for i in order_v]
        out = _solve_once(intervals_sorted_v, aft_v, T_GRID, FINE_GRID, W_FINE, w_vec, w_scalar,
                          args.k, args.tau_ready, args.nmin, args.delta_min, args.c_retest)
        rows_thresh.append({
            "Y_threshold": v,
            "feasible": int(out["feasible"]),
            "avg_cost": (out["avg_cost"] if out["feasible"] else np.nan),
            "t_mean": (float(np.mean(out["t_adj"])) if out["feasible"] else np.nan),
            "Fbar_mean": (float(np.mean(out["groups_Fbar"])) if out["feasible"] else np.nan)
        })
    df_thresh = pd.DataFrame(rows_thresh)
    csv_thresh = os.path.join(args.outdir, "q3_sens_y_threshold.csv")
    df_thresh.to_csv(csv_thresh, index=False, encoding="utf-8-sig")

    noise_values = _parse_list_float(args.sens_y_noise_abs)
    rows_noise_raw = []
    seed0 = int(args.seed)
    for nv in noise_values:
        for r in range(int(args.sens_reps)):
            intervals_n = load_intervals_with_y_noise(excel_local, sheet_local, args.thresh, args.hweek,
                                                      noise_abs=nv, seed=seed0 + r, overrides=overrides)
            aft_n = AFTLogLogistic(l2=args.l2, maxiter=1000, verbose=False,
                                   pen_align_full=args.pen_align_full, pen_align_pos=args.pen_align_pos,
                                   pen_extra=args.pen_extra, pen_sigma=args.pen_sigma,
                                   norm_by_sigma=bool(args.norm_by_sigma))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aft_n.fit(intervals_n, args.hweek,
                          cont_rounds=args.align_rounds, cont_growth=args.align_growth,
                          target_mean_pos=args.align_target_mean, target_max_pos=args.align_target_max)
            order_n = np.argsort([s.bmi for s in intervals_n])
            intervals_sorted_n = [intervals_n[i] for i in order_n]
            out = _solve_once(intervals_sorted_n, aft_n, T_GRID, FINE_GRID, W_FINE, w_vec, w_scalar,
                              args.k, args.tau_ready, args.nmin, args.delta_min, args.c_retest)
            rows_noise_raw.append({
                "noise_abs": nv,
                "rep": r,
                "feasible": int(out["feasible"]),
                "avg_cost": (out["avg_cost"] if out["feasible"] else np.nan),
                "t_mean": (float(np.mean(out["t_adj"])) if out["feasible"] else np.nan),
                "Fbar_mean": (float(np.mean(out["groups_Fbar"])) if out["feasible"] else np.nan)
            })
    df_noise_raw = pd.DataFrame(rows_noise_raw)
    csv_noise_raw = os.path.join(args.outdir, "q3_sens_y_noise_raw.csv")
    df_noise_raw.to_csv(csv_noise_raw, index=False, encoding="utf-8-sig")

    summ_rows = []
    for nv in noise_values:
        g = df_noise_raw[df_noise_raw["noise_abs"]==nv]
        g = g[g["feasible"]==1]
        if len(g)==0:
            summ_rows.append({"noise_abs": nv, "n_feasible": 0})
            continue
        def q(v, a): return float(np.nanpercentile(v.dropna().values, a)) if v.notna().any() else np.nan
        summ_rows.append({
            "noise_abs": nv,
            "n_feasible": int(len(g)),
            "avg_cost_mean": float(g["avg_cost"].mean()),
            "avg_cost_q025": q(g["avg_cost"], 2.5),
            "avg_cost_q975": q(g["avg_cost"], 97.5),
            "t_mean_mean": float(g["t_mean"].mean()),
            "t_mean_q025": q(g["t_mean"], 2.5),
            "t_mean_q975": q(g["t_mean"], 97.5),
            "Fbar_mean_mean": float(g["Fbar_mean"].mean())
        })
    df_noise_summ = pd.DataFrame(summ_rows)
    csv_noise_summ = os.path.join(args.outdir, "q3_sens_y_noise_summary.csv")
    df_noise_summ.to_csv(csv_noise_summ, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    dft = df_thresh[df_thresh["feasible"]==1].sort_values("Y_threshold")
    if not dft.empty:
        ax.plot(dft["Y_threshold"], dft["avg_cost"], marker="o", label="avg_cost")
        ax2 = ax.twinx()
        ax2.plot(dft["Y_threshold"], dft["t_mean"], "--", marker="s", label="t_mean")
        ax.set_xlabel("Y 达标阈值"); ax.set_ylabel("人均总风险")
        ax2.set_ylabel("平均推荐周")
        _mpl_clean(ax); _mpl_clean(ax2)
        ax.set_title("敏感性：阈值")
    else:
        ax.text(0.5, 0.5, "无可行点", ha="center", va="center"); ax.axis("off")

    ax = axes[1]
    dfn = df_noise_summ[df_noise_summ["n_feasible"] > 0].sort_values("noise_abs").reset_index(drop=True)
    if not dfn.empty:
        mean_cost  = dfn["avg_cost_mean"].to_numpy(float)
        lo_cost    = dfn["avg_cost_q025"].to_numpy(float)
        hi_cost    = dfn["avg_cost_q975"].to_numpy(float)
        err_low_c  = np.nan_to_num(np.maximum(mean_cost - lo_cost, 0.0))
        err_high_c = np.nan_to_num(np.maximum(hi_cost - mean_cost, 0.0))

        ax.errorbar(dfn["noise_abs"], mean_cost,
                    yerr=[err_low_c, err_high_c],
                    fmt="o-", capsize=3, label="avg_cost")

        mean_t  = dfn["t_mean_mean"].to_numpy(float)
        lo_t    = dfn["t_mean_q025"].to_numpy(float)
        hi_t    = dfn["t_mean_q975"].to_numpy(float)
        err_low_t  = np.nan_to_num(np.maximum(mean_t - lo_t, 0.0))
        err_high_t = np.nan_to_num(np.maximum(hi_t - mean_t, 0.0))

        ax2 = ax.twinx()
        ax2.errorbar(dfn["noise_abs"], mean_t,
                     yerr=[err_low_t, err_high_t],
                     fmt="s--", capsize=3, label="t_mean")

        ax.set_xlabel("Y 测量绝对噪声 σ（浓度单位）"); ax.set_ylabel("人均总风险")
        ax2.set_ylabel("平均推荐周")
        _mpl_clean(ax); _mpl_clean(ax2)
        ax.set_title("敏感性：测量误差")
    else:
        ax.text(0.5, 0.5, "无可行点", ha="center", va="center"); ax.axis("off")

    fig.tight_layout()
    png_lines = os.path.join(args.outdir, "q3_sens_lines.png")
    fig.savefig(png_lines, dpi=dpi); plt.close(fig)

    return csv_thresh, csv_noise_raw, csv_noise_summ, png_lines
def plot_k_elbow(ints, R_sorted, F_sorted, T_GRID, args,
                 save_path: str, k_min: int = 2, k_max: int = 8, dpi: int = 150):
    Ks = list(range(k_min, k_max + 1))
    N = len(ints)
    avg_costs, feasible = [], []

    for K in Ks:
        bounds, tids, total_cost = dp_segment_with_ready_size(
            R_sorted, F_sorted, T_GRID, K, args.tau_ready, args.nmin
        )
        if bounds is None:
            avg_costs.append(np.nan); feasible.append(False); continue

        t_rec = [float(T_GRID[i]) for i in tids]
        _ = enforce_monotone_with_gap(t_rec, bounds, T_GRID, F_sorted, args.tau_ready, args.delta_min)

        if total_cost is None or not np.isfinite(total_cost):
            total_cost = 0.0
            for (s, e), tid in zip(bounds, tids):
                total_cost += float(R_sorted[s:e+1, tid].sum())
        avg_costs.append(float(total_cost) / max(1, N))
        feasible.append(True)

    avg_costs = np.array(avg_costs, float)

    rel_improve = np.full_like(avg_costs, np.nan)
    for i in range(1, len(Ks)):
        if feasible[i] and feasible[i-1] and np.isfinite(avg_costs[i-1]) and avg_costs[i-1] > 0:
            rel_improve[i] = (avg_costs[i-1] - avg_costs[i]) / avg_costs[i-1] * 100.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Ks, avg_costs, marker="o")
    ax.set_xlabel("分组数 K")
    ax.set_ylabel("人均总风险")

    ax2 = ax.twinx()
    rel_plot = np.nan_to_num(rel_improve, nan=0.0) 
    ax2.bar(Ks, rel_plot, alpha=0.25)
    ax2.set_ylabel("相邻改进率 (%)")

    _mpl_clean(ax); _mpl_clean(ax2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", type=str, default=None)
    ap.add_argument("--sheet", type=str, default=None)
    ap.add_argument("--col_id", type=str); ap.add_argument("--col_week", type=str); ap.add_argument("--col_y", type=str)
    ap.add_argument("--col_bmi", type=str); ap.add_argument("--col_height", type=str); ap.add_argument("--col_weight", type=str)
    ap.add_argument("--col_age", type=str); ap.add_argument("--col_z13", type=str); ap.add_argument("--col_z18", type=str); ap.add_argument("--col_z21", type=str)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau_ready", type=float, default=0.95)
    ap.add_argument("--thresh", type=float, default=0.04)
    ap.add_argument("--hweek", type=float, default=27.0)
    ap.add_argument("--nmin", type=int, default=15)
    ap.add_argument("--tmin", type=float, default=15.0)
    ap.add_argument("--tmax", type=float, default=25.0)
    ap.add_argument("--tstep", type=float, default=0.5)
    ap.add_argument("--fine_step", type=float, default=0.25)
    ap.add_argument("--delta_min", type=float, default=0.2)
    ap.add_argument("--c_retest", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--l2", type=float, default=0.003)
    ap.add_argument("--pen_align_full", type=float, default=3.0)
    ap.add_argument("--pen_align_pos", type=float, default=6.0)
    ap.add_argument("--pen_extra", type=float, default=0.8)
    ap.add_argument("--pen_sigma", type=float, default=0.006)
    ap.add_argument("--norm_by_sigma", type=int, choices=[0,1], default=1)
    ap.add_argument("--align_rounds", type=int, default=6)
    ap.add_argument("--align_growth", type=float, default=2.0)
    ap.add_argument("--align_target_mean", type=float, default=0.015)
    ap.add_argument("--align_target_max", type=float, default=0.05)
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--sens_threshY", type=str, default="0.035,0.040,0.045,0.050")
    ap.add_argument("--sens_y_noise_abs", type=str, default="0.000,0.002,0.004,0.006")
    ap.add_argument("--sens_reps", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    excel = choose_excel_default(args.excel)
    sheet = choose_sheet_default(excel, args.sheet)
    args.excel = excel
    args.sheet = sheet

    overrides = {"col_id":args.col_id,"col_week":args.col_week,"col_y":args.col_y,
                 "col_bmi":args.col_bmi,"col_height":args.col_height,"col_weight":args.col_weight,
                 "col_age":args.col_age,"col_z13":args.col_z13,"col_z18":args.col_z18,"col_z21":args.col_z21}

    df = read_base_df(excel, sheet, overrides)
    intervals = intervals_from_df(df, args.thresh, args.hweek)
    intervals = [s for s in intervals if np.isfinite(s.bmi)]
    print(f"[INFO] N={len(intervals)} | 对齐目标(mean_pos≤{args.align_target_mean}, max_pos≤{args.align_target_max})")

    aft = AFTLogLogistic(
        l2=args.l2, maxiter=1200, verbose=args.verbose,
        pen_align_full=args.pen_align_full, pen_align_pos=args.pen_align_pos,
        pen_extra=args.pen_extra, pen_sigma=args.pen_sigma,
        norm_by_sigma=bool(args.norm_by_sigma)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aft.fit(intervals, args.hweek,
                cont_rounds=args.align_rounds, cont_growth=args.align_growth,
                target_mean_pos=args.align_target_mean, target_max_pos=args.align_target_max)

    print(f"[INFO] 对齐结果：mean_pos={aft.align_mean_pos_:.4f}, max_pos={aft.align_max_pos_:.4f}")

    T_GRID = np.round(np.arange(args.tmin, args.tmax+1e-8, args.tstep), 5)
    FINE_GRID = np.round(np.arange(8.0, args.hweek, args.fine_step), 5)
    def w_scalar(t):
        if t <= 12.0: return 0.0
        elif t <= 27.0: return args.alpha * (t - 12.0) / 15.0
        else: return args.alpha + args.beta * (t - 27.0) / (args.hweek - 27.0)
    w_vec = np.vectorize(w_scalar); W_FINE = w_vec(FINE_GRID)

    order = np.argsort([s.bmi for s in intervals]); ints = [intervals[i] for i in order]
    F_sorted = np.vstack([aft.F(T_GRID, AFTLogLogistic._features(s)) for s in ints])

    def expected_w_after_t_local(Fhat, x, t):
        mask = FINE_GRID > t
        if not np.any(mask): return 0.0
        ts = FINE_GRID[mask]; Ft = Fhat(ts, x)
        p = np.diff(np.concatenate([[0.0], Ft])); p = np.maximum(p, 0.0); p = p / (p.sum() + 1e-12)
        return float(np.sum(p * W_FINE[mask]))

    def risk_for_subject_local(subj):
        x = AFTLogLogistic._features(subj)
        Ft = aft.F(T_GRID, x)
        not_ready = 1.0 - Ft
        ew = np.array([expected_w_after_t_local(aft.F, x, float(t)) for t in T_GRID])
        return Ft * w_vec(T_GRID) + not_ready * (args.c_retest + ew)

    R_sorted = np.vstack([risk_for_subject_local(s) for s in ints])
    png_elbow = os.path.join(args.outdir, "q3_k_elbow.png")
    plot_k_elbow(ints, R_sorted, F_sorted, T_GRID, args,
                save_path=png_elbow, k_min=2, k_max=8, dpi=args.dpi)
    print("[OK] 肘部图：", png_elbow)

    bounds, tids, _ = dp_segment_with_ready_size(R_sorted, F_sorted, T_GRID, args.k, args.tau_ready, args.nmin)
    if bounds is None: raise RuntimeError("DP 不可行，请检查阈值/网格/人数约束。")
    t_rec = [float(T_GRID[i]) for i in tids]
    t_adj = enforce_monotone_with_gap(t_rec, bounds, T_GRID, F_sorted, args.tau_ready, args.delta_min)

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
        groups.append({
            "组别": g, "BMI下界": round(bmin,2), "BMI上界": round(bmax,2),
            "建议检测周": float(t1), "原始最优周": float(t0),
            "人数": int(ngrp), "≥35人数": n35, "≥35占比": round(p35,3),
            "组内平均F": round(Fbar,3)
        })
        for ii in range(s_idx, e_idx+1):
            assign.append({"组别":g,"孕妇代码":ints[ii].pid,"BMI":round(ints[ii].bmi,2),
                           "年龄":float(ints[ii].age),"Z13":float(ints[ii].z13),
                           "Z18":float(ints[ii].z18),"Z21":float(ints[ii].z21),
                           "区间L":float(ints[ii].L),"区间R":float(ints[ii].R),
                           "是否右删":int(ints[ii].right_censored),"建议检测周":float(t1)})

    groups_df = pd.DataFrame(groups); assign_df = pd.DataFrame(assign)
    out_groups = os.path.join(args.outdir, "q3_groups_opt.csv")
    out_assign = os.path.join(args.outdir, "q3_subject_assignments_opt.csv")
    groups_df.to_csv(out_groups, index=False, encoding="utf-8-sig")
    assign_df.to_csv(out_assign, index=False, encoding="utf-8-sig")
    print("[OK] 导出：", out_groups, " | ", out_assign)

    try:
        xs, ys = [], []
        for _, r in groups_df.iterrows():
            xs += [r["BMI下界"], r["BMI上界"]]
            ys += [r["建议检测周"], r["建议检测周"]]
        fig, ax = plt.subplots(figsize=(10,5))
        ax.step(xs, ys, where="post")
        ax.set_xlabel("BMI 区间边界"); ax.set_ylabel("建议检测周")
        _mpl_clean(ax); fig.tight_layout()
        png = os.path.join(args.outdir, "q3_t_vs_bmi_step.png")
        fig.savefig(png, dpi=args.dpi); plt.close(fig)
        print("[OK] 阶梯图：", png)
    except Exception as e:
        print("[WARN] 绘图失败：", e)


    plot_t_vs_bmi_steps(groups_df, os.path.join(args.outdir, "q3_t_vs_bmi_steps_opt.png"), dpi=args.dpi)
    X_sorted = np.vstack([AFTLogLogistic._features(s) for s in ints])
    median_x_by_group = [np.median(X_sorted[s:e+1, :], axis=0) for (s, e) in bounds]
    t_fine = np.linspace(args.tmin, args.hweek-1e-6, 200)
    plot_group_curves(groups_df, median_x_by_group, aft, t_fine,
                      os.path.join(args.outdir, "q3_group_curves_opt.png"), dpi=args.dpi)
    plot_passrate_bars(groups_df, os.path.join(args.outdir, "q3_group_passrate_bars_opt.png"), dpi=args.dpi)

    group_ranges = [(float(row["BMI下界"]), float(row["BMI上界"])) for _, row in groups_df.iterrows()]
    overall_mean, group_means = compute_ready_curves(intervals, aft, T_GRID, group_ranges)
    pd.DataFrame({"t": T_GRID, "mean": overall_mean}).to_csv(
        os.path.join(args.outdir, "q3_ready_overall_mean.csv"), index=False, encoding="utf-8-sig")

    overall_mat, groups_arr = bootstrap_ready(intervals, T_GRID, group_ranges, B=max(1, args.boot), seed=args.seed, l2=args.l2, maxiter=800)
    lo_overall, hi_overall = ci_from_boot(overall_mat, level=args.ci, axis=0)
    lo_groups,  hi_groups  = ci_from_boot(groups_arr,  level=args.ci, axis=0)
    pd.DataFrame({"t": T_GRID, "mean": overall_mean, "ci_lo": lo_overall, "ci_hi": hi_overall}).to_csv(
        os.path.join(args.outdir, "q3_ready_overall_ci.csv"), index=False, encoding="utf-8-sig")
    rows = []
    for gi in range(len(group_ranges)):
        rows.append(pd.DataFrame({"t": T_GRID, "group": gi+1,
                                  "mean": group_means[gi,:], "ci_lo": lo_groups[gi,:], "ci_hi": hi_groups[gi,:]}))
    pd.concat(rows, ignore_index=True).to_csv(
        os.path.join(args.outdir, "q3_ready_groups_ci.csv"), index=False, encoding="utf-8-sig")
    # 图
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    plot_curve_ci(T_GRID, overall_mean, lo_overall, hi_overall, "整体达标比例", ax, color="#374E8D")
    ax.set_xlabel("孕周"); ax.set_ylabel("达标比例（Y≥阈值）")
    _mpl_clean(ax); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "q3_ready_overall_ci.png"), dpi=args.dpi); plt.close(fig)
    plot_groups_ci_combined(T_GRID, group_means, lo_groups, hi_groups,
                            os.path.join(args.outdir, "q3_ready_groups_ci.png"), dpi=args.dpi)


    t_tau_overall_hat = time_to_level_from_curve(T_GRID, overall_mean, args.tau_ready)
    t_tau_overall_bs = np.array([time_to_level_from_curve(T_GRID, overall_mat[b,:], args.tau_ready)
                                 for b in range(overall_mat.shape[0])])
    lo_t_overall, hi_t_overall = ci_from_boot(t_tau_overall_bs[~np.isnan(t_tau_overall_bs)], level=args.ci, axis=0)
    t_tau_groups_hat, lo_t_groups, hi_t_groups = [], [], []
    for gi in range(len(group_ranges)):
        t_hat = time_to_level_from_curve(T_GRID, group_means[gi,:], args.tau_ready)
        t_tau_groups_hat.append(t_hat)
        grp_bs = groups_arr[:, gi, :]
        t_bs = np.array([time_to_level_from_curve(T_GRID, grp_bs[b,:], args.tau_ready) for b in range(grp_bs.shape[0])])
        t_bs = t_bs[~np.isnan(t_bs)]
        if t_bs.size>0: lo_t, hi_t = ci_from_boot(t_bs, level=args.ci, axis=0)
        else: lo_t, hi_t = (np.nan, np.nan)
        lo_t_groups.append(lo_t); hi_t_groups.append(hi_t)
    pd.DataFrame({
        "对象": ["整体"] + [f"组{g+1}" for g in range(len(group_ranges))],
        "t_at_tau_hat": [t_tau_overall_hat] + t_tau_groups_hat,
        "ci_lo": [lo_t_overall] + lo_t_groups,
        "ci_hi": [hi_t_overall] + hi_t_groups,
        "tau": args.tau_ready, "ci_level": args.ci
    }).to_csv(os.path.join(args.outdir, "q3_time_to_tau_ci.csv"), index=False, encoding="utf-8-sig")

    def t_quantiles_for_x(model: AFTLogLogistic, x_vec, ps=(0.5,0.95)):
        b = float(x_vec[1])
        mu = float(np.dot(x_vec, model.beta_))
        sigma = float(np.exp(model.s_[0] + model.s_[1]*b))
        return [math.exp(mu + sigma*logit(p)) for p in ps]
    q_rows = []
    for gi, x_med in enumerate(median_x_by_group, start=1):
        t50, t95 = t_quantiles_for_x(aft, x_med, (0.5, 0.95))
        q_rows += [{"对象": f"组{gi}典型", "p": 0.50, "t_hat": t50},
                   {"对象": f"组{gi}典型", "p": 0.95, "t_hat": t95}]
    pd.DataFrame(q_rows).to_csv(os.path.join(args.outdir, "q3_group_quantiles_typical.csv"),
                                index=False, encoding="utf-8-sig")

    csv_thresh, csv_noise_raw, csv_noise_summ, png_lines = run_sensitivity(
        args, intervals, aft, overrides, dpi=args.dpi
    )
    print("[OK] 灵敏度输出：")
    print("   ", csv_thresh)
    print("   ", csv_noise_raw)
    print("   ", csv_noise_summ)
    print("   ", png_lines)

if __name__ == "__main__":
    main()
