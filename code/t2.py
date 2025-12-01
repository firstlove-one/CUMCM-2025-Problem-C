import os, re, math, argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from scipy.optimize import minimize

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei","SimHei","PingFang SC","Noto Sans CJK SC",
                        "WenQuanYi Micro Hei","Arial Unicode MS","DejaVu Sans"],
    "axes.unicode_minus": False, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "mathtext.fontset": "stix",
})

def _try_register_cn_fonts():
    for p in [r"C:\Windows\Fonts\msyh.ttc",
              r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\simsun.ttc"]:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
            except Exception:
                pass

def parse_week_str(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    m = re.match(r"^\s*(\d+)\s*[wW周]?\s*\+\s*(\d+)\s*$", s)
    if m:
        w = int(m.group(1)); d = int(m.group(2))
        return w + d/7.0
    m2 = re.match(r"^\s*(\d+)\s*[wW周]\s*$", s)
    if m2:
        return float(m2.group(1))
    try:
        return float(s)
    except:
        return np.nan

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
    if miss: raise ValueError(f"Excel缺少列：{miss}")

    height_m = df['身高'].astype(float) / 100.0
    weight = df['体重'].astype(float)
    df['BMI_calc'] = weight / (height_m**2)
    df['孕周_周'] = df['检测孕周'].apply(parse_week_str)
    df['Y达标'] = df['Y染色体浓度'].astype(float) >= float(thresh)

    intervals: List[SubjectInterval] = []
    for pid, g in df.groupby('孕妇代码'):
        gi = g.sort_values('孕周_周').dropna(subset=['孕周_周'])
        if gi.empty: continue
        bmi = float(gi.iloc[0]['BMI_calc'])
        reached = gi[gi['Y达标']]
        if not reached.empty:
            R = float(reached.iloc[0]['孕周_周'])
            before = gi[(gi['孕周_周'] < R) & (~gi['Y达标'])]
            L = 0.0 if before.empty else float(before['孕周_周'].max())
        else:
            L = float(gi['孕周_周'].max()); R = float(h_week)
        if not (np.isnan(L) or np.isnan(R) or L<0 or R<=0 or L>R):
            intervals.append(SubjectInterval(pid=str(pid), bmi=bmi, L=L, R=R))
    return intervals

class TimeToReadyModel:
    def __init__(self, intervals: List[SubjectInterval], h_week: float, l2=5e-4):
        self.h_week = h_week
        self.b = np.array([x.bmi for x in intervals], float)
        self.L = np.array([x.L for x in intervals], float)
        self.R = np.array([x.R for x in intervals], float)
        self.l2 = l2
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
            FR = safe_sigmoid((self.R[mask_int] - mu[mask_int]) / s[mask_int])
            diff = np.maximum(FR - FL[mask_int], 1e-9)
            ll_int = np.log(diff).sum()
        mask_rc = ~mask_int
        ll_rc = 0.0
        if np.any(mask_rc):
            ll_rc = np.log(np.maximum(1.0 - FL[mask_rc], 1e-9)).sum()
        penalty = self.l2 * float(np.sum(np.square(params)))
        return -(ll_int + ll_rc) + penalty

    def fit(self, maxiter=2000, verbose=False):
        init = np.array([12.0, 0.05, 0.0, math.log(2.0), 0.0], float)
        bounds = [(-10, 40), (-1, 1), (-0.1, 0.1), (math.log(0.2), math.log(10.0)), (-0.2, 0.2)]
        self.nll_history = []

        def _cb(xk):
            self.nll_history.append(self._nll(xk))

        res = minimize(lambda p: self._nll(p), init, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": maxiter}, callback=_cb)
        if verbose:
            print("Optimization success:", res.success, "| msg:", res.message, "| nll:", res.fun)
        self.params = res.x
        # 追加最终一次（有的 SciPy 版本最后一次不会触发 callback）
        if len(self.nll_history)==0 or abs(self.nll_history[-1] - res.fun) > 1e-10:
            self.nll_history.append(res.fun)
        self.opt_result = res
        return res

    def F_hat(self, t, b):
        if self.params is None: raise RuntimeError("Model not fitted.")
        t = np.asarray(t, float); b = np.asarray(b, float)
        m0, m1, m2, s0, s1 = self.params
        mu = m0 + m1*b + m2*(b**2)
        s  = np.exp(s0 + s1*b)
        return safe_sigmoid((t - mu)/s)

def build_w_time(alpha: float, beta: float, h_week: float):
    def w_time(t):
        if t <= 12.0: return 0.0
        elif t <= 27.0: return alpha * (t - 12.0) / 15.0
        else: return alpha + beta * (t - 27.0) / (h_week - 27.0)
    return np.vectorize(w_time), w_time

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
            seg_sum_R = prefix_R[j+1]-prefix_R[i]
            if tau_ready > 0:
                meanF = (prefix_F[j+1]-prefix_F[i])/(j-i+1)
                feas = (meanF >= tau_ready)
                if not np.any(feas): continue
                seg_sum_R = np.where(feas, seg_sum_R, INF)
            t_idx = int(np.argmin(seg_sum_R))
            if not np.isfinite(seg_sum_R[t_idx]): continue
            CostStar[i,j] = float(seg_sum_R[t_idx]); TArg[i,j] = t_idx
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
        boundaries.append((s,e)); t_idx.append(TArg[s,e])
        j = i; g -= 1
    boundaries.reverse(); t_idx.reverse()
    # 单调化
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
            seg_sum_R = prefix_R[j+1]-prefix_R[i]
            t_idx = int(np.argmin(seg_sum_R))
            CostStar[i,j] = float(seg_sum_R[t_idx]); TArg[i,j] = t_idx
    dp = np.full((k_groups,n), np.inf); prev = np.full((k_groups,n), -1)
    for j in range(n): dp[0,j] = CostStar[0,j]
    for g in range(1,k_groups):
        for j in range(g,n):
            best=np.inf; bi=-1
            for i in range(g-1,j):
                val = dp[g-1,i] + CostStar[i+1,j]
                if val<best: best,bi = val,i
            dp[g,j]=best; prev[g,j]=bi
    boundaries, t_idx = [], []
    g, j = k_groups-1, n-1
    while g>=0:
        i=prev[g,j]; s=0 if i==-1 else i+1; e=j
        boundaries.append((s,e)); t_idx.append(TArg[s,e]); j=i; g-=1
    boundaries.reverse(); t_idx.reverse()
    t_vals = [float(t_grid[idx]) for idx in t_idx]
    for k in range(1,len(t_vals)):
        if t_vals[k]<t_vals[k-1]:
            t_vals[k]=t_vals[k-1]; t_idx[k]=int(np.argmin(np.abs(t_grid - t_vals[k])))
    return boundaries, t_idx

def setup_sns_no_grid():
    _try_register_cn_fonts()
    sns.set_theme(style="white", rc={
        "font.sans-serif": ["Microsoft YaHei","SimHei","PingFang SC","Noto Sans CJK SC",
                            "WenQuanYi Micro Hei","Arial Unicode MS","DejaVu Sans"],
        "axes.unicode_minus": False
    })

def plot_ready_curves(groups_df: pd.DataFrame, F_hat, fine_grid: np.ndarray, save_path: str):
    setup_sns_no_grid()
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.grid(False); ax.minorticks_off()
    for _, row in groups_df.iterrows():
        b_lo, b_hi = float(row["BMI下界"]), float(row["BMI上界"])
        t_rec = float(row["建议检测周"])
        Ft = F_hat(fine_grid, (b_lo+b_hi)/2.0)
        label = f'组{int(row["组别"])}：[{b_lo:.2f}, {b_hi:.2f}]'
        sns.lineplot(x=fine_grid, y=Ft, ax=ax, label=label)
        ax.axvline(t_rec, linestyle="--", linewidth=1.2)
    ax.set_xlabel("孕周（周）"); ax.set_ylabel("就绪比例（达到4%）")
    ax.set_ylim(0.0,1.02); ax.legend(title=None, frameon=False)
    sns.despine(ax=ax); fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)

def plot_fit_convergence(nll_hist: List[float], save_path: str):
    setup_sns_no_grid()
    it = np.arange(1, len(nll_hist)+1)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    sns.lineplot(x=it, y=nll_hist, marker="o", ax=ax)
    ax.set_xlabel("迭代次数"); ax.set_ylabel("负对数似然 NLL")
    sns.despine(ax=ax); fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)

def plot_group_risk_curves(boundaries, t_idx, Rmat_sorted: np.ndarray, t_grid: np.ndarray, save_path: str):
    setup_sns_no_grid()
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for g,(s,e),ti in zip(range(1,len(boundaries)+1), boundaries, t_idx):
        mean_risk = Rmat_sorted[s:e+1,:].mean(axis=0)
        label = f'组{g}：[{s+1}~{e+1}]人，均值'
        sns.lineplot(x=t_grid, y=mean_risk, ax=ax, label=label)
        ax.axvline(float(t_grid[ti]), linestyle="--", linewidth=1.2)
    ax.set_xlabel("孕周（周）"); ax.set_ylabel("人均风险")
    sns.despine(ax=ax); fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)

def plot_k_scan(k_list, mean_risks, k_chosen, save_path):
    setup_sns_no_grid()
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    sns.lineplot(x=k_list, y=mean_risks, marker="o", ax=ax)
    ax.set_xlabel("分组数 k"); ax.set_ylabel("人均风险")
    # 标注选择的 k
    if k_chosen in k_list:
        idx = k_list.index(k_chosen)
        ax.scatter([k_chosen],[mean_risks[idx]], s=80)
        ax.annotate(f"选择k={k_chosen}", (k_chosen, mean_risks[idx]),
                    xytext=(k_chosen+0.2, mean_risks[idx]+0.0008),
                    arrowprops=dict(arrowstyle="->", lw=1.0))
    sns.despine(ax=ax); fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)

def plot_formula10_curves(model, bmi_vals, t_range, h_week, save_path):
    setup_sns_no_grid()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    tmin, tmax = t_range
    tmin = max(8.0, float(tmin))
    tmax = min(float(h_week), float(tmax))
    tt = np.linspace(tmin, tmax, 400)


    if model.params is None:
        raise RuntimeError("model 尚未拟合，无法作图。")
    m0, m1, m2, s0, s1 = model.params
    def mu(b): return m0 + m1*b + m2*(b**2)
    def s_(b): return math.exp(s0 + s1*b)

    for b in bmi_vals:
        Ft = model.F_hat(tt, b)
        sns.lineplot(x=tt, y=Ft, ax=ax, label=f"BMI={b:.1f}")
        ax.axvline(mu(b), linestyle="--", linewidth=1.0)

    eq = (r"$F(t\,|\,b)=\sigma\!\left(\frac{t-\mu(b)}{s(b)}\right),\ "
          r"\mu(b)=m_0+m_1b+m_2b^2,\ s(b)=e^{\,s_0+s_1b}$")
    ax.text(0.02, 1.02, eq, transform=ax.transAxes, ha="left", va="bottom", fontsize=11)

    ax.set_xlabel("孕周 t（周）")
    ax.set_ylabel(r"$F(t\,|\,b)$（达到4%就绪的比例）")
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(0.0, 1.02)
    ax.legend(title=None, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_risk_definition_panel(
    groups_df: pd.DataFrame,
    boundaries, t_idx,            
    b_sorted: np.ndarray,           
    T_GRID: np.ndarray,             
    FINE_GRID: np.ndarray, W_FINE: np.ndarray,
    w_time_vec, w_time_scalar,    
    F_for_risk,                    
    c_retest: float,
    h_week: float,
    save_path: str
):

    setup_sns_no_grid()

    fig = plt.figure(figsize=(11.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 2.2], wspace=0.22)
    ax_w = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    t_demo = np.linspace(8.0, h_week, 400)
    w_demo = w_time_vec(t_demo)
    sns.lineplot(x=t_demo, y=w_demo, ax=ax_w)
    ax_w.set_xlabel("孕周（周）"); ax_w.set_ylabel(r"$w(t)$")
    ax_w.set_xlim(8, h_week)
    ax_w.set_ylim(0, max(1.05*np.max(w_demo), 0.1))
    ax_w.grid(False); ax_w.minorticks_off()
    sns.despine(ax=ax_w)

    piece_lines_math = [
        r"$w(t)=0,\quad t\leq 12$",
        r"$w(t)=\alpha\cdot\frac{t-12}{27-12},\quad 12<t\leq 27$",
        r"$w(t)=\alpha+\beta\cdot\frac{t-27}{H-27},\quad t>27$",
    ]
    try:
        ax_w.text(0.02, 0.96, "\n".join(piece_lines_math),
                  transform=ax_w.transAxes, va="top", ha="left",
                  fontsize=11, linespacing=1.4)
    except Exception:
        ax_w.text(0.02, 0.96,
                  "w(t)=0, t≤12\n"
                  "w(t)=α·(t−12)/(27−12), 12<t≤27\n"
                  "w(t)=α+β·(t−27)/(H−27), t>27",
                  transform=ax_w.transAxes, va="top", ha="left",
                  fontsize=11, linespacing=1.4)

    risk_eq_math = (
        r"$R(t|b)=\hat{F}(t|b)\,w(t)"
        r"+(1-\hat{F}(t|b))\,(C_{retest}+E[w(T^*)\,|\,T^*>t,b])$"
    )
    try:
        ax_r.text(0.01, 1.02, risk_eq_math, transform=ax_r.transAxes,
                  va="bottom", ha="left", fontsize=11)
    except Exception:
        ax_r.text(0.01, 1.02,
                  "R(t|b)=F(t|b)·w(t) + (1−F(t|b))·(C_retest + E[w(T*) | T*>t, b])",
                  transform=ax_r.transAxes, va="bottom", ha="left", fontsize=11)

    for g, ((s_idx, e_idx), tidx) in enumerate(zip(boundaries, t_idx), start=1):
        b_lo = float(b_sorted[s_idx]); b_hi = float(b_sorted[e_idx])
        b_med = float(np.median(b_sorted[s_idx:e_idx+1]))
        Rt = risk_at_times_for_b(
            b_med, T_GRID, F_for_risk, c_retest,
            FINE_GRID, W_FINE, w_time_vec, w_time_scalar
        )
        label = f"组{g}：BMI区间 [{b_lo:.2f}, {b_hi:.2f}]"
        sns.lineplot(x=T_GRID, y=Rt, ax=ax_r, label=label)
        ax_r.axvline(float(T_GRID[int(tidx)]), linestyle="--", linewidth=1.2)

    ax_r.set_xlabel("孕周（周）"); ax_r.set_ylabel(r"风险 $R(t|b)$")
    ax_r.grid(False); ax_r.minorticks_off()
    ax_r.legend(title=None, frameon=False, fontsize=9)
    sns.despine(ax=ax_r)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel",   type=str, default="附件.xlsx")
    ap.add_argument("--sheet",   type=str, default="男胎检测数据")
    ap.add_argument("--thresh",  type=float, default=0.04)
    ap.add_argument("--hweek",   type=float, default=30.0)
    ap.add_argument("--k",       type=int,   default=5)
    ap.add_argument("--tau_ready", type=float, default=0.95)
    ap.add_argument("--c_retest", type=float, default=1.0)
    ap.add_argument("--alpha",   type=float, default=1.0)
    ap.add_argument("--beta",    type=float, default=8.0)
    ap.add_argument("--tmin",    type=float, default=10.0)
    ap.add_argument("--tmax",    type=float, default=25.0)
    ap.add_argument("--tstep",   type=float, default=0.5)
    ap.add_argument("--fine_step", type=float, default=0.25)
    ap.add_argument("--kmin",    type=int, default=2, help="k 扫描下界")
    ap.add_argument("--kmax",    type=int, default=8, help="k 扫描上界")
    ap.add_argument("--outdir",  type=str, default=".")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)


    intervals = load_male_intervals(args.excel, args.sheet, args.thresh, args.hweek)
    if len(intervals)==0: raise RuntimeError("未构造到有效的男胎样本数据。")
    print(f"[INFO] 有效男胎样本数（按孕妇）：{len(intervals)}")

    model = TimeToReadyModel(intervals, h_week=args.hweek, l2=5e-4)
    res = model.fit(maxiter=2000, verbose=args.verbose)
    if args.verbose:
        print("params:", model.params)
    all_bmi = np.array([s.bmi for s in intervals], float)
    bmi_vals = np.round(np.quantile(all_bmi, [0.1, 0.3, 0.5, 0.7, 0.9]), 1)
    f10_png = os.path.join(args.outdir, "q2_formula10_F_t_given_b.png")
    plot_formula10_curves(model, bmi_vals, (args.tmin, args.tmax), args.hweek, f10_png)
    print("   ", f10_png)

    nll_csv = os.path.join(args.outdir, "q2_fit_nll_history.csv")
    pd.DataFrame({"iter": np.arange(1,len(model.nll_history)+1), "nll": model.nll_history}).to_csv(nll_csv, index=False, encoding="utf-8-sig")
    nll_png = os.path.join(args.outdir, "q2_fit_convergence.png")
    plot_fit_convergence(model.nll_history, nll_png)
    print("   ", nll_csv); print("   ", nll_png)

    T_GRID = np.round(np.arange(args.tmin, args.tmax + args.tstep/2, args.tstep), 5)
    FINE_GRID = np.round(np.arange(8.0, args.hweek, args.fine_step), 5)
    w_vec, w_scalar = build_w_time(args.alpha, args.beta, args.hweek)
    W_FINE = w_vec(FINE_GRID)

    b_list = np.array([s.bmi for s in intervals], float)
    pid_list = [s.pid for s in intervals]
    Rmat = np.zeros((len(intervals), len(T_GRID)), float)
    for i, s in enumerate(intervals):
        Rmat[i,:] = risk_at_times_for_b(s.bmi, T_GRID, model.F_hat, args.c_retest, FINE_GRID, W_FINE, w_vec, w_scalar)

    order = np.argsort(b_list)
    b_sorted = b_list[order]
    pid_sorted = [pid_list[i] for i in order]
    Rmat_sorted = Rmat[order,:]
    F_all_sorted = np.vstack([model.F_hat(T_GRID, b) for b in b_sorted])

    boundaries, t_idx = dp_segment_with_ready_constraint(Rmat_sorted, F_all_sorted, T_GRID, args.k, args.tau_ready)
    constrained_ok = (boundaries is not None)
    if not constrained_ok:
        print("[WARN] 带约束方案不可行，改用无约束。")
        boundaries, t_idx = dp_segment_unconstrained(Rmat_sorted, T_GRID, args.k)

    groups, assign = [], []
    for g, ((s_idx,e_idx), tidx) in enumerate(zip(boundaries, t_idx), start=1):
        bmin = float(b_sorted[s_idx]); bmax = float(b_sorted[e_idx]); t_rec = float(T_GRID[tidx])
        groups.append({"组别": g, "BMI下界": round(bmin,2), "BMI上界": round(bmax,2), "建议检测周": t_rec})
        for ii in range(s_idx, e_idx+1):
            assign.append({"组别": g, "孕妇代码": pid_sorted[ii], "BMI": round(float(b_sorted[ii]),2), "建议检测周": t_rec})
    groups_df = pd.DataFrame(groups); assign_df = pd.DataFrame(assign)
    g_csv = os.path.join(args.outdir, "q2_groups_constrained.csv" if constrained_ok else "q2_groups.csv")
    a_csv = os.path.join(args.outdir, "q2_subject_assignments_constrained.csv" if constrained_ok else "q2_subject_assignments.csv")
    groups_df.to_csv(g_csv, index=False, encoding="utf-8-sig")
    assign_df.to_csv(a_csv, index=False, encoding="utf-8-sig")
    print("[OK] 分组与分配已导出："); print("   ", g_csv); print("   ", a_csv)

    ready_png = os.path.join(args.outdir, "q2_summary_constrained.png" if constrained_ok else "q2_summary.png")
    plot_ready_curves(groups_df, model.F_hat, FINE_GRID, ready_png)
    print("   ", ready_png)

    risk_png = os.path.join(args.outdir, "q2_group_risk_curves.png")
    plot_group_risk_curves(boundaries, t_idx, Rmat_sorted, T_GRID, risk_png)
    print("   ", risk_png)

    risk_cols = {}
    for g,(s,e),ti in zip(range(1,len(boundaries)+1), boundaries, t_idx):
        mean_risk = Rmat_sorted[s:e+1,:].mean(axis=0)
        label = f"组{g}_[{b_sorted[s]:.2f},{b_sorted[e]:.2f}]"
        risk_cols[label] = mean_risk
        groups_df.loc[g-1, "推荐时点索引"] = int(ti)
    risk_df = pd.DataFrame({"孕周": T_GRID, **risk_cols})
    risk_csv = os.path.join(args.outdir, "q2_risk_summary.csv")
    risk_df.to_csv(risk_csv, index=False, encoding="utf-8-sig")
    print("   ", risk_csv)

    k_list = list(range(max(1,args.kmin), max(args.kmin,args.kmax)+1))
    mean_risks = []
    for kk in k_list:
        bnd_kk, tidx_kk = dp_segment_with_ready_constraint(Rmat_sorted, F_all_sorted, T_GRID, kk, args.tau_ready)
        if bnd_kk is None:  
            bnd_kk, tidx_kk = dp_segment_unconstrained(Rmat_sorted, T_GRID, kk)
        total = 0.0
        for (s,e),ti in zip(bnd_kk, tidx_kk):
            total += float(Rmat_sorted[s:e+1, ti].sum())
        mean_risks.append(total / Rmat_sorted.shape[0])
    k_csv = os.path.join(args.outdir, "q2_k_scan.csv")
    pd.DataFrame({"k": k_list, "mean_risk": mean_risks}).to_csv(k_csv, index=False, encoding="utf-8-sig")
    k_png = os.path.join(args.outdir, "q2_k_scan.png")
    plot_k_scan(k_list, mean_risks, args.k, k_png)
    print("   ", k_csv); print("   ", k_png)

    panel_png = os.path.join(args.outdir, "q2_risk_definition_panel.png")
    plot_risk_definition_panel(
        groups_df=groups_df,
        boundaries=boundaries, t_idx=t_idx,
        b_sorted=b_sorted,
        T_GRID=T_GRID,
        FINE_GRID=FINE_GRID, W_FINE=W_FINE,
        w_time_vec=w_vec, w_time_scalar=w_scalar,
        F_for_risk=model.F_hat,
        c_retest=args.c_retest,
        h_week=args.hweek,
        save_path=panel_png
    )
    print("   ", panel_png)

if __name__ == "__main__":
    main()
