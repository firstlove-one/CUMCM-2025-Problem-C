COLS = {
    "ID": ["孕妇代码","样本编号","ID","id"],
    "GA": ["检测孕周","孕周","GA"],
    "BMI": ["孕妇BMI","BMI"],
    "reads": ["原始读段数","总读段数","reads"],
    "map_rate": ["在参考基因组上比对的比例"],
    "dup_rate": ["重复读段的比例"],
    "filter_rate": ["被过滤掉读段数的比例"],
    "unique_reads": ["唯一比对的读段数  ","唯一比对的读段数"],
    "GC_all": ["GC含量","GC整体","GC%"],
    "Z_13": ["13号染色体的Z值","Z13","Z_13"],
    "Z_18": ["18号染色体的Z值","Z18","Z_18"],
    "Z_21": ["21号染色体的Z值","Z21","Z_21"],
    "Z_X": ["X染色体的Z值","Z_X"],
    "X_frac": ["X染色体浓度"],
    "GC_13": ["13号染色体的GC含量"],
    "GC_18": ["18号染色体的GC含量"],
    "GC_21": ["21号染色体的GC含量"],
    "AB": ["染色体的非整倍体"],
    "HEALTH": ["胎儿是否健康"],
}

def pick_col(df: pd.DataFrame, cand: List[str]) -> Optional[str]:
    s = set(df.columns)
    for k in cand:
        if k in s: return k
    norm = {c.replace(" ", ""): c for c in df.columns}
    for k in cand:
        kk = k.replace(" ", "")
        if kk in norm: return norm[kk]
    return None

def map_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {k: pick_col(df, v) for k, v in COLS.items()}

GA_PATTERNS = [
    re.compile(r"^\s*(\d+)\s*[wW周]\s*\+?\s*(\d+)\s*[dD天]?\s*$"),
    re.compile(r"^\s*(\d+)\s*\+\s*(\d+)\s*$"),
    re.compile(r"^\s*(\d+)\s*[wW周]\s*$"),
    re.compile(r"^\s*(\d+)\s*[dD天]\s*$"),
]
def parse_ga(x) -> float:
    if x is None: return np.nan
    if isinstance(x, (int, float)): return float(x) if x<=40 else float(x)/7.0
    s = str(x).strip().replace("＋","+").replace("：",":").replace("．",".")
    for pat in GA_PATTERNS:
        m = pat.match(s)
        if m:
            if pat in (GA_PATTERNS[0], GA_PATTERNS[1]): return float(m.group(1)) + float(m.group(2))/7.0
            if pat is GA_PATTERNS[2]: return float(m.group(1))
            if pat is GA_PATTERNS[3]: return float(m.group(1))/7.0
    try: return float(s)
    except: return np.nan

def to_num(x) -> float:
    if x is None or (isinstance(x,float) and np.isnan(x)): return np.nan
    if isinstance(x, (int,float)): return float(x)
    s = str(x).strip().replace(",","").replace("％","%")
    if s.endswith("%"):
        try: return float(s[:-1])/100.0
        except: return np.nan
    try: return float(s)
    except: return np.nan

def autoscale_fraction_columns(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.dropna().size == 0: continue
            q90 = np.quantile(x.dropna(), 0.90)
            if q90 > 1.5 and q90 <= 100.0:
                df[c] = x / 100.0

def clean_numeric_columns(df: pd.DataFrame, cm: Dict[str,str]) -> pd.DataFrame:
    numeric = ["BMI","reads","map_rate","dup_rate","filter_rate","unique_reads",
               "GC_all","Z_13","Z_18","Z_21","Z_X","X_frac","GC_13","GC_18","GC_21"]
    if cm.get("GA") and cm["GA"] in df.columns:
        df[cm["GA"]] = df[cm["GA"]].apply(parse_ga)
    for k in numeric:
        col = cm.get(k)
        if col and col in df.columns:
            df[col] = df[col].apply(to_num)
    autoscale_fraction_columns(df, [cm[x] for x in ["map_rate","dup_rate","filter_rate","GC_all","GC_13","GC_18","GC_21","X_frac"] if cm.get(x)])
    return df

@dataclass
class QCThresholds:
    gc_low: float; gc_high: float
    reads_min: float; map_min: float
    dup_max: float; filter_max: float
    uniq_min: Optional[float] = None

def robust_median_mad(x: np.ndarray) -> Tuple[float,float]:
    x = x[np.isfinite(x)]
    if x.size==0: return np.nan, np.nan
    med = np.median(x)
    mad = np.median(np.abs(x-med))*1.4826
    if mad<=1e-12:
        s = np.std(x); mad = s if s>1e-9 else 1.0
    return med, mad

def estimate_qc_thresholds(male_neg: pd.DataFrame, cm: Dict[str,str]) -> QCThresholds:
    def q(v,a,b):
        if v is None or v not in male_neg.columns: return (None,None,None)
        arr = male_neg[v].astype(float); arr = arr[np.isfinite(arr)]
        if arr.size==0: return (None,None,None)
        return arr.min(), np.quantile(arr,a), np.quantile(arr,b)
    _,gc1,gc99 = q(cm["GC_all"],0.01,0.99)
    gc_low  = float(gc1-0.002) if gc1 is not None else 0.38
    gc_high = float(gc99+0.002) if gc99 is not None else 0.42
    _,r10,_ = q(cm["reads"],0.10,0.90); reads_min = float(max(3e6, r10 if r10 is not None else 0))
    _,m1,_  = q(cm["map_rate"],0.01,0.99); map_min  = float(m1 if m1 is not None else 0.75)
    _,_,d99 = q(cm["dup_rate"],0.01,0.99); dup_max  = float(d99 if d99 is not None else 0.06)
    _,_,f99 = q(cm["filter_rate"],0.01,0.99); filter_max = float(f99 if f99 is not None else 0.05)
    uniq_min=None
    if cm["unique_reads"] and cm["reads"] and cm["unique_reads"] in male_neg.columns and cm["reads"] in male_neg.columns:
        ur = (male_neg[cm["unique_reads"]]/male_neg[cm["reads"]]).replace([np.inf,-np.inf],np.nan).clip(0,1)
        if ur.dropna().size>0: uniq_min = float(np.quantile(ur.dropna(),0.01))
    return QCThresholds(gc_low,gc_high,reads_min,map_min,dup_max,filter_max,uniq_min)

def qc0_status(row: pd.Series, cm: Dict[str,str], th: QCThresholds) -> str:
    bad=0
    if cm["GC_all"] and pd.notna(row.get(cm["GC_all"],np.nan)):
        if not (th.gc_low <= row[cm["GC_all"]] <= th.gc_high): bad+=1
    if cm["reads"] and pd.notna(row.get(cm["reads"],np.nan)):
        if row[cm["reads"]] < th.reads_min: bad+=1
    if cm["map_rate"] and pd.notna(row.get(cm["map_rate"],np.nan)):
        if row[cm["map_rate"]] < th.map_min: bad+=1
    if cm["dup_rate"] and pd.notna(row.get(cm["dup_rate"],np.nan)):
        if row[cm["dup_rate"]] > th.dup_max: bad+=1
    if cm["filter_rate"] and pd.notna(row.get(cm["filter_rate"],np.nan)):
        if row[cm["filter_rate"]] > th.filter_max: bad+=1
    if cm["unique_reads"] and cm["reads"]:
        denom=row.get(cm["reads"],np.nan); numer=row.get(cm["unique_reads"],np.nan)
        if np.isfinite(denom) and denom>0 and np.isfinite(numer):
            ur=float(numer)/float(denom)
            if th.uniq_min is not None and ur < th.uniq_min: bad+=1
    if bad==0: return "valid"
    if bad<=2: return "suspicious"
    return "invalid"

@dataclass
class XGateParams:
    k_sigma: float = 3.0

def detrend(y: pd.Series, X: Optional[pd.DataFrame]) -> pd.Series:
    if X is None or X.shape[1]==0: return y.astype(float).copy()
    Xf = X.copy()
    for c in Xf.columns:
        if Xf[c].isna().any(): Xf[c] = Xf[c].median()
    yy = y.values.astype(float)
    try:
        m = HuberRegressor(); m.fit(Xf.values, yy)
        return pd.Series(yy - m.predict(Xf.values), index=y.index)
    except Exception:
        return y.copy()

def fit_x_gate(male_neg: pd.DataFrame, cm: Dict[str,str]) -> Tuple[float,float]:
    zx = cm["Z_X"]
    if not zx or zx not in male_neg.columns: return np.nan, np.nan
    covars=[c for c in [cm["GC_all"],cm["reads"],cm["map_rate"],cm["dup_rate"],cm["filter_rate"],cm["BMI"],cm["GA"]] if c and c in male_neg.columns]
    resid = detrend(male_neg[zx].astype(float), male_neg[covars] if covars else None)
    core = resid.copy()
    if core.dropna().size>0:
        hi=np.quantile(core.dropna(),0.995); core=core[core<=hi]
    mu=float(np.nanmean(core)); sigma=float(np.nanstd(core,ddof=1))
    if not np.isfinite(sigma) or sigma<1e-9:
        _,mad=robust_median_mad(core.dropna().values); sigma=float(mad if np.isfinite(mad) else 1.0)
    return mu,sigma

def x_gate_status(row: pd.Series, cm: Dict[str,str], mu: float, sigma: float, xp: XGateParams) -> str:
    zx=cm["Z_X"]
    if not zx or zx not in row or not np.isfinite(mu) or not np.isfinite(sigma): return "unknown"
    val=row[zx]
    if not np.isfinite(val): return "unknown"
    return "pass" if abs(val-mu) <= xp.k_sigma*sigma else "fail"

def add_residual_Z(df: pd.DataFrame, ref_neg: pd.DataFrame, cm: Dict[str,str]) -> pd.DataFrame:
    covars=[c for c in [cm["GC_all"],cm["reads"],cm["map_rate"],cm["dup_rate"],cm["filter_rate"],cm["BMI"],cm["GA"]] if c and c in ref_neg.columns]
    for zk in ["Z_13","Z_18","Z_21"]:
        zc=cm[zk]
        if not zc or zc not in df.columns: continue
        if covars:
            try:
                model=HuberRegressor()
                Xref=ref_neg[covars].copy().fillna(ref_neg[covars].median())
                zref=ref_neg[zc].astype(float).values
                model.fit(Xref.values, zref)
                Xdf=df[covars].copy().fillna(ref_neg[covars].median())
                pred=model.predict(Xdf.values)
                df[f"{zc}_resid"]=df[zc].astype(float).values - pred
            except Exception:
                df[f"{zc}_resid"]=df[zc].astype(float).values
        else:
            df[f"{zc}_resid"]=df[zc].astype(float).values
    return df

def mahalanobis_tail_prob(X_ref: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    try:
        mcd=MinCovDet().fit(X_ref); d2=mcd.mahalanobis(X_test)
    except Exception:
        mu=np.nanmean(X_ref,axis=0,keepdims=True); diff=(X_test-mu); d2=np.sum(diff*diff,axis=1)
    k=X_ref.shape[1]; return 1.0-chi2.cdf(d2,df=k)

def robust_feature_tail(X_ref_df: pd.DataFrame, X_test_df: pd.DataFrame) -> np.ndarray:
    ps=[]
    for c in X_ref_df.columns:
        r=X_ref_df[c].values.astype(float); t=X_test_df[c].values.astype(float)
        r=r[np.isfinite(r)]
        if r.size==0: ps.append(np.ones_like(t)); continue
        med,mad=robust_median_mad(r); denom=mad if mad>1e-12 else (np.std(r)+1e-9)
        z=np.abs((t-med)/denom); p=2.0*(1.0-norm.cdf(z))
        ps.append(np.clip(p,1e-12,1.0))
    ps=np.array(ps); return 1.0-np.prod(1.0-ps,axis=0)

def weak_discriminator_scores(X0: np.ndarray, X1: np.ndarray, Xf: np.ndarray):
    y=np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])
    X=np.vstack([X0,X1])
    clf=LogisticRegression(max_iter=1000,class_weight="balanced",solver="lbfgs")
    clf.fit(X,y)
    s0=clf.predict_proba(X0)[:,1]; s1=clf.predict_proba(X1)[:,1]; sf=clf.predict_proba(Xf)[:,1]
    auc=roc_auc_score(y,np.concatenate([s0,s1])) if len(np.unique(y))>1 else np.nan
    ap =average_precision_score(y,np.concatenate([s0,s1])) if len(np.unique(y))>1 else np.nan
    return s0,s1,sf,float(auc if np.isfinite(auc) else np.nan),float(ap if np.isfinite(ap) else np.nan)

def top3_outliers_per_row(X_ref_df: pd.DataFrame, X_test_df: pd.DataFrame) -> List[str]:
    meds={}; mads={}
    for c in X_ref_df.columns:
        r=X_ref_df[c].values.astype(float); r=r[np.isfinite(r)]
        med,mad=robust_median_mad(r)
        if not np.isfinite(mad) or mad<1e-12: mad=np.std(r)+1e-9
        meds[c]=med; mads[c]=mad
    outs=[]
    for _,row in X_test_df.iterrows():
        zs={}
        for c in X_test_df.columns:
            v=row[c]
            if not np.isfinite(v): continue
            z=(v-meds[c])/(mads[c] if mads[c]>0 else 1.0)
            zs[c]=z
        if not zs: outs.append(""); continue
        top=sorted(zs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        outs.append("; ".join([f"{name}:{'+' if z>=0 else ''}{z:.2f}" for name,z in top]))
    return outs

def quantile_bins(series: pd.Series, q=[0,0.25,0.5,0.75,1.0], prefix="Q"):
    s=pd.to_numeric(series,errors="coerce")
    if s.dropna().size==0:
        return pd.Series(["UNK"]*len(series), index=series.index), None
    qs=np.quantile(s.dropna(), q)
    for i in range(1,len(qs)):
        if qs[i]<=qs[i-1]: qs[i]=qs[i-1]+1e-6
    bins=pd.cut(s,bins=qs,include_lowest=True,labels=[f"{prefix}{i}" for i in range(1,len(q))])
    return bins.astype(str).fillna("UNK"), qs

def apply_bins_with_edges(series: pd.Series, edges, prefix):
    s = pd.to_numeric(series, errors="coerce")
    if edges is None or len(edges)<2 or s.dropna().size==0:
        return pd.Series(["UNK"]*len(series), index=series.index)
    ed = np.array(edges).copy()
    for i in range(1,len(ed)):
        if ed[i]<=ed[i-1]: ed[i]=ed[i-1]+1e-6
    bins = pd.cut(s, bins=ed, include_lowest=True,
                  labels=[f"{prefix}{i}" for i in range(1,len(ed))])
    return bins.astype(str).fillna("UNK")

def compute_thresholds_by_mode(prob_neg_oof: np.ndarray,
                               bins_ga_neg: pd.Series, bins_bmi_neg: pd.Series,
                               fpr_mode: str, fpr_target: float, fpr_margin: float,
                               min_bin_n: int, shrink_lambda: float):
    res = {"mode": fpr_mode, "global": None, "ga": {}, "bmi": {}, "grid": {},
           "counts": {"ga":{}, "bmi":{}, "grid":{}}}
    q_level = 1.0 - min(max(fpr_target + max(fpr_margin, 0.0), 1e-6), 0.9999)
    tau_global = float(np.quantile(prob_neg_oof, q_level))
    res["global"] = tau_global
    def shrink_tau(arr, fallback):
        n = int(len(arr))
        if n == 0:
            return float(fallback), n
        tau_raw = float(np.quantile(arr, q_level))
        w = n / (n + shrink_lambda)
        tau = w * tau_raw + (1.0 - w) * float(fallback)
        if n < min_bin_n:
            tau = max(tau, float(fallback))
        return float(tau), n
    if fpr_mode in ["ga","ga_bmi"]:
        for g in sorted(pd.unique(bins_ga_neg.astype(str))):
            mask = (bins_ga_neg.astype(str)==g).values
            tau, n = shrink_tau(prob_neg_oof[mask], tau_global)
            res["ga"][g] = tau; res["counts"]["ga"][g] = n
    if fpr_mode in ["bmi","ga_bmi"]:
        for b in sorted(pd.unique(bins_bmi_neg.astype(str))):
            mask = (bins_bmi_neg.astype(str)==b).values
            tau, n = shrink_tau(prob_neg_oof[mask], tau_global)
            res["bmi"][b] = tau; res["counts"]["bmi"][b] = n
    if fpr_mode=="ga_bmi":
        for g in sorted(pd.unique(bins_ga_neg.astype(str))):
            for b in sorted(pd.unique(bins_bmi_neg.astype(str))):
                mask = ((bins_ga_neg.astype(str)==g) & (bins_bmi_neg.astype(str)==b)).values
                tau_gb, n = shrink_tau(prob_neg_oof[mask], max(res["ga"].get(g, tau_global),
                                                               res["bmi"].get(b, tau_global),
                                                               tau_global))
                tau_gb = float(max(tau_gb, res["ga"].get(g, tau_global), res["bmi"].get(b, tau_global), tau_global))
                res["grid"][(g,b)] = tau_gb; res["counts"]["grid"][(g,b)] = n
    return res

def select_threshold_for_row(thres_dict, fpr_mode, gbin:str, bbin:str):
    if fpr_mode=="global":
        return thres_dict["global"]
    if fpr_mode=="ga":
        return thres_dict["ga"].get(gbin, thres_dict["global"])
    if fpr_mode=="bmi":
        return thres_dict["bmi"].get(bbin, thres_dict["global"])
    if fpr_mode=="ga_bmi":
        return thres_dict["grid"].get((gbin,bbin),
               max(thres_dict["ga"].get(gbin, thres_dict["global"]),
                   thres_dict["bmi"].get(bbin, thres_dict["global"]),
                   thres_dict["global"]))
    return thres_dict["global"]

def q4_core(excel: str,
            male_sheet: str = "男胎检测数据",
            female_sheet: str = "女胎检测数据",
            fpr_target: float = 0.05,
            fpr_margin: float = 0.0,
            fpr_mode_in: str = "auto",
            min_bin_n: int = 30,
            shrink_lambda: float = 100.0):
    male = pd.read_excel(excel, sheet_name=male_sheet)
    female = pd.read_excel(excel, sheet_name=female_sheet)
    cm_m = map_columns(male); cm_f = map_columns(female)
    for k in COLS.keys():
        if not cm_m.get(k) and cm_f.get(k): cm_m[k] = cm_f[k]
    male   = clean_numeric_columns(male, cm_m)
    female = clean_numeric_columns(female, cm_m)
    if cm_m["AB"] is None or cm_m["AB"] not in male.columns:
        raise ValueError("缺少 AB 列")
    male["_ABbin"] = (~male[cm_m["AB"]].fillna("").astype(str).str.strip().eq("")).astype(int)
    male_neg0 = male[male["_ABbin"]==0].copy()
    th = estimate_qc_thresholds(male_neg0, cm_m)
    male["QC0"]   = male.apply(lambda r: qc0_status(r, cm_m, th), axis=1)
    female["QC0"] = female.apply(lambda r: qc0_status(r, cm_m, th), axis=1)
    male_neg = male[(male["_ABbin"]==0) & (male["QC0"]!="invalid")].copy()
    male_pos = male[(male["_ABbin"]==1) & (male["QC0"]!="invalid")].copy()
    mu_x, sigma_x = fit_x_gate(male_neg, cm_m)
    xp = XGateParams()
    female["QCX"] = female.apply(lambda r: x_gate_status(r, cm_m, mu_x, sigma_x, xp), axis=1)
    male_neg = add_residual_Z(male_neg, male_neg, cm_m)
    male_pos = add_residual_Z(male_pos, male_neg, cm_m)
    female  = add_residual_Z(female,  male_neg, cm_m)
    def assemble(df: pd.DataFrame, cm: Dict[str,str]) -> pd.DataFrame:
        feats=[]
        for zk in ["Z_13","Z_18","Z_21"]:
            zc=cm[zk]
            if zc and f"{zc}_resid" in df.columns: feats.append(f"{zc}_resid")
            elif zc: feats.append(zc)
        for k in ["GC_all","reads","map_rate","dup_rate","filter_rate","BMI","GA"]:
            if cm[k]: feats.append(cm[k])
        tmp=df[feats].copy()
        if cm["reads"]: tmp["log_reads"]=np.log1p(df[cm["reads"]].astype(float))
        for c in tmp.columns: tmp[c]=pd.to_numeric(tmp[c],errors="coerce")
        return tmp
    X0 = assemble(male_neg, cm_m); X1 = assemble(male_pos, cm_m); Xf = assemble(female, cm_m)
    med0 = X0.median(numeric_only=True)
    X0f = X0.fillna(med0); X1f = X1.fillna(med0); Xff = Xf.fillna(med0)
    scaler = RobustScaler().fit(X0f)
    X0s = scaler.transform(X0f); X1s = scaler.transform(X1f); Xfs = scaler.transform(Xff)
    p_maha_0 = mahalanobis_tail_prob(X0s, X0s)
    p_maha_1 = mahalanobis_tail_prob(X0s, X1s)
    p_maha_f = mahalanobis_tail_prob(X0s, Xfs)
    X0df = pd.DataFrame(X0s, columns=X0f.columns)
    X1df = pd.DataFrame(X1s, columns=X1f.columns)
    Xfdf = pd.DataFrame(Xfs, columns=Xff.columns)
    p_feat_0 = robust_feature_tail(X0df, X0df)
    p_feat_1 = robust_feature_tail(X0df, X1df)
    p_feat_f = robust_feature_tail(X0df, Xfdf)
    s_disc_0, s_disc_1, s_disc_f, auc_male, ap_male = weak_discriminator_scores(X0s, X1s, Xfs)
    X_fuse_train = np.vstack([np.column_stack([p_maha_0, p_feat_0, s_disc_0]),
                              np.column_stack([p_maha_1, p_feat_1, s_disc_1])])
    y_fuse_train = np.concatenate([np.zeros_like(p_maha_0, dtype=int),
                                   np.ones_like(p_maha_1, dtype=int)])
    fuse_clf = LogisticRegression(max_iter=1000,class_weight="balanced",solver="lbfgs")
    fuse_clf.fit(X_fuse_train, y_fuse_train)
    s_fuse_0 = fuse_clf.predict_proba(np.column_stack([p_maha_0, p_feat_0, s_disc_0]))[:,1]
    s_fuse_1 = fuse_clf.predict_proba(np.column_stack([p_maha_1, p_feat_1, s_disc_1]))[:,1]
    s_fuse_f = fuse_clf.predict_proba(np.column_stack([p_maha_f, p_feat_f, s_disc_f]))[:,1]
    s_all = np.concatenate([s_fuse_0, s_fuse_1])
    y_all = np.concatenate([np.zeros_like(s_fuse_0, dtype=int),
                            np.ones_like(s_fuse_1, dtype=int)])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob = np.zeros_like(s_all, dtype=float)
    for tr_idx, va_idx in skf.split(s_all.reshape(-1,1), y_all):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(s_all[tr_idx], y_all[tr_idx])
        oof_prob[va_idx] = iso.predict(s_all[va_idx])
    iso_full = IsotonicRegression(out_of_bounds="clip").fit(s_all, y_all)
    prob_fem = iso_full.predict(s_fuse_f)
    n0 = len(s_fuse_0)
    prob_neg_oof = oof_prob[:n0]
    bins_ga_neg,  ga_edges  = quantile_bins(male_neg[cm_m["GA"]] if cm_m["GA"] else pd.Series([np.nan]*len(male_neg)), prefix="GAQ")
    bins_bmi_neg, bmi_edges = quantile_bins(male_neg[cm_m["BMI"]] if cm_m["BMI"] else pd.Series([np.nan]*len(male_neg)), prefix="BMIQ")
    female_ga_bin  = apply_bins_with_edges(female[cm_m["GA"]] if cm_m["GA"] else pd.Series([np.nan]*len(female)), ga_edges,  "GAQ")
    female_bmi_bin = apply_bins_with_edges(female[cm_m["BMI"]] if cm_m["BMI"] else pd.Series([np.nan]*len(female)), bmi_edges, "BMIQ")
    if fpr_mode_in == "auto":
        cnt_grid = pd.crosstab(bins_ga_neg, bins_bmi_neg)
        min_cell = int(cnt_grid.values.min()) if cnt_grid.size>0 else 0
        if min_cell >= 80: fpr_mode = "ga_bmi"
        else:
            cnt_ga  = bins_ga_neg.value_counts()
            cnt_bmi = bins_bmi_neg.value_counts()
            if (cnt_ga.min() if len(cnt_ga)>0 else 0) >= 80: fpr_mode = "ga"
            elif (cnt_bmi.min() if len(cnt_bmi)>0 else 0) >= 80: fpr_mode = "bmi"
            else: fpr_mode = "global"
    else:
        fpr_mode = fpr_mode_in
    thres_dict = compute_thresholds_by_mode(prob_neg_oof,
                                            bins_ga_neg.astype(str), bins_bmi_neg.astype(str),
                                            fpr_mode, fpr_target, fpr_margin,
                                            min_bin_n, shrink_lambda)
    def calc_real_fpr(prob, mask, tau):
        if np.sum(mask)==0: return np.nan
        return float(np.mean(prob[mask] >= tau))
    if fpr_mode=="global":
        fpr_real = calc_real_fpr(prob_neg_oof, np.ones_like(prob_neg_oof, dtype=bool), thres_dict["global"])
        fpr_by = pd.DataFrame([{"mode":"global","count":len(prob_neg_oof),"tau":thres_dict["global"],"fpr":fpr_real}])
    elif fpr_mode=="ga":
        rows=[]
        for g in sorted(pd.unique(bins_ga_neg.astype(str))):
            mask=(bins_ga_neg.astype(str)==g).values
            tau=thres_dict["ga"].get(g, thres_dict["global"])
            rows.append({"GA_bin":g, "count":int(np.sum(mask)), "tau":tau, "fpr":calc_real_fpr(prob_neg_oof, mask, tau)})
        fpr_by=pd.DataFrame(rows)
        fpr_real=float(np.nanmean(fpr_by["fpr"]))
    elif fpr_mode=="bmi":
        rows=[]
        for b in sorted(pd.unique(bins_bmi_neg.astype(str))):
            mask=(bins_bmi_neg.astype(str)==b).values
            tau=thres_dict["bmi"].get(b, thres_dict["global"])
            rows.append({"BMI_bin":b, "count":int(np.sum(mask)), "tau":tau, "fpr":calc_real_fpr(prob_neg_oof, mask, tau)})
        fpr_by=pd.DataFrame(rows)
        fpr_real=float(np.nanmean(fpr_by["fpr"]))
    else:
        rows=[]
        for g in sorted(pd.unique(bins_ga_neg.astype(str))):
            for b in sorted(pd.unique(bins_bmi_neg.astype(str))):
                mask=((bins_ga_neg.astype(str)==g) & (bins_bmi_neg.astype(str)==b)).values
                tau=thres_dict["grid"].get((g,b), thres_dict["global"])
                rows.append({"GA_bin":g,"BMI_bin":b,"count":int(np.sum(mask)),"tau":tau,"fpr":calc_real_fpr(prob_neg_oof, mask, tau)})
        fpr_by=pd.DataFrame(rows)
        fpr_real=float(np.nanmean(fpr_by["fpr"])) if len(fpr_by)>0 else np.nan
    qc_invalid = (female["QC0"]=="invalid")
    x_fail     = (female["QCX"]=="fail")
    used_tau = []
    decisions = []
    for i in range(len(female)):
        if qc_invalid.iloc[i] or x_fail.iloc[i]:
            used_tau.append(np.nan); decisions.append("无效-复检"); continue
        gbin=str(female_ga_bin.iloc[i]) if female_ga_bin is not None else "UNK"
        bbin=str(female_bmi_bin.iloc[i]) if female_bmi_bin is not None else "UNK"
        tau = select_threshold_for_row(thres_dict, fpr_mode, gbin, bbin)
        used_tau.append(tau)
        decisions.append("建议复检" if prob_fem[i] >= tau else "通过")
    out = female.copy()
    rename_cols = {
        cm_m.get("ID"): "ID", cm_m.get("GA"): "GA", cm_m.get("BMI"): "BMI",
        cm_m.get("GC_all"): "GC_all", cm_m.get("reads"): "reads",
        cm_m.get("map_rate"): "map_rate", cm_m.get("dup_rate"): "dup_rate",
        cm_m.get("filter_rate"): "filter_rate",
        cm_m.get("Z_13"): "Z_13", cm_m.get("Z_18"): "Z_18", cm_m.get("Z_21"): "Z_21",
        cm_m.get("Z_X"): "Z_X", cm_m.get("X_frac"): "X_frac",
        cm_m.get("AB"): "AB", cm_m.get("HEALTH"): "HEALTH"
    }
    rename_cols = {k:v for k,v in rename_cols.items() if k in out.columns and v is not None}
    out = out.rename(columns=rename_cols)
    out["QC0"] = female["QC0"]; out["QCX"] = female["QCX"]
    out["score_mahalanobis"] = p_maha_f
    out["score_feature_tail"] = p_feat_f
    out["score_disc_weak"] = s_disc_f
    out["score_fuse_raw"] = s_fuse_f
    out["anomaly_prob"] = prob_fem
    out["GA_bin"]  = female_ga_bin
    out["BMI_bin"] = female_bmi_bin
    out["tau_used"] = used_tau
    out["decision"] = decisions
    out["top_outliers"] = top3_outliers_per_row(pd.DataFrame(X0s, columns=X0f.columns), pd.DataFrame(Xfs, columns=Xff.columns))
    summary = {
        "fpr_mode": fpr_mode,
        "fpr_target": fpr_target,
        "fpr_margin": fpr_margin,
        "achieved_fpr_on_male_neg_mean": float(fpr_real) if fpr_real is not None else None,
        "N_female_total": int(len(out)),
        "QC_invalid": int((out["QC0"]=="invalid").sum()),
        "X_fail": int((out["QCX"]=="fail").sum()),
        "evaluated": int(((out["QC0"]!="invalid") & (out["QCX"]!="fail")).sum()),
        "flagged_suspect": int(((out["decision"]=="建议复检") & (out["QC0"]!="invalid") & (out["QCX"]!="fail")).sum()),
        "male_fuse_auc": float(roc_auc_score(y_all, s_all)) if len(np.unique(y_all))>1 else None,
        "male_fuse_ap": float(average_precision_score(y_all, s_all)) if len(np.unique(y_all))>1 else None,
        "qc_thresholds": {"GC_all":[th.gc_low, th.gc_high],"reads_min": th.reads_min,"map_min": th.map_min,"dup_max": th.dup_max,"filter_max": th.filter_max},
        "x_gate": {"mu": float(mu_x), "sigma": float(sigma_x), "k_sigma": xp.k_sigma}
    }
    return out, thres_dict, fpr_by, summary
