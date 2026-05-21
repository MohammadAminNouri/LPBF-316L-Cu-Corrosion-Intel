import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

DEFAULT_FEATURES = [
    "Ecorr_V_AgAgCl",
    "log10_Icorr",
    "Rp_PDP_kohm_cm2",
    "Rs_ohm_cm2",
    "Rp_EIS_kohm_cm2",
    "Qp1_1e5_ohm_inv_cm2_s_n",
    "CPE_n",
    "Ceff_1e5_F_cm2",
    "deff_nm",
    "cu_wt_percent",
    "h2o2_present",
]


def prepare_matrix(df: pd.DataFrame, features=None):
    """Return standardized matrix from selected features.

    Missing values are filled by column medians to avoid fabricating missing
    values. This is only for visualization/prototype classification.
    """
    features = features or DEFAULT_FEATURES
    x = df[features].copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    return x, xs, scaler


def pca_fingerprint(df: pd.DataFrame, n_components=2, features=None):
    """PCA fingerprint map for the manuscript-derived feature matrix."""
    x, xs, scaler = prepare_matrix(df, features=features)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(xs)
    out = df[["sample_id", "alloy", "medium", "degradation_label"]].copy()
    out["PC1"] = pcs[:, 0]
    out["PC2"] = pcs[:, 1]
    return out, pca


def nearest_paper_state(df: pd.DataFrame, query: dict, features=None):
    """Nearest-prototype classifier using the four paper states.

    This is not a general ML predictor. It simply identifies which reported
    manuscript state is most similar to the query based on standardized
    electrochemical features.
    """
    features = features or DEFAULT_FEATURES
    working = df.copy()
    for col in features:
        if col not in working.columns:
            raise ValueError(f"Missing feature in dataset: {col}")
    if "log10_Icorr" not in working and "Icorr_uA_cm2" in working:
        working["log10_Icorr"] = np.log10(working["Icorr_uA_cm2"])
    x = working[features].copy().apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    q = pd.DataFrame([query])
    if "log10_Icorr" not in q and "Icorr_uA_cm2" in q:
        q["log10_Icorr"] = np.log10(q["Icorr_uA_cm2"])
    qx = q.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    qx = qx.fillna(x.median(numeric_only=True))
    qxs = scaler.transform(qx)
    distances = euclidean_distances(qxs, xs).ravel()
    idx = int(np.argmin(distances))
    return {
        "nearest_sample_id": working.loc[idx, "sample_id"],
        "nearest_label": working.loc[idx, "degradation_label"],
        "distance": float(distances[idx]),
    }
