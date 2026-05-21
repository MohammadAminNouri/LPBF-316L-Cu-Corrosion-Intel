import numpy as np
import pandas as pd

BADNESS_WEIGHTS = {
    "current_badness": 0.25,
    "pdp_resistance_badness": 0.20,
    "eis_resistance_badness": 0.20,
    "film_nonideality_badness": 0.15,
    "oxide_defect_proxy_badness": 0.10,
    "peroxide_factor": 0.05,
    "cu_factor": 0.05,
}


def minmax(series: pd.Series) -> pd.Series:
    """Min-max normalize a numeric pandas Series."""
    s = pd.to_numeric(series, errors="coerce")
    if s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add table-derived features using only manuscript values."""
    out = df.copy()
    out["log10_Icorr"] = np.log10(out["Icorr_uA_cm2"])
    return out


def severity_components(df: pd.DataFrame) -> pd.DataFrame:
    """Return interpretable relative badness components for each sample.

    This is a relative index inside the four manuscript states. It is not a universal
    biomedical corrosion score.
    """
    x = add_features(df)
    comp = pd.DataFrame(index=x.index)
    comp["current_badness"] = minmax(x["log10_Icorr"])
    comp["pdp_resistance_badness"] = 1 - minmax(x["Rp_PDP_kohm_cm2"])
    comp["eis_resistance_badness"] = 1 - minmax(x["Rp_EIS_kohm_cm2"])
    comp["film_nonideality_badness"] = 1 - minmax(x["CPE_n"])
    comp["oxide_defect_proxy_badness"] = minmax(x["deff_nm"])
    comp["peroxide_factor"] = x["h2o2_present"].astype(float)
    comp["cu_factor"] = minmax(x["cu_wt_percent"])
    return comp


def add_corrosion_severity_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add relative corrosion severity index from paper-derived features."""
    out = add_features(df)
    comp = severity_components(out)
    out["corrosion_severity_index_0_1"] = sum(comp[c] * w for c, w in BADNESS_WEIGHTS.items())
    out["corrosion_severity_rank"] = out["corrosion_severity_index_0_1"].rank(method="dense").astype(int)
    return out
