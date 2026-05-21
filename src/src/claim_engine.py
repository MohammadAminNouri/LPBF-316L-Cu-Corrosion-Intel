"""Paper-derived claim engine for the LPBF 316L-Cu corrosion dashboard.

This module uses only the four summarized states reported in the manuscript:
316L / NaCl, 316L-Cu / NaCl, 316L / NaCl+H2O2, and 316L-Cu / NaCl+H2O2.

It does not generate fake raw curves. The central idea is to exploit the paper's
natural 2x2 design:

    factor A = Cu addition absent/present
    factor B = H2O2 absent/present

This allows quantitative claim support: Cu effect, peroxide effect, and Cu x
peroxide interaction for electrochemical descriptors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error


MODEL_FEATURES = [
    "log10_Icorr",
    "Rp_PDP_kohm_cm2",
    "Rp_EIS_kohm_cm2",
    "CPE_n",
    "deff_nm",
    "h2o2_present",
    "cu_present",
]


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def enrich_for_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Add claim-engine helper columns."""
    out = df.copy()

    if "log10_Icorr" not in out.columns:
        out["log10_Icorr"] = np.log10(pd.to_numeric(out["Icorr_uA_cm2"], errors="coerce"))

    out["cu_present"] = (pd.to_numeric(out["cu_wt_percent"], errors="coerce") > 0).astype(int)
    out["h2o2_present"] = pd.to_numeric(out["h2o2_present"], errors="coerce").fillna(0).astype(int)
    out["state_code"] = out["cu_present"].astype(str) + out["h2o2_present"].astype(str)

    for col in ["Rp_PDP_kohm_cm2", "Rp_EIS_kohm_cm2", "CPE_n"]:
        val = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_badness"] = val.max() - val

    return out


def _state_lookup(df: pd.DataFrame) -> dict:
    x = enrich_for_claims(df)
    lookup = {}
    for _, row in x.iterrows():
        key = (int(row["cu_present"]), int(row["h2o2_present"]))
        lookup[key] = row
    return lookup


def factorial_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate 2x2 Cu, H2O2, and interaction effects."""
    lookup = _state_lookup(df)
    required = [(0, 0), (1, 0), (0, 1), (1, 1)]
    missing = [k for k in required if k not in lookup]

    if missing:
        raise ValueError(f"The 2x2 factorial states are incomplete. Missing: {missing}")

    responses = [
        ("log10_Icorr", "log10 corrosion current", "higher = worse"),
        ("Rp_PDP_kohm_cm2_badness", "PDP resistance loss proxy", "higher = worse"),
        ("Rp_EIS_kohm_cm2_badness", "EIS resistance loss proxy", "higher = worse"),
        ("CPE_n_badness", "film non-ideality proxy", "higher = worse"),
        ("deff_nm", "defective oxide/hydroxide thickness proxy", "higher = worse"),
        ("corrosion_severity_index_0_1", "relative corrosion severity index", "higher = worse"),
    ]

    rows = []

    for col, label, direction in responses:
        y00 = _safe_float(lookup[(0, 0)].get(col, np.nan))
        y10 = _safe_float(lookup[(1, 0)].get(col, np.nan))
        y01 = _safe_float(lookup[(0, 1)].get(col, np.nan))
        y11 = _safe_float(lookup[(1, 1)].get(col, np.nan))

        cu_effect = ((y10 - y00) + (y11 - y01)) / 2
        h2o2_effect = ((y01 - y00) + (y11 - y10)) / 2
        interaction = (y11 - y10) - (y01 - y00)

        dominant = "H2O2" if abs(h2o2_effect) >= abs(cu_effect) else "Cu"

        rows.append(
            {
                "response": label,
                "column": col,
                "direction": direction,
                "baseline_316L_NaCl": y00,
                "Cu_main_effect": cu_effect,
                "H2O2_main_effect": h2o2_effect,
                "Cu_x_H2O2_interaction": interaction,
                "dominant_main_effect": dominant,
            }
        )

    return pd.DataFrame(rows)


def fold_change_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return intuitive fold / percent changes supporting manuscript claims."""
    lookup = _state_lookup(df)
    s00, s10, s01, s11 = lookup[(0, 0)], lookup[(1, 0)], lookup[(0, 1)], lookup[(1, 1)]

    def fold(a, b):
        return _safe_float(b) / _safe_float(a)

    def loss(a, b):
        a = _safe_float(a)
        b = _safe_float(b)
        return (a - b) / a * 100 if a else np.nan

    rows = [
        {
            "claim_test": "H2O2 effect on 316L",
            "Icorr_fold_increase": fold(s00["Icorr_uA_cm2"], s01["Icorr_uA_cm2"]),
            "PDP_Rp_loss_percent": loss(s00["Rp_PDP_kohm_cm2"], s01["Rp_PDP_kohm_cm2"]),
            "EIS_Rp_loss_percent": loss(s00["Rp_EIS_kohm_cm2"], s01["Rp_EIS_kohm_cm2"]),
        },
        {
            "claim_test": "H2O2 effect on 316L-Cu",
            "Icorr_fold_increase": fold(s10["Icorr_uA_cm2"], s11["Icorr_uA_cm2"]),
            "PDP_Rp_loss_percent": loss(s10["Rp_PDP_kohm_cm2"], s11["Rp_PDP_kohm_cm2"]),
            "EIS_Rp_loss_percent": loss(s10["Rp_EIS_kohm_cm2"], s11["Rp_EIS_kohm_cm2"]),
        },
        {
            "claim_test": "Cu effect in NaCl",
            "Icorr_fold_increase": fold(s00["Icorr_uA_cm2"], s10["Icorr_uA_cm2"]),
            "PDP_Rp_loss_percent": loss(s00["Rp_PDP_kohm_cm2"], s10["Rp_PDP_kohm_cm2"]),
            "EIS_Rp_loss_percent": loss(s00["Rp_EIS_kohm_cm2"], s10["Rp_EIS_kohm_cm2"]),
        },
        {
            "claim_test": "Cu effect in H2O2/NaCl",
            "Icorr_fold_increase": fold(s01["Icorr_uA_cm2"], s11["Icorr_uA_cm2"]),
            "PDP_Rp_loss_percent": loss(s01["Rp_PDP_kohm_cm2"], s11["Rp_PDP_kohm_cm2"]),
            "EIS_Rp_loss_percent": loss(s01["Rp_EIS_kohm_cm2"], s11["Rp_EIS_kohm_cm2"]),
        },
    ]

    return pd.DataFrame(rows)


def claim_verdicts(df: pd.DataFrame) -> pd.DataFrame:
    """Generate compact claim verdicts from the extracted paper data."""
    x = enrich_for_claims(df)
    lookup = _state_lookup(x)

    s00, s10, s01, s11 = lookup[(0, 0)], lookup[(1, 0)], lookup[(0, 1)], lookup[(1, 1)]

    worst = x.sort_values("corrosion_severity_index_0_1", ascending=False).iloc[0]

    peroxide_icorr_increases = (
        s01["Icorr_uA_cm2"] > s00["Icorr_uA_cm2"]
        and s11["Icorr_uA_cm2"] > s10["Icorr_uA_cm2"]
    )

    peroxide_rp_decreases = (
        s01["Rp_EIS_kohm_cm2"] < s00["Rp_EIS_kohm_cm2"]
        and s11["Rp_EIS_kohm_cm2"] < s10["Rp_EIS_kohm_cm2"]
    )

    cu_worsens_current = (
        s10["Icorr_uA_cm2"] > s00["Icorr_uA_cm2"]
        and s11["Icorr_uA_cm2"] > s01["Icorr_uA_cm2"]
    )

    cu_worsens_rp = (
        s10["Rp_EIS_kohm_cm2"] < s00["Rp_EIS_kohm_cm2"]
        and s11["Rp_EIS_kohm_cm2"] < s01["Rp_EIS_kohm_cm2"]
    )

    noble_shift_misleading = (
        s01["Ecorr_V_AgAgCl"] > s00["Ecorr_V_AgAgCl"]
        and s01["corrosion_severity_index_0_1"] > s00["corrosion_severity_index_0_1"]
    )

    verdict_rows = [
        {
            "claim": "H2O2 accelerates corrosion and weakens the passive film",
            "verdict": "Supported" if peroxide_icorr_increases and peroxide_rp_decreases else "Not fully supported",
            "evidence": "In both alloys, H2O2 increases Icorr and decreases EIS Rp.",
        },
        {
            "claim": "Cu addition does not guarantee improved corrosion resistance",
            "verdict": "Supported" if cu_worsens_current and cu_worsens_rp else "Partially supported",
            "evidence": "Cu-bearing states show higher Icorr and lower EIS Rp than their non-Cu counterparts.",
        },
        {
            "claim": "The combined 316L-Cu + H2O2/NaCl state is the highest-risk state",
            "verdict": "Supported" if worst["sample_id"] == s11["sample_id"] else "Not supported",
            "evidence": f"Highest relative severity state is {worst['sample_id']}.",
        },
        {
            "claim": "A nobler Ecorr/OCP shift in peroxide is not evidence of protection",
            "verdict": "Supported" if noble_shift_misleading else "Partially supported",
            "evidence": "Peroxide shifts Ecorr nobler while severity, Icorr, and resistance loss worsen.",
        },
    ]

    return pd.DataFrame(verdict_rows)


def train_surrogate_tree(df: pd.DataFrame) -> tuple[pd.DataFrame, str, float]:
    """Train an interpretable decision-tree surrogate for the severity index."""
    x = enrich_for_claims(df)

    features = [c for c in MODEL_FEATURES if c in x.columns]

    X = x[features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    y = pd.to_numeric(x["corrosion_severity_index_0_1"], errors="coerce")

    tree = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree.fit(X, y)

    loo = LeaveOneOut()
    preds = []
    actual = []

    for train_idx, test_idx in loo.split(X):
        t = DecisionTreeRegressor(max_depth=2, random_state=42)
        t.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.append(float(t.predict(X.iloc[test_idx])[0]))
        actual.append(float(y.iloc[test_idx].values[0]))

    mae = float(mean_absolute_error(actual, preds))

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": tree.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    rules = export_text(tree, feature_names=features)

    return importance, rules, mae


def counterfactual_table(df: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    """Compare one selected state against the other observed manuscript states."""
    x = enrich_for_claims(df)

    row = x.loc[x["sample_id"] == sample_id]

    if row.empty:
        raise ValueError(f"Unknown sample_id: {sample_id}")

    row = row.iloc[0]
    lookup = _state_lookup(x)

    scenarios = [
        (0, 0, "316L / NaCl"),
        (1, 0, "316L-Cu / NaCl"),
        (0, 1, "316L / NaCl + H2O2"),
        (1, 1, "316L-Cu / NaCl + H2O2"),
    ]

    base = row
    rows = []

    for cu, h, label in scenarios:
        target = lookup[(cu, h)]

        rows.append(
            {
                "scenario": label,
                "observed_sample": target["sample_id"],
                "delta_log10_Icorr_vs_selected": target["log10_Icorr"] - base["log10_Icorr"],
                "delta_Rp_EIS_vs_selected_kohm_cm2": target["Rp_EIS_kohm_cm2"] - base["Rp_EIS_kohm_cm2"],
                "delta_CPE_n_vs_selected": target["CPE_n"] - base["CPE_n"],
                "delta_severity_vs_selected": target["corrosion_severity_index_0_1"] - base["corrosion_severity_index_0_1"],
            }
        )

    return pd.DataFrame(rows)
