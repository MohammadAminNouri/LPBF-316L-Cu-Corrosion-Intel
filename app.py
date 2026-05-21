from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.feature_engineering import (
    BADNESS_WEIGHTS,
    add_corrosion_severity_index,
    severity_components,
)
from src.ml_fingerprint import pca_fingerprint, nearest_paper_state
from src.claim_engine import (
    claim_verdicts,
    counterfactual_table,
    enrich_for_claims,
    factorial_effect_table,
    fold_change_summary,
    train_surrogate_tree,
)

st.set_page_config(
    page_title="LPBF 316L-Cu Corrosion Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "processed" / "paper_extracted_corrosion_features.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = add_corrosion_severity_index(df)
    df = enrich_for_claims(df)
    return df


df = load_data()

st.title("LPBF 316L-Cu Corrosion Intelligence Dashboard")
st.caption(
    "Paper-derived ML-assisted corrosion claim engine for LPBF 316L and Cu-bearing 316L in NaCl and H₂O₂/NaCl."
)

st.info(
    "New elevation layer: the app now uses the paper's natural 2×2 design "
    "(Cu absent/present × H₂O₂ absent/present) to quantify Cu effects, peroxide effects, "
    "Cu×H₂O₂ interaction, and a transparent ML surrogate supporting the manuscript claims. "
    "No fake raw electrochemical curves or SEM data are generated."
)

with st.sidebar:
    st.header("Dashboard mode")
    st.write("Use this app as a companion to the manuscript, not as a universal corrosion predictor.")
    st.metric("Experimental states", len(df))
    st.metric("Factors", "Cu × H₂O₂")
    st.metric("LPBF VED", f"{df['ved_J_mm3'].iloc[0]:.1f} J/mm³")
    st.divider()
    st.write("**Best safe wording:**")
    st.success("ML-assisted electrochemical fingerprinting and factorial claim support")
    st.write("**Avoid:**")
    st.error("General ML prediction of implant corrosion")

worst = df.sort_values("corrosion_severity_index_0_1", ascending=False).iloc[0]
best = df.sort_values("corrosion_severity_index_0_1", ascending=True).iloc[0]
folds = fold_change_summary(df)
claim_df = claim_verdicts(df)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Worst state", worst["sample_id"], worst["degradation_label"])
with c2:
    st.metric("Best state", best["sample_id"], best["degradation_label"])
with c3:
    h2o2_cu_fold = folds.loc[
        folds["claim_test"] == "H2O2 effect on 316L-Cu",
        "Icorr_fold_increase",
    ].iloc[0]
    st.metric("H₂O₂ effect on 316L-Cu", f"{h2o2_cu_fold:.1f}× Icorr")
with c4:
    cu_h2o2_fold = folds.loc[
        folds["claim_test"] == "Cu effect in H2O2/NaCl",
        "Icorr_fold_increase",
    ].iloc[0]
    st.metric("Cu effect in H₂O₂/NaCl", f"{cu_h2o2_fold:.1f}× Icorr")


tabs = st.tabs(
    [
        "1 | Paper Data",
        "2 | Claim Engine",
        "3 | Factorial Effects",
        "4 | ML Surrogate",
        "5 | PCA Fingerprint",
        "6 | Counterfactuals",
        "7 | Manuscript Text",
    ]
)

with tabs[0]:
    st.subheader("Manuscript-extracted feature table")
    st.write(
        "This table uses only the values reported in the manuscript: polarization parameters, "
        "EIS fitted parameters, LPBF processing values, Cu content, and electrolyte labels."
    )
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download enhanced feature table",
        csv,
        "enhanced_paper_corrosion_features.csv",
        "text/csv",
    )

    st.subheader("Publication-useful severity map")
    fig = px.bar(
        df.sort_values("corrosion_severity_index_0_1"),
        x="sample_id",
        y="corrosion_severity_index_0_1",
        color="medium",
        text="corrosion_severity_index_0_1",
        hover_data=[
            "alloy",
            "Icorr_uA_cm2",
            "Rp_PDP_kohm_cm2",
            "Rp_EIS_kohm_cm2",
            "CPE_n",
            "deff_nm",
            "degradation_label",
        ],
        labels={"corrosion_severity_index_0_1": "Relative severity index"},
        title="Paper-derived corrosion severity ranking",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Severity components")
    comp = severity_components(df)
    comp.insert(0, "sample_id", df["sample_id"])
    st.dataframe(comp, use_container_width=True)
    with st.expander("Show severity weights"):
        st.json(BADNESS_WEIGHTS)

with tabs[1]:
    st.subheader("Claim-support engine")
    st.write(
        "This section converts the paper's four-state dataset into explicit claim tests. "
        "It helps you argue the paper more strongly because every claim is linked to a numerical condition."
    )
    st.dataframe(claim_df, use_container_width=True, hide_index=True)

    st.subheader("Direct fold-change evidence")
    st.write(
        "Fold increase is used for Icorr. Percentage loss is used for polarization resistance because lower Rp means weaker protection."
    )
    st.dataframe(folds, use_container_width=True, hide_index=True)

    fig = px.scatter(
        df,
        x="Ecorr_V_AgAgCl",
        y="corrosion_severity_index_0_1",
        color="medium",
        symbol="alloy",
        text="sample_id",
        size="Icorr_uA_cm2",
        title="Why noble potential does not mean better corrosion resistance",
        labels={
            "Ecorr_V_AgAgCl": "Ecorr (V vs Ag/AgCl)",
            "corrosion_severity_index_0_1": "Relative corrosion severity",
        },
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "Paper claim supported: the peroxide-containing states shift to more positive Ecorr values, "
        "but they also show higher severity and lower resistance. This helps defend the sentence: "
        "a noble OCP/Ecorr shift is not direct evidence of protection in oxidizing inflammatory media."
    )

with tabs[2]:
    st.subheader("2×2 factorial effect decomposition")
    st.write(
        "The experiment has two binary factors: Cu addition and H₂O₂ exposure. "
        "The table below decomposes each electrochemical response into Cu main effect, "
        "H₂O₂ main effect, and Cu×H₂O₂ interaction. This is stronger than only showing bars."
    )

    effects = factorial_effect_table(df)
    st.dataframe(effects, use_container_width=True, hide_index=True)

    effect_long = effects.melt(
        id_vars=["response"],
        value_vars=["Cu_main_effect", "H2O2_main_effect", "Cu_x_H2O2_interaction"],
        var_name="effect_type",
        value_name="effect_value",
    )
    fig = px.bar(
        effect_long,
        x="response",
        y="effect_value",
        color="effect_type",
        barmode="group",
        title="Cu, H₂O₂, and interaction effects from the manuscript's 2×2 design",
    )
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    dominant = effects[effects["dominant_main_effect"] == "H2O2"].shape[0]
    st.info(
        f"H₂O₂ is the dominant main effect in {dominant}/{len(effects)} paper-derived response descriptors. "
        "This supports the interpretation that peroxide controls early electrochemical degradation, "
        "while Cu modifies/localizes the response."
    )

with tabs[3]:
    st.subheader("Interpretable ML surrogate model")
    st.write(
        "This is the new ML layer. A shallow decision-tree regressor is trained to reproduce the "
        "paper-derived severity index from electrochemical descriptors. Because the dataset has only four states, "
        "this model is used as an explanation engine, not as a generalized predictor."
    )

    importance, rules, mae = train_surrogate_tree(df)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Leave-one-out MAE", f"{mae:.3f}")
        st.caption("Stress-test error on only four paper states. Lower is better, but this is not a validation of broad predictivity.")
        fig = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Surrogate feature importance for corrosion severity",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.write("**Decision rules learned from the paper-derived states:**")
        st.code(rules, language="text")

    st.success(
        "Use this in the paper as: 'an interpretable ML surrogate was used to rank the descriptors controlling the "
        "relative corrosion-severity index.' Do not call it an independent predictive model."
    )

with tabs[4]:
    st.subheader("Unsupervised electrochemical fingerprint map")
    pca_df, pca = pca_fingerprint(df)
    explained = pca.explained_variance_ratio_
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="medium",
        symbol="alloy",
        text="sample_id",
        hover_data=["degradation_label"],
        title=f"PCA fingerprint map | PC1={explained[0] * 100:.1f}% / PC2={explained[1] * 100:.1f}%",
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        "The PCA map is unsupervised: it does not know the paper conclusion in advance. "
        "Separation of the H₂O₂ and Cu-bearing states supports the claim that the inflammatory electrolyte "
        "and Cu addition produce distinct electrochemical fingerprints."
    )

    st.subheader("Nearest manuscript-state classifier")
    st.write("Enter summarized electrochemical parameters and compare the input to the four manuscript states.")
    c1, c2, c3 = st.columns(3)
    with c1:
        alloy = st.selectbox("Alloy", ["AISI 316L", "AISI 316L-Cu"], key="nearest_alloy")
        cu = 3.42 if alloy == "AISI 316L-Cu" else 0.0
        h2o2 = st.selectbox("H₂O₂ present", [0, 1], format_func=lambda x: "Yes" if x else "No")
        ecorr = st.number_input("Ecorr (V vs Ag/AgCl)", value=0.17, step=0.01)
    with c2:
        icorr = st.number_input("Icorr (µA/cm²)", value=1.65e-6, format="%.2e")
        rp_pdp = st.number_input("Rp from PDP (kΩ·cm²)", value=11.4)
        rp_eis = st.number_input("Rp from EIS (kΩ·cm²)", value=17.36)
    with c3:
        rs = st.number_input("Rs (Ω·cm²)", value=60.6)
        cpe_n = st.number_input("CPE-n", value=0.85, min_value=0.0, max_value=1.0)
        deff = st.number_input("d_eff (nm)", value=1.81)

    query = {
        "Ecorr_V_AgAgCl": ecorr,
        "Icorr_uA_cm2": icorr,
        "Rp_PDP_kohm_cm2": rp_pdp,
        "Rs_ohm_cm2": rs,
        "Rp_EIS_kohm_cm2": rp_eis,
        "Qp1_1e5_ohm_inv_cm2_s_n": float(df["Qp1_1e5_ohm_inv_cm2_s_n"].median()),
        "CPE_n": cpe_n,
        "Ceff_1e5_F_cm2": float(df["Ceff_1e5_F_cm2"].median()),
        "deff_nm": deff,
        "cu_wt_percent": cu,
        "h2o2_present": h2o2,
    }
    result = nearest_paper_state(df, query)
    st.success(f"Nearest manuscript state: {result['nearest_sample_id']} — {result['nearest_label']}")
    st.caption(f"Standardized Euclidean distance: {result['distance']:.3f}")

with tabs[5]:
    st.subheader("Counterfactual claim explorer")
    st.write(
        "Choose one observed paper state. The app compares it against the other observed states as if Cu and/or H₂O₂ were toggled. "
        "This is not fake prediction; it is a transparent comparison across your actual 2×2 paper design."
    )
    selected = st.selectbox("Selected manuscript state", df["sample_id"].tolist(), index=3)
    cf = counterfactual_table(df, selected)
    st.dataframe(cf, use_container_width=True, hide_index=True)

    fig = px.bar(
        cf,
        x="scenario",
        y="delta_severity_vs_selected",
        color="scenario",
        title=f"Severity difference relative to {selected}",
        labels={"delta_severity_vs_selected": "Δ relative severity"},
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "For the paper discussion, this helps write a stronger sentence: the Cu-bearing peroxide condition is not just visually worse; "
        "it is the worst state after combining current density, resistance loss, CPE non-ideality, and defect-thickness proxy."
    )

with tabs[6]:
    st.subheader("Manuscript-ready paragraph")
    st.markdown(
        """
**Suggested section title:**  
Data-driven electrochemical fingerprinting and factorial claim analysis

**Suggested paragraph:**

To strengthen the interpretation of the summarized electrochemical data, a paper-derived corrosion-intelligence workflow was implemented using the polarization and impedance parameters reported in the manuscript. The four experimental states naturally form a 2×2 design, where the two factors are Cu addition and H₂O₂ exposure. This enabled a transparent factorial decomposition of the electrochemical response into Cu main effect, H₂O₂ main effect, and Cu×H₂O₂ interaction. A relative corrosion-severity index was also calculated from normalized descriptors including log(Icorr), polarization resistance loss, EIS resistance loss, CPE-n decrease, and effective oxide/hydroxide thickness. The analysis ranked the 316L-Cu specimen in H₂O₂/NaCl as the highest-severity condition, consistent with the polarization, EIS, and post-exposure observations. Importantly, the peroxide-containing states showed nobler corrosion potentials while simultaneously exhibiting higher current density and lower resistance, confirming that a positive potential shift in oxidizing inflammatory media should not be interpreted as improved passivity. An interpretable decision-tree surrogate was used only as an explanatory model to identify the descriptors controlling the paper-derived severity ranking, not as a generalized corrosion predictor.
        """
    )

    st.subheader("Safe claim")
    st.success(
        "A paper-derived ML-assisted and factorial electrochemical fingerprinting workflow supports the conclusion that H₂O₂ dominates passive-film degradation, while Cu addition intensifies the high-risk localized corrosion state under peroxide-containing inflammatory conditions."
    )

    st.subheader("Unsafe claim")
    st.error(
        "Machine learning predicts implant corrosion performance. Avoid this unless you later add raw curves, more conditions, replicates, or literature-expanded datasets."
    )
