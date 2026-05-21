import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from src.feature_engineering import add_corrosion_severity_index, severity_components, BADNESS_WEIGHTS
from src.ml_fingerprint import pca_fingerprint, nearest_paper_state, DEFAULT_FEATURES

st.set_page_config(page_title="LPBF 316L-Cu Corrosion Intelligence", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "processed" / "paper_extracted_corrosion_features.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = add_corrosion_severity_index(df)
    return df

df = load_data()

st.title("LPBF 316L-Cu Corrosion Intelligence Dashboard")
st.caption("Paper-based ML-assisted electrochemical fingerprinting. No fake raw curves are generated.")

st.warning(
    "This dashboard uses only the four summarized experimental states reported in the manuscript. "
    "It is suitable for transparent visualization, severity ranking, and nearest-prototype interpretation — not for strong generalized prediction."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Paper Data",
    "Severity Index",
    "PCA Fingerprint",
    "Nearest Paper State",
    "Manuscript Claims"
])

with tab1:
    st.subheader("Extracted feature table")
    st.write("The table below is derived from the manuscript's polarization and EIS parameter tables.")
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download processed CSV", csv, "paper_extracted_corrosion_features.csv", "text/csv")

    st.subheader("LPBF processing condition")
    st.metric("Volumetric energy density", f"{df['ved_J_mm3'].iloc[0]:.1f} J/mm³")
    st.write("Calculated as P / (v × h × t) using 95 W, 500 mm/s, 84 µm, and 30 µm.")

with tab2:
    st.subheader("Relative corrosion severity index")
    plot_df = df.sort_values("corrosion_severity_index_0_1")
    fig = px.bar(
        plot_df,
        x="sample_id",
        y="corrosion_severity_index_0_1",
        color="medium",
        hover_data=["alloy", "degradation_label", "Icorr_uA_cm2", "Rp_PDP_kohm_cm2", "Rp_EIS_kohm_cm2", "CPE_n", "deff_nm"],
        labels={"corrosion_severity_index_0_1":"Relative severity index"},
        title="Paper-derived corrosion severity ranking"
    )
    st.plotly_chart(fig, use_container_width=True)

    comp = severity_components(df)
    comp.insert(0, "sample_id", df["sample_id"])
    st.subheader("Severity components")
    st.dataframe(comp, use_container_width=True)
    st.write("Weights used in the relative index:")
    st.json(BADNESS_WEIGHTS)

with tab3:
    st.subheader("PCA electrochemical fingerprint map")
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
        title=f"PCA fingerprint map | PC1={explained[0]*100:.1f}% / PC2={explained[1]*100:.1f}%"
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        "Interpretation: separation in this map reflects differences in current density, impedance resistance, film non-ideality, "
        "oxide-defect proxy thickness, Cu content, and peroxide exposure."
    )

with tab4:
    st.subheader("Nearest paper state classifier")
    st.write("Enter summarized electrochemical parameters. The tool returns the closest of the four manuscript states.")
    c1, c2, c3 = st.columns(3)
    with c1:
        alloy = st.selectbox("Alloy", ["AISI 316L", "AISI 316L-Cu"])
        cu = 3.42 if alloy == "AISI 316L-Cu" else 0.0
        h2o2 = st.selectbox("H2O2 present", [0, 1], format_func=lambda x: "Yes" if x else "No")
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

with tab5:
    st.subheader("Safe manuscript claim")
    st.markdown(
        "> A machine-learning-assisted electrochemical fingerprinting workflow was implemented using manuscript-derived polarization and impedance parameters. "
        "The objective was not to train a generalized predictive model, but to provide a transparent, reproducible classification of the four experimentally observed degradation states and to identify the dominant descriptors associated with peroxide-assisted corrosion."
    )
    st.subheader("Unsafe claim to avoid")
    st.error("Machine learning predicts the corrosion behavior of LPBF 316L-Cu implants.")
    st.subheader("Why")
    st.write(
        "The present dataset contains four summarized conditions. This is enough for visualization, severity ranking, and nearest-state comparison, but not enough for a robust predictive model."
    )
