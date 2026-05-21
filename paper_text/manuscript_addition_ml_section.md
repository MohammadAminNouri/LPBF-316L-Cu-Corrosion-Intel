# Manuscript Addition: ML-Assisted Electrochemical Fingerprinting

## Suggested section title

**Machine-learning-assisted electrochemical fingerprinting of peroxide-mediated degradation states**

## Suggested placement

Place this after the EIS section and before the post-exposure morphology discussion, or as a short independent subsection after the surface characterization section.

## Methods text

To provide a reproducible and quantitative comparison of the electrochemical states observed in this study, a machine-learning-assisted electrochemical fingerprinting workflow was implemented using the summarized polarization and impedance parameters extracted from the experimental results. The purpose of this workflow was not to train a generalized predictive model, because the present dataset contains four experimental states, but rather to classify the relative degradation response of each alloy/electrolyte condition and identify the electrochemical descriptors most closely associated with peroxide-assisted corrosion.

The input feature matrix included Ecorr, Icorr, polarization resistance obtained from potentiodynamic polarization, EIS-derived solution resistance, EIS-derived polarization resistance, CPE exponent n, effective capacitance, calculated effective film thickness, Cu content, and the presence or absence of H2O2. The corrosion current density was log-transformed before analysis to reduce scale dominance. Missing passive-current or breakdown-potential values in the H2O2-containing electrolyte were not artificially reconstructed, because the absence of a stable passive region is itself an experimental observation.

A relative corrosion severity index was calculated from normalized electrochemical descriptors. Features associated with increased degradation, including higher log(Icorr), lower polarization resistance, lower EIS resistance, lower CPE-n, and higher effective defective-film thickness, were weighted and combined into a dimensionless score from 0 to 1. This index was used only as an internal ranking tool within the four conditions tested in this study and should not be interpreted as an absolute biomedical corrosion-rate scale.

Principal component analysis (PCA) was then applied to the standardized feature matrix to visualize the separation between degradation states. In parallel, a nearest-prototype classifier was implemented to identify the closest experimentally observed corrosion state based on the standardized Euclidean distance between electrochemical fingerprints. This approach enables transparent interpretation while avoiding overstatement of model generality.

## Results text

The paper-derived fingerprinting workflow ranked the degradation severity in the following order: AISI 316L in 0.9% NaCl < AISI 316L-Cu in 0.9% NaCl < AISI 316L in 0.9% NaCl + H2O2 < AISI 316L-Cu in 0.9% NaCl + H2O2. This ranking is consistent with the electrochemical results, where the H2O2-containing medium produced higher corrosion current densities and lower polarization/impedance resistances, while the Cu-bearing alloy in the inflammatory electrolyte exhibited the poorest overall electrochemical response.

The PCA fingerprint map separated the chloride-only conditions from the peroxide-containing conditions, indicating that the oxidizing inflammatory electrolyte imposed the dominant change in electrochemical state. Within each electrolyte, the Cu-bearing alloy shifted toward higher degradation severity, consistent with its higher Icorr, lower Rp, and lower CPE-n. The strongest descriptors contributing to the severity ranking were log(Icorr), Rp from polarization, Rp from EIS, CPE-n, and calculated effective film thickness. These descriptors collectively capture the transition from a comparatively stable passive film to a thicker but more defective oxide/hydroxide layer under peroxide exposure.

The ML-assisted fingerprint therefore supports the mechanistic interpretation that the noble OCP/Ecorr shift in the H2O2-containing electrolyte should not be interpreted as improved protection. Instead, peroxide changes the redox condition at the metal/electrolyte interface while simultaneously reducing film barrier properties. For AISI 316L-Cu, the additional Cu-related contribution further increases electrochemical heterogeneity and may delay or obstruct repassivation in localized attacked regions.

## Reviewer-safe limitation sentence

The current workflow is intentionally presented as a low-data, interpretable fingerprinting tool rather than a generalized predictive model. Future work using raw electrochemical time-series, additional replicates, extended immersion times, image-based pit statistics, and literature-expanded datasets would allow development of more robust supervised learning models.
