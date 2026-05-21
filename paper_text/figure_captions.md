# Suggested New Figure Captions

## Figure 8. ML-assisted electrochemical fingerprinting workflow

Schematic workflow used to convert manuscript-derived electrochemical descriptors into an interpretable corrosion-state fingerprint. The input layer includes polarization parameters, EIS fitting parameters, alloy composition, and electrolyte condition. The analysis layer includes log-transformation of corrosion current density, feature standardization, relative corrosion severity indexing, PCA visualization, and nearest-prototype classification. The output layer ranks the four degradation states and identifies the descriptors associated with peroxide-assisted passive-film destabilization.

## Figure 9. Paper-derived corrosion severity ranking

Relative corrosion severity index calculated from normalized electrochemical descriptors extracted from the polarization and EIS results. The index combines log(Icorr), polarization resistance, EIS-derived resistance, CPE-n, and calculated effective defective-film thickness. The ranking shows the lowest degradation severity for AISI 316L in 0.9% NaCl and the highest severity for AISI 316L-Cu in 0.9% NaCl + H2O2, consistent with the loss of stable passivity and increased localized degradation in the peroxide-containing electrolyte.

## Figure 10. PCA electrochemical fingerprint map

Principal component analysis of the standardized paper-derived feature matrix. The chloride-only and peroxide-containing conditions separate in the reduced feature space, indicating that H2O2 is the dominant environmental variable controlling the electrochemical state. Within each electrolyte, the Cu-bearing alloy shifts toward a higher-severity fingerprint, supporting the interpretation that Cu modifies the passive-film response and contributes to localized degradation under inflammatory conditions.

## Figure 11. Feature contribution map

Standardized heatmap of the electrochemical descriptors used for ML-assisted fingerprinting. High corrosion current density, low polarization resistance, low EIS resistance, lower CPE-n, and increased effective defective-film thickness collectively distinguish the peroxide-exposed specimens from the chloride-only controls. The AISI 316L-Cu specimen exposed to H2O2/NaCl presents the most severe combined fingerprint.
