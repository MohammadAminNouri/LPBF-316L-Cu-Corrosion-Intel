import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


def plot_severity(df, output_path=None):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = df["sample_id"].str.replace("_", " ")
    ax.bar(labels, df["corrosion_severity_index_0_1"])
    ax.set_ylabel("Relative corrosion severity index (0–1)")
    ax.set_xlabel("Paper-derived experimental state")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Paper-derived corrosion severity ranking")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_pca(pca_df, output_path=None):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for _, row in pca_df.iterrows():
        ax.scatter(row["PC1"], row["PC2"], s=90)
        ax.annotate(row["sample_id"].replace("_", " "), (row["PC1"], row["PC2"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("ML-assisted electrochemical fingerprint map")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_feature_heatmap(df, features, output_path=None):
    x = df[features].copy().apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))
    xs = StandardScaler().fit_transform(x)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    im = ax.imshow(xs, aspect="auto")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["sample_id"].str.replace("_", " "))
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8)
    ax.set_title("Standardized electrochemical feature heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("z-score")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
