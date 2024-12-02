import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Function to calculate KL divergence
def calculate_kl_divergence(p, q):
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    return entropy(p, q)

# Streamlit App
st.title("Advanced Anomaly Detection with KL Divergence")

# Upload data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    # Parameters
    st.sidebar.header("Parameters")
    window_size = st.sidebar.slider("Window Size", 50, 500, 100)
    step_size = st.sidebar.slider("Step Size", 10, 200, 50)
    threshold_factor = st.sidebar.slider("Threshold Factor (std)", 1.0, 5.0, 3.0)

    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    # Compute KL divergence
    num_windows = (len(normalized_data) - window_size) // step_size + 1
    reference_window = normalized_data[:window_size]
    ref_distribution = np.mean(reference_window, axis=0)

    kl_divergences = []
    feature_contributions = []
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window = normalized_data[start_idx:end_idx]
        current_distribution = np.mean(window, axis=0)
        kl_div = calculate_kl_divergence(ref_distribution, current_distribution)
        kl_divergences.append(kl_div)

        # Feature-level contributions
        feature_contributions.append(np.abs(ref_distribution - current_distribution))

    kl_divergences = np.array(kl_divergences)
    feature_contributions = np.array(feature_contributions)
    threshold = kl_divergences.mean() + threshold_factor * kl_divergences.std()
    anomalies = np.where(kl_divergences > threshold)[0]

    # Visualization: KL Divergence
    st.subheader("KL Divergence with Anomalies")
    plt.figure(figsize=(15, 6))
    plt.plot(kl_divergences, label="KL Divergence", color='blue')
    plt.axhline(y=threshold, color='red', linestyle='--', label="Anomaly Threshold")
    plt.scatter(anomalies, kl_divergences[anomalies], color='orange', label="Anomalies")
    plt.title("KL Divergence Across Sliding Windows")
    plt.xlabel("Window Index")
    plt.ylabel("KL Divergence")
    plt.legend()
    st.pyplot(plt)

    # Advanced Analysis: Root Cause Analysis
    st.subheader("Feature Contribution Heatmap")
    plt.figure(figsize=(12, 8))
    sns.heatmap(feature_contributions.T, cmap="coolwarm", cbar=True, xticklabels=False)
    plt.title("Feature Contribution to KL Divergence Across Sliding Windows")
    plt.xlabel("Window Index")
    plt.ylabel("Feature Index")
    st.pyplot(plt)

    # Feature-level anomaly scores
    st.subheader("Feature-Level Anomaly Scores")
    feature_anomaly_scores = feature_contributions[anomalies, :].mean(axis=0)
    feature_names = data.columns
    scores_df = pd.DataFrame({
        "Feature": feature_names,
        "Anomaly Score": feature_anomaly_scores
    }).sort_values(by="Anomaly Score", ascending=False)
    st.dataframe(scores_df)

    plt.figure(figsize=(10, 6))
    plt.barh(scores_df["Feature"], scores_df["Anomaly Score"], color='skyblue')
    plt.xlabel("Anomaly Score")
    plt.ylabel("Feature")
    plt.title("Feature-Level Anomaly Scores")
    st.pyplot(plt)

    # Dimensionality Reduction: PCA Visualization
    st.subheader("PCA Visualization of Anomalies")
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(normalized_data)
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5, label="Normal")
    plt.scatter(
        reduced_data[anomalies, 0], reduced_data[anomalies, 1],
        color="red", label="Anomalies", edgecolor="black"
    )
    plt.title("PCA Visualization of Anomalies")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    st.pyplot(plt)

    # Results
    st.subheader("Detected Anomalies")
    anomaly_windows = [{"Window": i, "Start Index": i * step_size, "End Index": i * step_size + window_size} for i in anomalies]
    st.dataframe(pd.DataFrame(anomaly_windows))
