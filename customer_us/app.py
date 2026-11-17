import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f8ff 0%, #edf7f4 100%);
    }
    [data-testid="stSidebar"] {
        background-color: #e0e7ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------
# LOAD MODELS
# ---------------------------------

@st.cache_resource
def load_models():

    kmeans = joblib.load("models/kmeans_model.pkl")

    pca = joblib.load("models/pca_model.pkl")

    scaler = joblib.load("models/scaler.pkl")

    return kmeans, pca, scaler


kmeans, pca, scaler = load_models()


# ---------------------------------
# APPLICATION TITLE
# ---------------------------------

st.title("🛍️ Customer Segmentation System")

st.write(
    "Analyze customer behaviour and identify "
    "customer groups using K-Means Clustering and PCA."
)

st.divider()


# ---------------------------------
# SIDEBAR INPUT
# ---------------------------------

st.sidebar.header("Enter Customer Details")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

income = st.sidebar.number_input(
    "Annual Income (k$)",
    min_value=1,
    max_value=200,
    value=50
)

spending = st.sidebar.slider(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)


# ---------------------------------
# PREDICTION BUTTON
# ---------------------------------

if st.sidebar.button("Predict Customer Segment"):

    # Create input dataframe

    input_data = pd.DataFrame(
        [[age, income, spending]],
        columns=[
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    # Step 1: Scaling

    scaled_data = scaler.transform(input_data)

    # Step 2: PCA transformation

    pca_data = pca.transform(scaled_data)

    # Step 3: KMeans prediction

    cluster = kmeans.predict(scaled_data)[0]

    # ---------------------------------
    # DISPLAY RESULTS
    # ---------------------------------

    st.success("Customer Segmentation Completed!")

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Cluster",
            f"Cluster {cluster}"
        )

    with col2:

        st.metric(
            "Customer Gender",
            gender
        )

    st.divider()

    st.subheader("Customer Information")

    st.dataframe(input_data, use_container_width=True)

    # ---------------------------------
    # PCA VISUALIZATION
    # ---------------------------------

    st.subheader("Customer Clusters in PCA Space")

    cluster_centers_pca = pca.transform(kmeans.cluster_centers_)
    n_clusters = len(kmeans.cluster_centers_)
    cluster_cmap = plt.get_cmap("tab10", n_clusters)

    all_x = np.append(cluster_centers_pca[:, 0], pca_data[0, 0])
    all_y = np.append(cluster_centers_pca[:, 1], pca_data[0, 1])
    x_padding = max((all_x.max() - all_x.min()) * 0.15, 0.5)
    y_padding = max((all_y.max() - all_y.min()) * 0.15, 0.5)
    x_limits = (all_x.min() - x_padding, all_x.max() + x_padding)
    y_limits = (all_y.min() - y_padding, all_y.max() + y_padding)

    grid_x, grid_y = np.meshgrid(
        np.linspace(*x_limits, 140),
        np.linspace(*y_limits, 140)
    )
    pca_grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    scaled_grid = pca.inverse_transform(pca_grid)
    region_labels = kmeans.predict(scaled_grid).reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contourf(
        grid_x,
        grid_y,
        region_labels,
        levels=np.arange(n_clusters + 1) - 0.5,
        cmap=cluster_cmap,
        alpha=0.16
    )
    ax.contour(
        grid_x,
        grid_y,
        region_labels,
        levels=np.arange(n_clusters - 1) + 0.5,
        colors="#64748b",
        linewidths=0.7,
        alpha=0.65
    )

    for cluster_id, center in enumerate(cluster_centers_pca):
        ax.scatter(
            center[0],
            center[1],
            color=cluster_cmap(cluster_id),
            s=150,
            marker="o",
            edgecolors="white",
            linewidths=1.5,
            label=f"Cluster {cluster_id} center",
            zorder=3
        )
        ax.annotate(
            f"Cluster {cluster_id}",
            (center[0], center[1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="#1f2937"
        )

    ax.scatter(
        pca_data[0, 0],
        pca_data[0, 1],
        s=280,
        marker="*",
        color="#111827",
        edgecolors="white",
        linewidths=1,
        label=f"New customer (Cluster {cluster})",
        zorder=4
    )
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Cluster Regions, Centers, and Your Customer")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    st.pyplot(fig)

    st.caption(
        "Colored regions show cluster predictions on the 2D PCA plane. "
        "The customer's displayed cluster is predicted using all three scaled features."
    )

    st.info(
        f"The customer belongs to Cluster {cluster}."
    )


# ---------------------------------
# FOOTER
# ---------------------------------

st.divider()

st.caption(
    "Customer Segmentation | "
    "K-Means Clustering + PCA | Streamlit"
)