
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# Laptop Price Predictor v2.0 - Portfolio Edition
# Author: Dinushi Senarath
# ============================================================

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Constants
# -----------------------------
EURO_TO_LKR = 330
R2_SCORE = 0.8886
MAE_EURO = 167.48
RMSE_EURO = 226.35
MODEL_NAME = "XGBoost Regressor"
GITHUB_URL = "https://github.com/DinushiSenarath/laptop-price-predictor"
LINKEDIN_URL = "https://www.linkedin.com/"
APP_VERSION = "2.0"

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .hero-box {
        padding: 2.2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #1f2937 0%, #111827 50%, #0f172a 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 45px rgba(0,0,0,0.25);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #cbd5e1;
        max-width: 900px;
        line-height: 1.6;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        margin: 0.25rem 0.2rem 0.1rem 0;
        border-radius: 999px;
        background-color: rgba(59,130,246,0.15);
        color: #bfdbfe;
        border: 1px solid rgba(59,130,246,0.35);
        font-size: 0.82rem;
        font-weight: 600;
    }

    .section-card {
        padding: 1.2rem;
        border-radius: 18px;
        background-color: #111827;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 12px 28px rgba(0,0,0,0.18);
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 18px;
        background: linear-gradient(145deg, #111827, #0f172a);
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        min-height: 135px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.65rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .metric-caption {
        color: #cbd5e1;
        font-size: 0.82rem;
    }

    .price-card {
        padding: 1.5rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #064e3b, #065f46, #047857);
        border: 1px solid rgba(16,185,129,0.35);
        text-align: center;
        box-shadow: 0 18px 40px rgba(0,0,0,0.28);
        margin-bottom: 1rem;
    }

    .price-title {
        color: #d1fae5;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .price-value {
        color: #ffffff;
        font-size: 2.7rem;
        font-weight: 900;
        margin-bottom: 0.2rem;
    }

    .warning-card {
        padding: 1rem;
        border-radius: 16px;
        background-color: rgba(245,158,11,0.12);
        border: 1px solid rgba(245,158,11,0.35);
        color: #fde68a;
        margin-bottom: 1rem;
    }

    .success-card {
        padding: 1rem;
        border-radius: 16px;
        background-color: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.35);
        color: #a7f3d0;
        margin-bottom: 1rem;
    }

    .footer {
        text-align: center;
        padding: 2rem 1rem;
        color: #94a3b8;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin-top: 2rem;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    div.stButton > button:first-child {
        width: 100%;
        border-radius: 14px;
        height: 3.3rem;
        font-weight: 800;
        font-size: 1.05rem;
        border: none;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
    }

    div.stButton > button:first-child:hover {
        border: none;
        color: white;
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load Model and Dataset
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("df.pkl", "rb") as df_file:
        df = pickle.load(df_file)

    return model, df


model, df = load_artifacts()

# -----------------------------
# Helper Functions
# -----------------------------
def safe_sorted_unique(dataframe, column):
    values = dataframe[column].dropna().unique()
    try:
        return sorted(values)
    except TypeError:
        return values


def format_euro(value):
    return f"€{value:,.2f}"


def format_lkr(value):
    return f"Rs. {value * EURO_TO_LKR:,.2f}"


def get_price_category(price):
    if price < 500:
        return "Budget Laptop"
    elif price < 900:
        return "Mid-range Laptop"
    elif price < 1400:
        return "Premium Laptop"
    elif price < 2200:
        return "High-end Laptop"
    else:
        return "Workstation / Ultra Premium"


def get_confidence_level(warnings_count):
    base_confidence = int(R2_SCORE * 100)

    if warnings_count == 0:
        return "High", base_confidence
    elif warnings_count == 1:
        return "Medium", max(base_confidence - 8, 70)
    else:
        return "Lower", max(base_confidence - 15, 60)


def get_prediction_range(price, warnings_count):
    margin = 0.10 if warnings_count == 0 else 0.15 if warnings_count == 1 else 0.20
    return price * (1 - margin), price * (1 + margin)


def make_query_dataframe(inputs):
    template = df[df["Company"] == inputs["Company"]].iloc[0:1].copy()

    for key, value in inputs.items():
        if key in template.columns:
            template[key] = value

    query_features = template.drop(columns=["Price_euros", "Product"], errors="ignore")
    return query_features


def validate_inputs(inputs):
    warnings = []

    if inputs["Ram"] <= 4 and ("i7" in str(inputs["CPU_model"]).lower() or "i9" in str(inputs["CPU_model"]).lower()):
        warnings.append("Low RAM with a high-performance CPU is uncommon.")

    if inputs["Ram"] <= 4 and inputs["PrimaryStorage"] >= 1000:
        warnings.append("Very high storage with very low RAM is unusual.")

    if inputs["Weight"] < 1.0 and inputs["Inches"] >= 15.0:
        warnings.append("Large screen with very low weight is uncommon.")

    if inputs["ScreenW"] >= 2560 and inputs["Ram"] <= 4:
        warnings.append("High-resolution display with low RAM may be rare in the dataset.")

    company_rows = df[df["Company"] == inputs["Company"]]
    if not company_rows.empty:
        same_cpu = company_rows[company_rows["CPU_model"] == inputs["CPU_model"]]
        if same_cpu.empty:
            warnings.append("This CPU model is rare for the selected brand in the dataset.")

    similar_rows = df[
        (df["Company"] == inputs["Company"]) &
        (df["Ram"] == inputs["Ram"]) &
        (df["CPU_company"] == inputs["CPU_company"]) &
        (df["PrimaryStorageType"] == inputs["PrimaryStorageType"])
    ]

    if len(similar_rows) < 3:
        warnings.append("This hardware combination appears rarely in the dataset.")

    return warnings


def get_similar_laptops(inputs, predicted_price, top_n=5):
    data = df.copy()

    data["score"] = 0.0

    categorical_cols = [
        "Company", "TypeName", "OS", "CPU_company", "CPU_model",
        "GPU_company", "GPU_model", "PrimaryStorageType", "SecondaryStorageType"
    ]

    for col in categorical_cols:
        if col in data.columns and col in inputs:
            data["score"] += (data[col] == inputs[col]).astype(float)

    numeric_cols = ["Ram", "Weight", "Inches", "ScreenW", "ScreenH", "CPU_freq", "PrimaryStorage", "SecondaryStorage"]

    for col in numeric_cols:
        if col in data.columns and col in inputs:
            col_range = data[col].max() - data[col].min()
            if col_range == 0:
                col_range = 1
            data["score"] += 1 - (abs(data[col] - inputs[col]) / col_range)

    price_range = data["Price_euros"].max() - data["Price_euros"].min()
    if price_range == 0:
        price_range = 1

    data["score"] += 1 - (abs(data["Price_euros"] - predicted_price) / price_range)

    result = data.sort_values("score", ascending=False).head(top_n)
    return result[["Company", "Product", "TypeName", "Ram", "CPU_model", "PrimaryStorage", "PrimaryStorageType", "Price_euros"]]


def plot_feature_importance(model):
    try:
        regressor = model.named_steps["regressor"]
        preprocessor = model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        importances = regressor.feature_importances_

        feat_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        rename_dict = {
            "remainder__Ram": "RAM",
            "remainder__CPU_freq": "CPU Frequency",
            "remainder__ScreenW": "Screen Width",
            "remainder__ScreenH": "Screen Height",
            "remainder__Weight": "Weight",
            "remainder__Inches": "Screen Size",
            "cat__PrimaryStorageType_SSD": "Primary Storage: SSD",
            "cat__SecondaryStorageType_SSD": "Secondary Storage: SSD",
            "cat__TypeName_Workstation": "Type: Workstation",
            "cat__TypeName_Notebook": "Type: Notebook",
        }

        feat_imp["Feature"] = feat_imp["Feature"].replace(rename_dict)
        feat_imp["Feature"] = (
            feat_imp["Feature"]
            .str.replace("cat__", "", regex=False)
            .str.replace("remainder__", "", regex=False)
            .str.replace("_", " ", regex=False)
        )

        feat_imp = feat_imp.sort_values("Importance", ascending=True).tail(10)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(feat_imp["Feature"], feat_imp["Importance"])

        ax.set_title("Top Features Influencing Laptop Price", pad=18, fontsize=14, fontweight="bold")
        ax.set_xlabel("Relative Importance Score")
        ax.set_ylabel("")

        for i, value in enumerate(feat_imp["Importance"]):
            ax.text(value, i, f" {value:.3f}", va="center", fontsize=9)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        return fig

    except Exception as error:
        st.warning(f"Feature importance could not be displayed: {error}")
        return None


def plot_model_comparison():
    comparison_df = pd.DataFrame({
        "Model": ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"],
        "R² Score": [0.81, 0.78, 0.87, R2_SCORE],
        "MAE (€)": [210.50, 245.30, 178.20, MAE_EURO]
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(comparison_df["Model"], comparison_df["R² Score"])
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison by R² Score", pad=15, fontsize=14, fontweight="bold")
    ax.set_ylabel("R² Score")

    for i, value in enumerate(comparison_df["R² Score"]):
        ax.text(i, value + 0.015, f"{value:.4f}", ha="center", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=20)
    plt.tight_layout()

    return fig, comparison_df


def display_metric_card(label, value, caption):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("💻 Project Info")
    st.markdown("**Laptop Price Predictor v2.0**")
    st.caption("Machine Learning Regression Web App")

    st.markdown("---")
    st.markdown("### 🧠 Model")
    st.write(f"**Algorithm:** {MODEL_NAME}")
    st.write(f"**R² Score:** {R2_SCORE}")
    st.write(f"**MAE:** {format_euro(MAE_EURO)}")

    st.markdown("---")
    st.markdown("### 🛠 Tech Stack")
    st.write("Python")
    st.write("Pandas / NumPy")
    st.write("Scikit-learn")
    st.write("XGBoost")
    st.write("Streamlit")
    st.write("Matplotlib")

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown(f"[GitHub Repository]({GITHUB_URL})")
    st.markdown(f"[LinkedIn Profile]({LINKEDIN_URL})")

# -----------------------------
# Hero Section
# -----------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">Laptop Price Predictor 💻</div>
        <div class="hero-subtitle">
            A machine learning web application that predicts laptop prices using hardware specifications,
            brand information, display features, processor details, storage type, and GPU information.
        </div>
        <br>
        <span class="badge">Machine Learning</span>
        <span class="badge">Regression</span>
        <span class="badge">XGBoost</span>
        <span class="badge">Streamlit</span>
        <span class="badge">Portfolio Project</span>
        <span class="badge">R² Score: 0.8886</span>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Dataset Dashboard
# -----------------------------
st.subheader("📌 Dataset Overview")

d1, d2, d3, d4, d5 = st.columns(5)

with d1:
    display_metric_card("Laptops", f"{len(df):,}", "Records in dataset")

with d2:
    display_metric_card("Brands", f"{df['Company'].nunique()}", "Unique manufacturers")

with d3:
    display_metric_card("Features", f"{df.shape[1] - 1}", "Input and metadata columns")

with d4:
    display_metric_card("Average Price", format_euro(df["Price_euros"].mean()), "Dataset mean")

with d5:
    display_metric_card("Median Price", format_euro(df["Price_euros"].median()), "Typical laptop price")

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("⚙️ Enter Laptop Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🖥️ Core Details")
    company = st.selectbox("Brand", safe_sorted_unique(df, "Company"))
    product_type = st.selectbox("Laptop Type", safe_sorted_unique(df, "TypeName"))
    os = st.selectbox("Operating System", safe_sorted_unique(df, "OS"))
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    touchscreen = st.selectbox("Touchscreen", safe_sorted_unique(df, "Touchscreen"))
    retina = st.selectbox("Retina Display", safe_sorted_unique(df, "RetinaDisplay"))

with col2:
    st.markdown("### ⚙️ Performance")
    cpu_company = st.selectbox("CPU Brand", safe_sorted_unique(df, "CPU_company"))
    cpu_model = st.selectbox("CPU Model", safe_sorted_unique(df, "CPU_model"))
    cpu_freq = st.selectbox("CPU Frequency (GHz)", safe_sorted_unique(df, "CPU_freq"))
    ram = st.selectbox("RAM (GB)", safe_sorted_unique(df, "Ram"))
    gpu_company = st.selectbox("GPU Brand", safe_sorted_unique(df, "GPU_company"))
    gpu_model = st.selectbox("GPU Model", safe_sorted_unique(df, "GPU_model"))

with col3:
    st.markdown("### 💾 Storage & Display")
    primary_storage = st.selectbox("Primary Storage (GB)", safe_sorted_unique(df, "PrimaryStorage"))
    primary_storage_type = st.selectbox("Primary Storage Type", safe_sorted_unique(df, "PrimaryStorageType"))
    secondary_storage = st.selectbox("Secondary Storage (GB)", safe_sorted_unique(df, "SecondaryStorage"))
    secondary_storage_type = st.selectbox("Secondary Storage Type", safe_sorted_unique(df, "SecondaryStorageType"))
    inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    screen = st.selectbox("Screen Type", safe_sorted_unique(df, "Screen"))
    screen_w = st.selectbox("Screen Width (Pixels)", safe_sorted_unique(df, "ScreenW"))
    screen_h = st.selectbox("Screen Height (Pixels)", safe_sorted_unique(df, "ScreenH"))
    ips_panel = st.selectbox("IPS Panel", safe_sorted_unique(df, "IPSpanel"))

st.markdown("---")

# -----------------------------
# Prediction Controls
# -----------------------------
currency = st.radio(
    "Display Currency",
    ["Both", "Euros (€)", "Sri Lankan Rupees (LKR)"],
    horizontal=True
)

inputs = {
    "Company": company,
    "TypeName": product_type,
    "OS": os,
    "Weight": weight,
    "Touchscreen": touchscreen,
    "RetinaDisplay": retina,
    "CPU_company": cpu_company,
    "CPU_model": cpu_model,
    "CPU_freq": cpu_freq,
    "Ram": ram,
    "GPU_company": gpu_company,
    "GPU_model": gpu_model,
    "PrimaryStorage": primary_storage,
    "PrimaryStorageType": primary_storage_type,
    "SecondaryStorage": secondary_storage,
    "SecondaryStorageType": secondary_storage_type,
    "Inches": inches,
    "Screen": screen,
    "ScreenW": screen_w,
    "ScreenH": screen_h,
    "IPSpanel": ips_panel,
}

predict_button = st.button("🚀 Predict Laptop Price")

# -----------------------------
# Prediction Output
# -----------------------------
if predict_button:
    with st.spinner("Analyzing specifications and running the XGBoost model..."):
        time.sleep(0.8)

        query_features = make_query_dataframe(inputs)
        predicted_price_euros = float(model.predict(query_features)[0])

        warnings = validate_inputs(inputs)
        confidence_label, confidence_score = get_confidence_level(len(warnings))
        lower_price, upper_price = get_prediction_range(predicted_price_euros, len(warnings))
        category = get_price_category(predicted_price_euros)

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if warnings:
        st.markdown(
            "<div class='warning-card'><b>⚠️ Prediction Reliability Notice</b><br>"
            + "<br>".join([f"• {warning}" for warning in warnings])
            + "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='success-card'><b>✅ Reliable Configuration</b><br>"
            "This configuration is reasonably represented in the dataset.</div>",
            unsafe_allow_html=True
        )

    price_col1, price_col2 = st.columns(2)

    with price_col1:
        if currency in ["Both", "Euros (€)"]:
            st.markdown(
                f"""
                <div class="price-card">
                    <div class="price-title">Estimated Price in Euros</div>
                    <div class="price-value">{format_euro(predicted_price_euros)}</div>
                    <div class="metric-caption">Predicted using {MODEL_NAME}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with price_col2:
        if currency in ["Both", "Sri Lankan Rupees (LKR)"]:
            st.markdown(
                f"""
                <div class="price-card">
                    <div class="price-title">Estimated Price in Sri Lankan Rupees</div>
                    <div class="price-value">{format_lkr(predicted_price_euros)}</div>
                    <div class="metric-caption">Exchange rate used: 1 EUR = Rs. {EURO_TO_LKR}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        display_metric_card("Expected Range", f"{format_euro(lower_price)} - {format_euro(upper_price)}", "Practical prediction interval")

    with r2:
        display_metric_card("Confidence", f"{confidence_label} ({confidence_score}%)", "Based on model score and input rarity")

    with r3:
        display_metric_card("Category", category, "Price segment")

    with r4:
        display_metric_card("Model", "XGBoost", "Final selected algorithm")

    st.markdown("### 📋 Selected Hardware Summary")

    summary_df = pd.DataFrame({
        "Specification": [
            "Brand", "Type", "Operating System", "CPU", "CPU Frequency", "RAM",
            "GPU", "Storage", "Display", "Weight"
        ],
        "Selected Value": [
            company,
            product_type,
            os,
            f"{cpu_company} {cpu_model}",
            f"{cpu_freq} GHz",
            f"{ram} GB",
            f"{gpu_company} {gpu_model}",
            f"{primary_storage} GB {primary_storage_type} + {secondary_storage} GB {secondary_storage_type}",
            f"{inches} inch, {screen_w}x{screen_h}, {screen}",
            f"{weight} kg"
        ]
    })

    st.table(summary_df)

    st.markdown("### 🔎 Similar Laptops from Dataset")
    similar_laptops = get_similar_laptops(inputs, predicted_price_euros)

    similar_laptops_display = similar_laptops.copy()
    similar_laptops_display["Price_euros"] = similar_laptops_display["Price_euros"].apply(format_euro)
    similar_laptops_display = similar_laptops_display.rename(columns={
        "Company": "Brand",
        "Product": "Product",
        "TypeName": "Type",
        "Ram": "RAM",
        "CPU_model": "CPU",
        "PrimaryStorage": "Storage",
        "PrimaryStorageType": "Storage Type",
        "Price_euros": "Actual Price"
    })

    st.dataframe(similar_laptops_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📊 Feature Importance")

    fig_importance = plot_feature_importance(model)
    if fig_importance:
        st.pyplot(fig_importance)

# -----------------------------
# Model Comparison
# -----------------------------
st.markdown("---")
st.subheader("🧠 Model Performance & Comparison")

m1, m2, m3, m4 = st.columns(4)

with m1:
    display_metric_card("Final Model", "XGBoost", "Gradient boosted trees")

with m2:
    display_metric_card("R² Score", f"{R2_SCORE:.4f}", "Explains about 89% variation")

with m3:
    display_metric_card("MAE", format_euro(MAE_EURO), "Average prediction error")

with m4:
    display_metric_card("RMSE", format_euro(RMSE_EURO), "Penalizes large errors")

with st.expander("🔍 Why XGBoost was selected"):
    st.write(
        "Several regression models were tested. XGBoost was selected because it gave the best balance "
        "between accuracy and ability to model complex relationships between laptop specifications and price."
    )

    fig_comparison, comparison_df = plot_model_comparison()
    st.pyplot(fig_comparison)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# -----------------------------
# Project Explanation
# -----------------------------
st.markdown("---")
st.subheader("📌 About This Project")

about_col1, about_col2 = st.columns(2)

with about_col1:
    st.markdown(
        """
        <div class="section-card">
        <h4>Project Objective</h4>
        <p>
        This project predicts laptop prices using machine learning regression.
        Users can enter laptop hardware specifications and receive an estimated price
        in Euros and Sri Lankan Rupees.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with about_col2:
    st.markdown(
        """
        <div class="section-card">
        <h4>Machine Learning Workflow</h4>
        <p>
        The workflow includes data preprocessing, categorical encoding, model training,
        model comparison, evaluation using R² and MAE, and deployment using Streamlit.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    f"""
    <div class="footer">
        <h3>Developed by Dinushi Senarath</h3>
        <p>Final Year Computer Engineering Undergraduate</p>
        <p>Machine Learning • Python • Scikit-learn • XGBoost • Streamlit</p>
        <p>
            <a href="{GITHUB_URL}" target="_blank">GitHub Repository</a>
            &nbsp; | &nbsp;
            <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>
        </p>
        <p class="small-muted">Laptop Price Predictor v{APP_VERSION}</p>
    </div>
    """,
    unsafe_allow_html=True
)
