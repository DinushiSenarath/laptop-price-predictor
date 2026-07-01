import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the exported model and dataset
model = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Make the webpage wide to fit all our new features beautifully
st.set_page_config(layout="wide")

st.title("Professional Laptop Price Predictor 💻")
st.write("Select your full hardware specifications below to get a highly accurate retail price estimate.")

# --- NEW FEATURE: Organized Dashboard Layout ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🖥️ Core Specs")
    company = st.selectbox('Brand', df['Company'].unique())
    os = st.selectbox('Operating System', df['OS'].unique())
    weight = st.number_input('Weight (in kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    touchscreen = st.selectbox('Touchscreen', df['Touchscreen'].unique())

with col2:
    st.subheader("⚙️ Processing Power")
    cpu_company = st.selectbox('CPU Brand', df['CPU_company'].unique())
    cpu_model = st.selectbox('CPU Model', df['CPU_model'].unique())
    cpu_freq = st.selectbox('CPU Frequency (GHz)', df['CPU_freq'].unique())
    gpu_model = st.selectbox('GPU Model', df['GPU_model'].unique())
    ram = st.selectbox('RAM (in GB)', df['Ram'].sort_values().unique())

with col3:
    st.subheader("💾 Storage & Display")
    primary_storage = st.selectbox('Primary Storage Size (GB)', df['PrimaryStorage'].unique())
    primary_storage_type = st.selectbox('Primary Storage Type', df['PrimaryStorageType'].unique())
    secondary_storage = st.selectbox('Secondary Storage Size (GB)', df['SecondaryStorage'].unique())
    inches = st.number_input('Screen Size (Inches)', min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    screen_w = st.selectbox('Screen Width (Pixels)', df['ScreenW'].unique())
    screen_h = st.selectbox('Screen Height (Pixels)', df['ScreenH'].unique())
    ips_panel = st.selectbox('IPS Display', df['IPSpanel'].unique())

st.markdown("---")

# Currency Selection
currency = st.radio("Select Display Currency", ["Euros (€)", "Sri Lankan Rupees (LKR)"])

# 3. Create a Prediction Button
if st.button('Predict Price'):
    # Grab a background template of the selected brand to fill in any minor missing gaps
    query_data = df[df['Company'] == company].iloc[0:1].copy()
    
    # Update the template with ALL the specific user inputs we collected above
    query_data['Company'] = company
    query_data['OS'] = os
    query_data['Weight'] = weight
    query_data['Touchscreen'] = touchscreen
    query_data['CPU_company'] = cpu_company
    query_data['CPU_model'] = cpu_model
    query_data['CPU_freq'] = cpu_freq
    query_data['GPU_model'] = gpu_model
    query_data['Ram'] = ram
    query_data['PrimaryStorage'] = primary_storage
    query_data['PrimaryStorageType'] = primary_storage_type
    query_data['SecondaryStorage'] = secondary_storage
    query_data['Inches'] = inches
    query_data['ScreenW'] = screen_w
    query_data['ScreenH'] = screen_h
    query_data['IPSpanel'] = ips_panel
    
    # Drop the target and text columns just like we did in Phase 4 training
    query_features = query_data.drop(columns=['Price_euros', 'Product'], errors='ignore')
    
    # 4. Make the prediction
    predicted_price_euros = model.predict(query_features)[0]
    
    # 5. Handle Currency Conversion and Display
    st.markdown("---")
    st.subheader("💻 Predicted Laptop Price")
    
    # Create a nice container for the result
    with st.container():
        euro_price = f"€{predicted_price_euros:,.2f}"
        lkr_price = f"Rs. {predicted_price_euros * 330:,.2f}"
        
        # Big, bold price display
        st.markdown(f"### {euro_price}  /  {lkr_price}")
        st.markdown("✓ Predicted using **XGBoost Regressor**")

   # --- POLISHED VISUALIZATION ---
    st.markdown("---")
    st.subheader("📊 Features That Most Influence Laptop Price")
    
    regressor = model.named_steps['regressor']
    preprocessor = model.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_
    
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    
    # --- CONSISTENT RENAME DICT ---
    rename_dict = {
        'remainder__Ram': 'RAM (GB)',
        'remainder__CPU_freq': 'CPU Frequency',
        'remainder__ScreenW': 'Screen Width',
        'cat__PrimaryStorageType_SSD': 'Primary Storage (SSD)',
        'cat__SecondaryStorageType_SSD': 'Secondary Storage (SSD)',
        'cat__TypeName_Workstation': 'Laptop Type: Workstation',
        'cat__TypeName_Notebook': 'Laptop Type: Notebook',
        'cat__GPU_model_GeForce GTX 1050 Ti': 'GPU (GTX 1050 Ti)',
        'cat__GPU_model_GeForce GTX 1070': 'GPU (GTX 1070)',
        'cat__Screen_Standard': 'Screen (Standard)'
    }
    
    feat_imp['Feature'] = feat_imp['Feature'].replace(rename_dict)
    feat_imp = feat_imp.sort_values(by='Importance', ascending=True).tail(10)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='#3182ce') 
    
    ax.set_title('Features That Most Influence Laptop Price', color='white', pad=15)
    ax.tick_params(colors='white')
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    
    # Hide x-axis ticks for minimalist presentation
    ax.set_xticks([])
    ax.set_xlabel('Relative Importance Score', color='white')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    
    plt.tight_layout()
    st.pyplot(fig)
    # Display Model Metrics
    st.markdown("---")
    st.subheader("⚙️ Under the Hood: Model Performance")
    st.write("Transparency is key in machine learning. This prediction is powered by an **XGBoost Regressor** with the following accuracy metrics on unseen testing data:")
    
    metric1, metric2, metric3 = st.columns(3)
    
    with metric1:
        st.metric(label="Algorithm", value="XGBoost")
        st.caption("Gradient boosted decision trees.")
        
    with metric2:
        st.metric(label="R² Score", value="0.8886")
        st.caption("Explains ~89% of the variation in laptop prices.")
        
    with metric3:
        st.metric(label="MAE", value="€167.48")
        st.caption("On average, predictions differ from the actual price by about €167.")

# --- MODEL COMPARISON ---
st.markdown("---")
with st.expander("🔍 Why did we choose XGBoost? (Model Comparison)"):
    st.write("To ensure the highest accuracy, I tested multiple regression models. XGBoost provided the best performance on this dataset.")
    
    # Updated with your actual experiment results
    comparison_data = {
        "Model": ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"],
        "R² Score": [0.81, 0.78, 0.87, 0.89] 
    }
    comparison_df = pd.DataFrame(comparison_data)
    
    st.table(comparison_df)
    st.write("XGBoost was selected as the final model because it demonstrated superior performance in capturing complex relationships within the laptop hardware data.")

    # --- FOOTER ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Developed by <b>Dinushi Senarath</b></p>
        <p>Machine Learning • Python • Streamlit • XGBoost</p>
        <p><a href="https://github.com/DinushiSenarath/laptop-price-predictor" target="_blank">Check out the source code on GitHub</a></p>
    </div>
    """,
    unsafe_allow_html=True
)