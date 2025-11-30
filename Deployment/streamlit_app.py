import streamlit as st
import pandas as pd
import joblib
import numpy as np
from tensorflow import keras

# -----------------------------
# Load Saved Components
# -----------------------------
encoders = joblib.load("label_encoders.joblib")
feature_order = joblib.load("feature_order.joblib")

# Models
rf = joblib.load("RandomForest_model.joblib")
dt = joblib.load("DecisionTree_model.joblib")
svr = joblib.load("SVR_model.joblib")
nn = keras.models.load_model("KerasModel.keras")

# Scalers
svr_scaler = joblib.load("standard_scaler_sklearn.joblib")
nn_scaler_x = joblib.load("standard_scaler_X_keras.joblib")
nn_scaler_y = joblib.load("standard_scaler_y_keras.joblib")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Superstore Sales Prediction – All Models")
st.write("Enter order details, and get predictions from Random Forest, Decision Tree, SVR, and Neural Network.")

st.subheader("Enter Order Details")

# -----------------------------
# Input Fields
# -----------------------------
ship_mode = st.selectbox("Ship Mode", encoders["Ship Mode"].classes_)
segment = st.selectbox("Segment", encoders["Segment"].classes_)
region = st.selectbox("Region", encoders["Region"].classes_)

category = st.selectbox("Category", encoders["Category"].classes_)
subcat = st.selectbox("Sub-Category", encoders["Sub-Category"].classes_)

product_name = st.selectbox("Product Name", encoders["Product Name"].classes_)
city = st.selectbox("City", encoders["City"].classes_)
state = st.selectbox("State", encoders["State"].classes_)

postal_code = st.number_input("Postal Code", 1000, 99999)
quantity = st.number_input("Quantity", 1, 50)
discount = st.number_input("Discount", 0.0, 0.9, step=0.01)
profit = st.number_input("Profit", -5000.0, 5000.0)

order_year = st.number_input("Order Year", 2014, 2024)
order_month = st.number_input("Order Month", 1, 12)
shipping_delay = st.number_input("Shipping Delay (days)", 0, 30)

# -----------------------------
# Combined Prediction
# -----------------------------
if st.button("Predict with All Models"):

    # Step 1: Construct Input Row
    row = {
        "Ship Mode": ship_mode,
        "Segment": segment,
        "Region": region,
        "Category": category,
        "Sub-Category": subcat,
        "Product Name": product_name,
        "City": city,
        "State": state,
        "Postal Code": postal_code,
        "Quantity": quantity,
        "Discount": discount,
        "Profit_clipped": profit,
        "Quantity_clipped": quantity,
        "OrderYear": order_year,
        "OrderMonth": order_month,
        "ShippingDelay": shipping_delay
    }

    df = pd.DataFrame([row])

    # Step 2: Label Encoding
    for col in encoders:
        df[col] = encoders[col].transform(df[col])

    # Step 3: Proper Feature Order
    df = df[feature_order]

    # -----------------------------
    # Model Predictions
    # -----------------------------

    # Random Forest
    pred_rf = rf.predict(df)[0]

    # Decision Tree
    pred_dt = dt.predict(df)[0]

    # SVR (scaled)
    df_scaled_svr = svr_scaler.transform(df)
    pred_svr = svr.predict(df_scaled_svr)[0]

    # Neural Network (xy scaling)
    x_scaled_nn = nn_scaler_x.transform(df)
    y_scaled_pred = nn.predict(x_scaled_nn)
    pred_nn = nn_scaler_y.inverse_transform(y_scaled_pred)[0][0]

    # -----------------------------
    # Display Results
    # -----------------------------
    st.subheader("🔮 Model Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"🌳 Random Forest: **${pred_rf:.2f}**")
        st.success(f"📘 Decision Tree: **${pred_dt:.2f}**")

    with col2:
        st.warning(f"🔵 SVR: **${pred_svr:.2f}**")
        st.error(f"🤖 Neural Network: **${pred_nn:.2f}**")