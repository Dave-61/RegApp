# file: RegApp_DH.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

st.title("House Price Prediction App")

# File uploader to allow custom dataset
uploaded_file = st.file_uploader("Upload a CSV file with 'size', 'age', and 'price' columns", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Quick sanity check
    if all(col in df.columns for col in ["size", "age", "price"]):
        x1 = df["size"].values
        x2 = df["age"].values
        y = df["price"].values
    else:
        st.error("CSV must have 'size', 'age', and 'price' columns.")
        st.stop()
    
    # Prepare data
    X = np.column_stack((x1, x2))
    y = np.array(y)
    
    # Fit linear model
    model_linear = LinearRegression().fit(X, y)
    
    # Fit quadratic model
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    model_quad = LinearRegression().fit(X_poly, y)
    
    # Evaluate
    lin_preds = model_linear.predict(X)
    quad_preds = model_quad.predict(X_poly)
    mse_lin = mean_squared_error(y, lin_preds)
    r2_lin = r2_score(y, lin_preds)
    mse_quad = mean_squared_error(y, quad_preds)
    r2_quad = r2_score(y, quad_preds)
    
    st.markdown("### Model Performance (on uploaded data)")
    st.write(f"**Linear Model**: MSE={mse_lin:.2f}, R²={r2_lin:.2f}")
    st.write(f"**Quadratic Model**: MSE={mse_quad:.2f}, R²={r2_quad:.2f}")
    
    # Let user input new house for predictions
    size_input = st.number_input("Enter house size (sq ft)", min_value=0.0, step=100.0)
    age_input = st.number_input("Enter house age (years)", min_value=0.0, step=1.0)
    
    if st.button("Predict"):
        pred_lin = model_linear.predict([[size_input, age_input]])[0]
        pred_quad = model_quad.predict(poly.transform([[size_input, age_input]]))[0]
        
        st.write(f"**Linear Prediction**: {pred_lin:.2f} thousand dollars")
        st.write(f"**Quadratic Prediction**: {pred_quad:.2f} thousand dollars")
    
    # Plotly 3D plot if you like
    # ...
else:
    st.info("Please upload a CSV file.")
