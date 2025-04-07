# RegApp_DH.py

import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# -----------------------
# 1. Title and Description
# -----------------------
st.title("3D House Price Prediction (Linear vs. Quadratic Model)")

st.write(
    """
    This app trains two regression models (linear and quadratic) on a small
    sample dataset. It displays:
    - Model performance (MSE, R²)
    - A 3D interactive plot with the data points and regression surfaces
    - A simple prediction tool for house size & age.
    """
)

# -----------------------
# 2. Define the Dataset
# -----------------------
y = [63, 65.1, 69.9, 76.8, 73.9, 77.9, 74.9, 78, 79, 63.4, 79.5, 83.9, 79.7,
     84.5, 96, 109.5, 102.5, 121, 104.9, 128, 129, 117.9, 140]
x1 = [1605, 2489, 1553, 2404, 1884, 1558, 1748, 3105, 1682, 2470, 1820,
      2143, 2121, 2485, 2300, 2714, 2463, 3076, 3048, 3267, 3069, 4765, 4540]
x2 = [35, 45, 20, 32, 25, 14, 8, 10, 28, 30, 2, 6, 14, 9, 19, 4, 5, 7, 3, 6,
      10, 11, 8]

# Prepare feature matrix
X = np.column_stack((x1, x2))
y_array = np.array(y)

# -----------------------
# 3. Fit Linear Model
# -----------------------
model_linear = LinearRegression().fit(X, y_array)
b0 = model_linear.intercept_
b1, b2 = model_linear.coef_

# Generate linear equation string for annotation
equation_linear = f"y = {b0:.2f}"
equation_linear += f" {'+' if b1>=0 else '-'} {abs(b1):.2f} x1"
equation_linear += f" {'+' if b2>=0 else '-'} {abs(b2):.2f} x2"

# -----------------------
# 4. Fit Quadratic Model
# -----------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
model_quad = LinearRegression().fit(X_poly, y_array)
coef_quad = model_quad.coef_
intercept_quad = model_quad.intercept_

display_terms = ["x1", "x2", "x1<sup>2</sup>", "x1 x2", "x2<sup>2</sup>"]
equation_quad = f"y = {intercept_quad:.2f}"

for c, term in zip(coef_quad, display_terms):
    equation_quad += f" {'+' if c>=0 else '-'} {abs(c):.2f} {term}"

# -----------------------
# 5. Evaluate Performance
# -----------------------
lin_preds = model_linear.predict(X)
quad_preds = model_quad.predict(X_poly)

mse_lin = mean_squared_error(y_array, lin_preds)
r2_lin = r2_score(y_array, lin_preds)
mse_quad = mean_squared_error(y_array, quad_preds)
r2_quad = r2_score(y_array, quad_preds)

st.subheader("Model Performance")
st.write(
    f"**Linear Model**: MSE = {mse_lin:.2f}, R² = {r2_lin:.2f}\n\n"
    f"**Quadratic Model**: MSE = {mse_quad:.2f}, R² = {r2_quad:.2f}\n"
)

# -----------------------
# 6. Create 3D Plotly Figure
# -----------------------
# Create meshgrid for surfaces
x1_range = np.linspace(min(x1), max(x1), 20)
x2_range = np.linspace(min(x2), max(x2), 20)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

# Linear predictions for surface
y_pred_linear = (
    b0 + b1 * x1_mesh + b2 * x2_mesh
)

# Quadratic predictions for surface
mesh_points = np.column_stack((x1_mesh.ravel(), x2_mesh.ravel()))
mesh_points_poly = poly.transform(mesh_points)
y_pred_quad = model_quad.predict(mesh_points_poly).reshape(x1_mesh.shape)

# Create Scatter3D for raw data
hover_text = [
    f"Price: {price_val}k<br>Size: {size_val} sq ft<br>Age: {age_val} years"
    for (price_val, size_val, age_val) in zip(y, x1, x2)
]

scatter_data = go.Scatter3d(
    x=x1, y=x2, z=y,
    mode="markers",
    marker=dict(size=5, color="blue"),
    name="Data Points",
    hovertext=hover_text,
    hoverinfo="text"
)

# Create Surfaces
surface_linear = go.Surface(
    x=x1_mesh, y=x2_mesh, z=y_pred_linear,
    opacity=0.8,
    colorscale="Viridis",
    showscale=False,
    visible=False,
    name="Linear Regression Plane"
)

surface_quad = go.Surface(
    x=x1_mesh, y=x2_mesh, z=y_pred_quad,
    opacity=0.8,
    colorscale="Plasma",
    showscale=False,
    visible=False,
    name="Quadratic Regression Surface"
)

# Annotations for equations
linear_annotation = dict(
    text=equation_linear,
    xref="paper", yref="paper",
    x=0.5, y=1.15,
    showarrow=False,
    font=dict(size=14)
)

quadratic_annotation = dict(
    text=equation_quad,
    xref="paper", yref="paper",
    x=0.5, y=1.25,
    showarrow=False,
    font=dict(size=14)
)

# Dropdown menu
updatemenus = [
    {
        "buttons": [
            {
                "method": "update",
                "args": [
                    {"visible": [True, True, False]},
                    {"annotations": [linear_annotation]}
                ],
                "label": "Show Linear Plane"
            },
            {
                "method": "update",
                "args": [
                    {"visible": [True, False, True]},
                    {"annotations": [quadratic_annotation]}
                ],
                "label": "Show Quadratic Surface"
            },
            {
                "method": "update",
                "args": [
                    {"visible": [True, True, True]},
                    {"annotations": [linear_annotation, quadratic_annotation]}
                ],
                "label": "Show Both"
            },
            {
                "method": "update",
                "args": [
                    {"visible": [True, False, False]},
                    {"annotations": []}
                ],
                "label": "Hide Surfaces"
            }
        ],
        "direction": "down",
        "showactive": True,
        "x": 0.1,
        "xanchor": "left",
        "y": 1.35,
        "yanchor": "top"
    }
]

fig = go.Figure(data=[scatter_data, surface_linear, surface_quad])
fig.update_layout(
    updatemenus=updatemenus,
    scene=dict(
        xaxis_title="House Size (sq ft)",
        yaxis_title="Age (years)",
        zaxis_title="Market Price (k $)"
    ),
    title=dict(
        text="3D Scatter Plot with Linear and Quadratic Models",
        x=0.5,
        y=0.95,
        xanchor="center",
        yanchor="top"
    ),
    width=900,
    height=700,
    showlegend=True
)

# Display the Plotly figure in Streamlit
st.subheader("3D Interactive Plot")
st.write("Select from the dropdown to show/hide each regression surface.")
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 7. User Inputs for Prediction
# -----------------------
st.subheader("Try a Custom Prediction")
user_size = st.number_input("Enter the house size (sq ft):", min_value=0.0, step=100.0)
user_age = st.number_input("Enter the house age (years):", min_value=0.0, step=1.0)

if st.button("Predict"):
    # Linear Prediction
    pred_linear = model_linear.predict([[user_size, user_age]])[0]
    # Quadratic Prediction
    pred_quad = model_quad.predict(poly.transform([[user_size, user_age]]))[0]

    st.write(f"**Linear Model**: {pred_linear:.2f} thousand dollars")
    st.write(f"**Quadratic Model**: {pred_quad:.2f} thousand dollars")
