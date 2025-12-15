import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("iris_random_forest.pkl", "rb"))

# Page config
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
/* Remove default white containers */
.block-container {
    padding-top: 2rem;
}

/* Background */
body {
    background-color: #0E1117;
}

/* Title */
.main-title {
    text-align: center;
    color: #9C27B0;
    font-size: 40px;
    font-weight: bold;
}

.sub-title {
    text-align: center;
    color: #9E9E9E;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Input labels */
label {
    color: #E0E0E0 !important;
    font-size: 16px;
}

/* Buttons */
.stButton>button {
    background-color: #1f6feb;
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 10px 20px;
    width: 100%;
}

/* Success box */
.stSuccess {
    background-color: #163d2b !important;
    color: #7CFFB2 !important;
    border-radius: 12px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown('<div class="main-title">🌸 Iris Flower Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Random Forest Classification with Streamlit</div>', unsafe_allow_html=True)

# ---------- INPUTS ----------
sepal_length = st.number_input("🌿 Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width  = st.number_input("🌿 Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("🌺 Petal Length (cm)", min_value=0.0, step=0.1)
petal_width  = st.number_input("🌺 Petal Width (cm)", min_value=0.0, step=0.1)

# ---------- PREDICT ----------
if st.button("🔍 Predict Flower"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.success(f"🌼 Predicted Iris Species: {prediction[0]}")

# ---------- FOOTER ----------
st.caption("Developed using Random Forest 🌳 | Streamlit App")
