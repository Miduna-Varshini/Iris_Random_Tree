import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("iris_random_forest.pkl", "rb"))

# Page configuration
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}

.main-title {
    text-align: center;
    color: #6A1B9A;
    font-size: 38px;
    font-weight: bold;
}

.sub-title {
    text-align: center;
    color: #555;
    font-size: 18px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    margin-top: 20px;
}

.predict-btn button {
    background-color: #6A1B9A;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown('<div class="main-title">🌸 Iris Flower Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Random Forest Classification with Streamlit</div>', unsafe_allow_html=True)

# ---------- INPUT CARD ----------
st.markdown('<div class="card">', unsafe_allow_html=True)

sepal_length = st.number_input("🌿 Sepal Length (cm)", 0.0, step=0.1)
sepal_width  = st.number_input("🌿 Sepal Width (cm)", 0.0, step=0.1)
petal_length = st.number_input("🌺 Petal Length (cm)", 0.0, step=0.1)
petal_width  = st.number_input("🌺 Petal Width (cm)", 0.0, step=0.1)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREDICT BUTTON ----------
st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
predict = st.button("🔍 Predict Flower")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- RESULT ----------
if predict:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    result = model.predict(input_data)

    st.success(f"🌼 **Predicted Iris Species:** {result[0]}")

# ---------- FOOTER ----------
st.markdown('<div class="footer">Developed using Random Forest Algorithm 🌳</div>', unsafe_allow_html=True)
