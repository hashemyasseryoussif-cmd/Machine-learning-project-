
import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Cancer Classification App",
    page_icon="",
    layout="centered"
)

st.title(" Cancer Classification App")
st.write(
    "This machine learning application predicts cancer subtypes "
    "based on protein expression values."
)

@st.cache_resource
def load_model():
    model = joblib.load("breast_cancer_model.pkl")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.sidebar.header("Input Features")

her2 = st.sidebar.number_input(
    "HER2 Expression",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

basal = st.sidebar.number_input(
    "Basal Expression",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

luminal_a = st.sidebar.number_input(
    "Luminal A Expression",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)

luminal_b = st.sidebar.number_input(
    "Luminal B Expression",
    min_value=0.0,
    max_value=100.0,
    value=40.0
)

input_data = np.array([
    [
        her2,
        basal,
        luminal_a,
        luminal_b
    ]
])


st.subheader("Input Data")

input_df = pd.DataFrame(
    input_data,
    columns=[
        "HER2",
        "Basal",
        "Luminal A",
        "Luminal B"
    ]
)

st.dataframe(input_df)


if st.button("Predict Cancer Type"):

    try:
        prediction = model.predict(input_data)

        st.subheader("Prediction Result")

        st.success(f"Predicted Class: {prediction[0]}")

        # Optional probability support
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            prob_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": probabilities
            })

            st.subheader("Prediction Probabilities")
            st.dataframe(prob_df)

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("Built with Streamlit and Scikit-learn")

