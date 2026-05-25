import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('feature_columns.pkl')

st.set_page_config(page_title='Breast Cancer Subtype Predictor', layout='centered')
st.title('🔬 Breast Cancer Subtype Predictor')
st.markdown('Enter gene expression values for the top 200 genes (or upload a file)')

# Option: Upload CSV
uploaded = st.file_uploader('Upload CSV with gene expression (genes as rows, samples as columns)', type='csv')
if uploaded:
    df_uploaded = pd.read_csv(uploaded, index_col=0)
    # Take first sample
    sample = df_uploaded.iloc[:, 0].to_dict()
    # Align with feature list
    input_vec = [sample.get(f, 0) for f in features]
    X_input = scaler.transform([input_vec])
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0]
    subtype = f"Subtype {int(pred)}"
    st.success(f'**Predicted Subtype:** {subtype}')
    st.write(f'Confidence: {max(prob):.1%}')

# Option: Manual entry (simplified - just few genes)
st.markdown('---')
st.markdown('Or enter values for key genes:')
vals = {}
cols = st.columns(4)
for i, f in enumerate(features[:8]):  # show only first 8 for simplicity
    with cols[i % 4]:
        vals[f] = st.number_input(f'{f[:20]}...', value=0.0, format='%.2f')

if st.button('Predict'):
    input_vec = [vals.get(f, 0) for f in features]
    X_input = scaler.transform([input_vec])
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0]
    subtype = f"Subtype {int(pred)}"
    st.success(f'**Predicted Subtype:** {subtype}')
    st.write(f'Confidence: {max(prob):.1%}')
