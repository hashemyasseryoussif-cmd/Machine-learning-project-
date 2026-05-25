import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Genomic Cancer Subtyping", layout="wide")
st.title("🧬 Visual Breast Cancer Subtype Predictor")
st.write("Modify the log-2 transformed intensity values for highly discriminatory gene markers to analyze profiles.")

# Load structural assets
try:
    pipeline = joblib.load('gradient_boosting_model.pkl')
    selected_genes = joblib.load('selected_genes.pkl')
    target_classes = joblib.load('target_classes.pkl')
    mean_profiles = joblib.load('mean_profiles.pkl')
except Exception as e:
    st.error(f"Error loading system artifacts! Check Phase 1 execution status. Details: {e}")
    st.stop()

# Layout Configuration split: Inputs on Left Sidebar, Graphics on the main window
st.sidebar.header("🔬 Input Probe Metrics")
input_values = {}
for probe in selected_genes:
    # Baseline set to standard 7.0 expression metric value
    input_values[probe] = st.sidebar.slider(f"Probe: {probe}", min_value=0.0, max_value=16.0, value=7.0, step=0.1)

# Application Dashboard Landing Screen Tabs
tab1, tab2 = st.tabs(["📊 Prediction Analysis Dashboard", "📈 Historical Cluster Benchmarks"])

with tab1:
    st.subheader("Execute Biomarker Classification Profile Analysis")
    if st.button("Compute Diagnostic Subtype", type="primary"):
        # Reconstruct baseline matrix dataframe dimensions for pipeline expectations
        original_features = pipeline.named_steps['scaler'].feature_names_in_
        baseline_row = {feat: 7.0 for feat in original_features}
        for probe, val in input_values.items():
            if probe in baseline_row:
                baseline_row[probe] = val
        input_df = pd.DataFrame([baseline_row])
        
        # Inference pipeline computations
        prediction_idx = pipeline.predict(input_df)[0]
        probabilities = pipeline.predict_proba(input_df)[0]
        predicted_subtype = target_classes[prediction_idx]
        
        # Visual Callout Card
        st.metric(label="Diagnosed Molecular Subtype Signature", value=predicted_subtype)
        
        # Create visual side-by-side plots for interpretation
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            st.markdown("#### Subtype Probability Breakdown")
            prob_df = pd.DataFrame({'Subtype': target_classes, 'Probability': probabilities})
            fig_bar = px.bar(prob_df, x='Probability', y='Subtype', orientation='h', 
                             color='Probability', color_continuous_scale='Blues', text_auto='.1%')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with fig_col2:
            st.markdown("#### Patient Expression Alignment (Radar Map)")
            fig_radar = go.Figure()
            # Plot patient current selection array vs benchmark averages
            user_series = [input_values[g] for g in selected_genes]
            fig_radar.add_trace(go.Scatterpolar(r=user_series, theta=selected_genes, fill='toself', name='Current Input Profile'))
            # Add target baseline reference map 
            mean_series = mean_profiles.loc[predicted_subtype].tolist()
            fig_radar.add_trace(go.Scatterpolar(r=mean_series, theta=selected_genes, line=dict(dash='dash'), name=f'Avg {predicted_subtype} Pattern'))
            
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 16])), showlegend=True)
            st.plotly_chart(fig_radar, use_container_width=True)

with tab2:
    st.subheader("Pre-computed Laboratory Reference Visualizations")
    col_static1, col_static2 = st.columns(2)
    with col_static1:
        st.markdown("#### Training Set Spatial Clustering (PCA)")
        st.image("pca_separation.png", use_container_width=True)
    with col_static2:
        st.markdown("#### Pipeline Model Verification Matrix")
        st.image("confusion_matrix.png", use_container_width=True)
