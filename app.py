from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)


import os

# Get the absolute path to the directory where app.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the models using absolute paths
model = joblib.load(os.path.join(BASE_DIR, 'breast_cancer_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'breast_cancer_scaler.pkl'))
le = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))
selected_genes = joblib.load(os.path.join(BASE_DIR, 'selected_genes.pkl'))



@app.route('/')
def home():
    return render_template('index.html', gene_count=len(selected_genes))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_string = request.form.get('gene_expressions')
        features = [float(x.strip()) for x in input_string.split(',')]

        if len(features) != len(selected_genes):
            error_msg = f"Error: Expected {len(selected_genes)} gene values, but got {len(features)}."
            return render_template('index.html', error=error_msg, gene_count=len(selected_genes))

        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prediction_encoded = model.predict(features_scaled)
        prediction_label = le.inverse_transform(prediction_encoded)[0]

        return render_template(
            'index.html',
            prediction_text=f'Predicted Subtype: {prediction_label}',
            gene_count=len(selected_genes)
        )

    except ValueError:
        return render_template('index.html', error="Error: Please ensure all values are numbers separated by commas.", gene_count=len(selected_genes))
    except Exception as e:
        return render_template('index.html', error=f"An unexpected error occurred: {str(e)}", gene_count=len(selected_genes))

if __name__ == "__main__":
    app.run(debug=True)
