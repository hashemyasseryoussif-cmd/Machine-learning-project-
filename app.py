
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('breast_cancer_scaler.pkl')
le = joblib.load('label_encoder.pkl')
selected_genes = joblib.load('selected_genes.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]

    final_features = [np.array(features)]

    prediction = model.predict(final_features)

    return render_template(
        'index.html',
        prediction_text=f'Prediction: {prediction[0]}'
    )

if __name__ == "__main__":
    app.run(debug=True)
