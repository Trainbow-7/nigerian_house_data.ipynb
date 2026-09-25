
import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model, label encoder, and scaler
# Make sure these files are in the same directory or accessible path
model_path = 'gradient_boosting_model.pkl'
encoder_path = 'label_encoder.pkl'
scaler_path = 'scaler.pkl'

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    # Conditional loading of scaler
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = None # No scaler needed for tree-based models
    print('Model, encoder, and scaler loaded successfully.')
except FileNotFoundError as e:
    print(f'Error loading essential files: {e}')
    model = None # Ensure model is None if loading fails
    label_encoder = None
    scaler = None


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoder is None:
        return jsonify({'error': 'Model or encoder not loaded'}), 500

    data = request.get_json(force=True)
    # Assuming input format: {'bedrooms': 4, 'bathrooms': 3, 'toilets': 3, 'parking_space': 2}
    features = [data['bedrooms'], data['bathrooms'], data['toilets'], data['parking_space']]
    new_data = np.array([features])

    if scaler is not None:
        new_data_processed = scaler.transform(new_data)
    else:
        new_data_processed = new_data
        
    prediction_encoded = model.predict(new_data_processed)
    prediction_proba = model.predict_proba(new_data_processed)
    predicted_class = label_encoder.inverse_transform(prediction_encoded)[0]
    confidence = prediction_proba[0].max() * 100
    
    return jsonify({
        'predicted_property_type': predicted_class,
        'confidence': f'{confidence:.2f}%',
        'probabilities': {cls: f'{prob*100:.2f}%' for cls, prob in zip(label_encoder.classes_, prediction_proba[0])}
    })

@app.route('/')
def home():
    return 'Model Prediction API is running!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
