import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from web frontend

# Path to the trained house price regression model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'house_price_model.pkl')

model = None
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("House price regression model loaded successfully.")
    else:
        print(f"Warning: {MODEL_PATH} not found yet. Please place the model file in the project folder.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'House price model is not loaded on the server'}), 500

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Construct DataFrame matching the trained regression features
        input_df = pd.DataFrame([{
            'bedrooms': float(data.get('bedrooms', 0)),
            'bathrooms': float(data.get('bathrooms', 0)),
            'toilets': float(data.get('toilets', 0)),
            'parking_space': float(data.get('parking_space', 0)),
            'title': data.get('title') or data.get('house_type') or data.get('property_type', ''),
            'town': data.get('town', ''),
            'state': data.get('state', '')
        }])

        # Predict price (in NGN)
        raw_prediction = float(model.predict(input_df)[0])
        predicted_price = max(0.0, raw_prediction)

        return jsonify({
            'status': 'success',
            'predicted_price': round(predicted_price, 2),
            'formatted_price': f"₦{predicted_price:,.2f}",
            'currency': 'NGN',
            'inputs': {
                'bedrooms': data.get('bedrooms'),
                'bathrooms': data.get('bathrooms'),
                'toilets': data.get('toilets'),
                'parking_space': data.get('parking_space'),
                'house_type': data.get('title') or data.get('house_type') or data.get('property_type'),
                'town': data.get('town'),
                'state': data.get('state')
            }
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 400

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Nigerian House Price Prediction API is running!',
        'model_loaded': model is not None,
        'status': 'healthy'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
