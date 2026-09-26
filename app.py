import os
import re
import pickle
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Base frontend origin to seamlessly mirror without revealing base44 URL
BASE44_ORIGIN = "https://realestatevaluation.base44.app"

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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': 'GradientBoostingRegressor (Nigeria Real Estate Valuation)'
    })

# Transparent Reverse Proxy for the original interface without revealing base44
@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'])
def proxy(path):
    target_url = f"{BASE44_ORIGIN}/{path}"
    if request.query_string:
        target_url += f"?{request.query_string.decode('utf-8')}"

    # Copy relevant headers
    headers = {k: v for k, v in request.headers if k.lower() not in ['host', 'content-length', 'connection']}
    headers['Host'] = 'realestatevaluation.base44.app'
    headers['Referer'] = BASE44_ORIGIN

    req_data = request.get_data() if request.method in ['POST', 'PUT', 'PATCH'] else None
    req = urllib.request.Request(
        target_url,
        data=req_data,
        headers=headers,
        method=request.method
    )

    try:
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            content_type = resp.headers.get('Content-Type', 'text/html')

            # Clean and sanitize HTML response
            if 'text/html' in content_type:
                html = content.decode('utf-8', errors='ignore')
                # Remove base44 watermark badge script
                html = re.sub(r'<script[^>]*badge\.js[^>]*>[\s\S]*?</script>', '', html, flags=re.IGNORECASE)
                # Update title to clean brand name
                html = re.sub(r'<title>[\s\S]*?</title>', '<title>Nigeria Real Estate Valuation</title>', html, flags=re.IGNORECASE)
                content = html.encode('utf-8')

            excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection', 'x-frame-options']
            response_headers = [(name, value) for name, value in resp.headers.items() if name.lower() not in excluded_headers]

            return Response(content, status=resp.status, headers=response_headers, content_type=content_type)
    except urllib.error.HTTPError as e:
        error_content = e.read()
        return Response(error_content, status=e.code, content_type=e.headers.get('Content-Type', 'text/html'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
