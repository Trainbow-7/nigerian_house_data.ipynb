import os
import re
import gzip
import pickle
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Base frontend origin to seamlessly mirror
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
        print(f"Warning: {MODEL_PATH} not found. Please place the model file in the project folder.")
except Exception as e:
    print(f"Error loading model: {e}")

def normalize_title(raw_title):
    """
    Maps various user titles and aliases to standard categories present in the training set.
    """
    clean = str(raw_title or '').strip().lower()
    mapping = {
        'flat': 'block of flats',
        'flats': 'block of flats',
        'apartment': 'block of flats',
        'apartments': 'block of flats',
        'block of flats': 'block of flats',
        'block of flat': 'block of flats',
        'duplex': 'detached duplex',
        'mansion': 'detached duplex',
        'detached duplex': 'detached duplex',
        'terraced duplex': 'terraced duplexes',
        'terraced duplexes': 'terraced duplexes',
        'semi detached duplex': 'semi detached duplex',
        'semi-detached duplex': 'semi detached duplex',
        'bungalow': 'detached bungalow',
        'detached bungalow': 'detached bungalow',
        'semi detached bungalow': 'semi detached bungalow',
        'semi-detached bungalow': 'semi detached bungalow',
        'terraced bungalow': 'terraced bungalow',
        'terraced bungalows': 'terraced bungalow',
    }
    return mapping.get(clean, clean)

def execute_prediction(data):
    """
    Executes prediction using the loaded regression model.
    Accepts arbitrary dictionary formats and returns formatted prediction payload.
    """
    if model is None:
        return None, "House price model is not loaded on the server"

    if not data or not isinstance(data, dict):
        return None, "No valid input data provided"

    try:
        bedrooms = max(1.0, float(data.get('bedrooms') or 1))
        bathrooms = max(1.0, float(data.get('bathrooms') or 1))
        toilets = max(1.0, float(data.get('toilets') or bathrooms))
        parking_space = max(0.0, float(data.get('parking_space') if data.get('parking_space') is not None else (data.get('parking') or 1)))

        raw_title = str(data.get('title') or data.get('house_type') or data.get('property_type') or 'Detached Duplex').strip()
        raw_town = str(data.get('town') or '').strip()
        raw_state = str(data.get('state') or '').strip()

        norm_title = normalize_title(raw_title)
        norm_town = raw_town.lower()
        norm_state = raw_state.lower()

        input_df = pd.DataFrame([{
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'toilets': toilets,
            'parking_space': parking_space,
            'title': norm_title,
            'town': norm_town,
            'state': norm_state
        }])

        raw_prediction = float(model.predict(input_df)[0])
        predicted_price = max(0.0, raw_prediction)
        formatted_price = f"₦{predicted_price:,.2f}"

        bed_str = f"{int(bedrooms)} Bedroom" if bedrooms == 1 else f"{int(bedrooms) if bedrooms.is_integer() else bedrooms} Bedrooms"
        bath_str = f"{int(bathrooms)} Bathroom" if bathrooms == 1 else f"{int(bathrooms) if bathrooms.is_integer() else bathrooms} Bathrooms"
        prop_type = raw_title if raw_title else "Property"
        location_str = f"{raw_town}, {raw_state}" if (raw_town and raw_state) else (raw_town or raw_state or "Nigeria")

        rationale = f"Valuation estimate for a {bed_str}, {bath_str} {prop_type} in {location_str} generated using HistGradientBoosting regression."

        result_payload = {
            'status': 'success',
            'predicted_price': round(predicted_price, 2),
            'price': round(predicted_price, 2),
            'predicted_house_price': round(predicted_price, 2),
            'formatted_price': formatted_price,
            'currency': 'NGN',
            'confidence': 'High',
            'rationale': rationale,
            'inputs': {
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'toilets': toilets,
                'parking_space': parking_space,
                'title': raw_title,
                'house_type': raw_title,
                'town': raw_town,
                'state': raw_state
            }
        }
        return result_payload, None
    except Exception as e:
        return None, f"Prediction computation failed: {str(e)}"

def format_prediction_response(result_payload):
    """
    Wraps the prediction payload so it is compatible with both
    Base44 frontend expectations (n.data.result) and standard direct APIs.
    """
    response = {
        'status': 'success',
        'result': result_payload,
        'predicted_price': result_payload['predicted_price'],
        'price': result_payload['price'],
        'formatted_price': result_payload['formatted_price'],
        'currency': result_payload['currency'],
        'confidence': result_payload['confidence'],
        'rationale': result_payload['rationale'],
        'inputs': result_payload['inputs']
    }
    return jsonify(response)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return ('', 204)
    data = request.get_json(silent=True, force=True) or {}
    res, err = execute_prediction(data)
    if err:
        return jsonify({'error': err}), 400
    return format_prediction_response(res)

@app.route('/api/apps/<path:subpath>', methods=['POST', 'OPTIONS'])
@app.route('/apps/<path:subpath>', methods=['POST', 'OPTIONS'])
@app.route('/api/functions/<path:subpath>', methods=['POST', 'OPTIONS'])
@app.route('/functions/<path:subpath>', methods=['POST', 'OPTIONS'])
def handle_app_functions(subpath):
    if request.method == 'OPTIONS':
        return ('', 204)
    if 'predict' in subpath.lower() or 'price' in subpath.lower():
        data = request.get_json(silent=True, force=True) or {}
        res, err = execute_prediction(data)
        if err:
            return jsonify({'error': err}), 400
        return format_prediction_response(res)
    
    # Forward non-prediction function calls to proxy
    return proxy(request.path.lstrip('/'))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': 'GradientBoostingRegressor (Nigeria Real Estate Valuation)'
    })

# Transparent Reverse Proxy for the frontend interface
@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'])
def proxy(path):
    # Intercept any POST requests that target predictions
    if request.method == 'POST' and ('predict' in path.lower() or 'price' in path.lower()):
        data = request.get_json(silent=True, force=True) or {}
        res, err = execute_prediction(data)
        if err:
            return jsonify({'error': err}), 400
        return format_prediction_response(res)

    target_url = f"{BASE44_ORIGIN}/{path}"
    if request.query_string:
        target_url += f"?{request.query_string.decode('utf-8')}"

    # Filter out headers that could conflict with upstream
    headers = {
        k: v for k, v in request.headers 
        if k.lower() not in ['host', 'content-length', 'connection', 'accept-encoding']
    }
    headers['Host'] = 'realestatevaluation.base44.app'
    headers['Referer'] = BASE44_ORIGIN
    headers['Accept-Encoding'] = 'identity'

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
            
            # Decompress if upstream compressed with gzip
            if resp.headers.get('Content-Encoding') == 'gzip' or (len(content) >= 2 and content[:2] == b'\x1f\x8b'):
                try:
                    content = gzip.decompress(content)
                except Exception:
                    pass

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
        if e.headers.get('Content-Encoding') == 'gzip' or (len(error_content) >= 2 and error_content[:2] == b'\x1f\x8b'):
            try:
                error_content = gzip.decompress(error_content)
            except Exception:
                pass
        return Response(error_content, status=e.code, content_type=e.headers.get('Content-Type', 'text/html'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
