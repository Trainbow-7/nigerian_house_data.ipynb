import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from web frontend

# Path to the trained house price regression model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'house_price_model.pkl')

model = None
categories_info = {
    'types': ["Block of Flats", "Detached Bungalow", "Detached Duplex", "Semi Detached Bungalow", "Semi Detached Duplex", "Terraced Bungalow", "Terraced Duplexes"],
    'states': ["Abia", "Abuja", "Akwa Ibom", "Anambara", "Bayelsa", "Borno", "Delta", "Edo", "Ekiti", "Enugu", "Imo", "Kaduna", "Kano", "Katsina", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Osun", "Oyo", "Plateau", "Rivers"],
    'towns': ["Aba", "Abeokuta North", "Abeokuta South", "Abraka", "Ado-Ekiti", "Ado-Odo/Ota", "Afijio", "Agbara", "Agbara-Igbesa", "Agege", "Ajah", "Akinyele", "Akure", "Alimosho", "Amuwo Odofin", "Aniocha South", "Apapa", "Apo", "Arepo", "Asaba", "Asokoro District", "Ayobo", "Badagry", "Bwari", "Central Business District", "Chikun", "Dakibiyu", "Dakwo", "Danja", "Dape", "Dei-Dei", "Dekina", "Diplomatic Zones", "Duboyi", "Durumi", "Dutse", "Ede South", "Egbe", "Egbeda", "Egor", "Ejigbo", "Eket", "Eko Atlantic City", "Eleme", "Enugu", "Epe", "Ethiope West", "Ewekoro", "Gaduwa", "Galadimawa", "Garki", "Gbagada", "Gudu", "Guzamala", "Guzape District", "Gwagwalada", "Gwarinpa", "Ibadan", "Ibadan North", "Ibadan North-East", "Ibadan North-West", "Ibadan South-West", "Ibafo", "Ibarapa North", "Ibeju", "Ibeju Lekki", "Idimu", "Ido", "Idu Industrial", "Ifako-Ijaiye", "Ifo", "Ijaiye", "Ijebu Ode", "Ijede", "Ijesha", "Ijoko", "Ikeja", "Ikorodu", "Ikot Ekpene", "Ikotun", "Ikoyi", "Ikpoba Okha", "Ikwerre", "Ilorin East", "Ilorin South", "Ilorin West", "Ilupeju", "Imota", "Ipaja", "Isheri", "Isheri North", "Isolo", "Jabi", "Jahi", "Jikwoyi", "Jos North", "Jos South", "KM 46", "Kabusa", "Kado", "Kaduna North", "Kaduna South", "Kafe", "Kagini", "Karmo", "Karsana", "Karshi", "Karu", "Katampe", "Kaura", "Keffi", "Ketu", "Kosofe", "Kubwa", "Kuje", "Kukwaba", "Kurudu", "Kusada", "Kyami", "Lagos Island", "Lekki", "Life Camp", "Lokogoma District", "Lokoja", "Lugbe District", "Mabushi", "Magboro", "Magodo", "Maitama District", "Mararaba", "Maryland", "Mbora (Nbora)", "Mowe Ofada", "Mowe Town", "Mpape", "Mushin", "Nasarawa", "Nassarawa", "Nyanya", "Obafemi Owode", "Obio-Akpor", "Ogijo", "Ogudu", "Ohaji/Egbema", "Ojo", "Ojodu", "Ojota", "Oke-Aro", "Oke-Odo", "Okene", "Okpe", "Oluyole", "Oredo", "Orile", "Orozo", "Oshodi", "Osogbo", "Ovia North-East", "Owerri Municipal", "Owerri North", "Owerri West", "Oyigbo", "Oyo West", "Paikoro", "Port Harcourt", "Sagamu", "Sango Ota", "Shomolu", "Simawa", "Surulere", "Udu", "Ughelli South", "Uhunmwonde", "Umuahia", "Utako", "Uvwie", "Uyo", "Victoria Island (VI)", "Warri", "Wumba", "Wuse", "Wuse 2", "Wuye", "Yaba", "Yenagoa", "Yewa South"]
}

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("House price regression model loaded successfully.")
    else:
        print(f"Warning: {MODEL_PATH} not found yet. Please place the model file in the project folder.")
except Exception as e:
    print(f"Error loading model: {e}")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nigeria House Price Predictor | ML Valuation Engine</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0f1d;
            --bg-card: rgba(18, 26, 47, 0.75);
            --bg-card-border: rgba(255, 255, 255, 0.08);
            --accent-green: #00d084;
            --accent-green-glow: rgba(0, 208, 132, 0.25);
            --accent-cyan: #06b6d4;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --input-border: rgba(255, 255, 255, 0.12);
            --radius-lg: 20px;
            --radius-md: 12px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(at 0% 0%, rgba(0, 208, 132, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.12) 0px, transparent 50%),
                radial-gradient(at 50% 50%, rgba(15, 23, 42, 0.5) 0px, transparent 100%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2.5rem 1rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container {
            width: 100%;
            max-width: 1000px;
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 0.75rem;
            background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 1.05rem;
            max-width: 600px;
            margin: 0 auto;
        }

        .grid-layout {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 1.75rem;
        }

        @media (max-width: 850px) {
            .grid-layout {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--bg-card-border);
            border-radius: var(--radius-lg);
            padding: 2rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
        }

        @media (max-width: 500px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group.full-width {
            grid-column: 1 / -1;
        }

        label {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        select, input {
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: var(--radius-md);
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            transition: all 0.2s ease;
            outline: none;
            width: 100%;
        }

        select:focus, input:focus {
            border-color: var(--accent-green);
            box-shadow: 0 0 0 3px var(--accent-green-glow);
        }

        select option {
            background: #0f172a;
            color: #fff;
        }

        .btn-predict {
            margin-top: 1.5rem;
            width: 100%;
            background: linear-gradient(135deg, #00d084 0%, #059669 100%);
            color: #032b1d;
            border: none;
            border-radius: var(--radius-md);
            padding: 1rem;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 25px rgba(0, 208, 132, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(0, 208, 132, 0.4);
            filter: brightness(1.08);
        }

        .btn-predict:active {
            transform: translateY(0);
        }

        .result-panel {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 380px;
        }

        .valuation-box {
            background: rgba(0, 208, 132, 0.05);
            border: 1px solid rgba(0, 208, 132, 0.2);
            border-radius: var(--radius-md);
            padding: 1.75rem 1.25rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .valuation-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--accent-green);
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .valuation-price {
            font-size: 2.2rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 0.25rem;
            letter-spacing: -0.5px;
        }

        .summary-list {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }

        .summary-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .summary-item span:first-child {
            color: var(--text-muted);
        }

        .summary-item span:last-child {
            font-weight: 600;
            color: var(--text-main);
        }

        .api-docs-box {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--input-border);
            border-radius: var(--radius-md);
            padding: 1rem;
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .api-docs-box code {
            color: var(--accent-cyan);
            font-family: monospace;
        }

        footer {
            margin-top: 3rem;
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        footer a {
            color: var(--accent-green);
            text-decoration: none;
        }

        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(3, 43, 29, 0.3);
            border-radius: 50%;
            border-top-color: #032b1d;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>Nigeria Real Estate Valuation</h1>
        <p class="subtitle">Machine Learning Model trained on 24,000+ verified Nigerian property listings</p>
    </header>

    <div class="grid-layout">
        <!-- Input Form Card -->
        <div class="card">
            <h2 class="card-title">Property Specifications</h2>
            <form id="predictionForm">
                <div class="form-grid">
                    <div class="form-group full-width">
                        <label for="title">Property Type</label>
                        <select id="title" required>
                            {% for t in types %}
                            <option value="{{ t }}" {% if t == 'Detached Duplex' %}selected{% endif %}>{{ t }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="state">State</label>
                        <select id="state" required>
                            {% for s in states %}
                            <option value="{{ s }}" {% if s == 'Lagos' %}selected{% endif %}>{{ s }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="town">Town / Neighborhood</label>
                        <input list="townList" id="town" placeholder="e.g. Lekki, Ikoyi, Ikeja..." value="Lekki" required>
                        <datalist id="townList">
                            {% for tw in towns %}
                            <option value="{{ tw }}">
                            {% endfor %}
                        </datalist>
                    </div>

                    <div class="form-group">
                        <label for="bedrooms">Bedrooms</label>
                        <input type="number" id="bedrooms" min="1" max="20" value="4" required>
                    </div>

                    <div class="form-group">
                        <label for="bathrooms">Bathrooms</label>
                        <input type="number" id="bathrooms" min="1" max="20" value="4" required>
                    </div>

                    <div class="form-group">
                        <label for="toilets">Toilets</label>
                        <input type="number" id="toilets" min="1" max="25" value="5" required>
                    </div>

                    <div class="form-group">
                        <label for="parking_space">Parking Spaces</label>
                        <input type="number" id="parking_space" min="0" max="20" value="3" required>
                    </div>
                </div>

                <button type="submit" class="btn-predict" id="submitBtn">
                    <span class="spinner" id="btnSpinner"></span>
                    <span id="btnText">Calculate Estimated Value</span>
                </button>
            </form>
        </div>

        <!-- Result Card -->
        <div class="card result-panel">
            <div>
                <h2 class="card-title">Valuation Result</h2>
                <div class="valuation-box">
                    <div class="valuation-label">Estimated Property Price</div>
                    <div class="valuation-price" id="priceDisplay">₦150,000,000</div>
                    <div style="font-size: 0.85rem; color: var(--text-muted);" id="priceWords">Click Calculate to run ML model</div>
                </div>

                <ul class="summary-list">
                    <li class="summary-item">
                        <span>Model Status</span>
                        <span style="color: var(--accent-green);">Active & Ready</span>
                    </li>
                    <li class="summary-item">
                        <span>Algorithm</span>
                        <span>Gradient Boosting Pipeline</span>
                    </li>
                    <li class="summary-item">
                        <span>Response Speed</span>
                        <span id="speedDisplay">&lt; 35ms</span>
                    </li>
                </ul>
            </div>

            <div class="api-docs-box">
                <div style="font-weight: 600; margin-bottom: 4px;">Developer API:</div>
                <code>POST /predict</code>
                <div style="margin-top: 4px; font-size: 0.75rem;">Direct JSON endpoint ready for web & mobile apps.</div>
            </div>
        </div>
    </div>

    <footer>
        <p>Built with Flask & Scikit-Learn | <a href="https://github.com/Trainbow-7/nigerian_house_data.ipynb" target="_blank">View GitHub Repo</a></p>
    </footer>
</div>

<script>
    const form = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnSpinner = document.getElementById('btnSpinner');
    const btnText = document.getElementById('btnText');
    const priceDisplay = document.getElementById('priceDisplay');
    const speedDisplay = document.getElementById('speedDisplay');
    const priceWords = document.getElementById('priceWords');

    async function runPrediction() {
        const payload = {
            bedrooms: parseFloat(document.getElementById('bedrooms').value) || 0,
            bathrooms: parseFloat(document.getElementById('bathrooms').value) || 0,
            toilets: parseFloat(document.getElementById('toilets').value) || 0,
            parking_space: parseFloat(document.getElementById('parking_space').value) || 0,
            title: document.getElementById('title').value,
            town: document.getElementById('town').value,
            state: document.getElementById('state').value
        };

        btnSpinner.style.display = 'inline-block';
        btnText.textContent = 'Calculating...';
        submitBtn.disabled = true;

        const startTime = performance.now();
        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            const elapsed = Math.round(performance.now() - startTime);

            if (res.ok && data.status === 'success') {
                priceDisplay.textContent = data.formatted_price;
                speedDisplay.textContent = elapsed + 'ms';
                priceWords.textContent = 'Calculated for ' + payload.title + ' in ' + payload.town + ', ' + payload.state;
            } else {
                priceDisplay.textContent = 'Error';
                priceWords.textContent = data.error || 'Failed to estimate price.';
            }
        } catch (err) {
            priceDisplay.textContent = 'Error';
            priceWords.textContent = 'Could not reach prediction server.';
        } finally {
            btnSpinner.style.display = 'none';
            btnText.textContent = 'Calculate Estimated Value';
            submitBtn.disabled = false;
        }
    }

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        runPrediction();
    });

    // Run initial valuation on load
    window.addEventListener('DOMContentLoaded', () => {
        runPrediction();
    });
</script>

</body>
</html>
"""

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

@app.route('/', methods=['GET'])
def home():
    # If client explicitly requests JSON (API client), return JSON status
    if request.headers.get('Accept') == 'application/json':
        return jsonify({
            'message': 'Nigerian House Price Prediction API is running!',
            'model_loaded': model is not None,
            'status': 'healthy'
        })
    # Otherwise render the full interactive live demo web UI
    return render_template_string(
        HTML_TEMPLATE,
        types=categories_info['types'],
        states=categories_info['states'],
        towns=categories_info['towns']
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
