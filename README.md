# Nigeria Real Estate Valuation & House Price Prediction ML API 🏠

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Render](https://img.shields.io/badge/Deployed-Render-46E3B7.svg)](https://nigerian-house-data-ipynb.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A Machine Learning-powered REST API and interactive web application for estimating Nigerian real estate market property values. Trained on **24,000+ verified Nigerian property listings** across major states and metropolitan areas.

---

### 🌐 Live Deployment & Links

- 🔗 **Live Web App & Valuation Tool:** [nigerian-house-data-ipynb.onrender.com](https://nigerian-house-data-ipynb.onrender.com)
- ⚡ **API Prediction Endpoint:** `POST` [https://nigerian-house-data-ipynb.onrender.com/predict](https://nigerian-house-data-ipynb.onrender.com/predict)
- 🩺 **API Health Check:** `GET` [https://nigerian-house-data-ipynb.onrender.com/health](https://nigerian-house-data-ipynb.onrender.com/health)
- 🐙 **GitHub Repository:** [Trainbow-7/nigerian_house_data.ipynb](https://github.com/Trainbow-7/nigerian_house_data.ipynb)

> 💡 **Note on Hosting:** Hosted on Render's free tier. If the instance is idle, it may take 30–60 seconds for the initial cold start.

---

## 🎯 Features

- **Accurate Property Valuation** — Generates instant real estate price predictions in Nigerian Naira (₦).
- **Comprehensive Geographical Coverage** — Trained across 24 Nigerian states and 186 towns/cities (Lagos, Abuja, Rivers, Oyo, Ogun, etc.).
- **Multi-Property Type Support** — Supports Detached Duplexes, Semi-Detached Duplexes, Terraced Duplexes, Bungalows, Blocks of Flats, and more.
- **Built-in Interactive Web Interface** — Beautiful, responsive UI directly accessible at the root URL.
- **RESTful JSON API** — Ready for seamless integration with mobile apps, web frontends, or CRM systems.
- **CORS Enabled** — Cross-Origin Resource Sharing enabled for modern frontend applications.

---

## 📊 Model Specifications

| Parameter | Specification |
| :--- | :--- |
| **Model Type** | Machine Learning Regression Pipeline |
| **Algorithm** | `GradientBoostingRegressor` |
| **Preprocessing** | `OneHotEncoder` (Categorical) + `ColumnTransformer` Pipeline |
| **Training Records** | 24,000+ Nigerian listings |
| **Numerical Features** | `bedrooms`, `bathrooms`, `toilets`, `parking_space` |
| **Categorical Features**| `title` (Property Type), `town`, `state` |
| **Target Variable** | `price` (in ₦ NGN) |

---

## 📡 API Endpoints & Usage

### 1. Make a Price Prediction

**`POST /predict`**

#### Request Headers
```http
Content-Type: application/json
```

#### Request Body
```json
{
  "bedrooms": 4,
  "bathrooms": 4,
  "toilets": 5,
  "parking_space": 3,
  "title": "Detached Duplex",
  "town": "Lekki",
  "state": "Lagos"
}
```

#### Response (`200 OK`)
```json
{
  "status": "success",
  "predicted_price": 119936381.82,
  "formatted_price": "₦119,936,381.82",
  "currency": "NGN",
  "inputs": {
    "bedrooms": 4,
    "bathrooms": 4,
    "toilets": 5,
    "parking_space": 3,
    "house_type": "Detached Duplex",
    "town": "Lekki",
    "state": "Lagos"
  }
}
```

---

### 2. Health Check

**`GET /health`**

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "GradientBoostingRegressor (Nigeria Real Estate Valuation)"
}
```

---

## 💻 Code Examples

### cURL
```bash
curl -X POST https://nigerian-house-data-ipynb.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 4,
    "bathrooms": 3,
    "toilets": 4,
    "parking_space": 2,
    "title": "Detached Duplex",
    "town": "Ikeja",
    "state": "Lagos"
  }'
```

### Python (`requests`)
```python
import requests

url = "https://nigerian-house-data-ipynb.onrender.com/predict"
payload = {
    "bedrooms": 3,
    "bathrooms": 3,
    "toilets": 4,
    "parking_space": 2,
    "title": "Semi Detached Duplex",
    "town": "Maitama District",
    "state": "Abuja"
}

response = requests.post(url, json=payload)
data = response.json()
print("Estimated Valuation:", data["formatted_price"])
```

### JavaScript / TypeScript (`fetch`)
```javascript
const response = await fetch("https://nigerian-house-data-ipynb.onrender.com/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    bedrooms: 4,
    bathrooms: 4,
    toilets: 5,
    parking_space: 3,
    title: "Detached Duplex",
    town: "Lekki",
    state: "Lagos"
  })
});

const result = await response.json();
console.log(`Valuation: ${result.formatted_price}`);
```

---

## 🚀 Local Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Trainbow-7/nigerian_house_data.ipynb.git
cd nigerian_house_data.ipynb
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser to test the interactive UI.

---

## 📁 Project Structure

```
├── app.py                     # Flask Web Application & REST API
├── main.py                    # Entrypoint wrapper
├── train_model.py             # ML Model Training & Pipeline script
├── house_price_model.pkl      # Trained Pipeline Model (Gradient Boosting)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🛠️ Technology Stack

- **Backend:** Flask, Flask-CORS, Gunicorn
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **Deployment:** Render Web Service
- **Environment:** Python 3.10+

---

## 👤 Author & Maintainer

- **Oyedeji Temitayo Samson**
- **GitHub:** [@Trainbow-7](https://github.com/Trainbow-7)
- **Repository:** [nigerian_house_data.ipynb](https://github.com/Trainbow-7/nigerian_house_data.ipynb)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
