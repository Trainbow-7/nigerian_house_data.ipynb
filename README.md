# Nigeria Houses ML API 🏠

A FastAPI-based REST API for predicting Nigerian real estate property types using machine learning. Trained on 24,326 real estate records with 87%+ accuracy.

🔗 **Live Demo:** [realestatevaluation.base44.app](https://realestatevaluation.base44.app)  
⚡ **API Docs (Swagger UI):** [nigerian-house-data-ipynb.onrender.com/docs](https://nigerian-house-data-ipynb.onrender.com/docs)

> Note: hosted on Render's free tier — the API may take 30–60 seconds to wake up if it's been inactive.

## 🎯 Features

- **Property Type Prediction** - Predicts house property type based on features
- **Batch Processing** - Make predictions for multiple properties at once
- **Model Info Endpoint** - Get details about the loaded ML model
- **Interactive Documentation** - Swagger UI at `/docs`
- **CORS Enabled** - Ready for frontend integration
- **Production Ready** - Deployed on Render

## 📊 Model Details

- **Input Features:** Bedrooms, Bathrooms, Toilets, Parking Spaces
- **Output:** Property type (Detached Duplex, Terraced Duplex, Semi Detached Duplex, etc.)
- **Algorithm:** Random Forest / Gradient Boosting / Logistic Regression
- **Training Data:** 24,326 Nigerian property listings

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/nigeria-houses-api.git
   cd nigeria-houses-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place model files**
   ```bash
   # Copy your trained model files to project root:
   - random_forest_model.pkl (or your model name)
   - label_encoder.pkl
   - scaler.pkl (optional)
   ```

5. **Run the API**
   ```bash
   python main.py
   ```

6. **Open documentation**
   ```
   http://localhost:8000/docs
   ```

## 📡 API Endpoints

### Health Check
```
GET /health
```

### Get Model Info
```
GET /info
```

### Single Prediction
```
POST /predict

Body:
{
  "bedrooms": 4,
  "bathrooms": 3,
  "toilets": 3,
  "parking_space": 2
}
```

### Batch Prediction
```
POST /predict-batch

Body:
[
  {"bedrooms": 4, "bathrooms": 3, "toilets": 3, "parking_space": 2},
  {"bedrooms": 3, "bathrooms": 2, "toilets": 2, "parking_space": 1}
]
```

## 🌐 Live API

Deployed on Render at: `https://nigeria-houses-api.onrender.com`

### Example Request
```bash
curl -X POST https://nigeria-houses-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 4,
    "bathrooms": 3,
    "toilets": 3,
    "parking_space": 2
  }'
```

### Example Response
```json
{
  "predicted_property_type": "Detached Duplex",
  "confidence": 87.5,
  "all_probabilities": {
    "Detached Duplex": 87.5,
    "Terraced Duplex": 10.2,
    "Semi Detached Duplex": 2.3
  },
  "input_features": {
    "bedrooms": 4,
    "bathrooms": 3,
    "toilets": 3,
    "parking_space": 2
  }
}
```

## 📁 Project Structure

```
.
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── render.yaml                # Render deployment config
├── random_forest_model.pkl    # Trained model
├── label_encoder.pkl          # Label encoder for predictions
├── scaler.pkl                 # Feature scaler (optional)
└── README.md                  # This file
```

## 🛠️ Technologies

- **FastAPI** - Modern web framework
- **Uvicorn** - ASGI server
- **Scikit-learn** - ML models
- **Pandas** - Data processing
- **Pydantic** - Data validation
- **Render** - Hosting platform

## 📚 Training the Model

The model was trained using the Nigeria houses dataset with these steps:

1. **Data Loading** - 24,326 property records
2. **Exploratory Analysis** - Feature distributions & correlations
3. **Preprocessing** - Feature scaling, train-test split
4. **Model Training** - Random Forest, Gradient Boosting, Logistic Regression
5. **Evaluation** - Cross-validation, confusion matrices
6. **Deployment** - Serialized model to pickle

See `nigeria_houses_model_colab.py` for full training code.

## 🚀 Deployment

### Deploy to Render

See `RENDER_DEPLOYMENT_GUIDE.md` for step-by-step instructions.

**Quick deploy:**
1. Push to GitHub
2. Connect repo to Render
3. Use `render.yaml` for auto-configuration
4. Done! 🎉

### Environment Variables

Currently, no environment variables required. For production with secrets:

```bash
# Add in Render dashboard → Settings → Environment Variables
DATABASE_URL=your_database_url
SECRET_KEY=your_secret_key
```

## 🔒 Security

- CORS enabled for all origins (change if needed)
- No API key authentication (add if needed)
- Input validation via Pydantic
- Error handling for edge cases

## 📈 Performance

- **Cold start:** ~2-3 seconds (free tier)
- **Inference time:** ~50ms per prediction
- **Memory usage:** ~200MB
- **Concurrent requests:** Depends on plan

## 🐛 Troubleshooting

### Model not loading?
- Ensure pickle files are in repository
- Check file names match: `*_model.pkl`, `label_encoder.pkl`

### Port errors?
- Start command must include `$PORT` variable
- For local: `python main.py` (uses port 8000)

### Slow responses?
- Free tier Render apps spin down after 15 min inactivity
- Upgrade to paid tier for always-on service

## 📝 Logging

All API requests are logged. View logs in Render dashboard:
- Deployment logs
- Runtime logs
- Error logs

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add price prediction endpoint
- Implement request caching
- Add database for prediction history
- Build web frontend
- Add more models

## 📄 License

MIT License - feel free to use this project!

## 👤 Author

- **Temitayo Oyedeji**
- UAV HUB SYSTEMS LTD
- Email: [your-email]
- GitHub: [@your-username]

## 📞 Support

- API Documentation: `/docs` endpoint
- Issues: GitHub Issues
- Email: your-email@example.com

## 🎓 Learning Resources

- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Scikit-learn Docs](https://scikit-learn.org/)
- [Render Docs](https://render.com/docs)
- [REST API Best Practices](https://restfulapi.net/)

---

**Made with ❤️ for the Nigerian real estate market**
