import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_and_save_model():
    # Look for dataset in Desktop, current folder, or Downloads
    dataset_candidates = [
        r"C:\Users\hp\Desktop\nigeria_houses_data.csv",
        "nigeria_houses_data.csv",
        r"C:\Users\hp\Downloads\nigeria_houses_data.csv"
    ]
    
    csv_path = None
    for path in dataset_candidates:
        if os.path.exists(path):
            csv_path = path
            break
            
    if not csv_path:
        raise FileNotFoundError("Could not find nigeria_houses_data.csv")

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean & normalize column names
    df.columns = df.columns.str.strip().str.lower()
    print("Columns found:", df.columns.tolist())

    # Map column names if needed
    # Expected: bedrooms, bathrooms, toilets, parking_space, title, town, state, price
    numeric_features = ['bedrooms', 'bathrooms', 'toilets', 'parking_space']
    categorical_features = ['title', 'town', 'state']
    
    # Drop rows where target (price) is NaN or invalid
    df = df.dropna(subset=['price'])
    df = df[df['price'] > 0]
    
    # Fill any missing values in features
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0
            
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('')
        else:
            df[col] = ''

    X = df[numeric_features + categorical_features]
    y = df['price']

    print(f"Dataset shape: {X.shape}, Target samples: {len(y)}")

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # Pipeline with Regressor
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])

    print("Training GradientBoostingRegressor model pipeline...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    model_pipeline.fit(X_train, y_train)

    train_score = model_pipeline.score(X_train, y_train)
    test_score = model_pipeline.score(X_test, y_test)
    print(f"Model R^2 Score - Train: {train_score:.4f}, Test: {test_score:.4f}")

    output_path = os.path.join(os.path.dirname(__file__), 'house_price_model.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(model_pipeline, f)

    print(f"Successfully exported regression model to {output_path}")

if __name__ == '__main__':
    train_and_save_model()
