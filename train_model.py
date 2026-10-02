import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_percentage_error

def train_and_save_model():
    dataset_candidates = [
        r"C:\Users\hp\Desktop\nigeria_houses_data.csv",
        "nigeria_houses_data.csv",
        r"C:\Users\hp\Downloads\nigeria_houses_data.csv",
        os.path.join(os.path.dirname(__file__), "nigeria_houses_data.csv")
    ]
    
    csv_path = None
    for path in dataset_candidates:
        if os.path.exists(path):
            csv_path = path
            break
            
    if not csv_path:
        raise FileNotFoundError("Could not find nigeria_houses_data.csv in Desktop, project dir, or Downloads")

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean & normalize column names
    df.columns = df.columns.str.strip().str.lower()
    print("Columns found:", df.columns.tolist())

    numeric_features = ['bedrooms', 'bathrooms', 'toilets', 'parking_space']
    categorical_features = ['title', 'town', 'state']
    
    # Drop rows where target (price) is NaN or invalid
    df = df.dropna(subset=['price'])

    # Clean extreme outliers & corrupt data for realistic Nigerian real estate valuation
    # Prices below 1M NGN are usually monthly rents/test entries; prices above 25B are extreme anomalies
    clean_df = df[(df['price'] >= 1_000_000) & (df['price'] <= 25_000_000_000)].copy()
    print(f"Retained {len(clean_df)} / {len(df)} samples after outlier filtering.")

    # Fill and sanitize numeric features (clipping unreasonable extremes e.g. > 20 rooms)
    for col in numeric_features:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(1).clip(lower=0, upper=20)
        else:
            clean_df[col] = 1

    # Normalize categorical features (lowercase & strip whitespace)
    for col in categorical_features:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str).str.strip().str.lower()
        else:
            clean_df[col] = ''

    X = clean_df[numeric_features + categorical_features]
    y = clean_df['price']

    # Preprocessor using OrdinalEncoder with unknown category handling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ]
    )

    # Features: 0: bedrooms, 1: bathrooms, 2: toilets, 3: parking_space, 4: title, 5: town, 6: state
    # Apply monotonic increasing constraints to numeric features (+1) and unconstrained (0) to categoricals
    hgb_regressor = HistGradientBoostingRegressor(
        categorical_features=[4, 5, 6],
        monotonic_cst=[1, 1, 1, 1, 0, 0, 0],
        max_iter=350,
        learning_rate=0.06,
        max_leaf_nodes=45,
        min_samples_leaf=10,
        random_state=42
    )

    # TransformedTargetRegressor applies log1p to price during training and expm1 at prediction time.
    # This prevents extreme luxury outliers from dominating MSE loss and ensures every property feature
    # creates sensible, proportional price adjustments.
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', TransformedTargetRegressor(
            regressor=hgb_regressor,
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])

    print("Training HistGradientBoostingRegressor model pipeline with log-transformed target...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    model_pipeline.fit(X_train, y_train)

    y_pred_train = model_pipeline.predict(X_train)
    y_pred_test = model_pipeline.predict(X_test)

    train_log_r2 = r2_score(np.log1p(y_train), np.log1p(y_pred_train))
    test_log_r2 = r2_score(np.log1p(y_test), np.log1p(y_pred_test))
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test)

    print(f"Model Log R^2 Score - Train: {train_log_r2:.4f}, Test: {test_log_r2:.4f}")
    print(f"Test MAPE: {test_mape:.4f}")

    output_path = os.path.join(os.path.dirname(__file__), 'house_price_model.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(model_pipeline, f)

    print(f"Successfully exported regression model to {output_path}")

if __name__ == '__main__':
    train_and_save_model()
