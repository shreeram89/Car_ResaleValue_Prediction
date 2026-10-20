"""
=============================================================================
Used Car Resale Value Prediction - Training & Evaluation Pipeline
=============================================================================
This script trains a Machine Learning model to predict used car prices
based on brand, model, year, kilometers driven, fuel, transmission,
ownership, city, condition, and technical specifications.

Algorithms evaluated:
1. Linear Regression (Baseline Model)
2. Random Forest Regressor (Final Recommended Model)
=============================================================================
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib

# Fix Windows console UTF-8 output encoding for currency symbols
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_dataset(csv_path="data/car_data.csv"):
    """Loads the preprocessed car dataset."""
    print("=" * 70)
    print("STEP 1: LOADING DATASET")
    print("=" * 70)
    df = pd.read_csv(csv_path)
    print(f"Dataset successfully loaded from: {csv_path}")
    print(f"Total Rows (Records): {df.shape[0]}")
    print(f"Total Columns (Features): {df.shape[1]}")
    print(f"Features: {list(df.columns)}")
    return df


def build_preprocessor(num_cols, cat_cols):
    """
    Constructs a ColumnTransformer to handle:
    - Numerical columns: Passthrough (Random Forest handles scale automatically)
    - Categorical columns: One-Hot Encoding (ignoring unknown categories at test time)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    return preprocessor


def evaluate_model(name, model, X_test, y_test):
    """Calculates and prints standard regression metrics."""
    predictions = model.predict(X_test)
    # Ensure no negative predictions (salvage floor at Rs 20,000)
    predictions = np.clip(predictions, 20000, None)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

    print(f"\n--- {name} Performance ---")
    print(f"R² Score (Variance Explained) : {r2:.4f}  ({r2 * 100:.2f}%)")
    print(f"Mean Absolute Error (MAE)      : Rs. {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE) : Rs. {rmse:,.2f}")
    print(f"Mean Absolute % Error (MAPE)   : {mape:.2f}%")

    return {
        "model": name,
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
    }


def main():
    # 1. Load Data
    df = load_dataset()

    # 2. Separate Features and Target
    # Features (X) will use car_age instead of raw year for better generalization
    X = df.drop(columns=["selling_price", "year"])
    y = df["selling_price"]

    num_cols = ["car_age", "km_driven", "mileage_kmpl", "engine_cc", "max_power_bhp", "seats"]
    cat_cols = ["brand", "model", "fuel", "seller_type", "transmission", "owner", "city", "condition"]

    print("\n" + "=" * 70)
    print("STEP 2: FEATURE SPLITTING & PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"Numerical Features   ({len(num_cols)}): {num_cols}")
    print(f"Categorical Features ({len(cat_cols)}): {cat_cols}")

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # 3. Train-Test Split (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set size : {X_train.shape[0]} samples")
    print(f"Testing set size  : {X_test.shape[0]} samples")

    # 4. Model 1: Baseline Linear Regression
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING BASELINE MODEL (LINEAR REGRESSION)")
    print("=" * 70)
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_metrics = evaluate_model("Linear Regression", lr_pipeline, X_test, y_test)

    # 5. Model 2: Core Model - Random Forest Regressor
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING ENSEMBLE MODEL (RANDOM FOREST REGRESSOR)")
    print("=" * 70)
    print("Training 100 Decision Trees with random subsampling...")
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_metrics = evaluate_model("Random Forest Regressor", rf_pipeline, X_test, y_test)

    # 6. Comparison Table
    print("\n" + "=" * 70)
    print("STEP 5: MODEL COMPARISON SUMMARY")
    print("=" * 70)
    comparison_df = pd.DataFrame([lr_metrics, rf_metrics])
    print(comparison_df.to_string(index=False))

    # 7. Precompute Brand-Model Default Specs Lookup
    # This helps the web app auto-populate technical specs when a brand/model is selected
    specs_lookup = (
        df.groupby(["brand", "model"])[
            ["mileage_kmpl", "engine_cc", "max_power_bhp", "seats"]
        ]
        .median()
        .reset_index()
    )

    # 8. Save artifacts for Web Deployment
    print("\n" + "=" * 70)
    print("STEP 6: SERIALIZING TRAINED MODEL & METADATA")
    print("=" * 70)
    model_save_path = "models/car_price_pipeline.joblib"
    specs_save_path = "models/specs_lookup.joblib"
    metrics_save_path = "models/metrics.json"

    joblib.dump(rf_pipeline, model_save_path)
    joblib.dump(specs_lookup, specs_save_path)

    metadata = {
        "best_model": "Random Forest Regressor",
        "features": {
            "numerical": num_cols,
            "categorical": cat_cols
        },
        "brands": sorted(df["brand"].unique().tolist()),
        "fuels": sorted(df["fuel"].unique().tolist()),
        "transmissions": sorted(df["transmission"].unique().tolist()),
        "owners": [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner"
        ],
        "cities": sorted(df["city"].unique().tolist()),
        "conditions": ["Excellent", "Good", "Fair"],
        "metrics": {
            "linear_regression": lr_metrics,
            "random_forest": rf_metrics
        }
    }

    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"1. Model pipeline saved to: '{model_save_path}'")
    print(f"2. Specifications lookup saved to: '{specs_save_path}'")
    print(f"3. Metadata & metrics saved to: '{metrics_save_path}'")

    # 9. Quick Verification Test
    print("\n" + "=" * 70)
    print("STEP 7: SANITY TEST ON SAMPLE CAR")
    print("=" * 70)
    sample_car = pd.DataFrame([{
        "brand": "Maruti",
        "model": "Swift",
        "car_age": 7,           # 2019 model in 2026 (2026 - 2019 = 7 years)
        "km_driven": 45000,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "city": "Delhi",
        "condition": "Good",
        "mileage_kmpl": 23.4,
        "engine_cc": 1248,
        "max_power_bhp": 74.0,
        "seats": 5
    }])

    predicted_price = rf_pipeline.predict(sample_car)[0]
    lakhs = predicted_price / 100000.0
    print(f"Input: 2019 Maruti Swift Diesel (1st Owner, 45,000 km, Delhi)")
    print(f"Predicted Valuation: Rs. {predicted_price:,.2f}  (~Rs. {lakhs:.2f} Lakhs)")
    print("=" * 70)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
