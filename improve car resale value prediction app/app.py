"""
=============================================================================
CarDekho Valuation Tool - Flask Web Application
=============================================================================
This Flask application provides an interactive web interface for predicting
the resale price of used cars based on machine learning (Random Forest).
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "car_price_pipeline.joblib")
SPECS_PATH = os.path.join(BASE_DIR, "models", "specs_lookup.joblib")
META_PATH = os.path.join(BASE_DIR, "models", "metrics.json")


def load_artifacts():
    """Loads the trained model pipeline, specs lookup, and metadata."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SPECS_PATH):
        return None, None, {}

    model = joblib.load(MODEL_PATH)
    specs = joblib.load(SPECS_PATH)

    metadata = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, specs, metadata


# Global artifacts loaded at startup
model, specs_lookup, metadata = load_artifacts()

# Precompute brand-model mapping for fast dynamic dropdowns
brand_models_map = {}
if specs_lookup is not None:
    for brand_name, group in specs_lookup.groupby("brand"):
        brand_models_map[brand_name] = sorted(group["model"].unique().tolist())


def format_indian_currency(amount):
    """Formats numeric price into Indian currency words (Lakhs / Crores)."""
    if amount >= 10000000:
        crores = amount / 10000000.0
        return f"₹ {crores:.2f} Crore"
    elif amount >= 100000:
        lakhs = amount / 100000.0
        return f"₹ {lakhs:.2f} Lakh"
    else:
        return f"₹ {amount:,.0f}"


def get_specs_for_car(brand, model_name):
    """Returns default engine specs for a given brand and model."""
    if specs_lookup is not None:
        matched = specs_lookup[(specs_lookup["brand"] == brand) & (specs_lookup["model"] == model_name)]
        if not matched.empty:
            return {
                "mileage_kmpl": round(float(matched["mileage_kmpl"].values[0]), 1),
                "engine_cc": int(matched["engine_cc"].values[0]),
                "max_power_bhp": round(float(matched["max_power_bhp"].values[0]), 1),
                "seats": int(matched["seats"].values[0])
            }
    # Standard fallback
    return {
        "mileage_kmpl": 18.0,
        "engine_cc": 1197,
        "max_power_bhp": 80.0,
        "seats": 5
    }


def compute_prediction_result(data):
    """Performs ML inference and computes summary insights."""
    if model is None:
        raise ValueError("Model pipeline is not loaded. Please train the model first.")

    current_year = 2026
    year = int(data.get("year", 2021))
    car_age = max(1, current_year - year)

    km_driven = float(data.get("km_driven", 45000))
    brand = str(data.get("brand", "Maruti"))
    car_model = str(data.get("model", "Swift"))
    fuel = str(data.get("fuel", "Diesel"))
    seller_type = str(data.get("seller_type", "Individual"))
    transmission = str(data.get("transmission", "Manual"))
    owner = str(data.get("owner", "First Owner"))
    city = str(data.get("city", "Delhi"))
    condition = str(data.get("condition", "Good"))
    mileage_kmpl = float(data.get("mileage_kmpl", 18.0))
    engine_cc = float(data.get("engine_cc", 1197.0))
    max_power_bhp = float(data.get("max_power_bhp", 80.0))
    seats = int(data.get("seats", 5))

    input_df = pd.DataFrame([{
        "brand": brand,
        "model": car_model,
        "car_age": car_age,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "city": city,
        "condition": condition,
        "mileage_kmpl": mileage_kmpl,
        "engine_cc": engine_cc,
        "max_power_bhp": max_power_bhp,
        "seats": seats
    }])

    # Predict price using trained pipeline
    raw_price = model.predict(input_df)[0]
    predicted_price = max(25000.0, float(raw_price))

    # Fair market range (+/- 6%)
    lower_bound = predicted_price * 0.94
    upper_bound = predicted_price * 1.06

    avg_yearly_km = km_driven / car_age
    usage_status = "Normal" if avg_yearly_km <= 15000 else "High"

    return {
        "predicted_price": round(predicted_price, 2),
        "formatted_price": format_indian_currency(predicted_price),
        "raw_formatted_price": f"₹ {predicted_price:,.0f}",
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "formatted_range": f"{format_indian_currency(lower_bound)} - {format_indian_currency(upper_bound)}",
        "car_age": car_age,
        "manufactured_year": year,
        "km_driven": int(km_driven),
        "avg_yearly_km": int(round(avg_yearly_km)),
        "usage_status": usage_status,
        "brand": brand,
        "model": car_model,
        "fuel": fuel,
        "transmission": transmission,
        "owner": owner,
        "city": city,
        "condition": condition,
        "seller_type": seller_type,
        "mileage_kmpl": mileage_kmpl,
        "engine_cc": int(engine_cc),
        "max_power_bhp": max_power_bhp,
        "seats": seats,
        "insights": {
            "brand_retention": f"{brand} holds consistent and resilient demand in the Indian secondary market.",
            "ownership_impact": f"As a {owner}, the valuation reflects standard depreciation for this tier.",
            "usage_level": f"At {int(km_driven):,} km, your car is running {'within normal limits' if avg_yearly_km <= 15000 else 'above typical annual mileage'} (~{int(avg_yearly_km):,} km/year)."
        }
    }


@app.route("/")
def index():
    """Renders the main valuation dashboard."""
    current_year = 2026
    year_options = list(range(current_year, 2005, -1))

    brands = metadata.get("brands", sorted(brand_models_map.keys())) if brand_models_map else []
    default_brand = "Maruti" if "Maruti" in brands else (brands[0] if brands else "")
    models_for_default = brand_models_map.get(default_brand, ["Swift"])
    default_model = "Swift" if "Swift" in models_for_default else (models_for_default[0] if models_for_default else "")

    cities = metadata.get("cities", ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Pune", "Chennai", "Kolkata"])
    fuels = metadata.get("fuels", ["Petrol", "Diesel", "CNG", "LPG"])
    transmissions = metadata.get("transmissions", ["Manual", "Automatic"])
    owners = metadata.get("owners", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
    conditions = metadata.get("conditions", ["Excellent", "Good", "Fair"])
    seller_types = ["Individual", "Dealer", "Trustmark Dealer"]

    default_specs = get_specs_for_car(default_brand, default_model)

    rf_metrics = metadata.get("metrics", {}).get("random_forest", {
        "r2_score": 0.9439,
        "mae": 75057.59,
        "mape": 15.88,
        "rmse": 208831.36
    })
    lr_metrics = metadata.get("metrics", {}).get("linear_regression", {
        "r2_score": 0.8894,
        "mae": 136773.57,
        "mape": 32.69,
        "rmse": 293176.73
    })

    return render_template(
        "index.html",
        brands=brands,
        models=models_for_default,
        years=year_options,
        cities=cities,
        fuels=fuels,
        transmissions=transmissions,
        owners=owners,
        conditions=conditions,
        seller_types=seller_types,
        default_brand=default_brand,
        default_model=default_model,
        default_year=2021,
        default_city="Delhi",
        default_km=45000,
        default_owner="First Owner",
        default_condition="Good",
        default_seller="Individual",
        default_fuel="Diesel",
        default_transmission="Manual",
        default_specs=default_specs,
        rf_metrics=rf_metrics,
        lr_metrics=lr_metrics,
        result=None
    )


@app.route("/api/models")
def api_models():
    """Returns available car models for a selected brand."""
    brand = request.args.get("brand", "").strip()
    available_models = brand_models_map.get(brand, [])
    if not available_models and specs_lookup is not None:
        matched = specs_lookup[specs_lookup["brand"] == brand]
        available_models = sorted(matched["model"].unique().tolist())
    return jsonify({
        "status": "success",
        "brand": brand,
        "models": available_models if available_models else ["Other"]
    })


@app.route("/api/specs")
def api_specs():
    """Returns default engine specs for a given brand and model."""
    brand = request.args.get("brand", "").strip()
    car_model = request.args.get("model", "").strip()
    specs = get_specs_for_car(brand, car_model)
    return jsonify({
        "status": "success",
        "brand": brand,
        "model": car_model,
        "specs": specs
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Handles car resale price valuation request via AJAX JSON or Form POST."""
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        result = compute_prediction_result(data)

        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({
                "status": "success",
                "result": result
            })
        else:
            # Fallback for standard HTML form submission
            current_year = 2026
            year_options = list(range(current_year, 2005, -1))
            brands = metadata.get("brands", sorted(brand_models_map.keys()))
            cities = metadata.get("cities", ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Pune", "Chennai", "Kolkata"])
            fuels = metadata.get("fuels", ["Petrol", "Diesel", "CNG", "LPG"])
            transmissions = metadata.get("transmissions", ["Manual", "Automatic"])
            owners = metadata.get("owners", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
            conditions = metadata.get("conditions", ["Excellent", "Good", "Fair"])
            seller_types = ["Individual", "Dealer", "Trustmark Dealer"]

            selected_brand = result["brand"]
            models_for_brand = brand_models_map.get(selected_brand, [result["model"]])

            return render_template(
                "index.html",
                brands=brands,
                models=models_for_brand,
                years=year_options,
                cities=cities,
                fuels=fuels,
                transmissions=transmissions,
                owners=owners,
                conditions=conditions,
                seller_types=seller_types,
                default_brand=result["brand"],
                default_model=result["model"],
                default_year=result["manufactured_year"],
                default_city=result["city"],
                default_km=result["km_driven"],
                default_owner=result["owner"],
                default_condition=result["condition"],
                default_seller=result["seller_type"],
                default_fuel=result["fuel"],
                default_transmission=result["transmission"],
                default_specs={
                    "mileage_kmpl": result["mileage_kmpl"],
                    "engine_cc": result["engine_cc"],
                    "max_power_bhp": result["max_power_bhp"],
                    "seats": result["seats"]
                },
                rf_metrics=metadata.get("metrics", {}).get("random_forest", {}),
                lr_metrics=metadata.get("metrics", {}).get("linear_regression", {}),
                result=result
            )

    except Exception as e:
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 400
        else:
            return f"An error occurred during valuation: {str(e)}", 500


if __name__ == "__main__":
    print("==================================================================")
    print("🚗 Starting CarDekho Resale Price Valuation Tool (Flask Server)")
    print("==================================================================")
    print("• Access the web app at: http://127.0.0.1:5000")
    print("==================================================================")
    app.run(host="0.0.0.0", port=5000, debug=True)
