"""
Unit and Integration Tests for Flask Car Valuation Web App
"""

import unittest
import json
from app import app


class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        """Test GET / returns 200 and renders HTML."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("Used Car Resale Price Valuation Tool", html)
        self.assertIn("brandSelect", html)
        self.assertIn("Random Forest", html)

    def test_api_models(self):
        """Test GET /api/models returns valid models for a brand."""
        response = self.app.get('/api/models?brand=Maruti')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("brand"), "Maruti")
        self.assertIsInstance(data.get("models"), list)
        self.assertIn("Swift", data["models"])

    def test_api_specs(self):
        """Test GET /api/specs returns technical specs."""
        response = self.app.get('/api/specs?brand=Maruti&model=Swift')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get("status"), "success")
        specs = data.get("specs", {})
        self.assertIn("mileage_kmpl", specs)
        self.assertIn("engine_cc", specs)
        self.assertIn("max_power_bhp", specs)
        self.assertIn("seats", specs)
        self.assertGreater(specs["engine_cc"], 500)

    def test_predict_json(self):
        """Test POST /predict with JSON payload."""
        payload = {
            "brand": "Maruti",
            "model": "Swift",
            "year": 2021,
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
        }
        response = self.app.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json',
            headers={'X-Requested-With': 'XMLHttpRequest'}
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get("status"), "success")
        result = data.get("result", {})
        self.assertIn("predicted_price", result)
        self.assertGreater(result["predicted_price"], 25000)
        self.assertIn("₹", result["formatted_price"])
        self.assertIn("formatted_range", result)
        self.assertEqual(result["brand"], "Maruti")
        self.assertEqual(result["model"], "Swift")
        self.assertEqual(result["car_age"], 5)  # 2026 - 2021 = 5

    def test_predict_form(self):
        """Test POST /predict with standard HTML Form submission."""
        form_data = {
            "brand": "Hyundai",
            "model": "i20",
            "year": "2020",
            "km_driven": "35000",
            "fuel": "Petrol",
            "seller_type": "Individual",
            "transmission": "Manual",
            "owner": "First Owner",
            "city": "Mumbai",
            "condition": "Excellent",
            "mileage_kmpl": "18.5",
            "engine_cc": "1197",
            "max_power_bhp": "82.0",
            "seats": "5"
        }
        response = self.app.post('/predict', data=form_data)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("Estimated Fair Resale Value", html)
        self.assertIn("Hyundai", html)
        self.assertIn("i20", html)


if __name__ == '__main__':
    unittest.main()
