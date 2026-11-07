import json
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

URL = "http://127.0.0.1:5000"


class DeliveryAPITests(unittest.TestCase):
    def post_api(self, endpoint, payload):
        request = Request(
            URL + endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def get_api(self, endpoint):
        with urlopen(URL + endpoint, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def valid_payload(self, **overrides):
        payload = {
            "customer_name": "API Test User",
            "distance_km": 25,
            "weight_kg": 3,
            "transportation": "Motorcycle",
        }
        payload.update(overrides)
        return payload

    # POST /predict (5 test cases)
    def test_01_motorcycle_prediction(self):
        status, data = self.post_api("/predict", self.valid_payload(
            customer_name="Juan Dela Cruz", transportation="Motorcycle"
        ))
        self.assertEqual(status, 200)
        self.assertGreater(data["estimated_delivery_minutes"], 0)
        self.assertEqual(data["customer_name"], "Juan Dela Cruz")
        self.assertEqual(data["transportation"], "Motorcycle")

    def test_02_bicycle_prediction(self):
        status, data = self.post_api("/predict", self.valid_payload(
            customer_name="Ana Santos", distance_km=2, weight_kg=1,
            transportation="Bicycle"
        ))
        self.assertEqual(status, 200)
        self.assertGreater(data["estimated_delivery_minutes"], 0)

    def test_03_car_prediction(self):
        status, data = self.post_api("/predict", self.valid_payload(
            customer_name="Mark Reyes", distance_km=10, weight_kg=4,
            transportation="Car"
        ))
        self.assertEqual(status, 200)
        self.assertEqual(data["transportation"], "Car")

    def test_04_van_prediction(self):
        status, data = self.post_api("/predict", self.valid_payload(
            customer_name="Liza Cruz", distance_km=20, weight_kg=7,
            transportation="Van"
        ))
        self.assertEqual(status, 200)
        self.assertIsInstance(data["estimated_delivery_minutes"], int)
        self.assertGreater(data["estimated_delivery_minutes"], 0)

    def test_05_invalid_distance(self):
        with self.assertRaises(HTTPError) as context:
            self.post_api("/predict", self.valid_payload(distance_km=0))
        self.assertEqual(context.exception.code, 400)
        error_data = json.loads(context.exception.read().decode("utf-8"))
        self.assertIn("Distance", error_data["error"])

    # POST /compare-transport (5 test cases)
    def test_06_compare_motorcycle_and_car(self):
        status, data = self.post_api("/compare-transport", {
            "customer_name": "Comparison Test 1",
            "distance_km": 25, "weight_kg": 3,
            "transportation_1": "Motorcycle",
            "transportation_2": "Car",
        })
        self.assertEqual(status, 200)
        self.assertGreater(data["estimated_minutes_1"], 0)
        self.assertGreater(data["estimated_minutes_2"], 0)
        self.assertIn(data["faster_transportation"], ["Motorcycle", "Car", "Tie"])

    def test_07_compare_bicycle_and_motorcycle(self):
        status, data = self.post_api("/compare-transport", {
            "customer_name": "Comparison Test 2",
            "distance_km": 5, "weight_kg": 1,
            "transportation_1": "Bicycle",
            "transportation_2": "Motorcycle",
        })
        self.assertEqual(status, 200)
        self.assertGreater(data["estimated_minutes_1"], 0)
        self.assertGreater(data["estimated_minutes_2"], 0)

    def test_08_compare_car_and_van(self):
        status, data = self.post_api("/compare-transport", {
            "customer_name": "Comparison Test 3",
            "distance_km": 40, "weight_kg": 8,
            "transportation_1": "Car",
            "transportation_2": "Van",
        })
        self.assertEqual(status, 200)
        self.assertGreaterEqual(data["difference_minutes"], 0)

    def test_09_comparison_requires_different_transports(self):
        with self.assertRaises(HTTPError) as context:
            self.post_api("/compare-transport", {
                "customer_name": "Comparison Test 4",
                "distance_km": 25, "weight_kg": 3,
                "transportation_1": "Car",
                "transportation_2": "Car",
            })
        self.assertEqual(context.exception.code, 400)

    def test_10_comparison_invalid_distance(self):
        with self.assertRaises(HTTPError) as context:
            self.post_api("/compare-transport", {
                "customer_name": "Comparison Test 5",
                "distance_km": 0, "weight_kg": 3,
                "transportation_1": "Motorcycle",
                "transportation_2": "Car",
            })
        self.assertEqual(context.exception.code, 400)

    # POST /delay-estimate (5 test cases)
    def test_11_delay_estimation_with_added_delay(self):
        status, data = self.post_api("/delay-estimate", self.valid_payload(
            additional_delay_minutes=15
        ))
        self.assertEqual(status, 200)
        self.assertGreater(data["model_predicted_minutes"], 0)
        self.assertEqual(data["additional_delay_minutes"], 15)
        self.assertEqual(
            data["adjusted_estimated_minutes"],
            data["model_predicted_minutes"] + 15,
        )
        self.assertEqual(data["delivery_status"], "DELAY ADDED")

    def test_12_delay_estimation_without_added_delay(self):
        status, data = self.post_api("/delay-estimate", self.valid_payload(
            additional_delay_minutes=0
        ))
        self.assertEqual(status, 200)
        self.assertEqual(
            data["adjusted_estimated_minutes"],
            data["model_predicted_minutes"],
        )
        self.assertEqual(data["delivery_status"], "ON ESTIMATE")

    def test_13_invalid_additional_delay(self):
        with self.assertRaises(HTTPError) as context:
            self.post_api("/delay-estimate", self.valid_payload(
                additional_delay_minutes=-5
            ))
        self.assertEqual(context.exception.code, 400)
        error_data = json.loads(context.exception.read().decode("utf-8"))
        self.assertIn("Additional delay", error_data["error"])

    def test_14_delay_estimation_with_long_delay(self):
        status, data = self.post_api("/delay-estimate", self.valid_payload(
            additional_delay_minutes=120
        ))
        self.assertEqual(status, 200)
        self.assertEqual(data["additional_delay_minutes"], 120)
        self.assertEqual(
            data["adjusted_estimated_minutes"],
            data["model_predicted_minutes"] + 120,
        )
        self.assertEqual(data["delivery_status"], "DELAY ADDED")

    def test_15_delay_estimation_with_small_delay(self):
        status, data = self.post_api("/delay-estimate", self.valid_payload(
            additional_delay_minutes=5
        ))
        self.assertEqual(status, 200)
        self.assertEqual(data["additional_delay_minutes"], 5)
        self.assertEqual(
            data["adjusted_estimated_minutes"],
            data["model_predicted_minutes"] + 5,
        )
        self.assertEqual(data["delivery_status"], "DELAY ADDED")

    # GET /history
    def test_12_prediction_history(self):
        status, data = self.get_api("/history")
        self.assertEqual(status, 200)
        self.assertIsInstance(data, dict)
        self.assertIn("count", data)
        self.assertIn("history", data)
        self.assertEqual(data["count"], len(data["history"]))
        self.assertIsInstance(data["history"], list)

    # GET /health
    def test_13_health(self):
        status, data = self.get_api("/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["offline"])


if __name__ == "__main__":
    try:
        with urlopen(URL + "/health", timeout=2):
            pass
    except (URLError, TimeoutError):
        raise SystemExit("Cannot connect. Start the API with: python app.py")

    unittest.main(verbosity=2)
