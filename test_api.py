"""Five API test cases. Start app.py in another terminal before running this."""
import json
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

URL = "http://127.0.0.1:5000"

class DeliveryAPITests(unittest.TestCase):
    def post_prediction(self, payload):
        request = Request(URL + "/predict", data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_01_motorcycle(self):
        status, data = self.post_prediction({"customer_name":"Juan Dela Cruz","distance_km":25,"weight_kg":3,"transportation":"Motorcycle"})
        self.assertEqual(status, 200); self.assertGreater(data["estimated_delivery_minutes"], 0)
        self.assertEqual(data["customer_name"], "Juan Dela Cruz")

    def test_02_bicycle(self):
        status, data = self.post_prediction({"customer_name":"Ana Santos","distance_km":2,"weight_kg":1,"transportation":"Bicycle"})
        self.assertEqual(status, 200); self.assertGreater(data["estimated_delivery_minutes"], 0)

    def test_03_car(self):
        status, data = self.post_prediction({"customer_name":"Mark Reyes","distance_km":10,"weight_kg":4,"transportation":"Car"})
        self.assertEqual(status, 200); self.assertEqual(data["transportation"], "Car")

    def test_04_van(self):
        status, data = self.post_prediction({"customer_name":"Liza Cruz","distance_km":20,"weight_kg":7,"transportation":"Van"})
        self.assertEqual(status, 200); self.assertIsInstance(data["estimated_delivery_minutes"], int)

    def test_05_invalid_distance(self):
        try:
            self.post_prediction({"customer_name":"Test User","distance_km":0,"weight_kg":2,"transportation":"Car"})
        except HTTPError as e:
            self.assertEqual(e.code, 400)
            self.assertIn("Distance", json.loads(e.read().decode("utf-8"))["error"])
        else:
            self.fail("Invalid distance should return HTTP 400")

    def test_health(self):
        with urlopen(URL + "/health", timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])

if __name__ == "__main__":
    try:
        with urlopen(URL + "/health", timeout=2):
            pass
    except (URLError, TimeoutError):
        raise SystemExit("Cannot connect. Start the API with: python app.py")
    unittest.main(verbosity=2)
