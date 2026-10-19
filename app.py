"""Offline local HTTP API and browser UI. Standard-library-only."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
import json
import pickle
import math

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "delivery_model.pkl"
HOST, PORT = "127.0.0.1", 5000

def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Saved model is missing. Run: python train_model.py")
    with MODEL_FILE.open("rb") as f:
        return pickle.load(f)

def predict(model, distance, weight, transport):
    if transport not in model["transports"]:
        raise ValueError("Transportation must be Bicycle, Motorcycle, Car, or Van.")
    x = [1.0, float(distance), float(weight)] + [
        1.0 if transport == t else 0.0 for t in model["transports"][1:]
    ]
    result = sum(a*b for a, b in zip(model["coefficients"], x))
    return max(1, int(round(result)))

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "templates"), **kwargs)

    def send_json(self, status, payload):
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "null")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return super().do_GET()
        if path == "/health":
            self.send_json(200, {"status": "ok", "model_loaded": MODEL_FILE.exists(), "offline": True})
            return
        self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/predict":
            self.send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                raise ValueError("Please send valid JSON input.")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            name = str(data.get("customer_name", "")).strip()
            if not name or len(name) > 100:
                raise ValueError("Customer name is required (maximum 100 characters).")
            distance = float(data.get("distance_km", 0))
            weight = float(data.get("weight_kg", 0))
            if not math.isfinite(distance) or not 0 < distance <= 1000:
                raise ValueError("Distance must be greater than 0 and no more than 1000 km.")
            if not math.isfinite(weight) or not 0 < weight <= 500:
                raise ValueError("Weight must be greater than 0 and no more than 500 kg.")
            transport = str(data.get("transportation", ""))
            model = load_model()
            minutes = predict(model, distance, weight, transport)
            self.send_json(200, {
                "customer_name": name,
                "distance_km": distance,
                "weight_kg": weight,
                "transportation": transport,
                "estimated_delivery_minutes": minutes
            })
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            self.send_json(400, {"error": str(e)})
        except FileNotFoundError as e:
            self.send_json(500, {"error": str(e)})
        except Exception:
            self.send_json(500, {"error": "Prediction failed. Check that the model was trained successfully."})

if __name__ == "__main__":
    if not MODEL_FILE.exists():
        print("Saved model not found. Run 'python train_model.py' first.")
        raise SystemExit(1)
    print(f"Offline Delivery Time Prediction System running at http://{HOST}:{PORT}")
    print("Keep this terminal open while using the system. Press Ctrl+C to stop.")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
