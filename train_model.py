"""Train and save a delivery-time regression model using Python's standard library only."""
import csv
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "delivery_dataset.csv"
MODEL_FILE = ROOT / "models" / "delivery_model.pkl"
TRANSPORTS = ["Bicycle", "Motorcycle", "Car", "Van"]

def features(distance, weight, transport):
    # Intercept, distance, weight, and one-hot transportation indicators.
    return [1.0, float(distance), float(weight)] + [1.0 if transport == t else 0.0 for t in TRANSPORTS[1:]]

def solve_linear_system(matrix, vector):
    """Solve Ax=b with Gaussian elimination and partial pivoting."""
    n = len(vector)
    a = [list(map(float, matrix[i])) + [float(vector[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("Training data cannot produce a stable model.")
        a[col], a[pivot] = a[pivot], a[col]
        divisor = a[col][col]
        a[col] = [v / divisor for v in a[col]]
        for row in range(n):
            if row != col:
                factor = a[row][col]
                a[row] = [a[row][j] - factor * a[col][j] for j in range(n + 1)]
    return [a[i][-1] for i in range(n)]

def train():
    with DATA_FILE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 20:
        raise ValueError("The training dataset must contain at least 20 records.")
    x = [features(float(r["distance_km"]), float(r["weight_kg"]), r["transportation"]) for r in rows]
    y = [float(r["delivery_time_minutes"]) for r in rows]
    columns = len(x[0])
    xtx = [[sum(row[i] * row[j] for row in x) for j in range(columns)] for i in range(columns)]
    xty = [sum(x[k][i] * y[k] for k in range(len(x))) for i in range(columns)]
    coefficients = solve_linear_system(xtx, xty)
    model = {"coefficients": coefficients, "transports": TRANSPORTS, "training_records": len(rows),
             "model_type": "Multiple Linear Regression (ordinary least squares)"}
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_FILE.open("wb") as f:
        pickle.dump(model, f)
    print(f"Training complete: {len(rows)} records used.")
    print(f"Saved model to: {MODEL_FILE}")
    print("Model type: Multiple Linear Regression (implemented with Python standard library).")

if __name__ == "__main__":
    train()
