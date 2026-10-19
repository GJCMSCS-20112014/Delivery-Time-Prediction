# Delivery Time Prediction System (Offline)

## Requirements
- Python 3.9 or later recommended
- No pip install required
- No Flask, NumPy, pandas, scikit-learn, or internet required
- Uses Python standard-library modules only

## Project files
- `data/delivery_dataset.csv`: 30 sample training records
- `train_model.py`: trains a multiple linear regression model using built-in Python
- `models/delivery_model.pkl`: generated saved model
- `app.py`: local HTTP API and website
- `templates/index.html`: user input interface and popup markup
- `templates/style.css`: all interface styling, including the history table popup
- `templates/script.js`: prediction form behavior and history table loading
- `data/prediction_history.csv`: persistent local prediction history
- `test_api.py`: API tests (five prediction/validation cases plus health check)

## Run it for the first time
Open VS Code in this project folder and use the Terminal:

1. Train and save the model:
   `python train_model.py`
2. Start the API (leave this terminal open):
   `python app.py`
3. Open your browser and go to:
   `http://127.0.0.1:5000`
4. Enter the customer name, distance, weight, and transportation type, then click Predict Delivery Time.

## Test the API
With `app.py` still running, open a SECOND VS Code terminal and run:
`python test_api.py`

## Restart test
1. Stop the server using Ctrl+C.
2. Start it again with `python app.py` (do NOT run `train_model.py` again).
3. Open the website and make another prediction. The app loads the saved `delivery_model.pkl`.

## Example JSON output
The values below are illustrative; your actual estimate is calculated by the trained model.

```json
{
  "customer_name": "Juan Dela Cruz",
  "distance_km": 25,
  "weight_kg": 3,
  "transportation": "Motorcycle",
  "estimated_delivery_minutes": 60
}
```

## Endpoints
- `GET /`: local user interface
- `GET /health`: server and saved-model status
- `POST /predict`: validates input, saves the prediction, and returns JSON
- `GET /history`: returns saved predictions in newest-first order

## Important notes
- The included records are synthetic sample data for an academic demonstration, not real courier records.
- The model is trained using ordinary least squares regression implemented with standard-library Python.
- Pickle files should only be loaded from sources you trust.
