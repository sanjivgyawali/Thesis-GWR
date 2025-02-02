from flask import Flask, request, render_template
import numpy as np
from joblib import load
from mgwr.gwr import GWR

# Load the trained GWR model, bandwidth, and data
try:
    gwr_results = load('gwr_model.pkl')  # GWR model results
    gwr_bw = load('gwr_bw.pkl')  # Bandwidth
    exp_vars, y, coords = load('gwr_data.pkl')  # Training data
except Exception as e:
    print(f"Error loading model or data: {e}")
    gwr_results, gwr_bw, exp_vars, y, coords = None, None, None, None

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from form (WITHOUT review_scores_rating)
        accommodates = int(request.form['accommodates'])
        bedrooms = int(request.form['bedrooms'])
        beds = int(request.form['beds'])
        lat = round(float(request.form['latitude']), 5)  # Round to 5 decimal places
        lon = round(float(request.form['longitude']), 5)  # Round to 5 decimal places

        # Debugging: Print input values
        print(f"Received Latitude: {lat}, Longitude: {lon}")

        # Fix: Ensure exp_vars has the same number of features as pred_exp_vars
        exp_vars_updated = exp_vars[:, :3]  # Keep only first 3 columns (drop extra feature)

        print(f"Original exp_vars shape: {exp_vars.shape}")
        print(f"Updated exp_vars shape (without review_scores_rating): {exp_vars_updated.shape}")

        # Prepare input for prediction (WITH correct number of features)
        pred_exp_vars = np.array([[accommodates, bedrooms, beds]])  # No intercept term
        pred_coords = np.array([[lon, lat]])  # New location coordinates

        # Perform GWR prediction by recalculating weights for the new location
        gwr_model = GWR(coords, y, exp_vars_updated, gwr_bw)  
        gwr_fitted = gwr_model.fit()  

        # Get local parameters for the given location
        pred_results = gwr_model.predict(pred_coords, pred_exp_vars)  
        predicted_price = np.exp(float(pred_results.predictions[0]))  

        return render_template('index.html', prediction_text=f'Predicted Price: ${predicted_price:.2f}')

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
