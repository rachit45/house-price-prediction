from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    longitude = float(request.form["longitude"])
    latitude = float(request.form["latitude"])
    housing_median_age = float(request.form["housing_median_age"])
    total_rooms = float(request.form["total_rooms"])
    total_bedrooms = float(request.form["total_bedrooms"])
    population = float(request.form["population"])
    households = float(request.form["households"])
    median_income = float(request.form["median_income"])
    ocean_proximity = request.form["ocean_proximity"]

    # Create input data
    data = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity_<1H OCEAN": 0,
        "ocean_proximity_INLAND": 0,
        "ocean_proximity_ISLAND": 0,
        "ocean_proximity_NEAR BAY": 0,
        "ocean_proximity_NEAR OCEAN": 0
    }

    # Set selected ocean proximity to 1
    column_name = "ocean_proximity_" + ocean_proximity
    data[column_name] = 1

    # Convert to DataFrame
    input_data = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=f"{prediction:,.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)