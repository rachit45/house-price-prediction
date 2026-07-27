from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_FILE = "house_price_model.pkl"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"{MODEL_FILE} not found. "
        "Please keep the model file in the same folder as app.py."
    )

model = joblib.load(MODEL_FILE)


# ============================================================
# VALIDATION RANGES
# ============================================================

# California Housing Dataset ranges
MIN_LONGITUDE = -124.5
MAX_LONGITUDE = -114.0

MIN_LATITUDE = 32.0
MAX_LATITUDE = 42.5

MIN_HOUSING_AGE = 1
MAX_HOUSING_AGE = 52

MIN_ROOMS = 1
MAX_ROOMS = 40000

MIN_BEDROOMS = 1
MAX_BEDROOMS = 10000

MIN_POPULATION = 1
MAX_POPULATION = 40000

MIN_HOUSEHOLDS = 1
MAX_HOUSEHOLDS = 10000

MIN_INCOME = 0.5
MAX_INCOME = 16.0

VALID_OCEAN_OPTIONS = [
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN"
]


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Default values
    values = {
        "longitude": "",
        "latitude": "",
        "housing_median_age": "",
        "total_rooms": "",
        "total_bedrooms": "",
        "population": "",
        "households": "",
        "median_income": "",
        "ocean_proximity": "<1H OCEAN"
    }

    try:

        # ----------------------------------------------------
        # GET FORM VALUES
        # ----------------------------------------------------

        values["longitude"] = request.form.get("longitude", "").strip()
        values["latitude"] = request.form.get("latitude", "").strip()
        values["housing_median_age"] = request.form.get(
            "housing_median_age", ""
        ).strip()

        values["total_rooms"] = request.form.get(
            "total_rooms", ""
        ).strip()

        values["total_bedrooms"] = request.form.get(
            "total_bedrooms", ""
        ).strip()

        values["population"] = request.form.get(
            "population", ""
        ).strip()

        values["households"] = request.form.get(
            "households", ""
        ).strip()

        values["median_income"] = request.form.get(
            "median_income", ""
        ).strip()

        values["ocean_proximity"] = request.form.get(
            "ocean_proximity", ""
        ).strip()


        # ----------------------------------------------------
        # CHECK EMPTY VALUES
        # ----------------------------------------------------

        errors = []

        required_fields = {
            "Longitude": values["longitude"],
            "Latitude": values["latitude"],
            "Housing Median Age": values["housing_median_age"],
            "Total Rooms": values["total_rooms"],
            "Total Bedrooms": values["total_bedrooms"],
            "Population": values["population"],
            "Households": values["households"],
            "Median Income": values["median_income"]
        }

        for field_name, field_value in required_fields.items():

            if field_value == "":
                errors.append(
                    f"{field_name} is required."
                )


        # If empty values exist, stop here
        if errors:

            return render_template(
                "index.html",
                errors=errors,
                values=values
            )


        # ----------------------------------------------------
        # CONVERT TO NUMBERS
        # ----------------------------------------------------

        longitude = float(values["longitude"])
        latitude = float(values["latitude"])

        housing_median_age = float(
            values["housing_median_age"]
        )

        total_rooms = float(
            values["total_rooms"]
        )

        total_bedrooms = float(
            values["total_bedrooms"]
        )

        population = float(
            values["population"]
        )

        households = float(
            values["households"]
        )

        median_income = float(
            values["median_income"]
        )

        ocean_proximity = values["ocean_proximity"]


        # ----------------------------------------------------
        # LONGITUDE VALIDATION
        # ----------------------------------------------------

        if not (
            MIN_LONGITUDE <= longitude <= MAX_LONGITUDE
        ):

            errors.append(
                f"Longitude must be between "
                f"{MIN_LONGITUDE} and {MAX_LONGITUDE}."
            )


        # ----------------------------------------------------
        # LATITUDE VALIDATION
        # ----------------------------------------------------

        if not (
            MIN_LATITUDE <= latitude <= MAX_LATITUDE
        ):

            errors.append(
                f"Latitude must be between "
                f"{MIN_LATITUDE} and {MAX_LATITUDE}."
            )


        # ----------------------------------------------------
        # HOUSING AGE
        # ----------------------------------------------------

        if not (
            MIN_HOUSING_AGE
            <= housing_median_age
            <= MAX_HOUSING_AGE
        ):

            errors.append(
                f"Housing Median Age must be between "
                f"{MIN_HOUSING_AGE} and {MAX_HOUSING_AGE} years."
            )


        # ----------------------------------------------------
        # TOTAL ROOMS
        # ----------------------------------------------------

        if not (
            MIN_ROOMS <= total_rooms <= MAX_ROOMS
        ):

            errors.append(
                f"Total Rooms must be between "
                f"{MIN_ROOMS} and {MAX_ROOMS}."
            )


        # ----------------------------------------------------
        # TOTAL BEDROOMS
        # ----------------------------------------------------

        if not (
            MIN_BEDROOMS
            <= total_bedrooms
            <= MAX_BEDROOMS
        ):

            errors.append(
                f"Total Bedrooms must be between "
                f"{MIN_BEDROOMS} and {MAX_BEDROOMS}."
            )


        # Bedrooms cannot exceed rooms
        if total_bedrooms > total_rooms:

            errors.append(
                "Total Bedrooms cannot be greater than "
                "Total Rooms."
            )


        # ----------------------------------------------------
        # POPULATION
        # ----------------------------------------------------

        if not (
            MIN_POPULATION
            <= population
            <= MAX_POPULATION
        ):

            errors.append(
                f"Population must be between "
                f"{MIN_POPULATION} and {MAX_POPULATION}."
            )


        # ----------------------------------------------------
        # HOUSEHOLDS
        # ----------------------------------------------------

        if not (
            MIN_HOUSEHOLDS
            <= households
            <= MAX_HOUSEHOLDS
        ):

            errors.append(
                f"Households must be between "
                f"{MIN_HOUSEHOLDS} and {MAX_HOUSEHOLDS}."
            )


        # Households cannot exceed population
        if households > population:

            errors.append(
                "Households cannot be greater than Population."
            )


        # ----------------------------------------------------
        # MEDIAN INCOME
        # ----------------------------------------------------

        if not (
            MIN_INCOME
            <= median_income
            <= MAX_INCOME
        ):

            errors.append(
                f"Median Income must be between "
                f"{MIN_INCOME} and {MAX_INCOME}."
            )


        # ----------------------------------------------------
        # OCEAN PROXIMITY
        # ----------------------------------------------------

        if ocean_proximity not in VALID_OCEAN_OPTIONS:

            errors.append(
                "Please select a valid Ocean Proximity."
            )


        # ----------------------------------------------------
        # IF VALIDATION FAILED
        # ----------------------------------------------------

        if errors:

            return render_template(
                "index.html",
                errors=errors,
                values=values
            )


        # ----------------------------------------------------
        # CREATE MODEL INPUT
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "longitude": [longitude],

            "latitude": [latitude],

            "housing_median_age": [
                housing_median_age
            ],

            "total_rooms": [
                total_rooms
            ],

            "total_bedrooms": [
                total_bedrooms
            ],

            "population": [
                population
            ],

            "households": [
                households
            ],

            "median_income": [
                median_income
            ],

            "ocean_proximity_<1H OCEAN": [
                int(ocean_proximity == "<1H OCEAN")
            ],

            "ocean_proximity_INLAND": [
                int(ocean_proximity == "INLAND")
            ],

            "ocean_proximity_ISLAND": [
                int(ocean_proximity == "ISLAND")
            ],

            "ocean_proximity_NEAR BAY": [
                int(ocean_proximity == "NEAR BAY")
            ],

            "ocean_proximity_NEAR OCEAN": [
                int(ocean_proximity == "NEAR OCEAN")
            ]

        })


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]


        # ----------------------------------------------------
        # NEGATIVE PREDICTION PROTECTION
        # ----------------------------------------------------

        if prediction < 0:

            return render_template(
                "index.html",

                errors=[
                    "The model could not produce a reliable "
                    "positive house price for these inputs. "
                    "Please use realistic values close to the "
                    "California Housing Dataset."
                ],

                values=values
            )


        # ----------------------------------------------------
        # FORMAT PRICE
        # ----------------------------------------------------

        prediction = round(float(prediction), 2)


        # ----------------------------------------------------
        # SHOW RESULT
        # ----------------------------------------------------

        return render_template(

            "index.html",

            prediction=f"{prediction:,.2f}",

            values=values

        )


    # ========================================================
    # INVALID NUMBER ERROR
    # ========================================================

    except ValueError:

        return render_template(

            "index.html",

            errors=[
                "Please enter valid numeric values "
                "in all numeric fields."
            ],

            values=values
        )


    # ========================================================
    # OTHER ERROR
    # ========================================================

    except Exception as e:

        return render_template(

            "index.html",

            errors=[
                f"Something went wrong: {str(e)}"
            ],

            values=values
        )


# ============================================================
# RUN FLASK APP
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )