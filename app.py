
import streamlit as st
import pandas as pd
import joblib

# Load trained preprocessing + XGBoost pipeline
model = joblib.load("model.pkl")

# Get exact categories used during training
encoder = (
    model
    .named_steps["preprocessor"]
    .named_transformers_["categorical"]
)

categorical_columns = [
    "CountyOrParish2",
    "FlooringType2",
    "AttachedGarageYN",
    "FireplaceYN",
    "NewConstructionYN",
    "PoolPrivateYN",
    "ViewYN"
]

category_options = dict(
    zip(categorical_columns, encoder.categories_)
)

st.set_page_config(
    page_title="California Home Price Predictor",
    page_icon="🏠"
)

st.title("🏠 California Home Price Predictor")

st.write(
    "Enter the property and market information below "
    "to estimate the expected sale price."
)

# --------------------------------------------------
# Property Information
# --------------------------------------------------

st.subheader("Property Information")

living_area = st.number_input(
    "Living Area (sq ft)",
    min_value=0.0,
    value=1800.0
)

lot_size = st.number_input(
    "Lot Size (sq ft)",
    min_value=0.0,
    value=7000.0
)

age = st.number_input(
    "Property Age",
    min_value=0,
    value=30
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=0,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=0,
    value=2
)

stories = st.number_input(
    "Stories",
    min_value=0,
    value=1
)

garage_spaces = st.number_input(
    "Garage Spaces",
    min_value=0,
    value=2
)

parking_total = st.number_input(
    "Total Parking Spaces",
    min_value=0,
    value=2
)

# --------------------------------------------------
# Location
# --------------------------------------------------

st.subheader("Location")

latitude = st.number_input(
    "Latitude",
    value=34.0522,
    format="%.6f"
)

longitude = st.number_input(
    "Longitude",
    value=-118.2437,
    format="%.6f"
)

county = st.selectbox(
    "County",
    category_options["CountyOrParish2"]
)

school_count = st.number_input(
    "Public Schools Within 2 Miles",
    min_value=0,
    value=10
)

# --------------------------------------------------
# Additional Property Features
# --------------------------------------------------

st.subheader("Additional Features")

flooring = st.selectbox(
    "Flooring Type",
    category_options["FlooringType2"]
)

attached_garage = st.selectbox(
    "Attached Garage",
    category_options["AttachedGarageYN"],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

fireplace = st.selectbox(
    "Fireplace",
    category_options["FireplaceYN"],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

new_construction = st.selectbox(
    "New Construction",
    category_options["NewConstructionYN"],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

private_pool = st.selectbox(
    "Private Pool",
    category_options["PoolPrivateYN"],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

view = st.selectbox(
    "View",
    category_options["ViewYN"],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# --------------------------------------------------
# Market Conditions
# --------------------------------------------------

st.subheader("Market Conditions")

cpi = st.number_input(
    "CPI",
    value=320.0
)

fed_rate = st.number_input(
    "Federal Interest Rate",
    value=4.5
)

mortgage_rate = st.number_input(
    "30-Year Mortgage Rate",
    value=6.5
)

unemployment_rate = st.number_input(
    "Unemployment Rate",
    value=5.0
)

sales_tax_rate = st.number_input(
    "Sales Tax Rate",
    value=7.25
)

lot_density = st.number_input(
    "Lot Density",
    value=0.25
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Sale Price"):

    input_data = pd.DataFrame(
        [{
            "Age": age,
            "CPI": cpi,
            "FedInterestRate": fed_rate,
            "Latitude": latitude,
            "LivingArea": living_area,
            "Longitude": longitude,
            "LotDensity": lot_density,
            "LotSizeSquareFeet": lot_size,
            "MortgageRate30Fixed": mortgage_rate,
            "SalesTaxRate": sales_tax_rate,
            "UnemploymentRate": unemployment_rate,

            "BathroomsTotalInteger": bathrooms,
            "BedroomsTotal": bedrooms,
            "Stories": stories,
            "GarageSpaces": garage_spaces,
            "ParkingTotal": parking_total,
            "SchoolCount2Miles": school_count,

            "CountyOrParish2": county,
            "FlooringType2": flooring,
            "AttachedGarageYN": attached_garage,
            "FireplaceYN": fireplace,
            "NewConstructionYN": new_construction,
            "PoolPrivateYN": private_pool,
            "ViewYN": view
        }]
    )

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Sale Price: ${prediction:,.0f}"
    )
