
import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = "tourism_project/deployment/best_model.pkl"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Tourism Package Prediction")
st.write(
    "Enter the customer details below to predict whether the "
    "customer is likely to purchase the tourism package."
)


# --------------------------------------------------
# Numerical / Binary inputs
# --------------------------------------------------

age = st.number_input("Age", min_value=18, max_value=100, value=30)

city_tier = st.selectbox("City Tier", [1, 2, 3])

duration_of_pitch = st.number_input(
    "Duration of Pitch (minutes)",
    min_value=0.0,
    value=15.0
)

number_of_person = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    value=2
)

number_of_followups = st.number_input(
    "Number of Follow-ups",
    min_value=0,
    value=3
)

preferred_property_star = st.selectbox(
    "Preferred Property Star Rating",
    [3, 4, 5]
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=0,
    value=2
)

passport = st.selectbox(
    "Has Passport?",
    ["No", "Yes"]
)

pitch_satisfaction = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

own_car = st.selectbox(
    "Owns Car?",
    ["No", "Yes"]
)

number_of_children = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    value=0
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=25000.0
)


# --------------------------------------------------
# Categorical inputs
# --------------------------------------------------

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Enquiry"]
)

occupation = st.selectbox(
    "Occupation",
    [
        "Free Lancer",
        "Large Business",
        "Salaried",
        "Small Business"
    ]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

product_pitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Deluxe",
        "King",
        "Standard",
        "Super Deluxe"
    ]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Divorced",
        "Married",
        "Single",
        "Unmarried"
    ]
)

designation = st.selectbox(
    "Designation",
    [
        "AVP",
        "Executive",
        "Manager",
        "Senior Manager",
        "VP"
    ]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict"):

    # DataFrame containing EXACTLY the columns
    # used during model training
    input_data = pd.DataFrame([{

        "Age": age,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "NumberOfPersonVisiting": number_of_person,
        "NumberOfFollowups": number_of_followups,
        "PreferredPropertyStar": preferred_property_star,
        "NumberOfTrips": number_of_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": number_of_children,
        "MonthlyIncome": monthly_income,

        # Type of Contact
        "TypeofContact_Self Enquiry":
            1 if type_of_contact == "Self Enquiry" else 0,

        # Occupation
        "Occupation_Large Business":
            1 if occupation == "Large Business" else 0,

        "Occupation_Salaried":
            1 if occupation == "Salaried" else 0,

        "Occupation_Small Business":
            1 if occupation == "Small Business" else 0,

        # Gender
        "Gender_Female":
            1 if gender == "Female" else 0,

        "Gender_Male":
            1 if gender == "Male" else 0,

        # Product Pitched
        "ProductPitched_Deluxe":
            1 if product_pitched == "Deluxe" else 0,

        "ProductPitched_King":
            1 if product_pitched == "King" else 0,

        "ProductPitched_Standard":
            1 if product_pitched == "Standard" else 0,

        "ProductPitched_Super Deluxe":
            1 if product_pitched == "Super Deluxe" else 0,

        # Marital Status
        "MaritalStatus_Married":
            1 if marital_status == "Married" else 0,

        "MaritalStatus_Single":
            1 if marital_status == "Single" else 0,

        "MaritalStatus_Unmarried":
            1 if marital_status == "Unmarried" else 0,

        # Designation
        "Designation_Executive":
            1 if designation == "Executive" else 0,

        "Designation_Manager":
            1 if designation == "Manager" else 0,

        "Designation_Senior Manager":
            1 if designation == "Senior Manager" else 0,

        "Designation_VP":
            1 if designation == "VP" else 0
    }])


    # --------------------------------------------------
    # Make prediction
    # --------------------------------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    st.divider()

    if prediction == 1:
        st.success("Potential Customer ✅")
    else:
        st.warning("Customer is unlikely to purchase the package.")

    st.write(
        f"Probability of purchasing: **{probability:.2%}**"
    )
