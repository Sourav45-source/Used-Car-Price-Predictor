import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

model = pickle.load(open('vechile_price_predictor_model.pkl', 'rb'))
model_columns  = pickle.load(open('model_columns.pkl', 'rb'))   # list of strings

st.title("🚗 Used‑Car Price Predictor")

brands = [
    'Maruti', 'Hyundai', 'Ford', 'Renault', 'Mini', 'Mercedes-Benz',
    'Toyota', 'Volkswagen', 'Honda', 'Mahindra', 'Datsun', 'Tata',
    'Kia', 'BMW', 'Audi', 'Land Rover', 'Jaguar', 'MG', 'Isuzu',
    'Porsche', 'Skoda', 'Volvo', 'Lexus', 'Jeep', 'Maserati',
    'Bentley', 'Nissan', 'ISUZU', 'Ferrari', 'Mercedes-AMG',
    'Rolls-Royce', 'Force'
]

models_dict = {
    'Maruti'        : ['Swift', 'Baleno', 'Ertiga', 'Alto', 'Dzire'],
    'Hyundai'       : ['Creta', 'i20', 'Verna', 'Venue', 'Grand i10'],
    'Ford'          : ['Figo', 'EcoSport', 'Endeavour', 'Mustang', 'Freestyle'],
    'Renault'       : ['Kwid', 'Duster', 'Triber', 'Kiger'],
    'Mini'          : ['Cooper', 'Countryman', 'Clubman', 'Convertible'],
    'Mercedes-Benz' : ['C‑Class', 'E‑Class', 'GLA', 'GLE', 'S‑Class'],
    'Toyota'        : ['Fortuner', 'Innova', 'Glanza', 'Camry', 'Urban Cruiser'],
    'Volkswagen'    : ['Polo', 'Vento', 'Taigun', 'Tiguan', 'Virtus'],
    'Honda'         : ['City', 'Civic', 'Amaze', 'Jazz', 'WR‑V'],
    'Mahindra'      : ['Scorpio', 'XUV500', 'Thar', 'Bolero', 'XUV700'],
    'Datsun'        : ['GO', 'redi‑GO', 'GO Plus'],
    'Tata'          : ['Nexon', 'Altroz', 'Harrier', 'Tiago', 'Safari'],
    'Kia'           : ['Seltos', 'Sonet', 'Carens', 'EV6'],
    'BMW'           : ['3 Series', '5 Series', 'X1', 'X5', 'X7'],
    'Audi'          : ['A4', 'A6', 'Q3', 'Q5', 'Q7'],
    'Land Rover'    : ['Defender', 'Discovery', 'Range Rover Evoque', 'Range Rover Sport'],
    'Jaguar'        : ['XE', 'XF', 'F‑Pace', 'E‑Pace', 'F‑Type'],
    'MG'            : ['Hector', 'Astor', 'Gloster', 'ZS EV'],
    'Isuzu'         : ['D‑Max', 'MU‑X', 'V‑Cross'],
    'Porsche'       : ['911', 'Cayenne', 'Macan', 'Panamera', 'Taycan'],
    'Skoda'         : ['Slavia', 'Kushaq', 'Octavia', 'Superb', 'Kodiaq'],
    'Volvo'         : ['XC40', 'XC60', 'XC90', 'S60', 'S90'],
    'Lexus'         : ['ES', 'RX', 'LX', 'NX', 'UX'],
    'Jeep'          : ['Compass', 'Wrangler', 'Meridian', 'Grand Cherokee'],
    'Maserati'      : ['Ghibli', 'Quattroporte', 'Levante', 'MC20'],
    'Bentley'       : ['Continental GT', 'Flying Spur', 'Bentayga'],
    'Nissan'        : ['Magnite', 'Kicks', 'Terrano', 'GT‑R', 'Micra'],
    'ISUZU'         : ['D‑Max', 'MU‑X', 'V‑Cross'],         # same as Isuzu, different casing
    'Ferrari'       : ['Roma', 'F8 Tributo', 'SF90 Stradale', '812 Superfast'],
    'Mercedes-AMG'  : ['A45', 'C63', 'GT R', 'G63'],
    'Rolls-Royce'   : ['Phantom', 'Ghost', 'Cullinan', 'Wraith', 'Dawn'],
    'Force'         : ['Gurkha', 'Trax Cruiser']
}


brand       = st.selectbox("Brand", sorted(brands),key="brand_sel")
model_name  = st.selectbox("Model", models_dict[brand], key="model_sel")

km_driven   = st.number_input("Kilometers Driven", min_value=0, step=1, key="km_input")

fuel_type   = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'],key="fuel_sel")

current_year = datetime.now().year
buy_year     = st.number_input(
    "Buying / Manufacturing Year",
    min_value=1990,
    max_value=current_year,
    value=2015,
    step=1,key="buy_year_input"
)

input_dict = {
    'brand'        : brand,
    'model'        : model_name,
    'km_driven'    : km_driven,
    'fuel_type'    : fuel_type,
    'Buying_year'  : buy_year
}

df = pd.DataFrame([input_dict])

# one‑hot‑encode exactly like training
df_dummies = pd.get_dummies(df)

# re‑index to training column order, fill missing cols with 0
df_model = df_dummies.reindex(columns=model_columns, fill_value=0)

# 4.  Predict
if st.button("Predict Selling Price"):
    price_pred = model.predict(df_model)[0]
    st.success(f"Estimated Selling Price: ₹ {price_pred:,.2f} Rupees")
