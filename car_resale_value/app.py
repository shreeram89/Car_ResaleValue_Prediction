import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
import joblib
import streamlit as st

df = pd.read_csv(r"D:\shreeram codes\AIML\car_resale_value\car_resale_expanded.csv")

#Exploring dataset
# print(df.head())
# print((df.shape))
# print(df.info())
# print(df.isnull().sum())
# print(df.describe())
# print(df.duplicated())
# print(df["Brand"].unique())

X = df.drop(columns='Resale_Price')
y = df['Resale_Price']
X_train ,X_test ,y_train , y_test = train_test_split(X,y,test_size=0.25,random_state=42)

# print("Training rows:", len(X_train))
# print("Test rows:", len(X_test))

#Categorize the numerical and categorical Features

numeric_features = ["Car_Age","KM_Driven","Showroom_Price"]
categorical_features = ["Fuel_Type","Transmission","Brand","Model" ]

numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

model_pipeline = Pipeline([
    ("prep", preprocessor),
    ("reg", LinearRegression())
])

model_pipeline.fit(X_train,y_train)

accuracy_score = model_pipeline.score(X_test,y_test)
# print(f"test accuracy: {accuracy_score:.1%}")

joblib.dump(model_pipeline,"car_resale_predict.joblib")
# print("Model saved to car_resale_predict.joblib")

#---------------------------------------------------------------------------------------------------------------#
st.title("Car Resale Price Predictor")
st.subheader("Welcome to the car resale price predictor")

# Define a dictionary mapping Brands to their specific Models
brand_model_mapping = {
    'Skoda': ['Kodiaq', 'Superb', 'Octavia', 'Slavia', 'Kushaq'],
    'Mahindra': ['XUV400', 'XUV700', 'Marazzo', 'XUV300', 'Thar', 'Bolero', 'Scorpio'],
    'Audi': ['Q5', 'A4', 'A6', 'Q3', 'Q7'],
    'Honda': ['Civic', 'WR-V', 'Amaze', 'Jazz', 'City', 'Elevate'],
    'Mercedes-Benz': ['E-Class', 'GLA', 'GLE', 'GLC', 'C-Class', 'S-Class'],
    'Tata': ['Punch', 'Tigor', 'Harrier', 'Safari', 'Tiago', 'Curvv', 'Altroz', 'Nexon'],
    'Volkswagen': ['Taigun', 'Virtus', 'Polo', 'Tiguan', 'Vento'],
    'Hyundai': ['Tucson', 'Creta', 'Grand i10 Nios', 'Aura', 'Venue', 'Verna', 'Alcazar', 'Exter', 'i20'],
    'Maruti Suzuki': ['Alto', 'Dzire', 'Brezza', 'WagonR', 'Ertiga', 'Swift', 'Ignis', 'Grand Vitara', 'Ciaz', 'Baleno'],
    'BMW': ['X3', 'X1', '7 Series', 'X5', '5 Series', '3 Series']
}

# Get the list of unique brands for the first dropdown
brands = list(brand_model_mapping.keys())
Brand = st.selectbox("Select Car Brand", brands)

# Dynamically filter models based on the selected brand
available_models = brand_model_mapping[Brand]
Model = st.selectbox("Select Car Model", available_models)

# Remaining fields
Car_Age = st.number_input("Car Age (Years)", min_value=1, max_value=15, value=1)
KM_Driven = st.number_input("Kilometers Driven", min_value=0, value=0, step=1000)
Fuel_Type = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'CNG'])
Transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])
Showroom_Price = st.number_input("Showroom Price (in Lakhs/Rs.)", min_value=0.0, value=0.0, step=0.5)

if st.button("Predict"):
    input_row = pd.DataFrame([{
        "Showroom_Price": Showroom_Price,
        "Transmission": Transmission,
        "Fuel_Type": Fuel_Type,
        "Brand": Brand,
        "Model": Model,
        "KM_Driven": KM_Driven,
        "Car_Age": Car_Age
    }])
    st.write("Data ready for prediction:", input_row)

    
    prediction = model_pipeline.predict(input_row)[0]


    if Brand and Model and Car_Age and KM_Driven and Fuel_Type and Transmission and Showroom_Price:
        st.success(f"Resale price is: ₹{prediction:,.2f}")
    else:
        st.error("Fill all Fields")