import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

# Load the trained models
sampah_model = joblib.load('Machine/model_sampahregresi.pkl')
suhu_model = joblib.load('Machine/model_suhuregresi.pkl')

# Fungsi pra-pemrosesan data
def preprocess_data(df, columns):
    for col in columns:
        if df[col].dtype == 'O':
            non_numeric_values = df[col][pd.to_numeric(df[col], errors='coerce').isna()].unique()
            for value in non_numeric_values:
                if '-' in value:
                    lower, upper = map(int, value.split('-'))
                    mean_value = (lower + upper) / 2
                    df[col] = df[col].replace(value, mean_value)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Fungsi prediksi timbulan sampah dengan model regresi
def predict_timbulan(year, avg_per_day):
    X_new = np.array([[year, avg_per_day]])
    pred_timbulan = sampah_model.predict(X_new)[0]
    return pred_timbulan

# Fungsi prediksi suhu harian dengan model regresi
def predict_suhu(date):
    date_ordinal = date.toordinal()  # Convert date to ordinal for prediction
    X_new = np.array([[date_ordinal]])
    pred_suhu = suhu_model.predict(X_new)[0]
    return pred_suhu

# Streamlit UI
st.title("Prediksi Timbulan Sampah dan Suhu Harian")

# Input tahun untuk prediksi timbulan sampah
year = st.number_input("Masukkan tahun prediksi:", min_value=2000, max_value=2100, value=2025, step=1)
avg_per_day = st.number_input("Rata-rata timbulan sampah per hari (ton):", min_value=0.0, value=500.0)

# Prediksi timbulan sampah
if st.button("Prediksi Timbulan Sampah"):
    result_timbulan = predict_timbulan(year, avg_per_day)
    st.write(f"Prediksi total timbulan sampah untuk tahun {year}: {result_timbulan:.2f} ton")

# Input tanggal untuk prediksi suhu
date_input = st.date_input("Masukkan tanggal prediksi suhu harian:", value=datetime.today())

# Prediksi suhu harian
if st.button("Prediksi Suhu Harian"):
    result_suhu = predict_suhu(date_input)
    st.write(f"Prediksi suhu rata-rata harian untuk tanggal {date_input}: {result_suhu:.2f} °C")