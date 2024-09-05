import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from datetime import datetime
 import matplotlib.pyplot as plt

# Membaca data dari file CSV (dengan penanganan kesalahan)
try:
    df_ledakan = pd.read_csv('Data/data_ledakan.csv')
    df_suhu_rata_rata = pd.read_csv('Data/data_suhu.csv')
    df_timbulan = pd.read_csv('Data/data_timbulankota.csv')
    df_produksi_metana = pd.read_csv('Data/data_sampah.csv')
    df_kelembaban = pd.read_csv('Data/data_kelembabanudara.csv')
except FileNotFoundError:
    st.error("File data tidak ditemukan. Pastikan file CSV berada di direktori yang benar.")
    st.stop()  # Hentikan eksekusi jika file tidak ditemukan

# Fungsi pra-pemrosesan data (dengan docstring)
def preprocess_data(df, columns):
    """
    Membersihkan data dengan mengubah nilai non-numerik menjadi nilai numerik rata-rata.

    Args:
        df: DataFrame yang akan diproses.
        columns: Daftar kolom yang akan dibersihkan.

    Returns:
        DataFrame yang sudah dibersihkan.
    """
    for col in columns:
        if df[col].dtype == 'O':  # Jika kolom bertipe objek (string)
            non_numeric_values = df[col][pd.to_numeric(df[col], errors='coerce').isna()].unique()

            for value in non_numeric_values:
                if '-' in value:  # Hanya proses nilai dengan tanda "-"
                    lower, upper = map(int, value.split('-'))
                    mean_value = (lower + upper) / 2
                    df[col] = df[col].replace(value, mean_value)

        df[col] = pd.to_numeric(df[col], errors='coerce')  # Konversi ke numerik, abaikan error

    return df
# Fungsi untuk prediksi timbulan sampah
def predict_timbulan(year, df_timbulan):
    """
    Memprediksi timbulan sampah untuk tahun tertentu berdasarkan data historis.

    Args:
        year: Tahun yang akan diprediksi.
        df_timbulan: DataFrame data timbulan sampah.

    Returns:
        Prediksi timbulan sampah atau rata-rata jika tidak ada data untuk tahun tersebut.
    """
    avg_timbulan = df_timbulan[df_timbulan['Kabupaten/Kota'] == 'Kartamantul-gupro']['Timbulan Sampah Tahunan(ton)'].mean()
    last_timbulan_df = df_timbulan[(df_timbulan['Tahun'] <= year)]

    if not last_timbulan_df.empty:
        last_timbulan = last_timbulan_df.groupby('Tahun')['Timbulan Sampah Tahunan(ton)'].sum().iloc[-1]
    else:
        last_timbulan = None

    return last_timbulan if last_timbulan is not None else avg_timbulan

# Fungsi prediksi suhu rata-rata harian
def predict_suhu(date, df_suhu_rata_rata):
    """
    Memprediksi suhu rata-rata harian untuk tanggal tertentu.

    Args:
        date: Tanggal yang akan diprediksi.
        df_suhu_rata_rata: DataFrame data suhu rata-rata.

    Returns:
        Prediksi suhu rata-rata harian atau None jika tidak ada data.
    """
    year = date.year
    month_name = date.strftime('%B')

    matching_data = df_suhu_rata_rata[df_suhu_rata_rata['Tahun'] == year]
    if not matching_data.empty:
        predicted_suhu = matching_data[f'Bulan {month_name}'].values[0]
    else:
        predicted_suhu = None  # Atau berikan nilai default lainnya jika diperlukan

    return predicted_suhu

# Fungsi prediksi kelembaban udara rata-rata harian
def predict_lembab(date, df_lembab_rata_rata):
    """
    Memprediksi kelembaban udara rata-rata harian untuk tanggal tertentu.

    Args:
        date: Tanggal yang akan diprediksi.
        df_lembab_rata_rata: DataFrame data kelembaban rata-rata.

    Returns:
        Prediksi kelembaban udara rata-rata harian atau None jika tidak ada data.
    """
    year = date.year
    month_name = date.strftime('%B')  # Mendapatkan nama bulan

    # Filter data berdasarkan tahun
    matching_data = df_lembab_rata_rata[df_lembab_rata_rata['Tahun'] == year]

    if not matching_data.empty:
        # Ambil nilai kelembaban dari kolom bulan yang sesuai
        predicted_lembab = matching_data[f'Bulan {month_name}'].values[0]
    else:
        predicted_lembab = None 

    return predicted_lembab

# Fungsi menghitung kepadatan sampah
def calculate_density(berat_sampah_kg, luas_tpa_m2, kedalaman_timbunan_m):
    """
    Menghitung kepadatan sampah.

    Args:
        berat_sampah_kg: Berat sampah dalam kilogram.
        luas_tpa_m2: Luas TPA dalam meter persegi.
        kedalaman_timbunan_m: Kedalaman timbunan dalam meter.

    Returns:
        Kepadatan sampah dalam kg/m^3.
    """
    volume_timbunan_m3 = luas_tpa_m2 * kedalaman_timbunan_m
    kepadatan = berat_sampah_kg / volume_timbunan_m3
    return kepadatan

# Pra-pemrosesan data
columns_to_preprocess = ['Kandungan Metan (%)', 'Suhu (C)', 'Kelembaban (%)', 'Kepadatan/Densitas (kg/m)']
df_ledakan = preprocess_data(df_ledakan, columns_to_preprocess)

# Menghitung nilai statistik
median_values = df_ledakan[columns_to_preprocess].median()
mean_values = df_ledakan[columns_to_preprocess].mean()
max_values = df_ledakan[columns_to_preprocess].max()

# Membuat layout aplikasi Streamlit
st.title('Prediksi Potensi Ledakan TPA Piyungan')

# Widget input
col1, col2 = st.columns(2)

# Widget input tanggal
with col1:
    try:
        input_date = st.date_input('Masukkan Tanggal (YYYY-MM-DD):')
        if input_date > datetime.today().date():
            st.error("Tanggal tidak boleh melebihi hari ini.")
    except ValueError:
        st.error("Format tanggal tidak valid. Pastikan formatnya YYYY-MM-DD.")
# Widget input suhu
with col2:
    input_suhu = st.number_input('Masukkan Suhu (dalam derajat Celsius):', value=30, step=1)
    if input_suhu < 0 or input_suhu > 50:
        st.error("Suhu harus berada dalam rentang 0-50 derajat Celsius.")

# Logika untuk menentukan tanggal dan suhu
date = input_date if input_date else datetime.today()

suhu = int(input_suhu) if input_suhu else predict_suhu(date, df_suhu_rata_rata)
if suhu is None:
    st.warning("Tidak ada data suhu untuk tanggal tersebut. Menggunakan suhu input atau default.")
    suhu = int(input_suhu) if input_suhu else 30  # Ganti 30 dengan nilai default yang sesuai

# Prediksi dan perhitungan
year = date.year
berat_relatif = predict_timbulan(year, df_timbulan)
berat_relatif = pd.to_numeric(berat_relatif, errors='coerce')  # Konversi dengan pandas, ganti nilai non-numerik dengan NaN
# atau
berat_relatif = float(berat_relatif)  # Konversi dengan fungsi float()
Kandungan_Metan = 50 * berat_relatif / 1000
Kelembaban = predict_lembab
berat_sampah = berat_relatif / 1000
luas_TPA_Piyungan = 12.5  # dalam hektar
luas_TPA_Piyungan_m2 = luas_TPA_Piyungan * 10000
berat_sampah_kg = berat_sampah * 1000
Kepadatan = calculate_density(berat_sampah_kg, luas_TPA_Piyungan_m2, 5)  # Kedalaman timbunan 5 meter

# Menghitung laju emisi metana
MCF = berat_relatif
DOC = 0.6
DOCF = 0.2
F = 0.5
Emisi = MCF * DOC * DOCF * F * 16/12

# Membandingkan nilai metana
if Kandungan_Metan >= max_values['Kandungan Metan (%)']:
    hasil_perbandingan_metan = "Kandungan metan sangat tinggi!"
elif Kandungan_Metan == median_values['Kandungan Metan (%)']:
    hasil_perbandingan_metan = "Kandungan metan cukup tinggi"
elif Kandungan_Metan < median_values['Kandungan Metan (%)']:
    hasil_perbandingan_metan = "Kandungan metan rendah"

# Logika prediksi
if Kandungan_Metan > 40 and suhu > 30 and Kelembaban > 60 and Kepadatan > 400:
    hasil_perbandingan = "Sangat berpotensi meledak!"
elif 20 < Kandungan_Metan <= 40 and 25 < suhu <= 30 and 50 < Kelembaban <= 60 and 200 < Kepadatan <= 400:
    hasil_perbandingan = "Berpotensi meledak!"
else:
    hasil_perbandingan = "Relatif aman, namun tetap perlu pemantauan berkala."

# Menampilkan hasil
st.write(f"""
{hasil_perbandingan_metan} {hasil_perbandingan} 
Kandungan metan saat ini ada pada angka {Kandungan_Metan:.2f}% dengan suhu wilayah sekitar {suhu:.2f}°C dan kepadatan/densitas sebesar {Kepadatan:.2f} kg/m³.
Sementara Besaran laju emisi metana ada pada angka {Emisi:.2f} ton/tahun.
""")
# Fungsi untuk membuat visualisasi timbulan sampah
def visualize_timbulan_sampah(df_timbulan):
    """
    Membuat visualisasi data historis dan prediksi timbulan sampah.

    Args:
        df_timbulan: DataFrame data timbulan sampah.
    """
# Menentukan rentang tahun untuk data historis dan prediksi
last_year = df_timbulan['Tahun'].max()
first_year = df_timbulan['Tahun'].min()

# Prediksi 5 tahun ke depan dan visualisasi data beberapa tahun sebelumnya
years_to_predict = list(range(last_year + 1, last_year + 6))  # 5 tahun ke depan
years_to_visualize = list(range(first_year, last_year + 1))  # Tahun historis

# Membuat list kosong untuk prediksi timbulan sampah
predicted_timbulan = []

# Iterasi untuk prediksi timbulan di tahun yang akan datang
for year in years_to_predict:
    prediction = predict_timbulan(year, df_timbulan)
    predicted_timbulan.append(prediction)

# Data historis dari DataFrame asli
df_historis = df_timbulan[['Tahun', 'Timbulan Sampah Tahunan(ton)']].copy()
df_prediksi = pd.DataFrame({'Tahun': years_to_predict, 'Timbulan Sampah Tahunan(ton)': predicted_timbulan})

# Menggabungkan data historis dan prediksi ke dalam satu DataFrame
df_visualisasi = pd.concat([df_historis, df_prediksi])

# Visualisasi menggunakan Altair
chart = alt.Chart(df_visualisasi).mark_line(point=True).encode(
    x='Tahun:O',
    y='Timbulan Sampah Tahunan(ton):Q',
    tooltip=['Tahun', 'Timbulan Sampah Tahunan(ton)']
).properties(
    title='Data Historis dan Prediksi Timbulan Sampah'
).interactive()

# Simpan visualisasi ke file JSON
chart.save('prediksi_timbulan_sampah.json')

# Tampilkan chart di Streamlit
st.altair_chart(chart, use_container_width=True)


def visualize_produksi_metana(df_ledakan):
    """
    Membuat visualisasi produksi metana tahunan.

    Args:
        df_ledakan: DataFrame data ledakan.
    """
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Menampilkan 5 baris pertama
print(df_ledakan.head().to_markdown(index=False, numalign="left", stralign="left"))

# Menampilkan informasi tentang kolom (nama dan tipe data)
print(df_ledakan.info())

import altair as alt

# 1. Extract `Tahun` from the `Tanggal Kejadian` column
df_ledakan['Tahun'] = pd.to_datetime(df_ledakan['Tanggal Kejadian'], format='%A,%B %d,%Y').dt.year

# 2. Create a new column `Produksi Metana Harian (ton)` 
df_ledakan['Produksi Metana Harian (ton)'] = df_ledakan['Kandungan Metan (%)'] * 10 

# 3. Group the data by `Tahun` and sum the `Produksi Metana Harian (ton)`
df_produksi_metana_tahunan = df_ledakan.groupby('Tahun', as_index=False)['Produksi Metana Harian (ton)'].sum()

# 4. Create a line chart using Altair
chart = alt.Chart(df_produksi_metana_tahunan).mark_line(point=True).encode(
    x='Tahun:O',
    y='Produksi Metana Harian (ton):Q',
    tooltip=['Tahun', 'Produksi Metana Harian (ton)']
)

# 5. Add title to the chart
chart = chart.properties(title='Grafik Produksi Metana Tahunan')

# 6. Save the chart as a JSON file
chart.save('produksi_metana_tahunan_line_chart.json')

# Penjelasan
