import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"

import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

st.set_page_config(
    page_title="Prediksi Tip Taksi NYC",
    page_icon="🚖",
    layout="centered"
)

@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("TaxiDashboard") \
        .master("local[*]") \
        .getOrCreate()

@st.cache_resource
def load_models():
    reg_model   = RandomForestRegressionModel.load("../models/taxi_reg_model")
    class_model = RandomForestClassificationModel.load("../models/taxi_class_model")
    return reg_model, class_model

spark = get_spark()
model_reg, model_class = load_models()

# ─── UI ───────────────────────────────────────────────────────────
st.title("🚖 Prediksi Tip Taksi NYC")
st.caption("Kappa Architecture · Spark MLlib · Random Forest")
st.markdown("---")

st.subheader("Masukkan Data Perjalanan")

col1, col2 = st.columns(2)
with col1:
    dist  = st.number_input("Jarak Perjalanan (miles)", min_value=0.1, max_value=100.0, value=3.2, step=0.1)
    pax   = st.slider("Jumlah Penumpang", 1, 6, 1)
    pu    = st.number_input("Zona Pickup (1–263)", min_value=1, max_value=263, value=140)

with col2:
    do    = st.number_input("Zona Dropoff (1–263)", min_value=1, max_value=263, value=236)
    hour  = st.slider("Jam Pickup (0–23)", 0, 23, 9)
    day   = st.selectbox("Hari", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"],
                         index=1)

day_map = {"Senin": 1, "Selasa": 2, "Rabu": 3, "Kamis": 4,
           "Jumat": 5, "Sabtu": 6, "Minggu": 7}

st.markdown("---")

if st.button("🔮 Prediksi Tip", use_container_width=True):
    # Kappa Architecture: fareAmount TIDAK dipakai sebagai fitur
    FEATURE_COLS = ["tripDistance", "passengerCount", "pickup_zone",
                    "dropoff_zone",  "pickup_hour",    "pickup_day"]

    schema = StructType([
        StructField("tripDistance",   DoubleType()),
        StructField("passengerCount", IntegerType()),
        StructField("pickup_zone",    DoubleType()),
        StructField("dropoff_zone",   DoubleType()),
        StructField("pickup_hour",    IntegerType()),
        StructField("pickup_day",     IntegerType()),
    ])

    data = [(float(dist), int(pax), float(pu), float(do), int(hour), int(day_map[day]))]
    df   = spark.createDataFrame(data, schema)

    assembler   = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
    df_features = assembler.transform(df)

    pred_nominal = model_reg.transform(df_features).select("prediction").collect()[0][0]
    pred_cat     = model_class.transform(df_features).select("prediction").collect()[0][0]

    label_map = {0.0: "🔵 Rendah ($0–$2)", 1.0: "🟡 Menengah ($2–$5)", 2.0: "🟢 Tinggi (>$5)"}

    m1, m2 = st.columns(2)
    m1.metric("💰 Estimasi Tip Nominal", f"${pred_nominal:.2f}")
    m2.metric("🏷️ Kategori Tip", label_map.get(pred_cat, "N/A"))

    st.success("Prediksi selesai menggunakan Kappa Architecture + Spark MLlib Random Forest")

    with st.expander("ℹ️ Catatan Metode"):
        st.markdown("""
        **Kappa Architecture** — Pipeline ini menggunakan satu jalur streaming:
        - Data masuk via **Kafka** (stream tunggal, tidak ada batch terpisah)
        - Model dilatih dari **ClickHouse** (Serving Layer yang diisi oleh stream)
        - Prediksi real-time via **Spark Structured Streaming**

        **Fitur yang digunakan:**
        - Jarak, penumpang, zona pickup/dropoff, jam, hari
        - `fareAmount` **tidak digunakan** (menghindari data leakage)

        **Evaluasi Model:**
        - RMSE, MAE, R² untuk regresi
        - Akurasi, F1, **Cohen's Kappa (κ)** untuk klasifikasi
        """)