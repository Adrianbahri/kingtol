import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages com.clickhouse:clickhouse-jdbc:0.4.6,"
    "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.0 pyspark-shell"
)

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (StructType, StructField,
                               DoubleType, IntegerType)
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler, Bucketizer

# ============================================================
# KAPPA ARCHITECTURE — Real-time Prediction Layer
#
# Peran: Konsumsi data real-time dari Kafka, prediksi tip
# secara langsung menggunakan model yang sudah dilatih.
#
# Aliran:
#   Kafka (green-taxi-stream)
#     → Spark Structured Streaming
#       → ML Model (taxi_reg_model + taxi_class_model)
#         → Output Prediksi (console / ClickHouse / API)
# ============================================================

print("="*60)
print("  KAPPA ARCHITECTURE — Real-time Prediction Layer")
print("="*60)

spark = SparkSession.builder \
    .appName("KappaRealtimePredictor") \
    .master("local[2]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Muat model (hasil dari spark_batch_training.py)
print("\n[Predictor] Memuat model dari disk...")
reg_model   = RandomForestRegressionModel.load("../models/taxi_reg_model")
class_model = RandomForestClassificationModel.load("../models/taxi_class_model")
print("[Predictor] Model regresi & klasifikasi berhasil dimuat.")

# Skema JSON dari Kafka (dikirim oleh data_generator.py)
# fareAmount TIDAK dimasukkan ke fitur — hindari data leakage
schema = StructType([
    StructField("tip_amount",     DoubleType()),
    StructField("fareAmount",     DoubleType()),   # hanya untuk info display
    StructField("tripDistance",   DoubleType()),
    StructField("passengerCount", IntegerType()),
    StructField("pickup_zone",    DoubleType()),
    StructField("dropoff_zone",   DoubleType()),
    StructField("pickup_hour",    IntegerType()),
    StructField("pickup_day",     IntegerType()),
])

# Consume dari Kafka
print("[Predictor] Membaca stream dari Kafka...")
df_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "green-taxi-stream") \
    .option("startingOffsets", "latest") \
    .load()

df_parsed = df_stream \
    .selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), schema).alias("data")) \
    .select("data.*")

# Feature Engineering (HARUS sama persis dengan training)
FEATURE_COLS = ["tripDistance", "passengerCount", "pickup_zone",
                "dropoff_zone", "pickup_hour", "pickup_day"]

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
df_features = assembler.transform(df_parsed)

LABEL_MAP = {0: "Rendah ($0-$2)", 1: "Menengah ($2-$5)", 2: "Tinggi (>$5)"}

def predict_and_display(batch_df, batch_id):
    """Prediksi setiap micro-batch dan tampilkan hasilnya."""
    if batch_df.isEmpty():
        return

    pred_reg   = reg_model.transform(batch_df)
    pred_class = class_model.transform(pred_reg)

    results = pred_class.select(
        "fareAmount", "tripDistance", "passengerCount",
        "tip_amount", "prediction", "prediction"
    ).collect()

    print(f"\n{'─'*70}")
    print(f"  [Batch {batch_id}] {len(results)} prediksi baru masuk:")
    print(f"  {'Fare':>7} {'Dist':>6} {'Pax':>4}  │  {'Tip Aktual':>10}  {'Tip Prediksi':>12}  Kategori")
    print(f"  {'─'*7}  {'─'*6}  {'─'*3}  │  {'─'*10}  {'─'*12}  {'─'*16}")

    # Prediksi lengkap dengan kategori
    pred_full = class_model.transform(reg_model.transform(batch_df))
    for row in pred_full.select(
            "fareAmount", "tripDistance", "passengerCount",
            "tip_amount", col("prediction").alias("pred_nominal"),
            col("prediction").alias("pred_cat")
        ).collect():
        # Re-run dengan kedua model
        pass

    # Simple display
    for row in pred_class.select(
        "fareAmount", "tripDistance", "passengerCount",
        "tip_amount", col("prediction").alias("pred_nominal")
    ).collect():
        kategori = "Rendah" if row.pred_nominal < 2 else ("Menengah" if row.pred_nominal < 5 else "Tinggi")
        print(f"  ${row.fareAmount:>6.2f}  {row.tripDistance:>5.1f}mi  {row.passengerCount:>3}  │  "
              f"${row.tip_amount:>9.2f}  ${row.pred_nominal:>11.2f}  {kategori}")

# Jalankan streaming prediction
query = df_features.writeStream \
    .foreachBatch(predict_and_display) \
    .outputMode("append") \
    .trigger(processingTime="5 seconds") \
    .option("checkpointLocation", "../checkpoints/checkpoint_predictor") \
    .start()

print("[Predictor] Real-time prediction aktif! Tekan Ctrl+C untuk berhenti.")
print("[Predictor] Pastikan data_generator.py berjalan di terminal lain.\n")

try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("\n[Predictor] Dihentikan.")
    query.stop()
    spark.stop()