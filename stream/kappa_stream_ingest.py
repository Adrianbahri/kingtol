import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages com.clickhouse:clickhouse-jdbc:0.4.6,"
    "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.0 pyspark-shell"
)

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, current_timestamp
from pyspark.sql.types import (StructType, StructField,
                               DoubleType, IntegerType, FloatType)

# ============================================================
# KAPPA ARCHITECTURE — Stream Ingest Layer
#
# Peran: Satu-satunya jalur data masuk ke ClickHouse.
# Tidak ada batch ingest terpisah.
#
# Aliran:
#   Kafka (green-taxi-stream)
#     → Spark Structured Streaming
#       → ClickHouse (Serving Layer)
# ============================================================

print("="*60)
print("  KAPPA ARCHITECTURE — Stream Ingest to ClickHouse")
print("="*60)

spark = SparkSession.builder \
    .appName("KappaStreamIngest") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Skema data dari Kafka Producer (data_generator.py)
schema = StructType([
    StructField("tip_amount",     DoubleType()),
    StructField("fareAmount",     DoubleType()),
    StructField("tripDistance",   DoubleType()),
    StructField("passengerCount", IntegerType()),
    StructField("pickup_zone",    DoubleType()),
    StructField("dropoff_zone",   DoubleType()),
    StructField("pickup_hour",    IntegerType()),
    StructField("pickup_day",     IntegerType()),
])

print("\n[Stream] Membaca dari Kafka topic: green-taxi-stream")
df_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "green-taxi-stream") \
    .option("startingOffsets", "earliest") \
    .load()

# Parse JSON dari Kafka
df_parsed = df_stream \
    .selectExpr("CAST(value AS STRING) as json_str", "timestamp") \
    .select(from_json(col("json_str"), schema).alias("data"), "timestamp") \
    .select("data.*", "timestamp")

# Filter data valid (sama dengan filter training)
df_clean = df_parsed \
    .filter(col("fareAmount") > 0) \
    .filter(col("fareAmount") < 500) \
    .filter(col("tripDistance") > 0) \
    .filter(col("tripDistance") < 100) \
    .filter(col("passengerCount") > 0) \
    .filter(col("tip_amount") >= 0)

import clickhouse_connect

def write_to_clickhouse(batch_df, batch_id):
    """
    Tulis setiap micro-batch dari Kafka ke ClickHouse.
    Ini adalah 'foreachBatch' sink — semua data masuk via stream.
    """
    if batch_df.isEmpty():
        return

    rows = batch_df.select(
        "tip_amount", "fareAmount", "tripDistance",
        "passengerCount", "pickup_zone", "dropoff_zone",
        "pickup_hour", "pickup_day"
    ).collect()

    import pandas as pd
    from datetime import datetime

    pdf = pd.DataFrame([r.asDict() for r in rows])
    # Tambahkan kolom yang dibutuhkan skema ClickHouse
    now = datetime.now()
    pdf["VendorID"]            = "stream"
    pdf["lpepPickupDatetime"]  = now
    pdf["lpepDropoffDatetime"] = now
    pdf["passenger_count"]     = pdf["passengerCount"].astype(float)
    pdf["tripDistance"]        = pdf["tripDistance"].astype(float)
    pdf["puLocationId"]        = pdf["pickup_zone"].astype(int).astype(str)
    pdf["doLocationId"]        = pdf["dropoff_zone"].astype(int).astype(str)
    pdf["RatecodeID"]          = "1"
    pdf["payment_type"]        = "1"
    pdf["fareAmount"]          = pdf["fareAmount"].astype(float)
    pdf["tip_amount"]          = pdf["tip_amount"].astype(float)

    kolom_ch = [
        "VendorID", "lpepPickupDatetime", "lpepDropoffDatetime",
        "passenger_count", "tripDistance", "puLocationId",
        "doLocationId", "RatecodeID", "payment_type",
        "fareAmount", "tip_amount"
    ]

    client = clickhouse_connect.get_client(
        host="localhost", port=8123,
        username="mahasiswa", password="bigdata123",
        database="taxi_db"
    )
    client.insert_df("green_taxi", pdf[kolom_ch])
    print(f"[Batch {batch_id}] ✅ {len(pdf):,} baris ditulis ke ClickHouse")

# Tulis ke ClickHouse via foreachBatch (satu-satunya jalur data masuk)
query = df_clean.writeStream \
    .foreachBatch(write_to_clickhouse) \
    .outputMode("append") \
    .trigger(processingTime="10 seconds") \
    .option("checkpointLocation", "../checkpoints/checkpoint_ingest") \
    .start()

print("[Stream] Ingest berjalan... Tekan Ctrl+C untuk berhenti")
print("[Stream] Data dari Kafka akan masuk ke ClickHouse setiap 10 detik\n")

try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("\n[Stream] Ingest dihentikan.")
    query.stop()
    spark.stop()
