import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"

from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

# 1. Inisialisasi
spark = SparkSession.builder.appName("TaxiTipPredict").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# 2. Muat Model Regresi (prediksi nilai tip dalam $)
model = RandomForestRegressionModel.load("../models/taxi_reg_model")

# 3. Simulasi Input Data Baru
# CATATAN: fareAmount TIDAK dimasukkan ke fitur (untuk hindari data leakage)
# Fitur yang digunakan: tripDistance, passengerCount, pickup_zone, dropoff_zone, pickup_hour, pickup_day

contoh_trip = [
    # (dist_km, penumpang, zona_pickup, zona_dropoff, jam,  hari)
    (3.2,       1,         12.0,        140.0,         9,    2),   # Pagi weekday, jarak dekat
    (8.5,       2,         45.0,        230.0,         18,   5),   # Sore jumat, jarak menengah
    (15.0,      1,         82.0,        180.0,         22,   6),   # Malam sabtu, jarak jauh
    (1.5,       4,         10.0,        11.0,          12,   1),   # Siang senin, jarak sangat dekat
]

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

df_input   = spark.createDataFrame(contoh_trip, schema)
assembler  = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
df_features = assembler.transform(df_input)

hasil = model.transform(df_features)

print("\n" + "="*60)
print("  PREDIKSI TIP TAKSI NYC")
print("="*60)
print(f"  {'Jarak':>6} {'Pax':>4} {'PU':>5} {'DO':>5} {'Jam':>4} {'Hari':>5}  |  Prediksi Tip")
print("-"*60)

rows = hasil.select(*FEATURE_COLS, "prediction").collect()
for r in rows:
    print(f"  {r.tripDistance:>6.1f} {r.passengerCount:>4d} "
          f"{int(r.pickup_zone):>5} {int(r.dropoff_zone):>5} "
          f"{r.pickup_hour:>4}   {r.pickup_day:>5}  |  ${r.prediction:.2f}")

print("="*60)
spark.stop()