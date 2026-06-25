import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages com.clickhouse:clickhouse-jdbc:0.4.6 pyspark-shell"

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Bucketizer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator

# ============================================================
# KAPPA ARCHITECTURE — Periodic Retraining
#
# Dalam Kappa Architecture, tidak ada Batch Layer terpisah.
# Retraining dilakukan dengan membaca dari ClickHouse
# (Serving Layer) — satu-satunya sumber kebenaran data.
# Data di ClickHouse diisi hanya oleh kappa_stream_ingest.py
#
# Aliran:
#   ClickHouse (diisi oleh Stream Layer)
#     → Spark ML Training
#       → Model tersimpan → dipakai realtime_predictor.py
# ============================================================

print("="*60)
print("  KAPPA ARCHITECTURE — Periodic Model Retraining")
print("  Sumber data: ClickHouse (diisi hanya dari stream Kafka)")
print("="*60)

spark = SparkSession.builder \
    .appName("KappaModelRetraining") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Baca dari ClickHouse (Serving Layer — satu-satunya sumber data di Kappa)
print("\n[Training] Membaca data dari ClickHouse (Serving Layer)...")
query = """(
    SELECT
        tip_amount       AS tipAmount,
        tripDistance,
        passenger_count  AS passengerCount,
        toFloat32(puLocationId)          AS pickup_zone,
        toFloat32(doLocationId)          AS dropoff_zone,
        toHour(lpepPickupDatetime)       AS pickup_hour,
        toDayOfWeek(lpepPickupDatetime)  AS pickup_day
    FROM green_taxi
    WHERE
        payment_type = '1'
        AND fareAmount > 0 AND fareAmount < 500
        AND tripDistance > 0 AND tripDistance < 100
        AND passenger_count > 0
        AND tip_amount >= 0
) AS taxi_data"""

df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:clickhouse://localhost:8123/taxi_db") \
    .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
    .option("user", "mahasiswa") \
    .option("password", "bigdata123") \
    .option("dbtable", query) \
    .load()

total = df.count()
print(f"[Training] Dataset: {total:,} baris dari ClickHouse")

if total < 1000:
    print("[Training] ⚠️  Data terlalu sedikit untuk training yang baik.")
    print("[Training] Jalankan kappa_stream_ingest.py + data_generator.py lebih lama.")
    spark.stop()
    exit(1)

# ----------------------------------------------------------------
# Feature Engineering
# FITUR: tripDistance, passengerCount, pickup_zone, dropoff_zone,
#        pickup_hour, pickup_day
# LABEL: tip_amount (nilai asli dari NYC TLC — BUKAN hitungan rumus)
#
# ⚠️  fareAmount TIDAK dimasukkan ke fitur karena tip berkorelasi
#     langsung dengan fare → akan menyebabkan data leakage!
# ----------------------------------------------------------------
FEATURE_COLS = ["tripDistance", "passengerCount", "pickup_zone",
                "dropoff_zone", "pickup_hour", "pickup_day"]

# Label Klasifikasi: 0=Rendah($0-$2), 1=Menengah($2-$5), 2=Tinggi(>$5)
bucketizer = Bucketizer(
    splits=[0.0, 2.0, 5.0, float('inf')],
    inputCol="tipAmount",
    outputCol="tipCategory"
)
df_labeled = bucketizer.transform(df)

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
data_ml = assembler.transform(df_labeled)

# Split 80/20 — reproducible dengan seed=42
train_data, test_data = data_ml.randomSplit([0.8, 0.2], seed=42)
print(f"[Training] Train: {train_data.count():,} | Test: {test_data.count():,}")

# ----------------------------------------------------------------
# Pelatihan Model Regresi (prediksi nilai tip nominal)
# ----------------------------------------------------------------
print("\n[Training] Melatih Model Regresi (Nominal Tip)...")
rf_reg = RandomForestRegressor(
    featuresCol="features",
    labelCol="tipAmount",
    numTrees=50,
    maxDepth=10,
    minInstancesPerNode=10,
    featureSubsetStrategy="sqrt",   # regularisasi: pakai subset fitur per split
    seed=42
)
model_reg = rf_reg.fit(train_data)

# ----------------------------------------------------------------
# Pelatihan Model Klasifikasi (prediksi kategori tip)
# ----------------------------------------------------------------
print("[Training] Melatih Model Klasifikasi (Kategori Tip)...")
rf_class = RandomForestClassifier(
    featuresCol="features",
    labelCol="tipCategory",
    numTrees=50,
    maxDepth=10,
    minInstancesPerNode=10,
    featureSubsetStrategy="sqrt",
    seed=42
)
model_class = rf_class.fit(train_data)

# ----------------------------------------------------------------
# Evaluasi pada Test Set
# ----------------------------------------------------------------
print("\n[Evaluasi] Menghitung metrik pada Test Set...")
pred_reg   = model_reg.transform(test_data)
pred_class = model_class.transform(test_data)

rmse = RegressionEvaluator(labelCol="tipAmount", metricName="rmse").evaluate(pred_reg)
mae  = RegressionEvaluator(labelCol="tipAmount", metricName="mae").evaluate(pred_reg)
r2   = RegressionEvaluator(labelCol="tipAmount", metricName="r2").evaluate(pred_reg)
acc  = MulticlassClassificationEvaluator(labelCol="tipCategory", metricName="accuracy").evaluate(pred_class)
f1   = MulticlassClassificationEvaluator(labelCol="tipCategory", metricName="f1").evaluate(pred_class)

# Cohen's Kappa — Metode Kappa untuk evaluasi klasifikasi
# κ = (Observed Agreement - Expected Agreement) / (1 - Expected Agreement)
# Mengukur seberapa baik model dibanding tebak acak
from pyspark.sql.functions import col as spark_col
from pyspark.sql import functions as F

n = pred_class.count()
num_classes = 3  # Rendah, Menengah, Tinggi

# Hitung po (observed agreement = akurasi)
po = acc

# Hitung pe (expected agreement by chance)
pe = 0.0
for c in [0.0, 1.0, 2.0]:
    actual_count = pred_class.filter(spark_col("tipCategory") == c).count()
    pred_count   = pred_class.filter(spark_col("prediction") == c).count()
    pe += (actual_count / n) * (pred_count / n)

kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

print("\n" + "="*60)
print("  HASIL EVALUASI MODEL (Kappa Architecture)")
print("="*60)
print(f"\n  [Regresi] Prediksi Nilai Tip Nominal")
print(f"   RMSE     : ${rmse:.4f}  (error rata-rata prediksi vs aktual)")
print(f"   MAE      : ${mae:.4f}  (error absolut rata-rata)")
print(f"   R²       : {r2:.4f}   (mendekati 1.0 = sempurna)")

print(f"\n  [Klasifikasi] Prediksi Kategori Tip")
print(f"   Akurasi  : {acc * 100:.2f}%")
print(f"   F1-Score : {f1:.4f}")
print(f"   Cohen's κ: {kappa:.4f}  ← Metrik Kappa")

# Interpretasi Cohen's Kappa
if kappa > 0.8:
    kappa_label = "Sangat Baik (Almost Perfect)"
elif kappa > 0.6:
    kappa_label = "Baik (Substantial)"
elif kappa > 0.4:
    kappa_label = "Cukup (Moderate)"
elif kappa > 0.2:
    kappa_label = "Lemah (Fair)"
else:
    kappa_label = "Sangat Lemah / Tidak lebih baik dari tebak acak"

print(f"   Interpretasi κ: {kappa_label}")

# Peringatan overfitting
if acc > 0.98:
    print("\n  ⚠️  PERINGATAN: Akurasi >98% — kemungkinan masih ada data leakage!")
elif acc > 0.85:
    print("\n  ✅ Akurasi sangat baik untuk data taksi nyata.")
else:
    print("\n  ✅ Akurasi wajar dan realistis untuk data taksi nyata.")

# Feature Importance
print(f"\n  [Feature Importance]")
for name, imp in zip(FEATURE_COLS, model_reg.featureImportances):
    bar = "█" * int(imp * 40)
    print(f"   {name:<20}: {imp:.4f} {bar}")

# Simpan Model
print("\n[Simpan] Menyimpan model ke disk...")
model_reg.write().overwrite().save("../models/taxi_reg_model")
model_class.write().overwrite().save("../models/taxi_class_model")

spark.stop()
print("\n✅ Kappa Retraining selesai! Model siap digunakan.")
print("   Jalankan realtime_predictor.py untuk prediksi real-time via Kafka.")