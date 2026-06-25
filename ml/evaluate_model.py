import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages com.clickhouse:clickhouse-jdbc:0.4.6 pyspark-shell"

from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler, Bucketizer
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.sql.functions import col

# ============================================================
# KAPPA ARCHITECTURE — Model Evaluation
#
# Evaluasi model menggunakan data dari ClickHouse
# (Serving Layer — diisi hanya oleh Kafka stream)
# Metrik: RMSE, MAE, R², Akurasi, F1, Cohen's Kappa
# ============================================================

print("="*60)
print("  KAPPA ARCHITECTURE — Model Evaluation")
print("="*60)

spark = SparkSession.builder \
    .appName("KappaModelEvaluation") \
    .master("local[*]") \
    .config("spark.jars.packages", "com.clickhouse:clickhouse-jdbc:0.4.6") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Muat Model
model_reg   = RandomForestRegressionModel.load("../models/taxi_reg_model")
model_class = RandomForestClassificationModel.load("../models/taxi_class_model")

# Muat data dari ClickHouse (TANPA fareAmount di fitur)
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
    LIMIT 30000
) AS taxi_data"""

df = spark.read.format("jdbc") \
    .option("url", "jdbc:clickhouse://localhost:8123/taxi_db") \
    .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
    .option("dbtable", query) \
    .option("user", "mahasiswa").option("password", "bigdata123").load()

FEATURE_COLS = ["tripDistance", "passengerCount", "pickup_zone",
                "dropoff_zone", "pickup_hour", "pickup_day"]

bucketizer = Bucketizer(
    splits=[0.0, 2.0, 5.0, float('inf')],
    inputCol="tipAmount", outputCol="tipCategory"
)
df_labeled  = bucketizer.transform(df)
assembler   = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
test_data   = assembler.transform(df_labeled)

# Prediksi
pred_reg   = model_reg.transform(test_data)
pred_class = model_class.transform(test_data)

# Metrik Regresi
rmse = RegressionEvaluator(labelCol="tipAmount", metricName="rmse").evaluate(pred_reg)
mae  = RegressionEvaluator(labelCol="tipAmount", metricName="mae").evaluate(pred_reg)
r2   = RegressionEvaluator(labelCol="tipAmount", metricName="r2").evaluate(pred_reg)

# Metrik Klasifikasi
acc = MulticlassClassificationEvaluator(labelCol="tipCategory", metricName="accuracy").evaluate(pred_class)
f1  = MulticlassClassificationEvaluator(labelCol="tipCategory", metricName="f1").evaluate(pred_class)

# ================================================================
# COHEN'S KAPPA — Metrik Kappa
# Mengukur kesepakatan antara prediksi dan label sebenarnya,
# dengan memperhitungkan peluang kebetulan.
#
# κ = (Po - Pe) / (1 - Pe)
#   Po = observed agreement (akurasi)
#   Pe = expected agreement (kebetulan)
#
# Interpretasi:
#   κ > 0.8  : Sangat Baik (Almost Perfect)
#   κ > 0.6  : Baik (Substantial)
#   κ > 0.4  : Cukup (Moderate)
#   κ > 0.2  : Lemah (Fair)
#   κ ≤ 0.2  : Sangat Lemah
# ================================================================
n  = pred_class.count()
po = acc  # Observed agreement = accuracy

pe = 0.0
for c in [0.0, 1.0, 2.0]:
    actual_frac = pred_class.filter(col("tipCategory") == c).count() / n
    pred_frac   = pred_class.filter(col("prediction")  == c).count() / n
    pe += actual_frac * pred_frac

kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

if kappa > 0.8:
    kappa_interp = "Sangat Baik (Almost Perfect)"
elif kappa > 0.6:
    kappa_interp = "Baik (Substantial)"
elif kappa > 0.4:
    kappa_interp = "Cukup (Moderate)"
elif kappa > 0.2:
    kappa_interp = "Lemah (Fair)"
else:
    kappa_interp = "Sangat Lemah"

# Output Hasil
print("\n" + "="*60)
print("  HASIL EVALUASI MODEL PREDIKSI TIP TAKSI NYC")
print("  (Kappa Architecture — Single Stream Pipeline)")
print("="*60)

print(f"\n  [Regresi] Prediksi Nilai Tip ($)")
print(f"  {'─'*40}")
print(f"   RMSE     : ${rmse:.4f}")
print(f"   MAE      : ${mae:.4f}")
print(f"   R²       : {r2:.4f}  (mendekati 1 = sempurna)")

print(f"\n  [Klasifikasi] Kategori Tip")
print(f"  {'─'*40}")
print(f"   Akurasi  : {acc * 100:.2f}%")
print(f"   F1-Score : {f1:.4f}")
print(f"   Cohen's κ: {kappa:.4f}")
print(f"   Interpretasi: {kappa_interp}")

print(f"\n  [Distribusi Aktual vs Prediksi]")
print(f"   Kategori 0 = Rendah ($0-$2)")
print(f"   Kategori 1 = Menengah ($2-$5)")
print(f"   Kategori 2 = Tinggi (>$5)")

# Confusion Matrix
print(f"\n  [Confusion Matrix]")
pred_class.crosstab("tipCategory", "prediction").show()

# Feature Importance
print("  [Feature Importance — Kontribusi tiap fitur]")
for name, imp in zip(FEATURE_COLS, model_reg.featureImportances):
    bar = "█" * int(imp * 40)
    print(f"   {name:<20}: {imp:.4f} {bar}")

# Overfitting check
print(f"\n  [Deteksi Overfitting]")
if acc > 0.98:
    print(f"   ⚠️  PERINGATAN: Akurasi {acc*100:.1f}% terlalu tinggi — kemungkinan data leakage!")
elif kappa > 0.95:
    print(f"   ⚠️  PERINGATAN: Kappa {kappa:.3f} terlalu sempurna — periksa fitur!")
else:
    print(f"   ✅ Tidak terdeteksi overfitting. Hasil realistis untuk data taksi nyata.")

spark.stop()