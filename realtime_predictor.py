import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.2 pyspark-shell"

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler

# 1. Inisialisasi
spark = SparkSession.builder.appName("TaxiRealtimePrediction").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# 2. Muat Model
reg_model = RandomForestRegressionModel.load("./taxi_reg_model")
class_model = RandomForestClassificationModel.load("./taxi_class_model")

# 3. Skema Data (Harus sesuai dengan format JSON di Kafka)
schema = StructType([
    StructField("fareAmount", DoubleType()), StructField("tripDistance", DoubleType()),
    StructField("passengerCount", IntegerType()), StructField("pickup_zone", DoubleType()),
    StructField("dropoff_zone", DoubleType()), StructField("pickup_hour", IntegerType()),
    StructField("pickup_day", IntegerType())
])

# 4. Consume Kafka
df_stream = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "green-taxi-stream") \
    .load()

df_parsed = df_stream.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

# 5. Transformasi Fitur (Wajib disamakan dengan training)
assembler = VectorAssembler(
    inputCols=["fareAmount", "tripDistance", "passengerCount", "pickup_zone", "dropoff_zone", "pickup_hour", "pickup_day"], 
    outputCol="features"
)
df_features = assembler.transform(df_parsed)

# 6. Prediksi Otomatis
pred_reg = reg_model.transform(df_features)
pred_class = class_model.transform(df_features)

# 7. Output ke Konsol (Real-time)
query = pred_reg.select("fareAmount", "prediction").writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()