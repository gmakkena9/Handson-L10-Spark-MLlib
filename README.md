"""
Task 5: Time-Based Fare Trend Prediction
ITCS 6190/8190 - Cloud Computing for Data Analysis, Summer 2026

Workflow:
  1. Offline Model Training: Aggregate training-dataset.csv into 5-minute windows,
     compute avg_fare per window, engineer hour_of_day and minute_of_hour features,
     train a LinearRegression model, and save it.
  2. Real-Time Inference: Apply the same 5-minute windowed aggregation + feature
     engineering to the live stream, load the saved model, and predict avg_fare
     for each incoming time window.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, avg, window, hour, minute
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, TimestampType
)

# MLlib imports
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

# ─────────────────────────────────────────────────────────────────
# Spark Session
# ─────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Task5_FareTrendPrediction") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Paths
MODEL_PATH = "models/fare_trend_model_v2"
TRAINING_DATA_PATH = "training-dataset.csv"

# ─────────────────────────────────────────────────────────────────
# PART 1: OFFLINE MODEL TRAINING
# ─────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"\n[Training Phase] Training new model with feature engineering using {TRAINING_DATA_PATH}...")

    # Load historical CSV data
    hist_df_raw = spark.read.csv(TRAINING_DATA_PATH, header=True, inferSchema=False)

    # Cast timestamp and fare_amount to correct types
    hist_df_processed = hist_df_raw \
        .withColumn("event_time", col("timestamp").cast(TimestampType())) \
        .withColumn("fare_amount", col("fare_amount").cast(DoubleType()))

    # Aggregate into 5-minute time windows, computing avg_fare per window
    hist_windowed_df = hist_df_processed \
        .groupBy(window(col("event_time"), "5 minutes")) \
        .agg(avg("fare_amount").alias("avg_fare"))

    # Feature engineering: extract hour_of_day and minute_of_hour from window start
    hist_features = hist_windowed_df \
        .withColumn("hour_of_day",    hour(col("window.start"))) \
        .withColumn("minute_of_hour", minute(col("window.start")))

    # VectorAssembler: combine time-based features into a single 'features' vector
    assembler = VectorAssembler(
        inputCols=["hour_of_day", "minute_of_hour"],
        outputCol="features"
    )
    train_df = assembler.transform(hist_features)

    # Train LinearRegression model with avg_fare as the label
    lr = LinearRegression(
        featuresCol="features",
        labelCol="avg_fare"
    )
    model = lr.fit(train_df)

    # Save the trained model to disk
    model.write().overwrite().save(MODEL_PATH)
    print(f"[Model Saved] -> {MODEL_PATH}")
    print(f"  Coefficients : {model.coefficients}")
    print(f"  Intercept    : {model.intercept:.4f}")
    print(f"  RMSE         : {model.summary.rootMeanSquaredError:.4f}")
    print(f"  R²           : {model.summary.r2:.4f}")
else:
    print(f"[Model Found] Using existing model at {MODEL_PATH}")

# ─────────────────────────────────────────────────────────────────
# PART 2: STREAMING INFERENCE
# ─────────────────────────────────────────────────────────────────
print("\n[Inference Phase] Starting real-time trend prediction stream...")

# Schema for incoming JSON ride events
schema = StructType([
    StructField("trip_id",     StringType()),
    StructField("driver_id",   IntegerType()),
    StructField("distance_km", DoubleType()),
    StructField("fare_amount", DoubleType()),
    StructField("timestamp",   StringType())
])

# Read raw streaming data from socket and parse JSON
raw_stream = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

parsed_stream = raw_stream \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*") \
    .withColumn("event_time", col("timestamp").cast(TimestampType()))

# Add watermark to handle late-arriving data (1-minute tolerance)
parsed_stream = parsed_stream.withWatermark("event_time", "1 minute")

# Apply same 5-minute windowed aggregation as training (slide every 1 minute)
windowed_df = parsed_stream \
    .groupBy(window(col("event_time"), "5 minutes", "1 minute")) \
    .agg(avg("fare_amount").alias("avg_fare"))

# Feature engineering: same as training phase
windowed_features = windowed_df \
    .withColumn("hour_of_day",    hour(col("window.start"))) \
    .withColumn("minute_of_hour", minute(col("window.start")))

# VectorAssembler matching training configuration
assembler_inference = VectorAssembler(
    inputCols=["hour_of_day", "minute_of_hour"],
    outputCol="features"
)
feature_df = assembler_inference.transform(windowed_features)

# Load the pre-trained regression model
trend_model = LinearRegressionModel.load(MODEL_PATH)

# Apply model to produce predictions
predictions = trend_model.transform(feature_df)

# Select final output columns
output_df = predictions.select(
    col("window.start").alias("window_start"),
    col("window.end").alias("window_end"),
    "avg_fare",
    col("prediction").alias("predicted_next_avg_fare")
)

# Write predictions to console
query = output_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .option("truncate", False) \
    .start()

query.awaitTermination()
