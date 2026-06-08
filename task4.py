"""
Task 4: Real-Time Fare Prediction Using MLlib Regression
ITCS 6190/8190 - Cloud Computing for Data Analysis, Summer 2026

Workflow:
  1. Offline Model Training: Train a LinearRegression model on training-dataset.csv
     using distance_km as the feature and fare_amount as the label. Save model to disk.
  2. Real-Time Inference: Read live ride data from socket, load the saved model,
     predict fare, and compute deviation (|actual - predicted|) for anomaly detection.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, abs as abs_diff
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)

# MLlib imports
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

# ─────────────────────────────────────────────────────────────────
# Spark Session
# ─────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Task4_FarePrediction") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Paths
MODEL_PATH = "models/fare_model"
TRAINING_DATA_PATH = "training-dataset.csv"

# ─────────────────────────────────────────────────────────────────
# PART 1: OFFLINE MODEL TRAINING
# ─────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"\n[Training Phase] No model found. Training a new model using {TRAINING_DATA_PATH}...")

    # Load training data from CSV
    train_df_raw = spark.read.csv(TRAINING_DATA_PATH, header=True, inferSchema=False)

    # Cast feature and label columns to DoubleType for ML processing
    train_df = train_df_raw \
        .withColumn("distance_km", col("distance_km").cast(DoubleType())) \
        .withColumn("fare_amount", col("fare_amount").cast(DoubleType()))

    # Create VectorAssembler: combines distance_km into a single 'features' vector
    assembler = VectorAssembler(
        inputCols=["distance_km"],
        outputCol="features"
    )
    train_data_with_features = assembler.transform(train_df)

    # Create and train LinearRegression model
    lr = LinearRegression(
        featuresCol="features",
        labelCol="fare_amount"
    )
    model = lr.fit(train_data_with_features)

    # Save the trained model to disk
    model.write().overwrite().save(MODEL_PATH)
    print(f"[Training Complete] Model saved to -> {MODEL_PATH}")
    print(f"  Coefficients : {model.coefficients}")
    print(f"  Intercept    : {model.intercept:.4f}")
    print(f"  RMSE         : {model.summary.rootMeanSquaredError:.4f}")
    print(f"  R²           : {model.summary.r2:.4f}")
else:
    print(f"[Model Found] Using existing model from {MODEL_PATH}")

# ─────────────────────────────────────────────────────────────────
# PART 2: STREAMING INFERENCE
# ─────────────────────────────────────────────────────────────────
print("\n[Inference Phase] Starting real-time fare prediction stream...")

# Schema for incoming JSON ride events
schema = StructType([
    StructField("trip_id",     StringType()),
    StructField("driver_id",   IntegerType()),
    StructField("distance_km", DoubleType()),
    StructField("fare_amount", DoubleType()),
    StructField("timestamp",   StringType())
])

# Read raw streaming data from socket
raw_stream = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Parse JSON payload
parsed_stream = raw_stream \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Load the pre-trained LinearRegressionModel from disk
model = LinearRegressionModel.load(MODEL_PATH)

# Apply the same VectorAssembler transformation used during training
assembler_inference = VectorAssembler(
    inputCols=["distance_km"],
    outputCol="features"
)
stream_with_features = assembler_inference.transform(parsed_stream)

# Generate fare predictions using the loaded model
predictions = model.transform(stream_with_features)

# Compute deviation: |actual_fare - predicted_fare|
predictions_with_deviation = predictions.withColumn(
    "deviation",
    abs_diff(col("fare_amount") - col("prediction"))
)

# Select final output columns
output_df = predictions_with_deviation.select(
    "trip_id",
    "driver_id",
    "distance_km",
    "fare_amount",
    col("prediction").alias("predicted_fare"),
    "deviation"
)

# Write results to console
query = output_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .option("truncate", False) \
    .start()

query.awaitTermination()
