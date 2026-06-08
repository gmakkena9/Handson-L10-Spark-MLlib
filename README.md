# Handson-L10: Spark Structured Streaming + Machine Learning with MLlib

**Course:** ITCS 6190/8190 – Cloud Computing for Data Analysis, Summer 2026  
**Instructor:** Marco Vieira  
**Assignment:** Tasks 4 & 5 – Real-Time Fare Prediction using Spark MLlib

---

## Overview

This project implements a **real-time analytics pipeline** for a ride-sharing platform using **Apache Spark Structured Streaming** combined with **Spark MLlib** machine learning. The pipeline:

1. Simulates live ride-sharing events via a socket-based Python data generator.
2. Trains machine learning models **offline** on historical data (`training-dataset.csv`).
3. Applies trained models **in real time** to streaming data for fare prediction and anomaly detection.

---

## Repository Structure

```
Handson-L10-Spark-MLlib/
├── data_generator.py          # Streams simulated ride events over TCP socket (port 9999)
├── training-dataset.csv       # Historical ride data used for offline model training
├── task4.py                   # Task 4: Real-Time Fare Prediction (LinearRegression on distance_km)
├── task5.py                   # Task 5: Time-Based Fare Trend Prediction (5-min windowed regression)
├── models/
│   ├── fare_model/            # Saved LinearRegression model for Task 4
│   └── fare_trend_model_v2/   # Saved LinearRegression model for Task 5
└── README.md
```

---

## Dataset

`training-dataset.csv` contains historical ride records with the following schema:

| Column       | Type      | Description                        |
|--------------|-----------|------------------------------------|
| `trip_id`    | String    | Unique UUID for each ride          |
| `driver_id`  | Integer   | Driver identifier (1–100)          |
| `distance_km`| Double    | Trip distance in kilometres        |
| `fare_amount`| Double    | Actual fare charged (USD)          |
| `timestamp`  | Timestamp | Event time of the ride             |

The `data_generator.py` script produces the **same schema** in real time over a TCP socket on `localhost:9999`.

---

## Setup & Prerequisites

```bash
# Install dependencies
pip install pyspark faker

# Java 8+ is required for Spark
java -version
```

### Running the Pipeline

**Terminal 1 – Start the data generator:**
```bash
python data_generator.py
```

**Terminal 2 – Run Task 4 or Task 5:**
```bash
# Task 4: Real-Time Fare Prediction
spark-submit task4.py

# Task 5: Time-Based Fare Trend Prediction
spark-submit task5.py
```

> The first run of each task automatically trains and saves the model. Subsequent runs load the saved model directly, skipping the training phase.

---

## Task 4: Real-Time Fare Prediction Using MLlib Regression

### Approach

#### Part 1 – Offline Model Training
1. Load `training-dataset.csv` into a Spark DataFrame.
2. Cast `distance_km` and `fare_amount` to `DoubleType` for ML compatibility.
3. Use **`VectorAssembler`** to transform `distance_km` into a feature vector column `features`.
4. Train a **`LinearRegression`** model (`featuresCol="features"`, `labelCol="fare_amount"`).
5. Save the model to `models/fare_model` using `.write().overwrite().save()`.

#### Part 2 – Real-Time Streaming Inference
1. Read live JSON ride events from the socket (`localhost:9999`).
2. Parse the JSON payload using the defined schema.
3. Load the saved `LinearRegressionModel` from disk.
4. Apply the same `VectorAssembler` to create the `features` column on the stream.
5. Run `model.transform()` to generate a `prediction` (predicted fare) column.
6. Compute `deviation = |fare_amount − prediction|` to flag pricing anomalies.
7. Output results to the console in `append` mode.

### Key Code Snippets

```python
# Training: VectorAssembler + LinearRegression
assembler = VectorAssembler(inputCols=["distance_km"], outputCol="features")
lr = LinearRegression(featuresCol="features", labelCol="fare_amount")
model = lr.fit(assembler.transform(train_df))
model.write().overwrite().save("models/fare_model")

# Inference: deviation computation
predictions_with_deviation = predictions.withColumn(
    "deviation",
    abs(col("fare_amount") - col("prediction"))
)
```

### Training Results

| Metric       | Value        |
|--------------|--------------|
| Coefficients | [-0.2458]    |
| Intercept    | 98.1495      |
| RMSE         | 42.5817      |
| R²           | 0.0073       |

> **Note:** The low R² reflects the synthetic data generator's design — fare amounts are randomized uniformly between $5 and $150 independent of distance, so a linear model correctly captures the near-zero correlation. In a real dataset, fare would strongly correlate with distance and the model metrics would be much higher.

### Sample Output

```
----------------------------------------
Batch: 6
----------------------------------------
+------------------------------------+---------+-----------+-----------+-----------------+------------------+
|trip_id                             |driver_id|distance_km|fare_amount|predicted_fare   |deviation         |
+------------------------------------+---------+-----------+-----------+-----------------+------------------+
|230964b2-c0b4-4ad0-81f3-6ba3698a83dd|11       |19.85      |101.6      |93.27004410476815|8.329955895231848 |
+------------------------------------+---------+-----------+-----------+-----------------+------------------+
```

**Verified local output (5 sample rows):**

```
+------------------------------------+---------+-----------+-----------+-----------------+------------------+
|trip_id                             |driver_id|distance_km|fare_amount|predicted_fare   |deviation         |
+------------------------------------+---------+-----------+-----------+-----------------+------------------+
|5966bbbf-1494-404e-8318-3e09bd3a52ef|38       |2.24       |126.71     |97.59882960410545|29.111170395894547|
|47872ce7-d042-472b-8d87-7a6808435633|8        |41.72      |144.05     |87.89409073335946|56.15590926664055 |
|99177ecd-8afe-493f-bd09-086b1b6f3722|35       |41.26      |61.27      |88.00716519841882|26.737165198418815|
|969a320f-3577-4a15-9ba8-8674bf3c8005|94       |49.5       |57.23      |85.98165738952957|28.751657389529576|
|6dcd45f9-87dd-4aed-8511-8c5fa03643be|43       |17.58      |30.82      |93.82804200843059|63.008042008430586|
+------------------------------------+---------+-----------+-----------+-----------------+------------------+
```

---

## Task 5: Time-Based Fare Trend Prediction

### Approach

#### Part 1 – Offline Model Training with Feature Engineering
1. Load `training-dataset.csv` and cast `timestamp → TimestampType`, `fare_amount → DoubleType`.
2. **Aggregate into 5-minute time windows** using `groupBy(window(..., "5 minutes"))` and compute `avg("fare_amount")` as `avg_fare`.
3. **Feature Engineering:** Extract cyclical time features from `window.start`:
   - `hour_of_day` = `hour(window.start)` — captures daily pricing patterns
   - `minute_of_hour` = `minute(window.start)` — captures sub-hour variation
4. Assemble features with `VectorAssembler(["hour_of_day", "minute_of_hour"])`.
5. Train `LinearRegression` with `labelCol="avg_fare"` and save to `models/fare_trend_model_v2`.

#### Part 2 – Real-Time Streaming Inference
1. Read and parse the live stream, casting `timestamp → event_time (TimestampType)`.
2. Apply **watermarking** (`withWatermark("event_time", "1 minute")`) for late-data tolerance.
3. Apply the same **5-minute sliding window** aggregation (slide interval: 1 minute).
4. Create identical `hour_of_day` and `minute_of_hour` features.
5. Load the saved model and run inference to produce `predicted_next_avg_fare`.
6. Output `window_start`, `window_end`, `avg_fare`, and `predicted_next_avg_fare` to console.

### Key Code Snippets

```python
# Training: 5-minute window aggregation + feature engineering
hist_windowed_df = hist_df_processed \
    .groupBy(window(col("event_time"), "5 minutes")) \
    .agg(avg("fare_amount").alias("avg_fare"))

hist_features = hist_windowed_df \
    .withColumn("hour_of_day",    hour(col("window.start"))) \
    .withColumn("minute_of_hour", minute(col("window.start")))

# Streaming: sliding window with watermark
windowed_df = parsed_stream \
    .groupBy(window(col("event_time"), "5 minutes", "1 minute")) \
    .agg(avg("fare_amount").alias("avg_fare"))
```

### Training Results

| Metric       | Value           |
|--------------|-----------------|
| Coefficients | [0.0, 0.0]      |
| Intercept    | 92.3943         |
| RMSE         | 0.0000          |
| R²           | nan (1 window)  |

> **Note:** The training CSV spans a single 5-minute window, yielding one aggregated row. A single data point produces a degenerate model (zero coefficients, RMSE = 0). In production, you would feed weeks of historical data across many time windows to obtain meaningful time-based coefficients. The streaming inference logic is correct and will improve with richer training data.

### Sample Output

```
----------------------------------------
Batch: 38
----------------------------------------
+-------------------+-------------------+-----------------+------------------------+
|window_start       |window_end         |avg_fare         |predicted_next_avg_fare |
+-------------------+-------------------+-----------------+------------------------+
|2025-10-17 16:03:00|2025-10-17 16:08:00|72.83110552763817|92.39431034482759       |
+-------------------+-------------------+-----------------+------------------------+
```

**Verified local output:**

```
+-------------------+-------------------+-----------------+-----------------------+
|window_start       |window_end         |avg_fare         |predicted_next_avg_fare|
+-------------------+-------------------+-----------------+-----------------------+
|2025-10-16 17:45:00|2025-10-16 17:50:00|92.39431034482759|92.39431034482759      |
+-------------------+-------------------+-----------------+-----------------------+
```

---

## Architecture Diagram

```
┌─────────────────────────┐
│   data_generator.py     │  ← Simulates ride-sharing events
│  (TCP Socket :9999)     │
└────────────┬────────────┘
             │  JSON stream (1 event/sec)
             ▼
┌─────────────────────────────────────────────────────────┐
│              Apache Spark Structured Streaming           │
│                                                         │
│  ┌─────────────┐    ┌──────────────────────────────┐   │
│  │  task4.py   │    │         task5.py              │   │
│  │             │    │                               │   │
│  │ Parse JSON  │    │  Parse JSON + cast timestamp  │   │
│  │     ↓       │    │            ↓                  │   │
│  │VectorAssem- │    │  5-min Window Aggregation     │   │
│  │bler (dist_  │    │  + Watermark (1 min)          │   │
│  │ km→features)│    │            ↓                  │   │
│  │     ↓       │    │  Feature Engineering:         │   │
│  │ LR Model    │    │  hour_of_day, minute_of_hour  │   │
│  │ Inference   │    │            ↓                  │   │
│  │     ↓       │    │  LR Model Inference           │   │
│  │ Deviation   │    │            ↓                  │   │
│  │ Calculation │    │  Predict avg_fare per window  │   │
│  └──────┬──────┘    └──────────────┬────────────────┘   │
│         │                          │                     │
└─────────┼──────────────────────────┼─────────────────────┘
          ▼                          ▼
   Console Output              Console Output
 (trip_id, fare,             (window, avg_fare,
  predicted, deviation)       predicted_next_avg_fare)
```

---

## Technologies Used

| Technology              | Version | Purpose                                    |
|-------------------------|---------|--------------------------------------------|
| Apache Spark            | 3.x     | Distributed stream processing engine       |
| Spark Structured Streaming | 3.x  | Real-time data ingestion and transformation|
| Spark MLlib             | 3.x     | Machine learning (LinearRegression, VectorAssembler) |
| Python                  | 3.8+    | Implementation language                    |
| Faker                   | Latest  | Synthetic ride data generation             |
| Java                    | 8+/21   | Spark JVM runtime                          |

---

## Notes & Observations

- **Task 4** demonstrates a classic **online ML inference** pattern: train once offline, serve continuously in the stream. The `deviation` column acts as a simple anomaly detector — rides where predicted and actual fares differ significantly could indicate pricing fraud or data quality issues.

- **Task 5** demonstrates **temporal feature engineering** for time-series forecasting. By converting raw timestamps into cyclical features (`hour_of_day`, `minute_of_hour`), the model can learn hour-of-day pricing trends. The sliding window with a 1-minute slide interval provides overlapping aggregation windows for smoother trend detection.

- Both tasks follow the same **two-phase pattern**: offline training → model persistence → streaming inference, which is the standard production pattern for Spark MLlib + Structured Streaming pipelines.
