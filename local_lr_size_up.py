# Databricks notebook source
# MAGIC %md
# MAGIC ### Size-up experiment on Feb 2024 Yellow Taxi Data

# COMMAND ----------

global c
c = 8 # for size-up, fix at 8 cores
sizes = [10, 20, 40, 80, 100]

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import matplotlib.pyplot as plt
# MAGIC import pyspark.sql.functions as F
# MAGIC from pyspark.sql import Row
# MAGIC
# MAGIC from pyspark.sql.types import DoubleType, DateType
# MAGIC
# MAGIC import pandas as pd
# MAGIC
# MAGIC from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession \
    .builder \
    .master(f"local[*]") \
    .appName("Local LR with {c} partitions") \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .getOrCreate()

# swith the latest spark version to older one so that it tolerates some data format issues
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

"""
in order to avoid "Parquet column cannot be converted" error, we need to disable vectorized reader when we have decimal values in our columns.
refer to https://learn.microsoft.com/en-us/answers/questions/853861/parquet-column-cannot-be-converted for further info
"""
# spark.conf.set("spark.sql.parquet.enableVectorizedReader", "false")

sc = spark.sparkContext

# COMMAND ----------

import os

directory = '/mnt/2024-team14/yellow_taxi'

cols = [ # cols to rename and reformat
    ("tpep_pickup_datetime", "Date", DateType()),
    ("trip_distance", "trip_miles", DoubleType()),
    ("fare_amount", "base_passenger_fare", DoubleType())
]

# Read the Parquet file with schema inference
df = spark.read.parquet(directory) \
.select("tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "fare_amount")

df = df.withColumns({n: F.col(o).cast(t) for o, n, t in cols}) \
.withColumns({
  "trip_time": F.unix_timestamp(F.col("tpep_dropoff_datetime")) - F.unix_timestamp(F.col("tpep_pickup_datetime")),
  "trip_km": F.col("trip_miles") * 0.621371
  }) \
  .drop(*["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "trip_miles", "Date", "fare_amount"])

# COMMAND ----------

df.count()

# COMMAND ----------

df.cache()

# COMMAND ----------

column_names = df.columns

# COMMAND ----------

from sklearn.linear_model import LinearRegression
import time

def build_model(partition_data_iter):
  start = time.time()

  partition_data_df = pd.DataFrame(partition_data_iter, columns=column_names)
  reg = LinearRegression()
  X_train = partition_data_df[['trip_km', 'trip_time']]
  y_train = partition_data_df["base_passenger_fare"]
  model = reg.fit(X_train.values,y_train.values)

  end = time.time()
  return [(model, end - start)]

# COMMAND ----------

for i in range(10): # experiment 10 times
  for fraction in sizes:
    sample_df = df.sample(fraction/100)
    # split data into train and test
    train, test = sample_df.randomSplit([0.7, 0.3], seed=555)

    train_rdd = train.rdd.repartition(c).cache()
    test_rdd = test.rdd.repartition(c)

    train_rdd.count()

    start = time.time()
    models_runtimes = train_rdd.mapPartitions(build_model).collect()

    _, runtimes = zip(*models_runtimes)
    end = time.time()

    # Printing at the end of the file a log with the number of cores, percentage,
    # building time and average runtime of mapPartitions:
    percentage = fraction
    with open("sizeup.csv", "a") as f:
        print(
            f"{c},{percentage},{end - start},{sum(runtimes)/len(runtimes)}",
            file=f,
        )