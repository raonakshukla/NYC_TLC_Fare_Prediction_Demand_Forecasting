# Databricks notebook source
# MAGIC %md
# MAGIC ### Scale-up experiment on Feb 2024 Yellow Taxi Data

# COMMAND ----------

cores = [1, 2, 4, 8, 10] # for scale up
sizes = [10, 20, 40, 80, 100]
directory = 'dbfs:/mnt/2024-team14/yellow_taxi' # used yellow taxi dataset

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

# cols to rename and reformat
cols = [
    ("tpep_pickup_datetime", "Date", DateType()),
    ("trip_distance", "trip_miles", DoubleType()),
    ("fare_amount", "base_passenger_fare", DoubleType())
]

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor
import time

def build_model(partition_data_iter):
  column_names = ['trip_km', 'trip_time', 'base_passenger_fare']
  
  start = time.time()

  partition_data_df = pd.DataFrame(partition_data_iter, columns=column_names)
  reg = DecisionTreeRegressor(max_depth=5)
  X_train = partition_data_df[['trip_km', 'trip_time']]
  y_train = partition_data_df["base_passenger_fare"]
  model = reg.fit(X_train.values,y_train.values)

  end = time.time()
  return [(model, end - start)]

# COMMAND ----------

def predict(instance):
  return[m.predict([instance[:-1]])[0] for m in models]

# COMMAND ----------

def agg_predictions(preds):
  mean = sum(preds) / len(preds)
  return float(mean)

# COMMAND ----------

def transform(instance):
  return Row(**instance.asDict(), \
             raw_prediction = agg_predictions(predict(instance)))

# COMMAND ----------

for _ in range(10): # perform experiments 10 times
  for i, c in enumerate(cores):
    fraction = sizes[i]
    spark = SparkSession \
        .builder \
        .master(f"local[{c}]") \
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

    # Read the Parquet file with schema inference
    df = spark.read.parquet(directory) \
    .select("tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "fare_amount")

    df = df.withColumns({n: F.col(o).cast(t) for o, n, t in cols}) \
    .withColumns({
      "trip_time": F.unix_timestamp(F.col("tpep_dropoff_datetime")) - F.unix_timestamp(F.col("tpep_pickup_datetime")),
      "trip_km": F.col("trip_miles") * 0.621371
      }).drop(*["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "trip_miles", "Date", "fare_amount"])
    
    df.cache()

    sample_df = df.sample(fraction/100)

    # split data into train and test
    train, test = sample_df.randomSplit([0.7, 0.3], seed=555)
    
    train_rdd = train.rdd.repartition(c).cache()
    test_rdd = test.rdd.repartition(c)

    train_rdd.count()
    
    start = time.time()
    models_runtimes = train_rdd.mapPartitions(build_model).collect()
    models, runtimes = zip(*models_runtimes)
    end = time.time()

    with open("scaleup_dt.csv", "a") as f:
        print(
            f"{c},{fraction},{end - start},{sum(runtimes)/len(runtimes)}",
            file=f,
        )