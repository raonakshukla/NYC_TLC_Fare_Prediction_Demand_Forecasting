# Databricks notebook source
# MAGIC %md
# MAGIC ## Feature Selection

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature engineering

# COMMAND ----------

(sc, spark)

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .master("local[*]") \
    .appName("MLlib lab") \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .getOrCreate()
    
# swith the latest spark version to older one so that it tolerates some data format issues
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

""" 
in order to avoid "Parquet column cannot be converted" error, we need to disable vectorized reader when we have decimal values in our columns. 
refer to https://learn.microsoft.com/en-us/answers/questions/853861/parquet-column-cannot-be-converted for further info
"""
spark.conf.set("spark.sql.parquet.enableVectorizedReader", "false") 

sc = spark.sparkContext


# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import matplotlib.pyplot as plt
# MAGIC import pyspark.sql.functions as F
# MAGIC from pyspark.sql import Row
# MAGIC
# MAGIC from pyspark.sql.types import StringType, TimestampNTZType, LongType, DoubleType, IntegerType, DateType
# MAGIC
# MAGIC import pandas as pd
# MAGIC
# MAGIC from pyspark.ml.feature import VectorAssembler
# MAGIC from pyspark.ml.stat import Correlation

# COMMAND ----------

year_range = (2019, 2025)

# COMMAND ----------

"""
We have encountered "Parquet column cannot be converted" error. As a workaround we decided to loop through directory and ensure there is no column type mismatch by checking file by file.
"""

directory = '/mnt/2024-team14/'

# Read the Parquet file with schema inference
df = spark.read.parquet(directory)

cols_to_drop_first = [
    "hvfhs_license_num", 
    "dispatching_base_num", 
    "originating_base_num",
    "request_datetime",
    "on_scene_datetime",
    "dropoff_datetime",
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag"
]

# (column with mismatch, desirable type)
mismatch_col = [
    ("airport_fee", "double"), 
    ("PULocationID", "bigint"), 
    ("DOLocationID", "bigint")
]

df = df.drop(*cols_to_drop_first) \
    .withColumns({c: F.col(c).cast(t) for c, t in mismatch_col}) \
    .withColumns({
    "trip_meters": F.col('trip_miles')*1609.35,
    "pickupdayofyear": F.dayofyear(F.col("pickup_datetime")),
    "pickupmonth": F.month(F.col("pickup_datetime")),
    "pickupyear": F.year(F.col("pickup_datetime")),
    "pickupdate": F.col("pickup_datetime").cast(DateType()),
    "pickuphour": F.hour("pickup_datetime"),
    "week": F.weekofyear("pickup_datetime") + (F.year("pickup_datetime") - 2019) * 52 - 4
    }) \
    .drop(*["pickup_datetime", "trip_miles"])

df.cache()

# COMMAND ----------

total_rows = df.count()
df.cache()
print(f"Total number of rows:{total_rows}")

# COMMAND ----------

df1 = df.groupBy("pickupdate") \
  .agg(
    F.count("*").alias("daily_numTrips")
  )

# COMMAND ----------

df = df.join(df1, ["pickupdate"])

# COMMAND ----------

import holidays

hds = []
for y in range(year_range[0], year_range[1]):
  hds += holidays.US(state="NY", years=y).keys()

# COMMAND ----------

df2 = df.withColumns({
  "isWeekend": F.when(F.col("pickupdate").isin(hds) | F.dayofweek(F.col("pickupdate")).isin([1, 7]), 1).otherwise(0),
  "isOvernight": F.when(F.col("pickuphour").isin(list(range(20, 24))+list(range(0, 6))), 1).otherwise(0)
  }) \
  .withColumn("isRushhour", 
              F.when(F.col("pickuphour").isin(list(range(16, 20))) | (F.col("isWeekend") == 0), 1).otherwise(0)
              )
df2.cache()
df.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Incorporating NY Daily Weather Data

# COMMAND ----------

weather_df = spark.read.csv(directory + "csvs/weather_data.csv", header=True)

# COMMAND ----------

weather_df = weather_df.drop(*["PRCP (Inches)", "SNOW (Inches)", "SNWD (Inches)"]) \
  .withColumns({
    "Date": F.to_date(F.col("Date"), "dd-MM-yyyy"),
    "TMAX (Degrees Fahrenheit)": F.col("TMAX (Degrees Fahrenheit)").cast(DoubleType()),
    "TMIN (Degrees Fahrenheit)": F.col("TMIN (Degrees Fahrenheit)").cast(DoubleType()),
    "PRCP (mm)": F.col("PRCP (mm)").cast(DoubleType()),
    "SNOW (mm)": F.col("SNOW (mm)").cast(DoubleType()),
    "SNWD (mm)": F.col("SNWD (mm)").cast(DoubleType())
  })

# COMMAND ----------

weather_df=weather_df.withColumns({
  "pickupdayofyear": F.dayofyear(F.col('Date')),
  "pickupmonth": F.month(F.col('Date')),
  "pickupyear": F.year(F.col('Date'))})

# COMMAND ----------

weather_df = weather_df \
  .drop(F.col("Date"))

display(weather_df)

# COMMAND ----------

joined_df = df2.join(weather_df, ["pickupyear", "pickupmonth", "pickupdayofyear"]).drop("pickupdate")

joined_df.cache()
df2.unpersist()

# COMMAND ----------

# create feature for extremity of the temperture (abs(temp-68) 20C is 68F)
joined_df1 = joined_df.withColumns({
  "TMAXExtremity": F.abs(F.col("TMAX (Degrees Fahrenheit)") - 68), 
  "TMINExtremity": F.abs(F.col("TMIN (Degrees Fahrenheit)") - 68)
  })

# COMMAND ----------

# MAGIC %md
# MAGIC #### Incorporating MTA Public Transport Daily Ridership Data

# COMMAND ----------

mta_df = spark.read.csv(directory + "csvs/MTA_2020mar_2024apr.csv", header=True)

# COMMAND ----------

mta_df.columns

# COMMAND ----------

mta_df = mta_df.drop(*[
    'Subways: % of Comparable Pre-Pandemic Day',
    'Buses: % of Comparable Pre-Pandemic Day',
    'LIRR: % of Comparable Pre-Pandemic Day',
    'Metro-North: % of Comparable Pre-Pandemic Day',
    'Access-A-Ride: % of Comparable Pre-Pandemic Day',
    'Bridges and Tunnels: % of Comparable Pre-Pandemic Day',
    'Staten Island Railway: % of Comparable Pre-Pandemic Day'
  ]) \
  .withColumn("Date", F.to_date(F.col('Date'), "MM/dd/yyyy"))

# COMMAND ----------

mta_df = mta_df.withColumnsRenamed({
  "Subways: Total Estimated Ridership": "subways_daily",
  "Buses: Total Estimated Ridership": 'buses_daily',
  "LIRR: Total Estimated Ridership": 'LIRR_daily',
  "Metro-North: Total Estimated Ridership": 'metro_north_daily',
  "Access-A-Ride: Total Scheduled Trips": 'Access-A-Ride_daily',
  "Bridges and Tunnels: Total Traffic": 'br_tunnel_traffic',
  "Staten Island Railway: Total Estimated Ridership": 'SIR_daily'
})

# COMMAND ----------

mta_df = mta_df.withColumns({
    "pickupdayofyear": F.dayofyear(F.col('Date')),
    "pickupmonth": F.month(F.col('Date')),
    "pickupyear": F.year(F.col('Date')),
    'subways_daily': F.col('subways_daily').cast("int"),
    'buses_daily': F.col('buses_daily').cast("int"),
    'LIRR_daily': F.col('LIRR_daily').cast("int"),
    'metro_north_daily': F.col('metro_north_daily').cast("int"),
    'Access-A-Ride_daily': F.col('Access-A-Ride_daily').cast("int"),
    'br_tunnel_traffic': F.col('br_tunnel_traffic').cast("int"),
    'SIR_daily': F.col('SIR_daily').cast("int")
  }) \
  .drop(F.col("Date"))

# COMMAND ----------

mta_df.cache()

# COMMAND ----------

joined_df2 = joined_df1.join(mta_df, ["pickupyear", "pickupmonth", "pickupdayofyear"])

# COMMAND ----------

joined_df2.cache()
joined_df1.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating pt_taxi_ratio which represents the ratio of the number of public transport users and taxi passengers in each day

# COMMAND ----------

# add an additional feature of public transport ridership (daily) / taxi trips 
joined_df3 = joined_df2.withColumn("pt_taxi_ratio", 
        (F.col("subways_daily") + F.col("buses_daily") + F.col("LIRR_daily") + F.col("metro_north_daily") + F.col("Access-A-Ride_daily") + F.col("SIR_daily")) / F.col("daily_numTrips")
    )
joined_df3.cache()
joined_df2.unpersist()

# COMMAND ----------

joined_df4 = joined_df3.drop(
  "subways_daily",
  "buses_daily",
  "LIRR_daily",
  "metro_north_daily", 
  "Access-A-Ride_daily", 
  "SIR_daily", 
  "TMAX (Degrees Fahrenheit)",
  "TMIN (Degrees Fahrenheit)"
  )

# COMMAND ----------

joined_df5 = df.join(joined_df4, ["pickupyear", "pickupmonth", "pickupdayofyear", "pickuphour"])

# COMMAND ----------

df6 = joined_df5.drop("pickupdate")
df6.cache()
joined_df3.unpersist()

# COMMAND ----------

df_rdd = df6.rdd.repartition(24).cache()

# COMMAND ----------

column_names = df6.columns

# COMMAND ----------

feature_cols = column_names
feature_cols.remove("base_passenger_fare")

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

def build_model(partition_data_it):
  partition_data_df = pd.DataFrame(partition_data_it,columns=column_names)
  X_train = partition_data_df[feature_cols]
  y_train = partition_data_df["base_passenger_fare"]
  sel = SelectFromModel(RandomForestRegressor(max_depth= 5, n_estimators = 100))
  model = sel.fit(X_train.values,y_train.values)
  selected_feat= X_train.columns[(sel.get_support())]
  return [selected_feat]

# COMMAND ----------

models = df_rdd.mapPartitions(build_model).collect()

# COMMAND ----------

feature_dict = {
    "pickupyear": 0,
    "pickupmonth": 0,
    "pickupdayofyear": 0,
    "pickuphour": 0,
    "PULocationID": 0,
    "DOLocationID": 0,
    "trip_time": 0,
    "base_passenger_fare": 0,
    "tolls": 0,
    "bcf": 0,
    "sales_tax": 0,
    "congestion_surcharge": 0,
    "airport_fee": 0,
    "tips": 0,
    "driver_pay": 0,
    "trip_meters": 0,
    "week": 0,
    "daily_numTrips": 0,
    "isWeekend": 0,
    "isOvernight": 0,
    "isRushhour": 0,
    "PRCP (mm)": 0,
    "SNOW (mm)": 0,
    "SNWD (mm)": 0,
    "TMAXExtremity": 0,
    "TMINExtremity": 0,
    "br_tunnel_traffic": 0,
    "pt_taxi_ratio": 0
}

# COMMAND ----------

for features in models:
  for f in features:
    feature_dict[f] += 1

feature_dict