# Databricks notebook source
# MAGIC %md
# MAGIC ## Exploratory Data Analysis and Creating a dataframe for heat map

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature engineering and correlation matrices

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

mta_df = spark.read.csv(directory + "csvs/MTA_2020mar_2024apr.csv", header=True)
weather_df = spark.read.csv(directory + "csvs/weather_data.csv", header=True)
taxi_zone_df = spark.read.csv(directory + 'csvs/taxi_zone_lookup.csv', header=True)

# Read the Parquet file with schema inference
df = spark.read.parquet(directory)

df = df.select(
    "pickup_datetime",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "PULocationID"
    ) \
    .withColumns({
    "pickupdayofyear": F.dayofyear(F.col("pickup_datetime")),
    "pickupmonth": F.month(F.col("pickup_datetime")),
    "pickupyear": F.year(F.col("pickup_datetime")),
    "pickupdate": F.col("pickup_datetime").cast(DateType()),
    "pickuphour": F.hour("pickup_datetime"),
    "speed": F.col("trip_miles") / F.col("trip_time") * 3600,
    "week": F.weekofyear("pickup_datetime") + (F.year("pickup_datetime") - 2019) * 52 - 4
    }) \
    .drop("pickup_datetime")

df.cache()

# COMMAND ----------

total_rows = df.count()
df.cache()
print(f"Total number of rows:{total_rows}")

# COMMAND ----------

df0 = df.groupBy("pickupdate", "pickupyear", "pickupmonth", "pickuphour") \
  .agg(
    F.count("*").alias("hourly_numTrips"),
    F.sum("trip_miles").alias("hourly_total_miles"), 
    F.sum("trip_time").alias("hourly_total_time"), 
    F.sum("base_passenger_fare").alias("hourly_total_base_fare"),
    F.mean("speed").alias("hourly_mean_speed")
  )

# COMMAND ----------

df0 = df0.join(df, ["pickupdate", "pickupyear", "pickupmonth", "pickuphour"])
df.cache()

# COMMAND ----------

df1 = df.groupBy("pickupdate", "pickupyear", "pickupmonth", "pickupdayofyear") \
  .agg(
    F.count("*").alias("daily_numTrips"), 
    F.sum("trip_miles").alias("daily_total_miles"), 
    F.sum("trip_time").alias("daily_total_time"), 
    F.sum("base_passenger_fare").alias("daily_total_base_fare"),
    F.mean("speed").alias("daily_mean_speed")
  )

# COMMAND ----------

df1 = df1.join(df0, ["pickupdate", "pickupyear", "pickupmonth", "pickupdayofyear"])
df1.cache()

# COMMAND ----------

import holidays

hds = []
for y in range(year_range[0], year_range[1]):
  hds += holidays.US(state="NY", years=y).keys()

# COMMAND ----------

df2 = df1.withColumns({
  "isWeekend": F.when(F.col("pickupdate").isin(hds) | F.dayofweek(F.col("pickupdate")).isin([1, 7]), 1).otherwise(0),
  "isOvernight": F.when(F.col("pickuphour").isin(list(range(20, 24))+list(range(0, 6))), 1).otherwise(0),
  "d_base_fare_per_mile": F.col("daily_total_base_fare") / F.col("daily_total_miles"),
  "d_base_fare_per_min": F.col("daily_total_base_fare") / F.col("daily_total_time") * 60,
  "h_base_fare_per_mile": F.col("hourly_total_base_fare") / F.col("hourly_total_miles"),
  "h_base_fare_per_min": F.col("hourly_total_base_fare") / F.col("hourly_total_time") * 60
  }) \
  .withColumn("isRushhour", 
              F.when(F.col("pickuphour").isin(list(range(16, 20))) | (F.col("isWeekend") == 0), 1).otherwise(0)
              )
df2.cache()
df1.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Incorporating NY Daily Weather Data

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

display(weather_df)

# COMMAND ----------

weather_df=weather_df.withColumns({
  "pickupdayofyear": F.dayofyear(F.col('Date')),
  "pickupmonth": F.month(F.col('Date')),
  "pickupyear": F.year(F.col('Date'))})

display(weather_df)

# COMMAND ----------

weather_df = weather_df \
  .drop(F.col("Date"))

display(weather_df)

# COMMAND ----------

joined_df = df2.join(weather_df, ["pickupyear", "pickupmonth", "pickupdayofyear"]).drop("pickupdate")

joined_df.cache()
df2.unpersist()

# COMMAND ----------

vectorAssembler = VectorAssembler(
    inputCols=joined_df.columns,
    outputCol="features",
    handleInvalid="skip"
    )
feature_vectors = vectorAssembler.transform(joined_df).select("features")

# create a correlation matrix
correlation = Correlation.corr(feature_vectors, "features").collect()[0][0]
corr_matrix = correlation.toArray().tolist()
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns=joined_df.columns, index=joined_df.columns)
corr_matrix_df.style.background_gradient(cmap='coolwarm')

# COMMAND ----------

# create feature for extremity of the temperture (abs(temp-68) 20C is 68F)
joined_df1 = joined_df.withColumns({
  "TMAXExtremity": F.abs(F.col("TMAX (Degrees Fahrenheit)") - 68), 
  "TMINExtremity": F.abs(F.col("TMIN (Degrees Fahrenheit)") - 68)
  })

# COMMAND ----------

vectorAssembler = VectorAssembler(
    inputCols=joined_df1.columns,
    outputCol="features",
    handleInvalid="skip"
    )
feature_vectors = vectorAssembler.transform(joined_df1).select("features")

# create a correlation matrix
correlation = Correlation.corr(feature_vectors, "features").collect()[0][0]
corr_matrix = correlation.toArray().tolist()
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns=joined_df1.columns, index=joined_df1.columns)
corr_matrix_df.style.background_gradient(cmap='coolwarm')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Incorporating MTA Public Transport Daily Ridership Data

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

display(mta_df)

# COMMAND ----------

mta_df.cache()

# COMMAND ----------

joined_df2 = joined_df1.join(mta_df, ["pickupyear", "pickupmonth", "pickupdayofyear"])

# COMMAND ----------

joined_df2.cache()
joined_df1.unpersist()

# COMMAND ----------

vectorAssembler = VectorAssembler(
    inputCols=joined_df2.columns,
    outputCol="features",
    handleInvalid="skip"
    )
feature_vectors = vectorAssembler.transform(joined_df2).select("features")

# create a correlation matrix
correlation = Correlation.corr(feature_vectors, "features").collect()[0][0]
corr_matrix = correlation.toArray().tolist()
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns=joined_df2.columns, index=joined_df2.columns)
corr_matrix_df.style.background_gradient(cmap='coolwarm')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating pt_taxi_ratio which represents the ratio of the number of public transport users and taxi passengers in each day

# COMMAND ----------

# add an additional feature of taxi trips / public transport ridership (daily)
joined_df3 = joined_df2.withColumn("pt_taxi_ratio", 
        (F.col("subways_daily") + F.col("buses_daily") + F.col("LIRR_daily") + F.col("metro_north_daily") + F.col("Access-A-Ride_daily") + F.col("SIR_daily")) / F.col("daily_numTrips")
    )
joined_df3.cache()
joined_df2.unpersist()

# COMMAND ----------

vectorAssembler = VectorAssembler(
    inputCols=joined_df3.columns,
    outputCol="features",
    handleInvalid="skip"
    )
feature_vectors = vectorAssembler.transform(joined_df3).select("features")

# create a correlation matrix
correlation = Correlation.corr(feature_vectors, "features").collect()[0][0]
corr_matrix = correlation.toArray().tolist()
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns=joined_df3.columns, index=joined_df3.columns)
corr_matrix_df.style.background_gradient(cmap='coolwarm')

# COMMAND ----------

joined_df4 = joined_df3.drop(
  "subways_daily",
  "buses_daily",
  "LIRR_daily",
  "metro_north_daily", 
  "Access-A-Ride_daily", 
  "SIR_daily", 
  "daily_numTrips", 
  "daily_total_miles", 
  "daily_total_time", 
  "daily_total_base_fare",
  "daily_mean_speed",
  "hourly_numTrips",
  "hourly_total_miles",
  "hourly_total_time",
  "hourly_total_base_fare",
  "hourly_mean_speed",
  "TMAX (Degrees Fahrenheit)",
  "TMIN (Degrees Fahrenheit)",
  "d_base_fare_per_mile",
  "d_base_fare_per_min",
  "h_base_fare_per_mile",
  "h_base_fare_per_min"
  )

# COMMAND ----------

joined_df5 = df.join(joined_df4, ["pickupyear", "pickupmonth", "pickupdayofyear", "pickuphour"])

# COMMAND ----------

df6 = joined_df5
df6.cache()
joined_df3.unpersist()

# COMMAND ----------

vectorAssembler = VectorAssembler(
    inputCols=df6.columns,
    outputCol="features",
    handleInvalid="skip"
    )
feature_vectors = vectorAssembler.transform(df6).select("features")

# create a correlation matrix
correlation = Correlation.corr(feature_vectors, "features").collect()[0][0]
corr_matrix = correlation.toArray().tolist()
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns=df6.columns, index=df6.columns)
corr_matrix_df.style.background_gradient(cmap='coolwarm')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dataframe for creating a heat map

# COMMAND ----------

taxi_zone_df = taxi_zone_df.drop("service_zone")
taxi_zone_df = taxi_zone_df.withColumns(
  {
    "PULocationID": F.col("LocationID"),
    "PUBorough": F.col("Borough"),
    "PUZone": F.col("Zone"),
    "DOLocationID": F.col("LocationID"),
    "DOBorough": F.col("Borough"),
    "DOZone": F.col("Zone")
  }) \
  .drop("LocationID", "Borough", "Zone")

# COMMAND ----------

df_for_count = df.select("pickup_datetime", "PULocationID") \
  .withColumns({
    "pickupdate": F.col("pickup_datetime").cast(DateType()),
    "pickuphour": F.hour(F.col("pickup_datetime"))}) \
  .withColumns({
    "isWeekend": F.when(F.col("pickupdate").isin(hds) | F.dayofweek(F.col("pickupdate")).isin([1, 7]), 1).otherwise(0),
    "isOvernight": F.when(F.col("pickuphour").isin(list(range(20, 24))+list(range(0, 6))), 1).otherwise(0)}) \
  .withColumn("isRushhour", 
              F.when(F.col("pickuphour").isin(list(range(16, 20))) | (F.col("isWeekend") == 0), 1)
              .otherwise(0)
              ) \
  .join(taxi_zone_df.select("PULocationID", "PUZone"), "PULocationID") \
  .drop("PULocationID", "pickup_datetime", "pickupdate", "pickuphour")

df_for_count.cache()

# COMMAND ----------

result_df = df_for_count.groupBy("PUZone") \
  .agg(
    F.sum(F.col("isOvernight")).alias("overnight_count"),
    F.sum(F.col("isRushhour")).alias("rushhour_count"),
    F.sum(F.col("isWeekend")).alias("weekend_count")
  )

display(result_df) # save the table for future use (e.g., creating heatmap)