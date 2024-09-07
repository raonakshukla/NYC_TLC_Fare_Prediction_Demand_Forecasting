# Databricks notebook source
# MAGIC %md
# MAGIC ### Data Visualisation

# COMMAND ----------

!pip install statsmodels

# COMMAND ----------

(sc, spark)

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .master("local[*]") \
    .appName("MLlib lab") \
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

# MAGIC %matplotlib inline
# MAGIC import matplotlib.pyplot as plt
# MAGIC import pyspark.sql.functions as F
# MAGIC from pyspark.sql import Row
# MAGIC from pyspark.sql.types import StructType, StructField, StringType, TimestampNTZType, LongType, DoubleType, IntegerType, DateType
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC from pyspark.ml.feature import VectorAssembler
# MAGIC from pyspark.ml.stat import Correlation
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.sql.types import *

import pandas as pd


directory = '/mnt/2024-team14/'


mta_df = spark.read.csv("dbfs:/mnt/2024-team14/csvs/MTA_2020mar_2024apr.csv", header=True)
weather_df = spark.read.csv("dbfs:/mnt/2024-team14/csvs/weather_data.csv", header=True)
df = spark.read.parquet(directory)

# (column with mismatch, desirable type)
mismatch_col = [
  ("wav_match_flag", "int"), 
  ("airport_fee", "double"), 
  ("PULocationID", "bigint"), 
  ("DOLocationID", "bigint")
  ]
df = df.withColumns({c: F.col(c).cast(t) for c, t in mismatch_col})

df.cache()

# COMMAND ----------

# extract the year,month,dayofyear,dayofweek and hour from pickup datetime
df=df.withColumns({
  "pickupyear" : F.year(df.pickup_datetime),
  "pickupdayofyear" : F.dayofyear(df.pickup_datetime),
  "pickuphour": F.hour(df.pickup_datetime),
  "pickupmonth": F.month(df.pickup_datetime),
  "pickupdayofweek":F.dayofweek(df.pickup_datetime)
  })\
    .drop(*['dispatching_base_num','originating_base_num','request_datetime','on_scene_datetime','pickup_datetime','dropoff_datetime','PULocationID','DOLocationID','trip_miles','trip_time','base_passenger_fare','tolls','bcf','sales_tax','congestion_surcharge','airport_fee','tips','driver_pay','shared_request_flag','shared_match_flag', 'access_a_ride_flag','wav_request_flag','wav_match_flag',])

# COMMAND ----------

df.columns

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Trip count yearly chart

# COMMAND ----------

# trip counts by year (line chart)
df_year = df.groupby("hvfhs_license_num","pickupyear").agg(F.count("*").alias("trip_count")).orderBy("pickupyear")
df_02=df_year.where("hvfhs_license_num = 'HV0002'").toPandas()
df_03=df_year.where("hvfhs_license_num = 'HV0003'").toPandas()
df_04=df_year.where("hvfhs_license_num = 'HV0004'").toPandas()
df_05=df_year.where("hvfhs_license_num = 'HV0005'").toPandas()
plt.figure(figsize=(10,6))
plt.plot(df_02['pickupyear'],df_02['trip_count'],color='red',linewidth=5)
plt.plot(df_03['pickupyear'],df_03['trip_count'],color='yellow',linewidth=5)
plt.plot(df_04['pickupyear'],df_04['trip_count'],color='blue',linewidth=5)
plt.plot(df_05['pickupyear'],df_05['trip_count'],color='green',linewidth=5)
plt.title('trips count by year')
plt.xlabel('year')
plt.ylabel('trip counts')
plt.legend(["Juno","Uber","Via","Lyft"])
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Trip count hourly chart

# COMMAND ----------

# trip count by hour (line chart)
df_hour = df.groupby("hvfhs_license_num","pickuphour").agg(F.count("*").alias("trip_count")).orderBy("pickuphour")
# Filter the data for each license number separately
df_02=df_hour.where("hvfhs_license_num = 'HV0002'").toPandas()
df_03=df_hour.where("hvfhs_license_num = 'HV0003'").toPandas()
df_04=df_hour.where("hvfhs_license_num = 'HV0004'").toPandas()
df_05=df_hour.where("hvfhs_license_num = 'HV0005'").toPandas()
plt.figure(figsize=(10,6))
plt.plot(df_02['pickuphour'],df_02['trip_count'])
plt.plot(df_03['pickuphour'],df_03['trip_count'])
plt.plot(df_04['pickuphour'],df_04['trip_count'])
plt.plot(df_05['pickuphour'],df_05['trip_count'])
plt.title('trips count by hour')
plt.xlabel('hour')
plt.ylabel('trip counts')
plt.legend(["Juno","Uber","Via","Lyft"])
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Trip count daily chart

# COMMAND ----------

# trip count by dates (line chart)
df_date = df.groupby("hvfhs_license_num","pickupdayofyear").agg(F.count("*").alias("trip_count")).orderBy("pickupdayofyear")
df_02=df_date.where("hvfhs_license_num = 'HV0002'").toPandas()
df_03=df_date.where("hvfhs_license_num = 'HV0003'").toPandas()
df_04=df_date.where("hvfhs_license_num = 'HV0004'").toPandas()
df_05=df_date.where("hvfhs_license_num = 'HV0005'").toPandas()
plt.figure(figsize=(10,6))
plt.plot(df_02['pickupdayofyear'],df_02['trip_count'],color='red',linewidth=5)
plt.plot(df_03['pickupdayofyear'],df_03['trip_count'],color='yellow',linewidth=5)
plt.plot(df_04['pickupdayofyear'],df_04['trip_count'],color='blue',linewidth=5)
plt.plot(df_05['pickupdayofyear'],df_05['trip_count'],color='green',linewidth=5)
plt.title('trips count by date')
plt.xlabel('dates')
plt.ylabel('trip counts')
plt.legend(["Juno","Uber","Via","Lyft"])
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Trip count monthly chart

# COMMAND ----------

# trip count by month
df_month = df.groupby("hvfhs_license_num","pickupmonth").agg(F.count("*").alias("trip_count")).orderBy("pickupmonth")
df_02=df_month.where("hvfhs_license_num = 'HV0002'").toPandas()
df_03=df_month.where("hvfhs_license_num = 'HV0003'").toPandas()
df_04=df_month.where("hvfhs_license_num = 'HV0004'").toPandas()
df_05=df_month.where("hvfhs_license_num = 'HV0005'").toPandas()
plt.figure(figsize=(10,6))
plt.plot(df_02['pickupmonth'],df_02['trip_count'])
plt.plot(df_03['pickupmonth'],df_03['trip_count'])
plt.plot(df_04['pickupmonth'],df_04['trip_count'])
plt.plot(df_05['pickupmonth'],df_05['trip_count'])
plt.title('trips count by months')
plt.xlabel('months')
plt.ylabel('trip counts')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Trip count montly chart by affiliated ride-share companies

# COMMAND ----------

# trip count by month (grouped bar chart)
df_companies_month=df.groupby("hvfhs_license_num","pickupmonth").count().orderBy("hvfhs_license_num","pickupmonth")
df_cm = df_companies_month.toPandas()
cm_df=df_cm.pivot(index='pickupmonth',columns='hvfhs_license_num',values='count')
cm_df.plot(kind='bar',figsize=(10,7),width=0.8)
plt.title('Trips by Companies and Month')
plt.xlabel('Month')
plt.ylabel('Number of Trips')
plt.xticks(rotation=0)  # Rotates the x labels to horizontal
plt.legend(["Juno","Uber","Via","Lyft"])
plt.tight_layout()
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Trip count weekly chart by affiliated ride-share companies

# COMMAND ----------

# trip count by month and dayofweek (stacked bar chart)
df_month_week = df.groupby("pickupmonth","pickupdayofweek").agg(F.count("*").alias("trip_count")).orderBy("pickupmonth","pickupdayofweek")
df_mw=df_month_week.toPandas()
pivoted_df=df_mw.pivot(index="pickupmonth",columns='pickupdayofweek',values='trip_count')
pivoted_df.columns = [ 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat','Sun']
pivoted_df.index= ['January','February','March','April','May','June','July','August','September','October','November','December']
pivoted_df.plot(kind='bar', stacked=True, figsize=(10, 7))
plt.title('Trips by Day and Month')
plt.xlabel('Month')
plt.ylabel('Number of Trips')
plt.legend(title='Day of Week')
plt.grid(True)
plt.show()