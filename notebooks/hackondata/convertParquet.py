# Databricks notebook source
# Replace with your values
#
# NOTE: Set the access to this notebook appropriately to protect the security of your keys.
# Or you can delete this cell after you run the mount command below once successfully.

MOUNT_NAME = "s3storage"


sqlContext.setConf("spark.sql.parquet.compression.codec.", "snappy")
DF = sqlContext.read.json("/mnt/%s/kcore_5.json.gz" % MOUNT_NAME).repartition(20).cache()
DF.write.parquet("/mnt/%s/parquetDataset/kcore_5.parq" % MOUNT_NAME)

# COMMAND ----------

MOUNT_NAME = "s3storage"
DF = sqlContext.read.csv("/mnt/%s/ratings_only_full.csv" % MOUNT_NAME)

# COMMAND ----------

DF.show(10)

# COMMAND ----------

from pyspark.sql.functions import col
sqlContext.setConf("spark.sql.parquet.compression.codec.", "snappy")
DF = DF.select(col("_c0").alias("asin"), col("_c1").alias("reviewerId"), col("_c2").alias("overall"), col("_c3").alias("timestamp")).cache()
DF.write.parquet("/mnt/%s/parquetDataset/ratings_only_full.parq"  % MOUNT_NAME)

# COMMAND ----------


import pandas as pd
import gzip

# functions
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)
    
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

# read in data
meta_df = getDF("/dbfs/mnt/%s/reviews_Books_5.json.gz" % MOUNT_NAME)

sqlContext.setConf("spark.sql.parquet.compression.codec.", "snappy")
sqlContext.createDataFrame(meta_df).repartition(10).write.parquet("/mnt/%s/parquetDataset/reviews_Books_5.parq" % MOUNT_NAME)

# COMMAND ----------

MOUNT_NAME = "s3storage"
DF = spark.sql("select * from reviews_books_5")
sqlContext.setConf("spark.sql.parquet.compression.codec.", "snappy")
DF.limit(500000).repartition(10).write.parquet("/mnt/%s/parquetDataset/reviews_Books_5_sample.parq" % MOUNT_NAME)

# COMMAND ----------

df = sqlContext.read.parquet("/mnt/s3storage/meta_Books.parq")

# COMMAND ----------

df.show(10)

# COMMAND ----------

df.limit(500000).repartition(1).write.parquet("/mnt/s3storage/out2.parq")

# COMMAND ----------

