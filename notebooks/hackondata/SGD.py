# Databricks notebook source
# just the variable...
MOUNT_NAME = "s3storage"
# data manipulation 
from pyspark.sql.functions import col,monotonically_increasing_id, split, regexp_extract, regexp_replace, split, explode, lit, when, count, avg, concat, round, array, udf, rank, from_unixtime, length, ltrim

from pyspark.sql.types import IntegerType, DoubleType, BooleanType, StringType
from pyspark.sql.window import Window

import numpy as np
import re

# machine learning 
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
#Farooq Dataset
rawMetasDF = spark.read.parquet("/mnt/%s/meta_Books.parq" % MOUNT_NAME).repartition(128).cache() # full dataset
#rawMetasDF = spark.read.parquet("/mnt/%s/out2.parq" % MOUNT_NAME).cache() # sample dataset: 500,000 lines
#rawMetasDF = spark.read.parquet("/mnt/%s/out.parq" % MOUNT_NAME).cache() # sample dataset: 50,000 lines


#Diego Dataset
rawRatingsDF = spark.read.parquet("/mnt/%s/parquetDataset/ratings_Books.parquet" % MOUNT_NAME).repartition(128).cache()


#Sai Dataset
rawReviewsDF = sqlContext.read.parquet("/mnt/%s/parquetDataset/reviews_Books_5_2.parq" % MOUNT_NAME).repartition(128).cache()

# rawMetasDF contains only products
rawMetasDF_products = (rawMetasDF
                       .select(col("asin"))
                       .withColumn('raw_MetasDF_asin', lit(1))
                       .distinct()
                      )





rawMetasDF_related_products = (rawMetasDF
                               .select(col('related'))
                               .filter('related is not null')
                               .withColumn('general', regexp_extract('related', '\\[(.*?)\\]', 1))
                               .select(explode(split(col("general"), ", ")).alias("asin"))
                               .filter("asin != ''")
                               .withColumn('asin', regexp_replace("asin", '\\"', ""))
                               .withColumn('raw_MetasDF_related', lit(1))
                               .distinct()
                               )






# rawRatingsDF contains both users and products
rawRatingsDF_users = (rawRatingsDF
                      .select(col("reviewerID"))
                      .withColumn('rawRatingsDF_reviewerID', lit(1))
                      .distinct()
                      )





rawRatingsDF_products = (rawRatingsDF
                         .select(col("asin"))
                         .withColumn('rawRatingsDF_asin', lit(1))
                         .distinct()
                        )






# rawReviewsDF contains both users and products
rawReviewsDF_users = (rawReviewsDF
                      .select(col("reviewerID"))
                      .withColumn('rawReviewsDF_reviewerID', lit(1))
                      .distinct()
                      )





rawReviewsDF_products = (rawReviewsDF
                         .select(col("asin"))
                         .withColumn('rawReviewsDF_asin', lit(1))
                         .distinct()
                        )


# users
all_users = (rawRatingsDF_users
             .join(rawReviewsDF_users, 'reviewerID', 'fullouter')
            )





# products
all_products = (rawMetasDF_products
                .join(rawMetasDF_related_products, 'asin', 'fullouter')
                .join(rawRatingsDF_products, 'asin', 'fullouter')
                .join(rawReviewsDF_products, 'asin', 'fullouter')
               )


userConversionTableDF = (all_users
                         .coalesce(1)
                         .select(col("reviewerID"))
                         .distinct()
                         .withColumn("newUserId", monotonically_increasing_id())
                         
                         )



productConversionTableDF = (all_products
                            .coalesce(1)
                            .select(col("asin"))
                            .distinct()
                            .withColumn("newProductId", monotonically_increasing_id())
                            
                            )


ratingsDF_newID = (rawRatingsDF
                   .join(userConversionTableDF, "reviewerID", "left")
                   .join(productConversionTableDF, "asin", "left")
                   .select(col('newProductId').cast(IntegerType()).alias("productId")
                           , col('newUserId').cast(IntegerType()).alias("userId")
                           , col("rating") 
                           , from_unixtime(col("Timestamp")).cast('date').alias("Timestamp")
                          )
                  
                  )



from pyspark.sql.functions import split, regexp_extract, regexp_replace, split, explode, lit, when

#print("raw data")
#rawMetasDF.show(5)

# extract data in a friendly format
temp = (rawMetasDF
        #.withColumn('general', regexp_extract('related', '\\[(.*?)\\]', 1))
        .withColumn('bought_together', regexp_extract('related', '"bought_together": \\[(.*?)\\]', 1))
        .withColumn('also_bought', regexp_extract('related', '"also_bought": \\[(.*?)\\]', 1))
        .withColumn('buy_after_viewing', regexp_extract('related', '"buy_after_viewing": \\[(.*?)\\]', 1))
        .withColumn('also_viewed', regexp_extract('related', '"also_viewed": \\[(.*?)\\]', 1))
        .select('asin'
                , 'salesRank'
                , 'imUrl'
                , 'categories'
                , 'title'
                ,'description'
                , 'price'
                , 'brand'
                , 'bought_together'
                , 'also_bought'
                , 'buy_after_viewing'
                , 'also_viewed'
               )
       )

#print("temp")
#temp.show(5)




# extract product attributes data and clean up
attributesDF = (temp
                .select('asin', 'title', 'brand', 'description', 'price', 'imUrl')
                .distinct()
               )













# ReviewsDF
reviewsDF_newID = (rawReviewsDF
                   .join(userConversionTableDF, "reviewerID", "left")
                   .join(productConversionTableDF, "asin", "left")
                   .select(col('newUserId').cast(IntegerType()).alias("userId"),
                           col('newProductId').cast(IntegerType()).alias("productId"),
                           col("overall"), 
                           col("reviewerName"), 
                           col("helpful"), 
                           col("reviewText"), 
                           col("summary"), 
                           from_unixtime(col("unixReviewTime")).cast('date').alias("Timestamp"),  
                           col("reviewTime")
                          )
                   
                  )



#unify the two datasets. 
ratingsDF_complete = (reviewsDF_newID
                      .select("productId"
                              , "userId"
                              , col("overall").alias("rating").cast(IntegerType())
                              
                             )
                      .unionAll(ratingsDF_newID
                                .select("productId"
                                        , "userId"
                                        , col("rating").cast(IntegerType())
                                        
                                       )
                               )
                      .distinct()
                      .filter("rating>=1 and rating<=5")
                      .cache()
                     )

print('ratingsDF_complete')

#inpDF = ratingsDF_complete.limit(100).cache()
rawMetasDF.unpersist()
rawRatingsDF.unpersist()
rawReviewsDF.unpersist()
userConversionTableDF.unpersist()
productConversionTableDF.unpersist()
ratingsDF_newID.unpersist()
#attributesDF_newID.unpersist()
reviewsDF_newID.unpersist()


# COMMAND ----------

import numpy as np
from pyspark.sql.functions import col
from pyspark import StorageLevel


class SSGD:
    """
    This is the implementation of the Stratified Stochastic Gradient Descent presented by IBM in the article
    "Large-Scale Matrix Factorization with Distributed Stochastic Gradient Descent". This is not meant to be
    a final implementation ready for the production environment, but just a proof of concept. Ultimately,
    this algorithm uses RDDs which means it would be faster if reimplemented in scala.

    """
    def __init__(self, maxIter=5, nWorkers=12, nFactors=8, beta=0.1, lambda_input=0.3):
        self.iter = maxIter
        self.nExecutors = nWorkers
        self.factors = nFactors
        self.beta = beta
        self.lambda_input = lambda_input
        self.total_iterations = -1
        self.Ni = -1
        self.Nj = -1

    def SGD_update_stratus(self, stratus):
      Warray=np.array()
      Harray=np.array()
      # SGD stepsize
      lmbd = self.lambda_input/(2**self.total_iterations)
      for block in stratus:
        V_block, W_block, H_block = block[1]
        Warray = Warray + np.asarray(W_block,dtype=np.float64)
        Harray = Harray + np.asarray(H_block,dtype=np.float64)
      I = (2**np.linalg.norm(Warray))+(2**np.linalg.norm(Harray))
      return iter(map(lambda block: (block[0], self.SGD_block_update(block[1],I,lmbd)),stratus))

    def SGD_block_update(self, t, I, lmbd):
        # get all three items
        V_block, W_block, H_block = t
        
        # for each item in each tuple
        for (product_id, user_id, rating) in V_block:
            
            Wi = W_block[product_id].copy()
            Hj = H_block[user_id].copy()

            W_block[product_id] = np.dot(np.dot((np.dot(Hj, Hj.T) + lmbd * I), Hj), rating)
            H_block[user_id] = np.dot(np.dot((np.dot(W_block[product_id].T,W_block[product_id]) + lmbd * I),Wi.T), rating)
        return (V_block, W_block, H_block)

    def customPartitioner(self, key):
          """
            This function is used for the partitioner to partition by diagonals and create strata
          """
          for i in range(0,self.nExecutors-1):
            for j in range(0,self.nExecutors-1):
              if (j-i >= 0 and j-i <= self.nExecutors-1):
                if key ==(j,j-i): return i
              if (self.nExecutors-i+j >=0 and self.nExecutors-i+j <= self.nExecutors-1):
                if key ==(j,self.nExecutors-i+j): return i
          
            
    def trainSSGD(self, VDF, userCol="userId", productCol="productId", ratingCol="rating"):
        

        # data in a dataframe
        ratings_data = VDF.select(col(productCol), col(userCol), col(ratingCol)).persist(StorageLevel.MEMORY_AND_DISK)

        # to calculate l2 loss
        self.Nj = ratings_data.rdd.keyBy(lambda x: x[0]).countByKey()
        self.Ni = ratings_data.rdd.keyBy(lambda x: x[1]).countByKey()

        num_products = ratings_data.select(productCol).distinct().count()
        num_users = ratings_data.select(userCol).distinct().count()

        

        # initilized W and H with same number of values as of number of users and movies
        # randomizing according to factors provided by the user
        W = ratings_data.select(col(productCol)).distinct().rdd.map(lambda x: (x[0], np.random.rand(self.factors)-0.5))
        H = ratings_data.select(col(userCol)).distinct().rdd.map(lambda x: (x[0], np.random.rand(self.factors)-0.5))

        #the initial values of W and H divided in blocks
        Wib = W.map(lambda x: (x[0] % self.nExecutors, [x])).reduceByKey(lambda a, b: a + b)
        Hib = H.map(lambda x: (x[0] % self.nExecutors, [x])).reduceByKey(lambda a, b: a + b)

        #we get the indexes of the partitions created (some could potentially be empty, those will be excluded)
        userBlocks = Hib.reduceByKey(lambda a, b: a).map(lambda x: x[0])
        productBlocks = Wib.reduceByKey(lambda a, b: a).map(lambda x: x[0])
        
        #keeping track of counters for the 
        nUserBlocks = userBlocks.count()
        nProductBlocks = productBlocks.count()
        
        #getting the same block replicated by the number of the partitions of the other matrix (I will need this later)
        W = Wib.cartesian(userBlocks).map(lambda x: ((x[0][0], x[1]), x[0][1]))
        H = Hib.cartesian(productBlocks).map(lambda x: ((x[1], x[0][0]), x[0][1]))

        # get blocks of data matrices divided by index.
        V_blocks = ratings_data.rdd.map(lambda x: ((x[0] % self.nExecutors, x[1] % self.nExecutors), [(x[0], x[1], x[2])]))

        #assemble the partitioned dataset getting one block per row in the RDD
        V_group = V_blocks.reduceByKey(lambda a, b: a + b).leftOuterJoin(W).leftOuterJoin(H).map(lambda x: (x[0], (x[1][0][0], dict(x[1][0][1]), dict(x[1][1]))))
        
        #partition by stratus 
        VWH_stratified = V_group.partitionBy(self.nExecutors, lambda k: self.customPartitioner(k)).cache()

        self.total_iterations = 0

        # to keep track of total number of SGD updates made across all strata
        curr_upd_count = V_blocks.count()

        # run till number of iterations provided by user
        while self.total_iterations < self.iter:
            print "Iterations: %d/%d" % (self.total_iterations+1, self.iter)

            # group Vblock, Wib and Hib to send it to SGD update
            #V_group = V_group.mapValues(self.SGD_update)
            VWH_stratified = VWH_stratified.mapPartitions(lambda stratus: SGD_update_stratus(stratus))

            # increment the loop
            self.total_iterations += 1

        W = VWH_stratified.flatMap(lambda x: x[1][1].items()).reduceByKey(lambda a, b: a + b).map(lambda v: (v[0], v[1] / float(nUserBlocks))).cache()
        H = VWH_stratified.flatMap(lambda x: x[1][2].items()).reduceByKey(lambda a, b: a + b).map(lambda v: (v[0], v[1] / float(nProductBlocks))).cache()

        return (W, H)


# COMMAND ----------

import numpy as np
from pyspark.sql.functions import col
from pyspark import StorageLevel


class SSGD:
    """
    This is the implementation of the Stratified Stochastic Gradient Descent presented by IBM in the article
    "Large-Scale Matrix Factorization with Distributed Stochastic Gradient Descent". This is not meant to be
    a final implementation ready for the production environment, but just a proof of concept. Ultimately,
    this algorithm uses RDDs which means it would be faster if reimplemented in scala.

    """
    def __init__(self, maxIter=5, nWorkers=12, nFactors=8, sigma=0.1, lambda_input=0.3):
        self.iter = maxIter
        self.nExecutors = nWorkers
        self.factors = nFactors
        self.sigma = sigma
        self.lambda_input = lambda_input
        self.total_update_count = -1


    def SGD_update(self, t):
        # get all three items
        V_block, W_block, H_block = t

        # for each item in each tuple
        for (product_id, user_id, rating) in V_block:

            Wi = W_block[product_id].copy()
            Hj = H_block[user_id].copy()
            
            W_block[product_id] = ((1-self.sigma)*W_block[product_id])+(self.sigma*(np.dot(np.dot(1.0/np.dot(Hj, Hj.T), Hj), rating)))
            H_block[user_id] = ((1-self.sigma)*H_block[user_id])+(self.sigma*(np.dot(np.dot(1.0/np.dot(W_block[product_id].T,W_block[product_id]),W_block[product_id].T), rating)))
            
        return (V_block, W_block, H_block)
      
    def filterWH(self, entry):
        """
          This function filters out the entries of W and H that do not contain any rating in the block
        """
        (V_block, W_block, H_block) = entry
        V_w = np.array(reduce(lambda a,b: a+b, map(lambda (pid, uid, r): [pid], V_block)))
        V_h = np.array(reduce(lambda a,b: a+b, map(lambda (pid, uid, r): [uid], V_block)))
        W_filtered = dict(filter(lambda (k,v): k in V_w, W_block.iteritems()))
        H_filtered = dict(filter(lambda (k,v): k in V_h, H_block.iteritems()))
        return (V_block, W_filtered, H_filtered)

    def trainSSGD(self, VDF, userCol="userId", productCol="productId", ratingCol="rating"):

        # data in a dataframe
        ratings_data = VDF.select(col(productCol), col(userCol), col(ratingCol)) #.persist(StorageLevel.MEMORY_AND_DISK)

        num_products = ratings_data.select(productCol).distinct().count()
        num_users = ratings_data.select(userCol).distinct().count()

        # global varibale to keep track of all previous itertaions
        self.total_update_count = 0

        # initilized W and H with same number of values as of number of users and movies
        # randomizing according to factors provided by the user
        W = ratings_data.select(col(productCol)).distinct().rdd.map(lambda x: (x[0], np.random.rand(self.factors)-0.5))
        H = ratings_data.select(col(userCol)).distinct().rdd.map(lambda x: (x[0], np.random.rand(self.factors)-0.5))

        #the initial values of W and H divided in blocks
        Wib = W.map(lambda x: (x[0] % self.nExecutors, [x])).reduceByKey(lambda a, b: a + b)
        Hib = H.map(lambda x: (x[0] % self.nExecutors, [x])).reduceByKey(lambda a, b: a + b)

        #we get the indexes of the partitions created (some could potentially be empty, those will be excluded)
        userBlocks = Hib.reduceByKey(lambda a, b: a).map(lambda x: x[0])
        productBlocks = Wib.reduceByKey(lambda a, b: a).map(lambda x: x[0])
        
        #getting the same block replicated by the number of the partitions of the other matrix (I will need this later)
        W = Wib.cartesian(userBlocks).map(lambda x: ((x[0][0], x[1]), x[0][1]))
        H = Hib.cartesian(productBlocks).map(lambda x: ((x[1], x[0][0]), x[0][1]))

        # get blocks of data matrices divided by index.
        V_blocks = ratings_data.rdd.map(lambda x: ((x[0] % self.nExecutors, x[1] % self.nExecutors), [(x[0], x[1], x[2])]))

        #assemble the partitioned dataset getting one block per row in the RDD
        V_group = V_blocks.reduceByKey(lambda a, b: a + b).leftOuterJoin(W).leftOuterJoin(H).map(lambda x: (x[0], (x[1][0][0], dict(x[1][0][1]), dict(x[1][1]))))
        
        #drop H and W entries that don't have at least one rating in the block
        VWH_group = V_group.mapValues(self.filterWH).cache()

        iterations = 0

        # to keep track of total number of SGD updates made across all strata
        curr_upd_count = V_blocks.count()

        # run till number of iterations provided by user
        while iterations < self.iter:
            print "Iterations: %d/%d" % (iterations+1, self.iter)

            # group Vblock, Wib and Hib to send it to SGD update
            VWH_group = VWH_group.mapValues(self.SGD_update)

            # update total updates or 'n' in algorithm 2 after each iteration
            self.total_update_count += curr_upd_count

            # increment the loop
            iterations += 1

        W_merged = VWH_group.flatMap(lambda x: x[1][1].items()).cache() 
        Ni = W_merged.countByKey()
        
        H_merged = VWH_group.flatMap(lambda x: x[1][2].items()).cache() 
        Nj = H_merged.countByKey()
        
        W = W_merged.reduceByKey(lambda a, b: a + b).map(lambda v: (v[0], v[1] / float(Ni[v[0]])))
        H = H_merged.reduceByKey(lambda a, b: a + b).map(lambda v: (v[0], v[1] / float(Nj[v[0]])))
        return (W, H)


# COMMAND ----------

seed = 1800009193L
(split_80_df, split_b_20_df) = ratingsDF_complete.randomSplit([0.5, 0.5], seed)

training_df = split_80_df.cache()
test_df = split_b_20_df.cache()
ssgd = SSGD()

#inputDF=ratingsDF_complete.limit(10000)
(W,H) = ssgd.trainSSGD(training_df)


# COMMAND ----------

W.take(100)

# COMMAND ----------

userCol="userId" 
productCol="productId" 
ratingCol="rating"
nExecutors = 12
factors = 8

ratings_data = training_df.limit(100).select(col(productCol), col(userCol), col(ratingCol)).persist(StorageLevel.MEMORY_AND_DISK)
display(ratings_data)

# COMMAND ----------

W = ratings_data.select(col(productCol)).distinct().rdd.map(lambda x: (x[0], np.random.rand(factors)))
print_rdd(W)

# COMMAND ----------

H = ratings_data.select(col(userCol)).distinct().rdd.map(lambda x: (x[0], np.random.rand(factors)))
print_rdd(H)

# COMMAND ----------

Wib = W.map(lambda x: (x[0] % nExecutors, [x])).reduceByKey(lambda a, b: a + b)
print_rdd(Wib)

# COMMAND ----------

Hib = H.map(lambda x: (x[0] % nExecutors, [x])).reduceByKey(lambda a, b: a + b)
print_rdd(Hib)

# COMMAND ----------

userBlocks = Hib.reduceByKey(lambda a, b: a).map(lambda x: x[0])
print_rdd(userBlocks)

# COMMAND ----------

productBlocks = Wib.reduceByKey(lambda a, b: a).map(lambda x: x[0])
print_rdd(productBlocks)

# COMMAND ----------

W = Wib.cartesian(userBlocks).map(lambda x: ((x[0][0], x[1]), x[0][1]))
print_rdd(W)

# COMMAND ----------

H = Hib.cartesian(productBlocks).map(lambda x: ((x[1], x[0][0]), x[0][1]))
print_rdd(H)

# COMMAND ----------

V_blocks = ratings_data.rdd.map(lambda x: ((x[0] % nExecutors, x[1] % nExecutors), [(x[0], x[1], x[2])]))
print_rdd(V_blocks)


# COMMAND ----------

WH = W.leftOuterJoin(H)
print_rdd(WH)

# COMMAND ----------

V_group = V_blocks.reduceByKey(lambda a, b: a + b).leftOuterJoin(WH).map(lambda x: (x[0], (x[1][0], dict(x[1][1][0]), dict(x[1][1][1])))).persist(StorageLevel.OFF_HEAP)
print_rdd(V_group)

# COMMAND ----------

print W.take(10)

# COMMAND ----------

sqlContext.createDataFrame(W.map(lambda x:(x[0], float(x[1][0]), float(x[1][1]),float(x[1][2]),float(x[1][3]),float(x[1][4]),float(x[1][5]),float(x[1][6]),float(x[1][7]),float(x[1][2])))).write.parquet("/mnt/%s/parquetDataset/results-productFeatures-full.parquet" % MOUNT_NAME) 
sqlContext.createDataFrame(H.map(lambda x:(x[0], float(x[1][0]), float(x[1][1]),float(x[1][2]),float(x[1][3]),float(x[1][4]),float(x[1][5]),float(x[1][6]),float(x[1][7]),float(x[1][2])))).write.parquet("/mnt/%s/parquetDataset/results-userFeatures-full.parquet" % MOUNT_NAME) 
#inputDF.write.parquet("/mnt/%s/parquetDataset/SGD-input.parquet" % MOUNT_NAME) 

# COMMAND ----------

print spark.sql.shuffle.partitions

# COMMAND ----------

print W.take(10)

# COMMAND ----------

print H.take(10)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC Farooq - Working on prediction function

# COMMAND ----------

# Initialize ALS algorithm
als = (ALS()
       #.setSeed(seed)
       .setParams(userCol="userId", itemCol="productId", ratingCol="rating")
       .setMaxIter(5)
       .setRegParam(0.03)
       .setRank(8)
      )

# Create the model with these parameters.
model = als.fit(inpDF)

# COMMAND ----------

model.userFactors.show(5)

# COMMAND ----------

model.itemFactors.show(5)

# COMMAND ----------

def predict(user, item, userFactors, itemFactors):
  """
  Inputs: 
    
    user (integer): userId
    
    item (integer): itemId
    
    userFactors (sparkDF): Must contains at least the following 2 columns:
                              id (integer) = userId
                              features (list) = 
                              
    itemFactors (sparkDF): Must contains at least the following 2 columns:
                              id (integer) = userId
                              features (list) = 
  Outputs: 
    
    user_item_rating (float): Predicted rating for item by user
    
    
  """
  
  # extract row for user
  user_row = (userFactors
              .filter(col('id')==user)
             )
  
  # extract row for item
  item_row = (itemFactors
              .filter(col('id')==item)
             )
  
  # extract features for user
  user_features = np.asarray(user_row.collect()[0][1]
                            )
  
  # extract features for item
  item_features = np.asarray(item_row.collect()[0][1]
                            )
  
  # dot product of user_features and item_features
  user_item_rating = user_features.dot(item_features)
  
  return user_item_rating

# COMMAND ----------

# sample data
userId = 483250
productId = 79200
userFactorsDF = model.userFactors
itemFactorsDF = model.itemFactors

# test function 
predict(user = 483250, item = 79200, userFactors = userFactorsDF, itemFactors = itemFactorsDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC Test on Diego W & H

# COMMAND ----------

W.count()

# COMMAND ----------

from pyspark.sql.types import *

schema = StructType([StructField('id', IntegerType()), 
                     StructField('features', StringType())
                    ])

df = sqlContext.createDataFrame(H, schema)

df.show(5)

# COMMAND ----------

# sample data
userId = 483250
productId = 79200

# need to turn these into dataframes
userFactorsDF = H.toDF()
# itemFactorsDF = W

userFactorsDF.show()

# test function 
# predict(user = 483250, item = 79200, userFactors = userFactorsDF, itemFactors = itemFactorsDF)

# COMMAND ----------

