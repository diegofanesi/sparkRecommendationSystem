# Databricks notebook source
# MAGIC %md 
# MAGIC # Hybrid, Scalable, Online Recommendations
# MAGIC 
# MAGIC _Diego Fanesi, Farooq Qaiser, Sai Chaitanya_  

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Introduction 
# MAGIC 
# MAGIC If you've ever shopped on Amazon, or listened to music on Spotify, or watched videos on Youtube, you've probably used their recommendation system, perhaps without even realizing it. A reccommendation system will typically point out one or two other pieces content that you might like, such as the ones below.

# COMMAND ----------

links = ['http://blog.sendblaster.com/wp-content/uploads/amazon-email-reccomend.jpg',
         'https://cdn.vox-cdn.com/thumbor/A-wdLgp-Wm0cQ3aM_lXvgfdDRqc=/cdn.vox-cdn.com/uploads/chorus_asset/file/4109214/Discover_Weekly_Snapshot.0.png',
         'https://cdn-images-1.medium.com/max/1248/0*xLPGdyqD3SoF-38C.', 
         'http://spc.columbiaspectator.com/sites/default/files/netflix_0.jpg'
        ]

html =  [("<img style='height:300px;' src ='" + link + "'>") for link in links]

displayHTML(''.join(html))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Recommendation systems have quickly become a ubiqitious part of the web experience. Almost everything you read, see, or buy on the internet these days has been selected by an algorithm. That includes news articles on Google, status updates on Facebook, products shown on Amazon, or movies on Netflix.
# MAGIC 
# MAGIC From a user standpoint, reccommendation systems allow users to quickly and easily find related/useful content, tailored specifically to their preferences. According to a [McKinsey report](http://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers) 35% of what consumers purchase on Amazon and 75% of what they watch on Netflix comes from product recommendations.  
# MAGIC 
# MAGIC The value of reccommendation systems for businesses is also well established in literature. For example, [Lee, D & Hosanagar, K. (2014)](https://www.researchgate.net/publication/289062986_Impact_of_recommender_systems_on_sales_volume_and_diversity) show that purchase-based collaborative filtering methods ("Consumers who bought this item also bought") caused a 25% lift in views and a 35% lift in the number of items purchased over the control group (no recommender) for those who purchase. Similarly, according to a [paper](http://dl.acm.org/citation.cfm?id=2843948) published by executives at Netflix (the popular on-demand video streaming service), they estimate that the company is able to save $1 billion per year by reccommending new content to viewers so they keep watching and don’t cancel their accounts.      
# MAGIC 
# MAGIC Given their incredible value and commercial importance, we chose to build a reccommendation system for our HackOn(Data) 2017 project.   
# MAGIC 
# MAGIC ### Problem statement
# MAGIC 
# MAGIC 
# MAGIC ![Amazon_home_page_mock](https://lh5.googleusercontent.com/J8vlDc1A9VuYSllInFrmhk05gAj1IghcrzC8RyxZpIObqR5kb-yEyE1QwKqTuzlWODQh03Cn=w1920-h971 "image description")
# MAGIC 
# MAGIC Let’s say we have a user called Molly and she’s currently browsing on Amazon. She and her husband have just had a baby and she’s looking for books to help her understand baby sign language. What she wants to be able to do is easily find other books about baby sign language. In addition, our data shows that she’s highly likely to be interested in romance novels as well so we reccommend some some romance novels she might also like.   
# MAGIC 
# MAGIC However, we don’t want to do this just for Molly! We want to do it for all of our other millions of users. In other words, we want to do this at scale!  
# MAGIC 
# MAGIC In addition, we also want to do this in real time. If Molly is looking at a book about baby sign language right now, we want to be able to reccommend baby products to her also right now. 
# MAGIC 
# MAGIC This is exactly what we were trying to do for our project. Below, we summarise the nature of the problems we tackled, our expermiments and the final solution.  
# MAGIC 
# MAGIC ![Problem_experiment_solution](https://lh4.googleusercontent.com/hSJao0xNdQMUmTWSMEvJgHCz6UkqG4QrzseGKcZ_uKxozzZrEedbILZWkmspLLtb9ohY2DLqOaaq2q8=w1366-h659 "image description")
# MAGIC 
# MAGIC ### How we’re solving the problem
# MAGIC 
# MAGIC We build a reccommendation engine that leverages:   
# MAGIC   
# MAGIC 1. A hybrid (content based and collaborative filtering) approach to produce robust reccommendations     
# MAGIC   
# MAGIC 2. The Spark framework to produce a scalable implementation  
# MAGIC   
# MAGIC 3. Advanced techniques such as Locality Sensitive Hashing (LSH) and Stochastic Gradient Descent (SGD) to demonstrate real-time capabilities  
# MAGIC 
# MAGIC ### How this notebook is laid out
# MAGIC 
# MAGIC 1. Data preparation
# MAGIC 2. Exploratory data analysis
# MAGIC 3. Baseline recommender
# MAGIC 4. Content based filtering (N^2) recommender
# MAGIC 5. Content based filtering (Locality Sensitive Hashing) recommender
# MAGIC 6. Collaborative filtering (Alternating Least Squares) recommender
# MAGIC 7. Collaborative filtering (Schotastic Gradient Descent) recommender
# MAGIC 8. Conclusions
# MAGIC 9. Future areas of improvement and research areas 
# MAGIC 
# MAGIC Next we introduce basic principles behind each type of recommendation system. 
# MAGIC 
# MAGIC ### A brief introduction to how reccommendation systems work  
# MAGIC 
# MAGIC Reccommendation engines can be broadly categorized as one of the following:  
# MAGIC   
# MAGIC 1. Content based methods  
# MAGIC    These make a recommendation on the basis of a user profile constructed from features of content they've previously viewed.    
# MAGIC      
# MAGIC 2. Collaborative filtering methods  
# MAGIC    These make a recommendation by finding users with similar tastes to other users and recommending items that those other users also liked.  
# MAGIC      
# MAGIC 3. Hybrid  methods  
# MAGIC    These are usually a combination of the previous two methods.
# MAGIC 
# MAGIC The majority of reccommendation engines in production use today are of the hybrid variety. Netflix, is again a good example of this. They make recommendations by comparing the viewing and searching habits of similar users (i.e. collaborative filtering) as well as by offering content that shares characteristics with content that a user has previously rated highly (i.e. content based filtering).  
# MAGIC 
# MAGIC That's it for now, we explain each of these approaches in  detail in their respective sections below.  
# MAGIC   
# MAGIC ### Credits
# MAGIC 
# MAGIC We would like to thank Julian McAuley for sharing the Amazon datasets that were used for this project.  
# MAGIC _R. He, J. McAuley. Modeling the visual evolution of fashion trends with one-class collaborative filtering. WWW, 2016_   
# MAGIC _J. McAuley, C. Targett, J. Shi, A. van den Hengel. Image-based recommendations on styles and substitutes. SIGIR, 2015_  

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Administrative stuff

# COMMAND ----------

# MAGIC %md 
# MAGIC Notebook parameters

# COMMAND ----------

# just the variable...
MOUNT_NAME = "s3storage"

# COMMAND ----------

seed = 1800009193L

# COMMAND ----------

# data manipulation 
from pyspark.sql.functions import col,monotonically_increasing_id, split, regexp_extract, regexp_replace, split, explode, lit, when, count, avg, concat, round, array, udf, rank, from_unixtime, length, ltrim

from pyspark.sql.types import IntegerType, DoubleType, BooleanType, StringType
from pyspark.sql.window import Window
from pyspark import StorageLevel

import pandas as pd
import numpy as np
import re

# machine learning 
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# COMMAND ----------

# MAGIC %md 
# MAGIC Load data

# COMMAND ----------

#Farooq Dataset
rawMetasDF = spark.read.parquet("/mnt/%s/meta_Books.parq" % MOUNT_NAME).cache() # full dataset
print rawMetasDF.count()

#Diego Dataset
rawRatingsDF = spark.read.parquet("/mnt/%s/parquetDataset/ratings_Books.parquet" % MOUNT_NAME).cache()
print rawRatingsDF.count()

#Sai Dataset
rawReviewsDF = sqlContext.read.parquet("/mnt/%s/parquetDataset/reviews_Books_5_2.parq" % MOUNT_NAME).cache()
print rawReviewsDF.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's take a quick look at our datasets. 

# COMMAND ----------

print('rawRatingsDF')
rawRatingsDF.show(5)

print('rawMetasDF')
rawMetasDF.show(5)

print('rawReviewsDF')
rawReviewsDF.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC There's a few things to note here.  
# MAGIC Firstly, most of our data is focused on the products, rather than the users.  
# MAGIC Secondly, we have missing data.  
# MAGIC Thirdly, there is some bad data (malformed rows), see below for a specific example at how reviewTime doesn't actually contain a time.  

# COMMAND ----------

rawReviewsDF.filter('reviewerID == "A1MCAHDE1F3Q6L" AND asin = "000100039X"').show(1)

# COMMAND ----------

# MAGIC %md 
# MAGIC Some of this data is going to be a little hard to work with.  
# MAGIC We'll take care of that in the Data Preparation section up next.  

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data preparation
# MAGIC 
# MAGIC In this section, we perform some basic data manipulation techniques to make it easier to extract data for modelling later on.   

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Useable data format  
# MAGIC First we transform our data into a useable shape.  
# MAGIC The rawRatingsDF looks like its in good shape. 
# MAGIC 
# MAGIC On the other hand, the rawMetasDF doesn't look so great. It would be more useful as severak dataframes as some columns are essentially denormalized structures. Let's fix that now.  

# COMMAND ----------

from pyspark.sql.functions import split, regexp_extract, regexp_replace, split, explode, lit, when


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

print("attributesDF")
attributesDF.show(5)




# extract categories data and clean up
# note each product can be associated with multiple categories
categoriesDF = (temp
                .select('asin', 'categories')
                .withColumn('categories', regexp_extract('categories', '\\[\\[(.*?)\\]', 1))
                .withColumn('categories', regexp_replace("categories", '\\"', ""))
                .filter("categories != ''")
                .select('asin', explode(split(col("categories"), ", ")).alias("categories"))
                .distinct()
                )

print("categoriesDF")
categoriesDF.show(5)




# extract salesRank data and clean up
salesRankDF = (temp
               .select('asin', col('salesRank').alias('salesRankMessy'))
               # extract everything between curly braces
               .withColumn('salesRankMessy', regexp_extract('salesRankMessy', '\\{(.*?)\\}', 1))
               # remove quotes
               .withColumn('salesRankMessy', regexp_replace("salesRankMessy", '\\"', ""))
               .filter("salesRankMessy != ''")
               # normalize structure
               .select('asin', explode(split(col("salesRankMessy"), ", ")).alias("salesRankMessy"))
               # extract salesRankCategory
               .withColumn('salesRankCategory', regexp_extract('salesRankMessy', '(\w+)', 1))
               # extract salesRank
               .withColumn('salesRank', regexp_extract('salesRankMessy', '(\d+)', 1))
               .select('asin', 'salesRankCategory', 'salesRank')
               .distinct()
              )

print("salesRank")
salesRankDF.show(5)




# explode each category
bought_together = (temp
                   .select('asin', explode(split(col("bought_together"), ", ")).alias("related_asin"))
                   .withColumn("related", lit("bought_together"))
                  )

also_bought = (temp
               .select('asin', explode(split(col("also_bought"), ", ")).alias("related_asin"))
               .withColumn("related", lit("also_bought"))
              )

buy_after_viewing = (temp
                     .select('asin', explode(split(col("buy_after_viewing"), ", ")).alias("related_asin"))
                     .withColumn("related", lit("buy_after_viewing"))
                    )

also_viewed = (temp
               .select('asin', explode(split(col("also_viewed"), ", ")).alias("related_asin"))
               .withColumn("related", lit("also_viewed"))
              )

# bring all categories together
relatedProductsDF = (bought_together
                     .unionAll(also_bought)
                     .unionAll(buy_after_viewing)
                     .unionAll(also_viewed)
                     .filter("related_asin != ''")
                     .withColumn('related_asin', regexp_replace("related_asin", '\\"', ""))
                     .select('asin', 'related', 'related_asin')
                    )

print("relatedProductsDF")
relatedProductsDF.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Convert userId and productId to integers
# MAGIC One of the requirements of some of the algorithms that we apply (e.g. ALS) is that userId and productId be in integer format.  
# MAGIC The datasets provided by James McAuley do not fulfil these requirements as the userId and productId are in alphanumeric format.  
# MAGIC As a result, our first stage of data preparation is to convert all userId and productId to an integer format across all of our datasets.  

# COMMAND ----------

# MAGIC %md 
# MAGIC First we extract all the userId and productId we can find across all of our datasets

# COMMAND ----------

# rawMetasDF contains only products
rawMetasDF_products = (rawMetasDF
                       .select(col("asin"))
                       .withColumn('raw_MetasDF_asin', lit(1))
                       .distinct()
                      )

# print('rawMetasDF_products')
# rawMetasDF_products.show(5)



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

# print('rawMetasDF_related_products')
# rawMetasDF_related_products.show(5)




# rawRatingsDF contains both users and products
rawRatingsDF_users = (rawRatingsDF
                      .select(col("reviewerID"))
                      .withColumn('rawRatingsDF_reviewerID', lit(1))
                      .distinct()
                      )

# print('rawRatingsDF_users')
# rawRatingsDF_users.show(5)



rawRatingsDF_products = (rawRatingsDF
                         .select(col("asin"))
                         .withColumn('rawRatingsDF_asin', lit(1))
                         .distinct()
                        )

# print('rawRatingsDF_products')
# rawRatingsDF_products.show(5)




# rawReviewsDF contains both users and products
rawReviewsDF_users = (rawReviewsDF
                      .select(col("reviewerID"))
                      .withColumn('rawReviewsDF_reviewerID', lit(1))
                      .distinct()
                      )

# print('rawReviewsDF_users')
# rawReviewsDF_users.show(5)



rawReviewsDF_products = (rawReviewsDF
                         .select(col("asin"))
                         .withColumn('rawReviewsDF_asin', lit(1))
                         .distinct()
                        )

# print('rawReviewsDF_products')
# rawReviewsDF_products.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC First let's build those master lists of all userId and productId across all of our datasets.  
# MAGIC Then we can check if there are any userId or productId that don't appear across all datasets.  

# COMMAND ----------

# users
all_users = (rawRatingsDF_users
             .join(rawReviewsDF_users, 'reviewerID', 'fullouter')
            )

# print('users not present in all datasets')
# (all_users
#  .where(col("rawRatingsDF_reviewerID").isNull() 
#         | col("rawReviewsDF_reviewerID").isNull() 
#        )
#  .show(5)
#  )



# products
all_products = (rawMetasDF_products
                .join(rawMetasDF_related_products, 'asin', 'fullouter')
                .join(rawRatingsDF_products, 'asin', 'fullouter')
                .join(rawReviewsDF_products, 'asin', 'fullouter')
               )

# print('products not present in all datasets')
# (all_products
#  .where(col("raw_MetasDF_asin").isNull() 
#         | col("raw_MetasDF_related").isNull() 
#         | col("rawRatingsDF_asin").isNull()
#         | col("rawReviewsDF_asin").isNull()
#        )
#  .show(5))

# COMMAND ----------

# MAGIC %md
# MAGIC Alright, let's use those master lists we built above and assign unique (integer-based) IDs to each user and product.  

# COMMAND ----------

userConversionTableDF = (all_users
                         .coalesce(1)
                         .select(col("reviewerID"))
                         .distinct()
                         .withColumn("newUserId", monotonically_increasing_id())
#                          .cache()
                         )

# print('userConversionTableDF')
# userConversionTableDF.show(5)

productConversionTableDF = (all_products
                            .coalesce(1)
                            .select(col("asin"))
                            .distinct()
                            .withColumn("newProductId", monotonically_increasing_id())
#                             .cache()
                            )

# print('productConversionTableDF')
# productConversionTableDF.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC Finally we convert the Id in all datasets to our new ID using the mapping tables we created above.  

# COMMAND ----------

ratingsDF_newID = (rawRatingsDF
                   .join(userConversionTableDF, "reviewerID", "left")
                   .join(productConversionTableDF, "asin", "left")
                   .select(col('newProductId').cast(IntegerType()).alias("productId")
                           , col('newUserId').cast(IntegerType()).alias("userId")
                           , col("rating") 
                           , from_unixtime(col("Timestamp")).cast('date').alias("Timestamp")
                          )
                   .cache()
                  )

print('ratingsDF_newID')
ratingsDF_newID.show(10)




attributesDF_newID = (attributesDF
                      .join(productConversionTableDF, "asin", "left")
                      .select(col('newProductId').cast(IntegerType()).alias("productId"),
                              'title', 
                              'brand', 
                              'description', 
                              'price', 
                              'imUrl')
                      .cache()
                     )

print("attributesDF_newID")
attributesDF_newID.show(5)




categoriesDF_newID = (categoriesDF
                      .join(productConversionTableDF, "asin", "left")
                      .select(col('newProductId').cast(IntegerType()).alias("productId"),
                              "categories")
                     )

print("categoriesDF_newID")
categoriesDF_newID.show(5)




salesRankDF_newID = (salesRankDF
                     .join(productConversionTableDF, "asin", "left")
                     .select(col('newProductId').cast(IntegerType()).alias("productId"),
                             'salesRankCategory', 
                             'salesRank')
                     .cache()
                    )

print("salesRankDF_newID")
salesRankDF_newID.show(5)




relatedProductsDF_newID = (relatedProductsDF
                           .join(productConversionTableDF, 
                                 "asin", 
                                 "left")
                           .join(productConversionTableDF.select('asin', col('newProductId').alias('new_related_asin_Id')), 
                                 relatedProductsDF.related_asin == productConversionTableDF.asin, 
                                 "left")
                           .select(col('newProductId').cast(IntegerType()).alias("productId"),
                                   'related',
                                   col('new_related_asin_Id').cast(IntegerType()).alias("related_productId")
                                  )
                           .cache()
                          )

print("relatedProductsDF_newID")
relatedProductsDF_newID.show(5)




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
                   .cache()
                  )

print('reviewsDF_newID')
reviewsDF_newID.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC There is ratings data in both rawRatingsDF and rawReviewsDF. Let's create a unified dataset of training data from these 2 datasets. 

# COMMAND ----------

#unify the two datasets. 
ratingsDF_complete = (reviewsDF_newID
                      .select("productId"
                              , "userId"
                              , col("overall").alias("rating").cast(IntegerType())
                              , "Timestamp"
                             )
                      .unionAll(ratingsDF_newID
                                .select("productId"
                                        , "userId"
                                        , col("rating").cast(IntegerType())
                                        , "Timestamp"
                                       )
                               )
                      .distinct()
                      .filter("rating>=1 and rating<=5")
                      .cache()
                     )

print('ratingsDF_complete')
ratingsDF_complete.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC Un-cache objects that are no longer necessary to save memory.  

# COMMAND ----------

rawMetasDF.unpersist()
rawRatingsDF.unpersist()
rawReviewsDF.unpersist()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exploratory Data Analysis  

# COMMAND ----------

users_val = userConversionTableDF.count()
items_val = productConversionTableDF.count()
ratings_val = ratingsDF_complete.count()
reviews_val = reviewsDF_newID.count()


print('#users: %s' %users_val)
print('#items: %s' %items_val)
print('#ratings: %s' %ratings_val)
print('#reviews: %s' %reviews_val)

# userConversionTableDF.unpersist()
# productConversionTableDF.unpersist()

# COMMAND ----------

# MAGIC %md 
# MAGIC This is big data.  

# COMMAND ----------

# create temporary views for R to read from for visualizations
categoriesDF_newID.createOrReplaceTempView("categoriesDF_newID")
ratingsDF_complete.createOrReplaceTempView("ratingsDF_complete")
reviewsDF_newID.createOrReplaceTempView("reviewsDF_newID")

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC # load some packages
# MAGIC library(ggplot2)
# MAGIC library(dplyr)
# MAGIC library(lubridate)
# MAGIC library(scales)
# MAGIC library(reshape2)
# MAGIC library(SparkR)
# MAGIC 
# MAGIC 
# MAGIC # color palette
# MAGIC color1 = '#0145ac' # dark blue
# MAGIC color1 = 'white' # white
# MAGIC color2 = '#82c7a5' # light blue
# MAGIC color3 = '#eece1a' # yellow
# MAGIC 
# MAGIC text_color1 = 'white'
# MAGIC 
# MAGIC bg_color1 = '#1b212c'
# MAGIC 
# MAGIC # text size
# MAGIC 
# MAGIC text_size1 = 12
# MAGIC 
# MAGIC axis_title_size1 = 16
# MAGIC 
# MAGIC title_size1 = 20
# MAGIC 
# MAGIC # theme for charts
# MAGIC theme.chart <- 
# MAGIC   theme_classic() + 
# MAGIC   theme(legend.position = "none", 
# MAGIC         axis.ticks = element_blank(), 
# MAGIC         axis.line.x = element_line(color = 'white'), 
# MAGIC         axis.line.y = element_line(color = 'white'), 
# MAGIC         axis.title = element_text(colour = text_color1, size = axis_title_size1),
# MAGIC         panel.background = element_rect(fill = bg_color1),
# MAGIC         plot.background = element_rect(fill = bg_color1), 
# MAGIC         plot.title = element_text(colour = text_color1, size = title_size1)
# MAGIC        )

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC sparkdf <- sql("
# MAGIC SELECT 
# MAGIC   categories AS Category
# MAGIC   , COUNT(*) AS Count
# MAGIC FROM 
# MAGIC   categoriesDF_newID 
# MAGIC GROUP BY 
# MAGIC   categories
# MAGIC HAVING
# MAGIC   COUNT(*)>5000
# MAGIC ")
# MAGIC 
# MAGIC rdf <- collect(sparkdf)
# MAGIC 
# MAGIC # some cleaning
# MAGIC temp <- rdf %>%
# MAGIC   dplyr::filter(Category!='Books')
# MAGIC 
# MAGIC temp$Category <- factor(temp$Category, 
# MAGIC                        levels = rev(temp$Category[order(temp$Count)]))
# MAGIC 
# MAGIC g <- ggplot(temp, aes(Category, Count)) + 
# MAGIC   geom_bar(stat="identity", width = 0.5, fill = color1) + 
# MAGIC   theme.chart +
# MAGIC   labs(title="Distribution of Products by Category") +
# MAGIC   theme(axis.text.x = element_text(angle = 90, vjust=0.6, color = text_color1),
# MAGIC         axis.text.y = element_text(color = text_color1)
# MAGIC        ) + 
# MAGIC   scale_y_continuous(labels = comma)
# MAGIC 
# MAGIC g

# COMMAND ----------

# MAGIC %md 
# MAGIC A few things to note about this visualization, a book can in several categories.  
# MAGIC Secondly, there is a long tail of book categories that is not fully visualized here.  
# MAGIC Lastly, most books aren't identified by any category.  

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC sparkdf <- sql("
# MAGIC SELECT 
# MAGIC   rating
# MAGIC   , COUNT(*) AS Count
# MAGIC FROM 
# MAGIC   ratingsDF_complete 
# MAGIC GROUP BY 
# MAGIC   rating
# MAGIC ")
# MAGIC 
# MAGIC rdf <- collect(sparkdf)
# MAGIC 
# MAGIC # some cleaning
# MAGIC temp <- rdf
# MAGIC 
# MAGIC temp$ratings <- factor(temp$rating, 
# MAGIC                        levels = rev(temp$rating[order(temp$rating)]))
# MAGIC 
# MAGIC g <- ggplot(temp, aes(rating, Count)) + 
# MAGIC   geom_bar(stat="identity", width = 0.5, fill = color1) + 
# MAGIC   labs(title="Distribution of Ratings") +
# MAGIC   theme.chart +
# MAGIC   theme(axis.text.x = element_text(size = 12, vjust=0.5, color = text_color1),
# MAGIC         axis.text.y = element_text(size = 12, color = text_color1)
# MAGIC        ) + 
# MAGIC   scale_y_continuous(labels = comma)
# MAGIC g

# COMMAND ----------

# MAGIC %md 
# MAGIC We see that ratings are skewed towards higher ratings.   
# MAGIC This isn't very surprising as this has been well noted in academic research.    

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC # collect data
# MAGIC sparkdf <- sql("
# MAGIC SELECT 
# MAGIC   categories
# MAGIC   , rating
# MAGIC   , COUNT(*) AS count
# MAGIC FROM 
# MAGIC   ratingsDF_complete AS a
# MAGIC   LEFT JOIN categoriesDF_newID AS b
# MAGIC     ON a.productId = b.productId
# MAGIC WHERE
# MAGIC   categories!='Books'
# MAGIC GROUP BY 
# MAGIC   categories
# MAGIC   , rating
# MAGIC HAVING 
# MAGIC   COUNT(*) > 20000
# MAGIC ")
# MAGIC 
# MAGIC rdf <- collect(sparkdf)
# MAGIC 
# MAGIC # some cleaning
# MAGIC temp <- rdf
# MAGIC 
# MAGIC temp$rating <- factor(temp$rating, 
# MAGIC                       levels = rev(temp$rating[order(temp$rating)]))
# MAGIC 
# MAGIC # plot
# MAGIC g <- ggplot(temp, aes(x = rating, y = count)) + 
# MAGIC   geom_bar(stat="identity", fill = color1) + 
# MAGIC   scale_x_discrete(drop = TRUE) +
# MAGIC   theme.chart +
# MAGIC   labs(title="Distribution of Ratings") +
# MAGIC   facet_wrap(~categories, scales = "free") + 
# MAGIC   theme(strip.text.x = element_text(size = 6, colour = "white"), 
# MAGIC         strip.background = element_blank(), 
# MAGIC         axis.text.x = element_text(vjust=0.5, color = text_color1),
# MAGIC         axis.text.y = element_text(color = text_color1)
# MAGIC        ) 
# MAGIC 
# MAGIC g

# COMMAND ----------

# MAGIC %md 
# MAGIC We see that this bias applies in general to all categories.  

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC sparkdf <- sql("
# MAGIC   SELECT
# MAGIC     YEAR(Timestamp) AS year
# MAGIC     , MONTH(Timestamp) AS month
# MAGIC     , AVG(Rating) AS rating
# MAGIC   FROM 
# MAGIC     ratingsDF_complete
# MAGIC   GROUP BY 
# MAGIC     YEAR(Timestamp)
# MAGIC     , MONTH(Timestamp)
# MAGIC ")
# MAGIC 
# MAGIC rdf <- collect(sparkdf)
# MAGIC 
# MAGIC # some cleaning
# MAGIC temp <- rdf %>%
# MAGIC   dplyr::mutate(date = lubridate::dmy(paste('01', month, year, sep = '/')))
# MAGIC   
# MAGIC g <- ggplot(temp, aes(x = date, y = rating)) + 
# MAGIC   geom_line(color = color1) +  
# MAGIC   geom_hline(yintercept = mean(temp$rating), linetype="dotted", color = color2) +
# MAGIC   theme.chart +
# MAGIC   labs(title="Average Product Rating Over Time", 
# MAGIC        x = 'Time', 
# MAGIC        y = 'Rating') +
# MAGIC   theme(axis.text.x = element_text(angle = 90, vjust=0.5, color = text_color1),
# MAGIC         axis.text.y = element_text(vjust=0.5, color = text_color1)
# MAGIC        ) +
# MAGIC   scale_y_continuous(limits = c(1,5)) +
# MAGIC   scale_x_date(date_breaks = "6 month", date_labels =  "%b %Y")
# MAGIC 
# MAGIC g

# COMMAND ----------

# MAGIC %md 
# MAGIC The average product rating has remained consistently high throughout the period this dataset is concerned with.  
# MAGIC The earliest part of the dataset shows a great deal of variance due to fewer observations.  

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC sparkdf <- sql("
# MAGIC   SELECT
# MAGIC     YEAR(Timestamp) AS year
# MAGIC     , MONTH(Timestamp) AS month
# MAGIC     , COUNT(*) AS reviews
# MAGIC   FROM 
# MAGIC     reviewsDF_newID
# MAGIC   GROUP BY 
# MAGIC     YEAR(Timestamp)
# MAGIC     , MONTH(Timestamp)
# MAGIC ")
# MAGIC 
# MAGIC rdf <- collect(sparkdf)
# MAGIC 
# MAGIC # some cleaning
# MAGIC temp <- rdf %>%
# MAGIC   dplyr::mutate(date = lubridate::dmy(paste('01', month, year, sep = '/')))
# MAGIC   
# MAGIC g <- ggplot(temp, aes(x = date, y = reviews)) + 
# MAGIC   geom_line(color = color1) +  
# MAGIC   theme.chart +
# MAGIC   labs(title="Number of Reviews over Time") +
# MAGIC   theme(axis.text.x = element_text(angle=90, vjust=0.6, color = text_color1),
# MAGIC         axis.text.y = element_text(color = text_color1)
# MAGIC        ) + 
# MAGIC   scale_y_continuous(labels = comma) +
# MAGIC   scale_x_date(date_breaks="1 year", date_labels="%Y")
# MAGIC 
# MAGIC g

# COMMAND ----------

# MAGIC %md
# MAGIC The number of reviews has grown steadily over the years.  

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC sparkdf <- sql("
# MAGIC 
# MAGIC WITH 
# MAGIC 
# MAGIC daily_count AS (
# MAGIC   SELECT
# MAGIC     YEAR(Timestamp) AS year
# MAGIC     , MONTH(Timestamp) AS month
# MAGIC     , DAY(Timestamp) AS day
# MAGIC     , COUNT(*) AS ratings
# MAGIC   FROM 
# MAGIC     ratingsDF_complete
# MAGIC   WHERE
# MAGIC     YEAR(Timestamp) >= 2013
# MAGIC   GROUP BY 
# MAGIC     YEAR(Timestamp)
# MAGIC     , MONTH(Timestamp)
# MAGIC     , DAY(Timestamp)
# MAGIC )
# MAGIC 
# MAGIC   SELECT
# MAGIC     day
# MAGIC     , AVG(ratings) AS ratings
# MAGIC   FROM 
# MAGIC     daily_count
# MAGIC   GROUP BY 
# MAGIC     day
# MAGIC 
# MAGIC ")
# MAGIC 
# MAGIC rdf <- collect(sparkdf)
# MAGIC 
# MAGIC # some cleaning
# MAGIC temp <- rdf %>%
# MAGIC   dplyr::mutate(date = lubridate::dmy(paste(day, '01', '2014', sep = '/')))
# MAGIC   
# MAGIC g <- ggplot(temp, aes(x = date, y = ratings)) + 
# MAGIC   geom_line(color = 'white') +
# MAGIC   geom_hline(yintercept = mean(temp$ratings), linetype="dotted", color = color2) +
# MAGIC   theme.chart +
# MAGIC   labs(title="Average Number of Daily Ratings Since January 2013"
# MAGIC        #,  x = "Day of Month",
# MAGIC        #, y = "Number of Products Rated"
# MAGIC       ) +
# MAGIC   theme(axis.text.x = element_text(vjust=0.6, color = text_color1, size = text_size1),
# MAGIC         axis.text.y = element_text(color = text_color1, size = text_size1),
# MAGIC         axis.title = element_blank()
# MAGIC        ) +
# MAGIC   scale_y_continuous(labels = comma, expand = c(0, 0), limits = c(0, 25000)) +
# MAGIC   scale_x_date(date_breaks="5 day", date_labels="%d")
# MAGIC 
# MAGIC g

# COMMAND ----------

# MAGIC %md 
# MAGIC 2014 - January - number of reviews/ratings coming in daily.  
# MAGIC max daily rate and average daily rate, from whole dataset.  

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC sparkdf <- sql("
# MAGIC SELECT
# MAGIC   userId
# MAGIC   , productId
# MAGIC   , rating
# MAGIC FROM 
# MAGIC   ratingsDF_complete
# MAGIC WHERE
# MAGIC  userId IN (SELECT DISTINCT userId FROM ratingsDF_complete LIMIT 300)
# MAGIC ")
# MAGIC 
# MAGIC rdf <- collect(sparkdf)
# MAGIC 
# MAGIC # some cleaning
# MAGIC temp <- rdf
# MAGIC 
# MAGIC temp$rating <- factor(temp$rating, 
# MAGIC                       levels = rev(temp$rating[order(temp$rating)]))
# MAGIC 
# MAGIC temp$userId <- factor(temp$userId, 
# MAGIC                       levels = rev(temp$userId[order(temp$userId)]))
# MAGIC 
# MAGIC temp$productId <- factor(temp$productId, 
# MAGIC                          levels = rev(temp$productId[order(temp$productId)]))
# MAGIC 
# MAGIC g <- ggplot(temp, aes(x = userId, y = productId)) + 
# MAGIC   geom_tile(fill = 'white') +
# MAGIC   theme.chart +
# MAGIC   labs(title="User - Product Matrix", 
# MAGIC        x = 'User',
# MAGIC        y = 'Product') +
# MAGIC   theme(axis.text = element_blank()
# MAGIC        )
# MAGIC   
# MAGIC g

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC total_number_of_ratings = nrow(temp)
# MAGIC 
# MAGIC total_number_of_products = nrow(temp %>% dplyr::select(productId) %>% dplyr::distinct())
# MAGIC 
# MAGIC total_number_of_users = nrow(temp %>% dplyr::select(userId) %>% dplyr::distinct())
# MAGIC 
# MAGIC total_number_of_potential_ratings = total_number_of_products * total_number_of_users
# MAGIC 
# MAGIC print(paste0('total number of ratings: ', total_number_of_ratings))
# MAGIC print(paste0('total number of products: ', total_number_of_products))
# MAGIC print(paste0('total number of users: ', total_number_of_users))
# MAGIC print(paste0('total number of potential ratings: ', total_number_of_potential_ratings))

# COMMAND ----------

# MAGIC %md 
# MAGIC As expected, the ratings dataset is quite sparse.  

# COMMAND ----------

# We could leverage the `histogram` function from the RDD api
histogram = (ratingsDF_complete
             .groupBy('userId')
             .count()
             .sample(False, 0.01, seed = seed)
            )

# print(histogram.count()) #80,128 records

histogram.createOrReplaceTempView("histogram")

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC sparkdf <- sql("SELECT * FROM histogram")
# MAGIC 
# MAGIC rdf <- collect(sparkdf)
# MAGIC 
# MAGIC # some cleaning
# MAGIC temp <- rdf 

# COMMAND ----------

# MAGIC %r 
# MAGIC 
# MAGIC g <- ggplot(data = temp, aes(x = count)) + 
# MAGIC   geom_histogram(fill = color1) +
# MAGIC   theme.chart +
# MAGIC   labs(title="Histogram of Number of Ratings by Users", 
# MAGIC        x = 'Number of Ratings by User',
# MAGIC        y = 'Frequency (log 10)') +
# MAGIC   theme(axis.text.x  = element_text(angle = 90, vjust=0.5, color = text_color1),
# MAGIC         axis.text.y  = element_text(vjust=0.5, color = text_color1)
# MAGIC        ) +
# MAGIC   scale_y_continuous(labels = comma
# MAGIC                      #, trans='log10'
# MAGIC                     ) + 
# MAGIC   scale_x_continuous(breaks =  scales::pretty_breaks(n = 10))
# MAGIC 
# MAGIC g

# COMMAND ----------

# MAGIC %r 
# MAGIC 
# MAGIC g <- 
# MAGIC   ggplot(data = temp %>% dplyr::filter(count<15), aes(x = count)) + 
# MAGIC   stat_count(fill = color1) +
# MAGIC   theme.chart +
# MAGIC   labs(title="Histogram of Number of Ratings by Users \n(Long Tail Removed)", 
# MAGIC        x = 'Number of Ratings by User',
# MAGIC        y = 'Frequency') +
# MAGIC   theme(axis.text.x  = element_text(vjust=0.5, color = text_color1),
# MAGIC         axis.text.y  = element_text(vjust=0.5, color = text_color1)) +
# MAGIC   scale_y_continuous(labels = comma) + 
# MAGIC   scale_x_continuous(breaks = scales::pretty_breaks(n = 16))
# MAGIC 
# MAGIC g

# COMMAND ----------

# MAGIC %md 
# MAGIC As you can see most users rate very few products.  

# COMMAND ----------

from pyspark.sql.functions import collect_set

# create temporary views for R to read from for visualizations
metaData = (attributesDF_newID
            .join(categoriesDF_newID.groupBy('productId').agg(collect_set('categories').alias('categories')), 
                  'productId', 'fullouter')
            .join(salesRankDF_newID.groupBy('productId').agg(collect_set('salesRank').alias('salesRank'))
                  , 'productId', 'fullouter')
            .join(relatedProductsDF_newID.groupBy('productId').agg(collect_set('related_productId').alias('related_productId'))
                  , 'productId', 'fullouter')
            #.sample(True, 0.0001)
           )

# print(metaData.count())
# metaData.show(5)

metaData.createOrReplaceTempView("metaData")

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC sparkdf <- sql("
# MAGIC SELECT
# MAGIC   title, 
# MAGIC   description
# MAGIC FROM 
# MAGIC   metaData
# MAGIC LIMIT 
# MAGIC   1000
# MAGIC ")
# MAGIC 
# MAGIC rdf <- collect(sparkdf)
# MAGIC 
# MAGIC ggplot_missing <- function(x){
# MAGIC   
# MAGIC   # Nicholas Tierney
# MAGIC   # https://github.com/njtierney
# MAGIC   
# MAGIC   x %>% 
# MAGIC     is.na %>%
# MAGIC     melt %>%
# MAGIC     ggplot(data = .,
# MAGIC            aes(x = Var2,
# MAGIC                y = Var1)) +
# MAGIC     geom_raster(aes(fill = value)) +
# MAGIC     scale_fill_manual(name = "",
# MAGIC                       labels = c("Data Present","Data Missing"), 
# MAGIC                       values = c(color3, bg_color1)) +
# MAGIC     theme.chart +
# MAGIC     theme(axis.text.x  = element_text(size = 16, vjust=0.5, color = text_color1),
# MAGIC           axis.text.y  = element_text(size = 16, vjust=0.5, color = text_color1),
# MAGIC           legend.background = element_rect(fill = bg_color1), 
# MAGIC           legend.text = element_text(size = 16, color = text_color1),
# MAGIC           legend.position = "right"
# MAGIC          ) + 
# MAGIC     labs(title = 'Missing-ness Diagram', 
# MAGIC          x = "Variables",
# MAGIC          y = "ProductId")
# MAGIC }
# MAGIC 
# MAGIC ggplot_missing(rdf)

# COMMAND ----------

# MAGIC %md
# MAGIC For our content based reccommenders, its important to see how much data is missing from potential features.  
# MAGIC As you can see, the situation isn't optimal. Description is often missing but we usually have the title.   
# MAGIC This tells us that a content based recommender approach along is not quite going to cut it.  
# MAGIC Interestingly, we have pretty good image information. This makes the case for an image features based content reccommender.   

# COMMAND ----------

from pyspark.sql.functions import size

tokenizer = (Tokenizer()
             .setInputCol("description")
             .setOutputCol("words")
            )

content_length = (tokenizer
                  .transform(attributesDF_newID
                             .filter(col('description').isNotNull())
                             .filter(length(ltrim(col('description')))!=0))
                  .withColumn('length_of_description', size('words'))
                  .sample(False, 0.1, seed = seed)
                 )

# print(content_length.count()) #106,615 records

content_length.createOrReplaceTempView("content_length")

# COMMAND ----------

# for character length
# from pyspark.sql.functions import length

# content_length = (attributesDF_newID
#                   .withColumn('length_of_description', length('description'))
#                   .sample(False, 0.01, seed = seed)
#                  )

# # print(content_length.count()) #23,254 records

# content_length.createOrReplaceTempView("content_length")

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC sparkdf <- sql("SELECT * FROM content_length")
# MAGIC 
# MAGIC rdf <- collect(sparkdf)
# MAGIC 
# MAGIC # some cleaning
# MAGIC temp <- rdf 

# COMMAND ----------

# MAGIC %r 
# MAGIC 
# MAGIC g <- ggplot(data = temp %>% 
# MAGIC                      dplyr::filter(!is.na(length_of_description)) %>% 
# MAGIC                      dplyr::filter(length_of_description<3000)
# MAGIC             , aes(x = length_of_description)) + 
# MAGIC   geom_histogram(fill = color1) +
# MAGIC   theme.chart +
# MAGIC   labs(title="Number of Words in Description", 
# MAGIC        x = 'Number of Words',
# MAGIC        y = 'Frequency') +
# MAGIC   theme(axis.text.x  = element_text(angle = 90, vjust=0.5, color = text_color1),
# MAGIC         axis.text.y  = element_text(vjust=0.5, color = text_color1)
# MAGIC        ) +
# MAGIC   scale_y_continuous(labels = comma) + 
# MAGIC   scale_x_continuous(breaks =  scales::pretty_breaks(n = 15))
# MAGIC 
# MAGIC g

# COMMAND ----------

# MAGIC %r 
# MAGIC 
# MAGIC g <- ggplot(data = temp %>% 
# MAGIC                      dplyr::mutate(!is.na(length_of_description)) %>% 
# MAGIC                      dplyr::filter(length_of_description<1000)
# MAGIC             , aes(x = 1, y = length_of_description)) + 
# MAGIC   geom_violin(fill = color1) +
# MAGIC   theme.chart +
# MAGIC   labs(title="Length of Product Descriptions", 
# MAGIC        x = '',
# MAGIC        y = 'Number of Words') +
# MAGIC   theme(axis.text.x  = element_blank(),
# MAGIC         axis.text.y  = element_text(vjust=0.5, color = text_color1)
# MAGIC        ) +
# MAGIC   scale_y_continuous(labels = comma, breaks =  scales::pretty_breaks(n = 15))
# MAGIC 
# MAGIC g

# COMMAND ----------

# MAGIC %md
# MAGIC Most descriptions are less than 300 words long.  That's not a lot of data to build features on but as we'll see later, that is plenty enough to give us good results.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Model

# COMMAND ----------

# MAGIC %md 
# MAGIC One very simple way to recommend movies is to always recommend the movies with the highest average rating.  
# MAGIC We filter our products with high ratings for only those that have more than 1000 unique ratings (because products with few ratings may not have broad appeal).  

# COMMAND ----------

highestRatedDF = (ratingsDF_complete
                  .groupBy('productId')
                  .agg(count(col('rating')).alias("count")
                       , avg(col('rating')).alias("average"))
                  .filter("count > 1000")
                  .join(attributesDF_newID, 
                       'productID'
                       , 'left').orderBy(col("average").desc())
                 )

highestRatedDF = (highestRatedDF
                  .withColumn("html", 
                              concat(lit("<img style='height:300px;' src ='"), col("imUrl"), lit("'>"))))

pictureOfBestFive = reduce(lambda row1,row2 : row1[:]+row2[:], highestRatedDF.select("html").limit(5).collect())

displayHTML(''.join(pictureOfBestFive))

# COMMAND ----------

# MAGIC %md 
# MAGIC A way to enhance this is to recommend the highest rated products with more than 500 ratings in a given category.  

# COMMAND ----------

window = (Window
          .partitionBy(['categories'])
          .orderBy(col('average').desc())
          )

highestRatedByCategoryDF = (ratingsDF_complete
                            .join(categoriesDF_newID, 'productID', 'left')
                            .groupBy('categories', 'productId')
                            .agg(count(col('rating')).alias("count")
                                 , avg(col('rating')).alias("average"))
                            .filter("count > 500")
                            .join(attributesDF_newID, 'productID', 'left')
                            .withColumn("rank", rank().over(window).alias('rank'))
                            .filter("rank <= 20")
                            .cache()
                           )

categoriesList = [i.categories for i in highestRatedByCategoryDF.select('categories').distinct().collect()]

for category in categoriesList: 
  print(category)
  highestRatedByCategoryDF.filter(col('categories')==category).show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC These are our baseline models.  

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Content Based Recommender (N^2 approach)
# MAGIC 
# MAGIC #### Brief overview of content based recommenders
# MAGIC 
# MAGIC If for example a user is currently viewing item A, one way of making a recommendation is to look for other items that are similar to Item A. Item B below shows a high similarity with Item A and therefore we can recommend item B back to the user. 
# MAGIC 
# MAGIC ![Introduction_content_based](https://lh6.googleusercontent.com/ZU3NuSmMtDPLvkfWDX9omnWea18_cqnwrUpAbpeKG4OYzsn0hv-1-zOb4GSJWWJHqUBbXxKnuyg9AV4=w1366-h659
# MAGIC  "image description")
# MAGIC 
# MAGIC For more of the technical details, see below.  
# MAGIC 
# MAGIC #### Brief overview of N^2 approach
# MAGIC 
# MAGIC In the N^2 approach, we compared every single product to every other product in our dataset to figure out which products are similar. This is computationally quite expensive and takes a very long time (in fact it never finished for us on the full dataset). For demonstration purposes we build this model on only a small subset of the data below.  
# MAGIC 
# MAGIC ![Introduction_N_squared](https://lh3.googleusercontent.com/2YOa0jnpB0UA4slOUEEKxLUzUbgqk0BFl80Vhsr81cnXF8PC4KtIAb2K41G7peXFAL-Mp_LIGdUFWh0=w1366-h659
# MAGIC  "image description")
# MAGIC 
# MAGIC For more of the technical details, see below.  
# MAGIC 
# MAGIC #### Vector Space Model 
# MAGIC 
# MAGIC Our content based recommender system will use a relatively simple  model, known as the Vector Space Model (VSM) with basic TF-IDF weighting. VSM is essentially a spatial representation of text documents where each document is represented by a vector in a n-dimensional space and each dimension corresponds to a term from the overall vocabulary of a given document collection.  
# MAGIC 
# MAGIC More formally, every document is represented as a vector of term weights, where each weight indicates the degree of association between the document and the term. To make this vector, you need a set of documents D ={d1,d2,...,dN}, and a dictionary of all of the words in the corpus T ={t1,t2,...,tn}. Both of these can be easily obtained using various natural language processing (NLP) techniques, such as tokenization and stopwords removal. More advanced models can make use of NLP techniques like stemming and ngrams.    
# MAGIC 
# MAGIC Each document dj is then represented as a vector in a n-dimensional vector space, so dj ={w1j,w2j,...,dnj}, where wkj is the weight for termt k in document dj. The most commonly used term weighting scheme is Term Frequency-Inverse Document Frequency weighting. TFIDF assumes words that occur frequently in one document (TF), but rarely in the rest of the corpus (IDF), are more likely to be relevant to the topic of the document. As an additional step, normalization of the resulting weight vectors prevents longer documents from having a better chance of retrieval.   
# MAGIC 
# MAGIC Content based recommender systems relying on VSM will have both user proﬁles and items represented as weighted term vectors. Prediction of a user’s interest in a particular item can then be derived by computing the similarity between the user profile and the item. Cosine distance is a commonly used measure here.   
# MAGIC 
# MAGIC Over the next few cells, we demonstrate an implementation of this methodology using Spark.  

# COMMAND ----------

# MAGIC %md 
# MAGIC Create a distinct dataset of productId and product description.   

# COMMAND ----------

# set to -1 to disable
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# udf_category ='Literature & Fiction'
udf_category = 'Entrepreneurship'
# udf_category = 'General'

# COMMAND ----------

# Test case
# col('productId')==3229580, product_description should be null
            
# create a table of products and description
products = (attributesDF_newID
            # add categories data
            .join(categoriesDF_newID, 'productID', 'left')
            # filter for only one category for illustrative purposes
            .filter(col('categories')==udf_category)
            # combine titles and description
            .withColumn('product_description', 
                        when(col('title').isNotNull() & col('description').isNotNull(), concat(col("title"), col("description")))
                        .when(col('title').isNotNull() & col('description').isNull(), col("title"))
                        .when(col('title').isNull() & col('description').isNotNull(), col("description"))
                        .otherwise(lit(None))
                       )
            # only keep observations that have a product description 
            .filter(col('product_description').isNotNull())
            .filter(length(ltrim(col('product_description')))!=0)
            )

# print('total number of rows = %s' %products.count())
products.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC Tokenize the product\_description, remove any stop words, stem words and create ngrams.  

# COMMAND ----------

# tokenize product description
tokenizer = (Tokenizer()
             .setInputCol("product_description")
             .setOutputCol("words")
            )

tokenizedDF = (tokenizer
               .transform(products)
              )
           
# remove stop words 
remover = (StopWordsRemover()
           .setInputCol("words")
           .setOutputCol("features")
          )

noStopWordsDF = (remover
                 .transform(tokenizedDF)
                )


print('total number of rows = %s' %noStopWordsDF.count())
noStopWordsDF.show(5)

# COMMAND ----------

# ngrams
# from pyspark.ml.feature import NGram

# ngram = (NGram()
#          .setN(2)
#          .setInputCol("words_filtered")
#          .setOutputCol("ngrams")
#         )

# ngramDF = (ngram
#            .transform(noStopWordsDF)
#            .select('productId', col('ngrams').alias('features'))
#           )

# COMMAND ----------

# MAGIC %md
# MAGIC Apply TF-IDF. 

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer, IDF

# Word count to vector for each wiki content
vocabSize = 1000000

cvModel = (CountVectorizer()
           .setInputCol("features")
           .setOutputCol("tf")
           .setMinDF(5)
           .setVocabSize(vocabSize)
           .fit(noStopWordsDF)
          )

# Function to return True/False depending on if a sparseVector is not all zero or not 
isNoneZeroVector = udf(lambda v: v.numNonzeros() > 0, BooleanType())

vectorizedDf = (cvModel
                .transform(noStopWordsDF)
                # filter out any rows where the features sparse vector is completely zero
                .filter(isNoneZeroVector(col("tf")))
                # cache for performance
                .cache()
               )

# print('total number of rows = %s' %vectorizedDf.count())
vectorizedDf.show(5)

# COMMAND ----------

# compute IDF
idf = (IDF(inputCol="tf", outputCol="idf", minDocFreq=2)
       .fit(vectorizedDf)
      )

tfidf = idf.transform(vectorizedDf)

# print('total number of rows = %s' %tfidf.count())
tfidf.show(5)

# COMMAND ----------

# sum TFIDF for all terms/(features) for each product/(row)
#sum_ = udf(lambda v: float(v.values.sum()), DoubleType())

#tfidf = (tfidf
#         .withColumn("tf_sum", sum_("tf"))
#         .withColumn("idf_sum", sum_("idf")))

#print('total number of rows = %s' %tfidf.count())
#tfidf.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC Compute cosine similarity for each item against everything other item.   

# COMMAND ----------

compareDF = (tfidf
             .select(col("productId").alias("productId_a"),
                     col("description").alias("description_a"),
                     col("imUrl").alias("imUrl_a"),
                     col("idf").alias("idf_a"), 
                    )
             # cartesian join to self
             .crossJoin(tfidf
                        .select(col("productId").alias("productId_b"),
                                col("description").alias("description_b"),
                                col("imUrl").alias("imUrl_b"),
                                col("idf").alias("idf_b")))
             #remove first row of easch group of asin as this is self
             .filter("productId_a != productId_b")
             # cache for performance
             #.persist(StorageLevel.OFF_HEAP)
             .cache()
            )

compareDF.show(5)

# COMMAND ----------

import math

# calculate dot product 
dotProd = udf(lambda a, b: float(a.dot(b)), DoubleType())

# calculate norm
norm = udf(lambda v: float(math.sqrt(v.dot(v))), DoubleType())

# calculate cosine similarity
cosineSim = udf(lambda dot_product_ab, norm_a, norm_b: float(dot_product_ab / (norm_a * norm_b)), DoubleType())

similarityDF = (compareDF
                # cosine similarity calculation
                .withColumn('dot_product_ab', dotProd(col('idf_a'), col('idf_b')))
                .withColumn('norm_a', norm(col('idf_a')))
                .withColumn('norm_b', norm(col('idf_b')))
                .withColumn('similarity', cosineSim(col('dot_product_ab'), col('norm_a'), col('norm_b')))
                .cache()
               )

#print('total number of rows = %s' %similarityDF.count())
similarityDF.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluation
# MAGIC 
# MAGIC First, let's do a visual comparison of the covers of the top most similar products.  

# COMMAND ----------

window = (Window
          .partitionBy(['productId_a'])
          .orderBy(col('similarity').desc())
          )

similarProductsDF = (similarityDF
                     # reduce dataset size for faster processing
                     .filter(col('similarity')>0.8)
                     # for each product_a, keep only the row with the most similar product_b
                     .withColumn("rank", rank().over(window).alias('rank'))
                     .filter(col('rank')==1)
                     # rank across the whole dataset
                     .orderBy(col('similarity').desc())
                     .coalesce(1)
                     .withColumn('id', monotonically_increasing_id())
                     #.cache()
                     )

# similarProductsDF.show(5)

# COMMAND ----------

def visualize_similar_products(df, id):
    """
    Args:
        df (sparkdf): a dataframe with columns id, imurl_a, imurl_b
        id (integer): id of row
        imurl_a (string): image link of product a
        imurl_b (string): image link of product b

    Returns:
        Returns images of products in id row.
    """
    pictures = (df
                .filter(col('id')==id)
                .withColumn("html_a", concat(lit("<img style='height:300px;' src ='"), col("imUrl_a"), lit("'>")))
                .withColumn("html_b", concat(lit("<img style='height:300px;' src ='"), col("imUrl_b"), lit("'>")))
                .select("html_a", "html_b")
               )

    picture1 = pictures.select("html_a").collect()[0][0]
    picture2 = pictures.select("html_b").collect()[0][0]
    pictures_temp = picture1 + picture2

    return displayHTML(pictures_temp)

# COMMAND ----------

visualize_similar_products(similarProductsDF, 1)

# COMMAND ----------

visualize_similar_products(similarProductsDF, 2)

# COMMAND ----------

visualize_similar_products(similarProductsDF, 3)

# COMMAND ----------

visualize_similar_products(similarProductsDF, 4)

# COMMAND ----------

visualize_similar_products(similarProductsDF, 5)

# COMMAND ----------

# MAGIC %md 
# MAGIC As you can see our content reccommendation engine does a good job of finding similar products. In fact, in one sense, it does too good a job and finds duplicate products in the dataset, i.e. the same product with different productIds. To avoid this, one would need to use a deduplicated dataset or avoid reccommending products that are too similar (e.g. only reccommend products with cosine similarity <0.98).   
# MAGIC 
# MAGIC We'd like to evaluate our results at a macro level. For this, we will use the "related products" data we got from the meta dataset. By comparing the products our model says are similar to products that are often bought/viewed together (this will be our gold\_standard), we can see how well our model is doing. One thing to bear in mind however is that we do not know what the threshold was for determining is a set of products can be considered as "often bought together" "often viewed together." This implies that there may be more pairs of products that we could compare to if the threshold was set too high.  

# COMMAND ----------

gold_standard = (relatedProductsDF_newID
                 .select(col('productId').alias('productId_a'), 
                         col('related_productId').alias('productId_b')
                        )
                 .distinct()
                 # keep only entrepeneurship for product_a
                 .join(categoriesDF_newID.select(col('productId').alias('productId_a'), 
                                                 col('categories').alias('categories_a'))
                       , 'productId_a', 'left')
                 .filter(col('categories_a')==udf_category)
                 # keep only entrepeneurship for product_b
                 .join(categoriesDF_newID.select(col('productId').alias('productId_b'), 
                                                 col('categories').alias('categories_b'))
                       , 'productId_b', 'left')
                 .filter(col('categories_b')==udf_category)
                 # identify as true recommendation
                 .withColumn('Truth', lit(1))
                 # select final columns
                 .select('productId_a', 'productId_b', 'Truth')
                 .cache()
                )

# print(gold_standard.count())
gold_standard.show(5)

# COMMAND ----------

gold_standard_predictions = (compareDF
                             .select('productId_a', 'productId_b')
                             # add data on which which products are truly bought/viewed together
                             .join(gold_standard, ['productId_a', 'productId_b'], 'left')
                             .withColumn('Truth', 
                                         when(col('Truth').isNull(), 0)
                                         .otherwise(col('Truth')))
                             # add similarity data from VSM model 
                             .join(similarityDF.select('productId_a', 'productId_b', 'similarity')
                                   , ['productId_a', 'productId_b'], 'left')
                             # cache temporarily
                             .cache()
                            )

#print(gold_standard_predictions.count())
gold_standard_predictions.show(5)

# COMMAND ----------

# range over which to test
thresholds = np.arange(0, 1.05, 0.05)

# initialize empty dataframe
results = pd.DataFrame([])

for i in np.arange(0, len(thresholds)):
  
  threshold = thresholds[i]
  
  temp = (gold_standard_predictions
          .withColumn('Threshold', lit(threshold))
          .withColumn('Prediction', 
                      when(col('similarity')>threshold, 1)
                      .otherwise(0)
                     )
          .withColumn('Class', 
                      when((col('Truth')==1) & (col('Prediction')==1), 'TP')
                      .when((col('Truth')==0) & (col('Prediction')==1), 'FP')
                      .when((col('Truth')==0) & (col('Prediction')==0), 'TN')
                      .when((col('Truth')==1) & (col('Prediction')==0), 'FN')
                      .otherwise(None)
                     )
          .groupBy('Threshold', 'Class')
          .count()
         )
  
  results = results.append(temp.toPandas())
  
# gold_standard_predictions.unpersist()

results

# COMMAND ----------

# create some helper functions
def precision(row):
  # Precision = true-positives / (true-positives + false-positives)
  result = row['TP'] / (row['TP'] + row['FP']) 
  result = float(result)
  return result

def recall(row):
  # Recall = true-positives / (true-positives + false-negatives)
  result = row['TP'] / (row['TP'] + row['FN']) 
  result = float(result)
  return result

def f_measure(row):
  # F-measure = 2 x Recall x Precision / (Recall + Precision)
  result = (2 * row['recall'] * row['precision']) / (row['recall'] + row['precision'])
  result = float(result)
  return result

# COMMAND ----------

temp = (results
           .pivot(index='Threshold', columns='Class', values='count')
           .fillna(value = 0)
           .reset_index()
          )

# apply functions
temp['precision'] = temp.apply(lambda row: precision(row), axis=1)
temp['recall'] = temp.apply(lambda row: recall(row), axis=1)
temp['f_measure'] = temp.apply(lambda row: f_measure(row), axis=1)

# see results
temp

# COMMAND ----------

display(temp.plot.line(x = 'Threshold', y = ['precision', 'recall', 'f_measure']).figure)

# COMMAND ----------

# for presentation slides
# my_colors = [(0,0.27,0.67), (0.51,0.78,0.65), (0.93,0.81,0.10)]

# ax = temp.plot.line(x = 'Threshold', y = ['precision', 'recall', 'f_measure'], color = my_colors)

# ax.tick_params(
#   axis='x',          # changes apply to the x-axis
#   which='both',      # both major and minor ticks are affected
#   bottom='off',      # ticks along the bottom edge are off
#   top='off',         # ticks along the top edge are off
#   labelbottom='on')  # labels along the bottom edge are off

# ax.tick_params(
#   axis='y',          # changes apply to the x-axis
#   which='both',      # both major and minor ticks are affected
#   left='off',        # ticks along the bottom edge are off
#   right='off',       # ticks along the top edge are off
#   labelbottom='on')  # labels along the bottom edge are off

# ax.spines['bottom'].set_color('white')
# ax.spines['top'].set_color('white') 
# ax.spines['right'].set_color('white')
# ax.spines['left'].set_color('white')

# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')

# ax.yaxis.label.set_color('white')
# ax.xaxis.label.set_color('white')

# display(ax.figure)

# COMMAND ----------

# MAGIC %md 
# MAGIC While these results aren't very impressive, even this level of performance indicates that this model is clearly working (albeit at a low level). Moreover, you have to bear in mind that the fact that if a customer saw product A and didn't see product B, that  does not necessarily mean that the customer is not interested in product B.  
# MAGIC 
# MAGIC Leaving aside the caveats, we can't expect excellent performance from a basic model like this. There are many ways to boost the performance here, e.g. building a model that is able to understand semantics.   

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Conclusions
# MAGIC 
# MAGIC Another use-case for our VSM model, given the results we're seeing, is as a deduplication algorithm.    
# MAGIC 
# MAGIC Based on our precision, recall and f-score curves, another conclusion we can draw is that product similarity is not necessarily the best way to make reccommendations. This makes intuitive sense when you consider cases where a customer may be looking to use the reccommendation engine as a discovery tool or when a customer is looking for reccommendations for complementary products to (e.g. a plate to go with a set of knives). In both these cases, this model would fail to give the desired reccommendations.  
# MAGIC 
# MAGIC While the N^2 approach to finding similar items is certainly accurate, it is ultimately too time-consuming, volume-intensive, and hardware-reliant for any scalable purpose. In fact, we were never able to estimate the time it would take to run this on the full books dataset available to us as our moderately powered cluster could not complete the computations. In the next section, we look at an alternative approach which sacrifices a little accuracy for speed.   
# MAGIC 
# MAGIC Other approaches we considered included using other features to reduce the number of comparisons. For example, you could use pricing data to compare only products in the same price range. The downside here is you end up limiting upselling oppportunities. A better candidate variable might have been the Category variable which could be used to compare only products in the same category. Unfortunately, this condition is not restrictive enough to reduce the number of comparisons to a level our cluster could handle.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Content Based Reccommender (LSH approach)
# MAGIC 
# MAGIC There are a number of methods for addressing the disadvantages of our previous approach. One of the most promising methods is involves a combination of MinHash functions, Locality Sensitive Hashing (or LSH) and Nearest Neighbour Search, which we were inspired to try after learning of [Uber's success](https://databricks.com/blog/2017/05/09/detecting-abuse-scale-locality-sensitive-hashing-uber-engineering.html) using this approach to detect fraudulent activity at scale.  
# MAGIC 
# MAGIC #### Brief overview of Locality Sensitive Hashing 
# MAGIC 
# MAGIC The gist of it is that we use the LSH algorithm to "bucket" items together and then compare only items in the same bucket. In the example below, you would compare only the circle and the cross items as they are in the same "bucket." In this way, you are able to drastically reduce the number of operations as compared to the N^2 approach (above). This is what we were looking for, something that is highly scalable and extremely fast. 
# MAGIC 
# MAGIC ![Intro_lsh](https://lh4.googleusercontent.com/4PoFVvfWkW81lpTUS3gW9PNSRi3kA_Ax9ryqYim1Wg66x1XnnWIZc6iq32dqNTZwHUY7eoYzj4yZ3vE=w1366-h659
# MAGIC  "image description")
# MAGIC 
# MAGIC For more details on the technicalities behind this approach, see below.  
# MAGIC 
# MAGIC #### MinHash  
# MAGIC   
# MAGIC The MinHash function involves using a collection of hash functions, where we evaluate each word of a document by inputting it into the N hash functions, producing N hashed values. We then choose the minimum hash value which represents the signature for the word out of the N generated hash values. In this case, the minimum function is the function that samples the hashes and chooses the representation. The resulting set of hashes of words is comprised of the minimum hash computed for a word which was chosen from the N hash functions. The resulting vector of minimum hash values is what we call the MinHash signature of the document. We choose the minimum hash value by convention and for simplicity more than anything. We could just as well choose the maximum hash value as the sampling signature, the decision is arbitrary. Whatever we choose though, the hash values needs to be principled and consistent.  
# MAGIC 
# MAGIC #### Locality Senstive Hashing
# MAGIC 
# MAGIC Locality Sensitive Hashing is an algorithm which samples the result of the MinHash algorithm and compresses the MinHash signatures into LSH buckets. This serves to further reduce the size of the number of features that need to be compared to determine if documents are candidates for being similar. The idea behind LSH is that if documents are similar they should hash to approximately the same value. So, given some similarity threshold and N hash functions, sample the MinHash function in such a way that two documents are candidate pairs for similarity if and only if at least one of their LSH buckets are identical and share the same offset within the signature.  
# MAGIC 
# MAGIC Note that LSH allows us to quickly compare documents that are potential candidate matches. However, we could just use the MinHash signatures and compare those values to determine similarity. However, if we don’t use LSH to give us candidate pairs for matching, we would need to compare all MinHash signatures of our documents to all of the other documents that have been MinHashed, which would take order O(n2) number of operations, the same problem we had before. If we treat LSH values as buckets, then we can determine potential candidate pairs in order O(n) time by binning those LSH values that match together, and only if two documents have the same LSH bin would we further compute the MinHash similarity.   
# MAGIC 
# MAGIC Over the next few cells, we demonstrate an implementation of this methodology using Spark. We already have the basic dataset vectorizedDf ready so we can go straight from there.   

# COMMAND ----------

from pyspark.ml.feature import MinHashLSH

mhlsh = (MinHashLSH()
         .setNumHashTables(100)
         .setInputCol("tf")
         .setOutputCol("hashValues")
         .fit(vectorizedDf)
#          .persist(StorageLevel.OFF_HEAP)
        )

      
#mhlsh.transform(vectorizedDf).show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC approxSimilarityJoin

# COMMAND ----------

# threshold is based on Jaccard distance (not Jaccard similarity)
# the higher the Jaccard distance between two objects, the less similar they are
threshold = 0.9

similarityDF_lsh = (mhlsh
                    .approxSimilarityJoin(vectorizedDf, vectorizedDf, threshold)
                    .filter("distCol != 0")
                    .filter(col('datasetA.productId')!=col('datasetB.productId'))
                    .select(col('datasetA.productId').alias('productId_a')
                            , col('datasetB.productId').alias('productId_b')
                            , col('distCol').alias('similarity')
                           )
                    .persist(StorageLevel.OFF_HEAP)
                   )

# print(similarityDF_lsh.count())
# similarityDF_lsh.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Evaluation  
# MAGIC Let's start off by visualizing some of the most similar products.  

# COMMAND ----------

tempMetaData_a = (attributesDF_newID
                  .select(col('productId').alias('productId_a'), 
                          col('description').alias('description_a'),
                          col('imUrl').alias('imUrl_a')
                         )
                 )

tempMetaData_b = (attributesDF_newID
                  .select(col('productId').alias('productId_b'), 
                          col('description').alias('description_b'), 
                          col('imUrl').alias('imUrl_b')
                         )
                 )  


window = (Window
          .partitionBy(['productId_a'])
          .orderBy(col('similarity').asc())
          )

similarProductsDF_lsh = (similarityDF_lsh
                         .filter(col('similarity')>0.5)
                         .withColumn("rank", rank().over(window).alias('rank'))
                         .filter(col('rank')==1)
                         .orderBy(col('similarity').asc())
                         .coalesce(1)
                         .withColumn('id', monotonically_increasing_id())
                         # add description for productId_a
                         .join(tempMetaData_a, 
                               'productId_a', 
                               'left')
                         # add description for productId_b
                         .join(tempMetaData_b, 
                               'productId_b', 
                               'left')
                        )

similarProductsDF_lsh.show(5)

# COMMAND ----------

visualize_similar_products(similarProductsDF_lsh, 1)

# COMMAND ----------

visualize_similar_products(similarProductsDF_lsh, 2)

# COMMAND ----------

visualize_similar_products(similarProductsDF_lsh, 3)

# COMMAND ----------

visualize_similar_products(similarProductsDF_lsh, 4)

# COMMAND ----------

visualize_similar_products(similarProductsDF_lsh, 5)

# COMMAND ----------

# MAGIC %md 
# MAGIC As we saw with the N^2 approach, our model does too good a job and finds duplicate products in the dataset.  

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Perfomance comparison of N^2 and LSH 

# COMMAND ----------

# performance comparison in terms of running time for Entrepreneurhsip category



# COMMAND ----------

# MAGIC %md 
# MAGIC #### Content Based Demo

# COMMAND ----------

# create a table of products and description
products = (attributesDF_newID
            # add categories data
            .join(categoriesDF_newID, 'productID', 'left')
            # filter for only one category for illustrative purposes
            #.filter(col('categories')==udf_category)
            # combine titles and description
            .withColumn('product_description', 
                        when(col('title').isNotNull() & col('description').isNotNull(), concat(col("title"), col("description")))
                        .when(col('title').isNotNull() & col('description').isNull(), col("title"))
                        .when(col('title').isNull() & col('description').isNotNull(), col("description"))
                        .otherwise(lit(None))
                       )
            # only keep observations that have a product description 
            .filter(col('product_description').isNotNull())
            .filter(length(ltrim(col('product_description')))!=0)
            )

# COMMAND ----------

# MAGIC %md 
# MAGIC Tokenize the product\_description amd remove any stop words.  

# COMMAND ----------

# tokenize product description
tokenizer = (Tokenizer()
             .setInputCol("product_description")
             .setOutputCol("words")
            )

tokenizedDF = (tokenizer
               .transform(products)
              )
           
# remove stop words 
remover = (StopWordsRemover()
           .setInputCol("words")
           .setOutputCol("features")
          )

noStopWordsDF = (remover
                 .transform(tokenizedDF)
                )

# COMMAND ----------

# MAGIC %md
# MAGIC Calculate TF

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer, IDF

# Word count to vector for each wiki content
vocabSize = 1000000

cvModel = (CountVectorizer()
           .setInputCol("features")
           .setOutputCol("tf")
           .setMinDF(5)
           .setVocabSize(vocabSize)
           .fit(noStopWordsDF)
          )

# Function to return True/False depending on if a sparseVector is not all zero or not 
isNoneZeroVector = udf(lambda v: v.numNonzeros() > 0, BooleanType())

vectorizedDf_demo = (cvModel
                     .transform(noStopWordsDF)
                     # filter out any rows where the features sparse vector is completely zero
                     .filter(isNoneZeroVector(col("tf")))
                     # cache for performance
                     .cache()
                    )

# print('total number of rows = %s' %vectorizedDf.count())
# vectorizedDf.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Train model  

# COMMAND ----------

from pyspark.ml.feature import MinHashLSH

mhlsh = (MinHashLSH()
         .setNumHashTables(100)
         .setInputCol("tf")
         .setOutputCol("hashValues")
         .fit(vectorizedDf_demo)
        )

# COMMAND ----------

# MAGIC %md 
# MAGIC Select a random productId

# COMMAND ----------

# select a random productId 
# seed = 2 is another great example
udf_productId = vectorizedDf_demo.select('productId').rdd.takeSample(False, 1, seed = seed)[0][0]
udf_productId

# COMMAND ----------

udf_productId = 89203
# udf_productId = 168082

# COMMAND ----------

udf_product_info = (vectorizedDf_demo
                    .filter(col('productId')==udf_productId)
                   )

# udf_product_info.show()

columns = ['productId', 'title', 'description', 'price', 'imUrl']

for column in columns:
  print(column + ': ' + str(udf_product_info.select(column).collect()[0][0]))
  print('')

# COMMAND ----------

imUrl_list = udf_product_info.select('imUrl').collect()
links = [str(i.imUrl) for i in imUrl_list]

html =  [("<img style='height:300px;' src ='" + link + "'>") for link in links]

displayHTML(''.join(html))

# COMMAND ----------

# MAGIC %md
# MAGIC Perform a nearest neighbour search for that ProductId

# COMMAND ----------

key = (vectorizedDf_demo
       .filter(col('productId')==udf_productId)
       .select('tf')
       .collect()[0][0]
      )

k = 10

similar = mhlsh.approxNearestNeighbors(vectorizedDf_demo, key, k).orderBy(col('distCol'))

similar.select('productId', 'title', 'brand', 'description', 'price', 'distCol').show(k)

# COMMAND ----------

imUrl_list = (similar
              .filter(col('productId')!=udf_productId)
              .select('imUrl')
              .collect()
             )

links = [str(i.imUrl) for i in imUrl_list]

html =  [("<img style='height:300px;' src ='" + link + "'>") for link in links]

displayHTML(''.join(html))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Conclusion 
# MAGIC 
# MAGIC 1. We were successfully able to decrease the training time for our content based recommender model by applying LSH techinques.  
# MAGIC 2. Our evaluation results with the content based reccommenders showed clear signs that the model were making useful recommendations. While we do note that there is siginificant room for improvement, a content based approach can only go so far. Making reccommendations solely on the basis of similar products is not necessarily a wise strategy. Consumers often are not looking for similar products but rather products that are different but that they might still be interested in. To make these kinds of reccommendations, we have to use other methods. One such method is collaborative filtering, which we explore in the next section.  

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Collaborative Filtering with Alternating Least Squares
# MAGIC 
# MAGIC #### A brief introduction to Collaborative Filtering
# MAGIC 
# MAGIC One method of collaborative filtering works by looking for similar users.   
# MAGIC In the example below, you can see user A is very similar to user C in that they like very similar items. 
# MAGIC In this sense, it is highly likely that if we were recommend Item 5 to user A, they would probably like it. 
# MAGIC 
# MAGIC ![Amazon_home_page_mock](https://lh6.googleusercontent.com/vqrjEvrGtiqgUBSO72RrHPH4sucoR4JwRDirJtt97X_jziqU3q0x81pES19MfxGZ2t6Sec6pfeOtqec=w1366-h659
# MAGIC  "image description")
# MAGIC 
# MAGIC Next up we discuss matrix factorization which is pretty fundamental to building a collaborative filtering model.  
# MAGIC 
# MAGIC #### Matrix Factorization 
# MAGIC 
# MAGIC We can represent our data as a (very) large matrix A, say of dimensions m x n. where each row of A represents a user (m users in total), and each columns of A corresponds to an item (n items in total). Naturally, this matrix will be quite sparse (as we have shown previously) since most users will have only bought a very small fraction of all (n) items in the dataset. So we will have that Aij (i.e. the (i,j)th entry of the matrix) will depict the the rating the ith user gave the jth item.  
# MAGIC 
# MAGIC The goal of Matrix Factorization models is to approximate A with two smaller matrices, U (k x m) and V (k x n), each of which represent the rows and columns of A respectively (users and items). The vectors ui and vi in U and V respectively are called "latent factors," and k represents the number of features we believe associate each user to the item. Each entry in the row vectors of U and V therefore expresses how much association each has with k features.  
# MAGIC 
# MAGIC We then predict the “rating” of user U of item V to be:  
# MAGIC ri = uTi vj  
# MAGIC 
# MAGIC And so our aim to is to approximate (and complete) A as follows:  
# MAGIC A ≈ UT V  
# MAGIC 
# MAGIC We say that we are completing matrix A since we start with a sparse matrix A, but product the UTV will yield a dense, matrix.  
# MAGIC 
# MAGIC Our attention now switches to find the matrices U and V that best approximate A.  
# MAGIC 
# MAGIC #### Alternating Least Squares Algorithm  
# MAGIC 
# MAGIC One approach to finding U and V is ALS.  
# MAGIC 
# MAGIC We want to select these two matrices such that the error for the users/movie pairs where we know the correct ratings is minimized.  The Alternating Least Squares algorithm does this by first randomly filling the users matrix with values and then optimizing the value of the movies such that the error is minimized.  Then, it holds the items matrix constant and optimizes the value of the user's matrix.  This alternation between which matrix to optimize is the reason for the "alternating" in the name.  
# MAGIC 
# MAGIC Over the next few cells, we demonstrate an implementation of this methodology using Spark.  

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data preparation
# MAGIC Create training, validation and testing datasets.   

# COMMAND ----------

(split_60_df, split_a_20_df, split_b_20_df) = ratingsDF_complete.randomSplit([0.6, 0.2, 0.2], seed)

training_df = split_60_df.cache()
validation_df = split_a_20_df.cache()
test_df = split_b_20_df.cache()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Build model
# MAGIC We use the training_df and validation_df to build a cross-validated model.  
# MAGIC 
# MAGIC The CrossValidator is capable to running the ALS Algorithm using a range of Regularization Parameters and, Rank in order to provide an ALS model which yeild the lowest error (RMSE). Although we could certainly use the model provided by the CrossValidator, it would be difficult to rebuild a model with the same parameter without rerunning the CrossValidator. In order to be able to rebuild the model on-demand, we decided to search over pairs of Regularization Parameters and Ranks in order to find the Parameters that yeild the lowest error (RMSE). 

# COMMAND ----------

# # Let's initialize our ALS learner
# als = ALS()

# # Now we set the parameters for the method
# als.setMaxIter(10)\
#    .setSeed(seed)\
#    .setRegParam(0.001)\
#    .setParams(userCol="userId", itemCol="productId", ratingCol="rating")
  

# # Create an RMSE evaluator using the label and predicted columns
# reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="rmse")

# tolerance = 0.03
# ranks = [4, 8, 12]
# errors = [0, 0, 0]
# models = [0, 0, 0]
# err = 0
# min_error = float('inf')
# best_rank = -1

# for rank in ranks:
#   # Set the rank here:ratingsDf
#   als.setParams(rank=rank)
#   # Create the model with these parameters.
#   model = als.fit(training_df)
#   # Run the model to create a prediction. Predict against the validation_df.
#   predict_df = model.transform(validation_df)

#   # Remove NaN values from prediction (due to SPARK-14489)
#   predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))

#   # Run the previously created RMSE evaluator, reg_eval, on the predicted_ratings_df DataFrame
#   error = reg_eval.evaluate(predicted_ratings_df)
#   errors[err] = error
#   models[err] = model
#   print 'For rank %s the RMSE is %s' % (rank, error)
#   if error < min_error:
#     min_error = error
#     best_rank = err
#   err += 1

# als.setRank(ranks[best_rank])
# print 'The best model was trained with rank %s' % ranks[best_rank]
# my_model = models[best_rank]

# COMMAND ----------

# # We can reuse the RegressionEvaluator, regEval, to judge the model based on the best Root Mean Squared Error
# # Let's create our CrossValidator with 3 fold cross validation
# crossval = CrossValidator(estimator=als, evaluator=reg_eval, numFolds=3)

# # Let's tune over our regularization parameter from 0.01 to 0.10
# regParam = [0.1,0.01,0.001,0.0001]
# rank = [2, 4, 8, 12, 24]

# # We'll create a paramter grid using the ParamGridBuilder, and add the grid to the CrossValidator
# paramGrid = (ParamGridBuilder()
#              .addGrid(als.regParam, regParam)
#              .addGrid(als.rank, rank)
#              .build())
# crossval.setEstimatorParamMaps(paramGrid)

# # Now let's find and return the best model
# cvModel = crossval.fit(training_df).bestModel

# predict_df = cvModel.transform(validation_df)

# # Remove NaN values from prediction (due to SPARK-14489)
# predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))

# # Run the previously created RMSE evaluator, reg_eval, on the predicted_ratings_df DataFrame
# error = reg_eval.evaluate(predicted_ratings_df)

# print error

# COMMAND ----------

# predict_df.show(100)
# usersCount = rawRatingsDF.select("reviewerId").distinct().count()
# productId = rawRatingsDF.select("asin").distinct().count()

# print ratingsDF.select("userId").distinct().count()
# print usersCount
# print ratingsDF.select("productId").distinct().count()
# print productId
# print ratingsDF.select("rating").distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Choosing Hyperparameters using learning curves
# MAGIC 
# MAGIC Overfitting is the term used to describe the situation where a statistical model describes random error or noise instead of the underlying relationship being modelled. This usually results in poor real-life performance, as can be simulated on our testing datasets. To overcome this challenge, we used learning curves to gauge the optimal hyperparameters for the ALS algorithm that would give us a model that neither overfits nor underfits.  
# MAGIC 
# MAGIC Among other things, a learning curve basically allows you to find the point at which the algorithm starts to learn. We generated our learning curves using the following process:  
# MAGIC 1. Train ALS model on a 5% subset of the training dataset 
# MAGIC 2. Calculate RMSE on testing dataset 
# MAGIC 3. Calculate RMSE on subset of the training dataset 
# MAGIC 4. Repeat from step 1 except increase subset of the training dataset by 5%.  
# MAGIC 
# MAGIC The code for this process is commented out below due to long execution time. 

# COMMAND ----------

# # ParamGrid

# # Code commented out - long execution

# regParams = [0.8, 0.5, 0.3, 0.1, 0.05, 0.01]
# ranks = [8, 9, 10, 11, 12]
# ranges = [[0.05, 0.95], [0.1, 0.9], [0.15, 0.85], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7], [0.35, 0.65], [0.4, 0.6], [0.45, 0.55], [0.5, 0.5], [0.55, 0.45], [0.6, 0.4], [0.65, 0.35], [0.7, 0.3], [0.75, 0.25], [0.8, 0.2], [0.85, 0.15], [0.9, 0.1], [0.95, 0.05]]

# def randomSplitTrainTestVal(df, rng):
#   (trn, tst) = df.randomSplit(rng, seed)
#   train = trn.cache()
#   test = tst.cache()
#   return (train, test)
  
# def get_train_test_errors(model, reg_eval, train, test):
#   predict_df = model.transform(test)
#   predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))
#   test_error = reg_eval.evaluate(predicted_ratings_df)
#   train_predict_df = model.transform(train)
#   train_predicted_ratings_df = train_predict_df.filter(train_predict_df.prediction != float('nan'))
#   train_error = reg_eval.evaluate(train_predicted_ratings_df)
#   return (train_error, test_error)

# def get_learning_curve(df, ranges, rank, regParam):
#   (train, test) = randomSplitTrainTestVal(df, [0.8, 0.2])
#   errors = []
#   reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="rmse")
#   als = (ALS()
#            .setParams(userCol="userId", itemCol="productId", ratingCol="rating")
#            .setMaxIter(10)
#            .setSeed(seed)
#            .setRegParam(regParam)
#            .setParams(rank=rank))
#   for r in ranges:
#     # print "range: ", r
#     (train_subset, other) = randomSplitTrainTestVal(train, r)
#     model = als.fit(train_subset)
#     (train_error, test_error) = get_train_test_errors(model, reg_eval, train_subset, test)
#     # print "train error", train_error, "test error", test_error
#     errors.append((train_error, test_error))
#   # 100% training set
#   # print "range: ", [1.0, 0.0]
#   (train_error, test_error) = get_train_test_errors(model, reg_eval, train_subset, test)
#   # print "train error", train_error, "test error", test_error
#   errors.append((train_error, test_error))
#   return errors

# for regParam in regParams:
#   for rank in ranks:
#     print "STARTED  rank:", rank, "regParam:", regParam
#     lc = get_learning_curve(reviewsDF_newID, ranges, rank, regParam)
#     print "FINISHED rank", rank, "regParam", regParam
#     print "lc: ", lc
#     print "-----"

# COMMAND ----------

# MAGIC %md 
# MAGIC Based on the results of the previos code block, we chose the following values for our tunable hyperparameters:   
# MAGIC - regularization = 0.3
# MAGIC - rank = 8
# MAGIC 
# MAGIC We believe these values achieve the right balance of model complexity and performance.  
# MAGIC The following graphs show the performance of our model with these hyper-parameters on each of the different data sets.  

# COMMAND ----------

labels = []
x = []
for i,rng in enumerate(range(19)):
  labels.append(str(int((i+1) * 0.05 * 100)) + "%")
  x.append(i)
print labels
print x

reviews_lc = [(0.8640580121589105, 4.384441927740864), (0.9162809231699517, 3.690484300482888), (0.9384299226557588, 3.0612438047015575), (0.9466430715381681, 2.360932502568802), (0.9692083871156905, 1.9229308523437973), (0.9903098247046015, 1.7327298344878108), (1.010487906427418, 1.5416729229638737), (1.02603529745848, 1.4531459751437417), (1.0413454194101428, 1.378061221149991), (1.0525018106038844, 1.3369410801994481), (1.058597756535187, 1.326793759852507), (1.0712226906806182, 1.2877614927599652), (1.078651407565941, 1.27308252851925), (1.0878179342151915, 1.2561668513171094), (1.0921919895577556, 1.2476713431206847), (1.0970279362647135, 1.2421737543609286), (1.1020407600642035, 1.2333738897994269), (1.1046411136327223, 1.2302246933071448), (1.1094361605595275, 1.2251588329161107), (1.1094361605595278, 1.2251588329161107)]

plt.close()
figure = plt.subplot()
plt.plot(reviews_lc)
figure.set_ylim((0,5))
figure.set_xlim((-1,20))
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Learning Curve - Reviews Only')
plt.xticks(x, labels, rotation='vertical')
plt.grid(False)
test_legend = mpatches.Patch(color='green', label='Test')
training_legend = mpatches.Patch(color='blue', label='Training')
plt.legend(handles=[test_legend, training_legend])
plt.show()
display()


# COMMAND ----------

# # for presentation
# temp = pd.DataFrame([])

# for i in np.arange(0, len(reviews_lc)):
  
#   temp = temp.append(pd.DataFrame({'% of Training Data Used': (i+1)*5, 
#                                    'Training Error': reviews_lc[i][0],
#                                    'Testing Error': reviews_lc[i][1]
#                                   }
#                                   , index = [i]
#                                  )
#                     )

# temp 


# my_colors = [(1,1,1), (0.93,0.81,0.10), (0.51,0.78,0.65), (0,0.27,0.67)]

# ax = temp.plot.line(x = '% of Training Data Used', y = ['Training Error', 'Testing Error'], color = my_colors)

# ax.tick_params(
#   axis='x',          # changes apply to the x-axis
#   which='both',      # both major and minor ticks are affected
#   bottom='off',      # ticks along the bottom edge are off
#   top='off',         # ticks along the top edge are off
#   labelbottom='on')  # labels along the bottom edge are off

# ax.tick_params(
#   axis='y',          # changes apply to the x-axis
#   which='both',      # both major and minor ticks are affected
#   left='off',        # ticks along the bottom edge are off
#   right='off',       # ticks along the top edge are off
#   labelbottom='on')  # labels along the bottom edge are off

# ax.spines['bottom'].set_color('white')
# ax.spines['top'].set_color('white') 
# ax.spines['right'].set_color('white')
# ax.spines['left'].set_color('white')

# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')

# ax.yaxis.label.set_color('white')
# ax.xaxis.label.set_color('white')

# ax.legend(loc='best', fancybox=False, framealpha=0.2)

# ax.set_ylim((0,5))
# # figure.set_xlim((-1,20))


# display(ax.figure)

# COMMAND ----------

ratings_lc = [(0.8512420488592438, 4.402178452360909), (0.8985128906858578, 4.0079660176590926), (0.8946599135995017, 3.3939064738820286), (0.8985310810979403, 2.9684426207134504), (0.9089132610834753, 2.7348769794625443), (0.9194935511451728, 2.5877722971335286), (0.9287053424392893, 2.4157057608007646), (0.9366717124747126, 2.252721805815866), (0.9447478338776744, 2.116982703636849), (0.952174018424869, 2.05916611810784), (0.9594409377306761, 2.0338574307914956), (0.9661796957765452, 1.9545542418518507), (0.9738437348248564, 1.8432566433493862), (0.9788196624337752, 1.8132234200019504), (0.9847788245948345, 1.7718882306687063), (0.9892288532410219, 1.7468879562807988), (0.994331713485135, 1.7082346522186365), (0.9980104983233303, 1.7028764604276119), (1.0021646699077813, 1.6776681678401146), (1.0021646699077813, 1.6776681678401142)]

plt.close()
figure = plt.subplot()
plt.plot(ratings_lc)
figure.set_ylim((0,5))
figure.set_xlim((-1,20))
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Learning Curve - Ratings Only')
plt.xticks(x, labels, rotation='vertical')
plt.grid(False)
test_legend = mpatches.Patch(color='green', label='Test')
training_legend = mpatches.Patch(color='blue', label='Training')
plt.legend(handles=[test_legend, training_legend])
plt.show()
display()


# COMMAND ----------

# # for presentation
# temp = pd.DataFrame([])

# for i in np.arange(0, len(ratings_lc)):
  
#   temp = temp.append(pd.DataFrame({'% of Training Data Used': (i+1)*5, 
#                                    'Training Error': ratings_lc[i][0],
#                                    'Testing Error': ratings_lc[i][1]
#                                   }
#                                   , index = [i]
#                                  )
#                     )

# temp 


# my_colors = [(1,1,1), (0.93,0.81,0.10), (0.51,0.78,0.65), (0,0.27,0.67)]

# ax = temp.plot.line(x = '% of Training Data Used', y = ['Training Error', 'Testing Error'], color = my_colors)

# ax.tick_params(
#   axis='x',          # changes apply to the x-axis
#   which='both',      # both major and minor ticks are affected
#   bottom='off',      # ticks along the bottom edge are off
#   top='off',         # ticks along the top edge are off
#   labelbottom='on')  # labels along the bottom edge are off

# ax.tick_params(
#   axis='y',          # changes apply to the x-axis
#   which='both',      # both major and minor ticks are affected
#   left='off',        # ticks along the bottom edge are off
#   right='off',       # ticks along the top edge are off
#   labelbottom='on')  # labels along the bottom edge are off

# ax.spines['bottom'].set_color('white')
# ax.spines['top'].set_color('white') 
# ax.spines['right'].set_color('white')
# ax.spines['left'].set_color('white')

# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')

# ax.yaxis.label.set_color('white')
# ax.xaxis.label.set_color('white')

# ax.legend(loc='best', fancybox=False, framealpha=0.2)

# ax.set_ylim((0,5))
# # figure.set_xlim((-1,20))


# display(ax.figure)

# COMMAND ----------

complete_lc = [(0.851692687540396, 4.428562988682532), (0.8888863926069245, 3.880530106413204), (0.8874734804981644, 3.291523523610593), (0.8989707921588207, 2.9918748961421504), (0.9188548437487172, 2.9241459476195453), (0.918096453656919, 2.5138877808372357), (0.9275799356490655, 2.3889831310562513), (0.9360103065779821, 2.2529617285635153), (0.9455891898283856, 2.2464164794173667), (0.9520339232268076, 2.1076277324519577), (0.9596608474711091, 1.9912922342326205), (0.9662722305220206, 1.9273690722135204), (0.972105789686793, 1.8703246703666923), (0.9808310056738392, 1.7909899312840658), (0.9840512588991773, 1.7788630634497657), (0.988626393352775, 1.7679194184820883), (0.9928639045269848, 1.7391293758274775), (0.9981451674753056, 1.693672279285886), (1.0029811023548745, 1.658773456140052), (1.0029811023548745, 1.658773456140052)]

plt.close()
figure = plt.subplot()
plt.plot(complete_lc)
figure.set_ylim((0,5))
figure.set_xlim((-1,20))
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Learning Curve - Ratings and Reviews')
plt.xticks(x, labels, rotation='vertical')
plt.grid(False)
test_legend = mpatches.Patch(color='green', label='Test')
training_legend = mpatches.Patch(color='blue', label='Training')
plt.legend(handles=[test_legend, training_legend])
plt.show()
display()


# COMMAND ----------

# # for presentation
# temp = pd.DataFrame([])

# for i in np.arange(0, len(complete_lc)):
  
#   temp = temp.append(pd.DataFrame({'% of Training Data Used': (i+1)*5, 
#                                    'Training Error': complete_lc[i][0],
#                                    'Testing Error': complete_lc[i][1]
#                                   }
#                                   , index = [i]
#                                  )
#                     )

# temp 


# my_colors = [(1,1,1), (0.93,0.81,0.10), (0.51,0.78,0.65), (0,0.27,0.67)]

# ax = temp.plot.line(x = '% of Training Data Used', y = ['Training Error', 'Testing Error'], color = my_colors)

# ax.tick_params(
#   axis='x',          # changes apply to the x-axis
#   which='both',      # both major and minor ticks are affected
#   bottom='off',      # ticks along the bottom edge are off
#   top='off',         # ticks along the top edge are off
#   labelbottom='on')  # labels along the bottom edge are off

# ax.tick_params(
#   axis='y',          # changes apply to the x-axis
#   which='both',      # both major and minor ticks are affected
#   left='off',        # ticks along the bottom edge are off
#   right='off',       # ticks along the top edge are off
#   labelbottom='on')  # labels along the bottom edge are off

# ax.spines['bottom'].set_color('white')
# ax.spines['top'].set_color('white') 
# ax.spines['right'].set_color('white')
# ax.spines['left'].set_color('white')

# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')

# ax.yaxis.label.set_color('white')
# ax.xaxis.label.set_color('white')

# ax.legend(loc='best', fancybox=False, framealpha=0.2)

# ax.set_ylim((0,5))
# # figure.set_xlim((-1,20))


# display(ax.figure)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation
# MAGIC 
# MAGIC Despite the fact that we got better results on the reviews only dataset, we perform our evaluations on all the ratings (i.e. ratings in the ratings and reviews datasets). This allows us to make predictions for more users while trading off slightly on accuracy.  

# COMMAND ----------

(train, test) = ratingsDF_complete.drop("Timestamp").randomSplit([0.7, 0.3], seed)

# Initialize ALS algorithm
als = (ALS()
       .setSeed(seed)
       .setParams(userCol="userId", itemCol="productId", ratingCol="rating")
       .setMaxIter(10)
       .setRegParam(0.3)
       .setRank(8)
      )

# Create the model with these parameters.
model = als.fit(train)

# Run the model to create a prediction. Predict against the validation_df.
predict_df = model.transform(test)

# Remove NaN values from prediction (due to SPARK-14489)
predicted_ratings_df = (predict_df
                        .filter(predict_df.prediction != float('nan'))
                       )

predicted_ratings_df.limit(5).show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Home Page Recommendation
# MAGIC Recommends the top 20 items that the user might be interested, which the user has not yet reviewed or rated.

# COMMAND ----------

# Print Home page prediction for given user
my_user_id = 6425496

user_ratings = ratingsDF_complete.filter(col("userId") == my_user_id)
print "total user ratings", user_ratings.count()
all_ratings = ratingsDF_complete.filter(col("userId") != my_user_id)
(_train, test) = all_ratings.randomSplit([0.8, 0.2], seed)
train = _train.unionAll(user_ratings)

als = (ALS()
       .setParams(userCol="userId", itemCol="productId", ratingCol="rating", nonnegative=True)
       .setPredictionCol("prediction").setMaxIter(10).setSeed(seed).setRank(8).setRegParam(0.3))

model = als.fit(train)

predictions = model.transform(test).filter(col('prediction') != float('nan'))

my_rated_product_ids = [x[0] for x in user_ratings.select(user_ratings.productId).collect()]
not_rated_df = productConversionTableDF.filter(~ productConversionTableDF["newProductId"].isin(my_rated_product_ids))

my_unrated_products_df = not_rated_df.withColumn('userId', lit(my_user_id)).withColumnRenamed("newProductId", "productId")
new_predictions = model.transform(my_unrated_products_df).filter(col("prediction") != float('nan'))

product_predictions = new_predictions.sort(col("prediction").desc()).limit(20)

joined_df = (product_predictions.withColumnRenamed("asin", "a")
             .join(productConversionTableDF, productConversionTableDF.newProductId == product_predictions.productId)
             .drop('newProductId')
             .drop("a"))

joined_meta_df = joined_df.withColumnRenamed("asin", "a").join(attributesDF, attributesDF.asin == col("a")).drop("a")

images = [x[0] for x in joined_meta_df.select('imUrl').filter(col("imUrl").isNotNull()).collect()]
html =  [("<img style='height:300px;' src ='" + link + "'>") for link in images]
displayHTML(''.join(html))

# COMMAND ----------

# MAGIC %md 
# MAGIC Products rated highly by user for comparison

# COMMAND ----------

# user's top rated items
my_user_id = 6425496
reviewerID = userConversionTableDF.filter(col("newUserId") == my_user_id) .collect()[0][0]

users_top_50 = rawReviewsDF.filter(col("reviewerID") == reviewerID).select(col("asin").alias("a")).filter("overall >= 3.0") # .sort(col("overall").desc())
users_top_50_products = users_top_50.join(attributesDF, attributesDF.asin == users_top_50.a).drop('a').select("imUrl")

images = [x[0] for x in users_top_50_products.filter(col("imUrl").isNotNull()).collect()]
html =  [("<img style='height:300px;' src ='" + link + "'>") for link in images]

displayHTML(''.join(html))

# COMMAND ----------

# MAGIC %md 
# MAGIC Baseline - Popularity Recommender

# COMMAND ----------

my_user_id = 6425496
user_ratings = ratingsDF_complete.filter(col("userId") == my_user_id)
my_rated_product_ids = [x[0] for x in user_ratings.select(user_ratings.productId).collect()]
not_rated_df = productConversionTableDF.filter(~ productConversionTableDF["newProductId"].isin(my_rated_product_ids))

popular_items_df = (salesRankDF_newID
 .join(not_rated_df, productConversionTableDF.newProductId == salesRankDF_newID.productId)
 .drop("newProductId")
 .withColumnRenamed("asin", "a")
 .withColumnRenamed("salesRank", "sr")
 .join(rawMetasDF, rawMetasDF.asin == col("a"))
 .drop("a")
 .drop("salesRank")
 .withColumnRenamed("sr", "salesRank")
 .select(col("salesRank").cast(IntegerType()), "asin", "imUrl", "title")).filter(col("salesRank") != float('nan')).sort(col("salesRank")).limit(20)

# print popular_items_df.sort(col("salesRank").desc()).head(15)
# Popular Books: 
images = [x[0] for x in popular_items_df.select('imUrl').filter(col("imUrl").isNotNull()).collect()]
html =  [("<img style='height:300px;' src ='" + link + "'>") for link in images]
displayHTML(''.join(html))

# COMMAND ----------

# MAGIC %md 
# MAGIC Discussion
# MAGIC 
# MAGIC User example above - romance novel why are. Model works but we could improve it better. How about SGD? 
# MAGIC The examples above shows the performance of the ALS Algorithm when compared to the Popularity recommender as a baseline. The first set of pictures shows the products predicted by the ALS Algorithm that might be of interest to the selected user. The second set of pictures show the products rated highly by this user. And finally, the last set of pictures show the top rated products across all products in Amazon. At a glance, we can see that there is some correlation between the ALS predictions and the products rated highly by the user. There is less correlation between the products rated highly by the user, and the popular products.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### RMSE

# COMMAND ----------

# Create an RMSE evaluator using the label and predicted columns
reg_eval = (RegressionEvaluator()
            .setPredictionCol("prediction")
            .setLabelCol("rating")
            .setMetricName("rmse")
           )

# Run the previously created RMSE evaluator, reg_eval, on the predicted_ratings_df DataFrame
error = reg_eval.evaluate(predicted_ratings_df)

print(error)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Precision - Recall 

# COMMAND ----------

# range over which to test
thresholds = np.arange(0, 5.5, 0.1)

# initialize empty dataframe
results = pd.DataFrame([])

for i in np.arange(0, len(thresholds)):
  
  threshold = thresholds[i]
  
  temp = (predicted_ratings_df
          .withColumn('Truth', 
                      when(col('rating')>=4, 1)
                      .otherwise(0)
                     )
          .withColumn('Threshold', lit(threshold))
          .withColumn('Prediction', 
                      when(col('prediction')>=threshold, 1)
                      .otherwise(0)
                     )
          .withColumn('Class', 
                      when((col('Truth')==1) & (col('Prediction')==1), 'TP')
                      .when((col('Truth')==0) & (col('Prediction')==1), 'FP')
                      .when((col('Truth')==0) & (col('Prediction')==0), 'TN')
                      .when((col('Truth')==1) & (col('Prediction')==0), 'FN')
                      .otherwise(None)
                     )
          .groupBy('Threshold', 'Class')
          .count()
         )
  
  results = results.append(temp.toPandas())
  
results.head()

# COMMAND ----------

# create some helper functions
def precision(row):
  # Precision = true-positives / (true-positives + false-positives)
  result = row['TP'] / (row['TP'] + row['FP']) 
  result = float(result)
  return result

def recall(row):
  # Recall = true-positives / (true-positives + false-negatives)
  result = row['TP'] / (row['TP'] + row['FN']) 
  result = float(result)
  return result

def f_measure(row):
  # F-measure = 2 x Recall x Precision / (Recall + Precision)
  result = (2 * row['recall'] * row['precision']) / (row['recall'] + row['precision'])
  result = float(result)
  return result

# COMMAND ----------

temp = (results
           .pivot(index='Threshold', columns='Class', values='count')
           .fillna(value = 0)
           .reset_index()
          )

# apply functions
temp['precision'] = temp.apply(lambda row: precision(row), axis=1)
temp['recall'] = temp.apply(lambda row: recall(row), axis=1)
temp['f_measure'] = temp.apply(lambda row: f_measure(row), axis=1)

# see results
temp.head()

# COMMAND ----------

display(temp.plot.line(x = 'Threshold', y = ['precision', 'recall', 'f_measure']).figure)

# COMMAND ----------

# # for presentation slides
# my_colors = [(0,0.27,0.67), (0.51,0.78,0.65), (0.93,0.81,0.10)]

# ax = temp.plot.line(x = 'Threshold', y = ['precision', 'recall', 'f_measure'], color = my_colors)

# ax.tick_params(
#   axis='x',          # changes apply to the x-axis
#   which='both',      # both major and minor ticks are affected
#   bottom='off',      # ticks along the bottom edge are off
#   top='off',         # ticks along the top edge are off
#   labelbottom='on')  # labels along the bottom edge are off

# ax.tick_params(
#   axis='y',          # changes apply to the x-axis
#   which='both',      # both major and minor ticks are affected
#   left='off',        # ticks along the bottom edge are off
#   right='off',       # ticks along the top edge are off
#   labelbottom='on')  # labels along the bottom edge are off

# ax.spines['bottom'].set_color('white')
# ax.spines['top'].set_color('white') 
# ax.spines['right'].set_color('white')
# ax.spines['left'].set_color('white')

# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')

# ax.yaxis.label.set_color('white')
# ax.xaxis.label.set_color('white')

# display(ax.figure)

# COMMAND ----------

# MAGIC %md 
# MAGIC old code below

# COMMAND ----------

# # range over which to test
# thresholds = np.arange(0, 5.5, 0.1)

# # initialize empty dataframe
# results = pd.DataFrame([])

# for i in np.arange(0, len(thresholds)):
  
#   threshold = thresholds[i]
  
#   temp = (predicted_ratings_df
#           .withColumn('Truth', 
#                       when(col('rating')>=4, 1)
#                       .otherwise(0)
#                      )
#           .withColumn('Reccommendation', 
#                       when(col('prediction')>=threshold, 1)
#                       .otherwise(0)
#                      )
#          )
  
#   tp = (temp
#         .filter(col('Truth')==1)
#         .filter(col('Reccommendation')==1)
#         .count()
#        )
  
#   fp = (temp
#         .filter(col('Truth')==0)
#         .filter(col('Reccommendation')==1)
#         .count()
#        )
  
#   tn = (temp
#         .filter(col('Truth')==0)
#         .filter(col('Reccommendation')==0)
#         .count()
#        )
  
#   fn = (temp
#         .filter(col('Truth')==1)
#         .filter(col('Reccommendation')==0)
#         .count()
#        )
  
#   results = results.append(pd.DataFrame({'threshold': threshold, 
#                                          'tp': tp,
#                                          'fp': fp,
#                                          'tn': tn,
#                                          'fn': fn
#                                         }
#                                         , index = [i]
#                                        )
#                           )
# results

# # apply functions
# results['precision'] = results.apply(lambda row: precision(row), axis=1)
# results['recall'] = results.apply(lambda row: recall(row), axis=1)
# results['f_measure'] = results.apply(lambda row: f_measure(row), axis=1)

# # see results
# results

# COMMAND ----------

# display(results.plot.line(x = 'threshold', y = ['precision', 'recall', 'f_measure']).figure)

# COMMAND ----------

# my_colors = [(0,0.27,0.67), (0.51,0.78,0.65), (0.93,0.81,0.10)]

# ax = results.plot.line(x = 'threshold', y = ['precision', 'recall', 'f_measure'], color = my_colors)

# ax.tick_params(
#   axis='x',          # changes apply to the x-axis
#   which='both',      # both major and minor ticks are affected
#   bottom='off',      # ticks along the bottom edge are off
#   top='off',         # ticks along the top edge are off
#   labelbottom='on')  # labels along the bottom edge are off

# ax.tick_params(
#   axis='y',          # changes apply to the x-axis
#   which='both',      # both major and minor ticks are affected
#   left='off',        # ticks along the bottom edge are off
#   right='off',       # ticks along the top edge are off
#   labelbottom='on')  # labels along the bottom edge are off

# ax.spines['bottom'].set_color('white')
# ax.spines['top'].set_color('white') 
# ax.spines['right'].set_color('white')
# ax.spines['left'].set_color('white')

# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')

# ax.yaxis.label.set_color('white')
# ax.xaxis.label.set_color('white')

# display(ax.figure)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Average Execution Times - rank v regParam

# COMMAND ----------

# commented large execution

# # execution times
# import time
# # ParamGrid
# regParams = [0.8, 0.5, 0.3, 0.1, 0.05, 0.01]
# ranks = [8, 9, 10, 11, 12]
# ranges = [[0.05, 0.95], [0.1, 0.9], [0.15, 0.85], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7], [0.35, 0.65], [0.4, 0.6], [0.45, 0.55], [0.5, 0.5], [0.55, 0.45], [0.6, 0.4], [0.65, 0.35], [0.7, 0.3], [0.75, 0.25], [0.8, 0.2], [0.85, 0.15], [0.9, 0.1], [0.95, 0.05]]

# results = []

# def predict(df, rank, regParam):
#   (train, test) = randomSplitTrainTestVal(df, [0.8, 0.2])
#   errors = []
#   reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="rmse")
#   als = (ALS()
#            .setParams(userCol="userId", itemCol="productId", ratingCol="rating")
#            .setMaxIter(10)
#            .setSeed(seed)
#            .setRegParam(regParam)
#            .setParams(rank=rank))
#   model = als.fit(train)
#   predict_df = model.transform(test)
#   predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))
#   test_error = reg_eval.evaluate(predicted_ratings_df)
#   return test_error

# for regParam in regParams:
#   for rank in ranks:
#     durations = []
#     for i in range(5):
#       start = time.time()
#       error = predict(reviewsDF_newID, rank, regParam)
#       end = time.time()
#       duration = end-start
#       durations.append(duration)
#     avgDuration = reduce(lambda x, y: x + y, durations) / len(durations)
#     result = dict(error = error, rank= rank, regParam= regParam, avgDuration=avgDuration, durations=durations)
#     print "result: ", result
#     results.append(result)
    
# print results

# COMMAND ----------

avgDurations = [{'regParam': 0.8, 'durations': [32.629246950149536, 33.00645589828491, 33.302823066711426, 33.27130699157715, 31.821557998657227], 'avgDuration': 32.80627818107605, 'rank': 8, 'error': 1.2203321705754255}, {'regParam': 0.8, 'durations': [34.6021089553833, 34.39513087272644, 33.51827096939087, 34.31026792526245, 34.902360916137695], 'avgDuration': 34.34562792778015, 'rank': 9, 'error': 1.221807579947982}, {'regParam': 0.8, 'durations': [34.56409692764282, 34.54246711730957, 34.77338194847107, 34.296891927719116, 36.073237895965576], 'avgDuration': 34.85001516342163, 'rank': 10, 'error': 1.2205288743565545}, {'regParam': 0.8, 'durations': [36.83195900917053, 36.01732587814331, 37.4742169380188, 35.665250062942505, 36.67460894584656], 'avgDuration': 36.53267216682434, 'rank': 11, 'error': 1.2183948385501917}, {'regParam': 0.8, 'durations': [37.799954891204834, 37.50326204299927, 38.34353280067444, 37.660502910614014, 39.029202938079834], 'avgDuration': 38.06729111671448, 'rank': 12, 'error': 1.2191598280939822}, {'regParam': 0.5, 'durations': [34.12103009223938, 33.55432200431824, 32.75436210632324, 34.23967003822327, 34.39057397842407], 'avgDuration': 33.81199164390564, 'rank': 8, 'error': 1.0615770262069468}, {'regParam': 0.5, 'durations': [35.5602240562439, 34.377705812454224, 33.606611013412476, 34.52962803840637, 32.89467096328735], 'avgDuration': 34.19376797676087, 'rank': 9, 'error': 1.0626657776889408}, {'regParam': 0.5, 'durations': [50.966663122177124, 35.4899218082428, 35.09305000305176, 34.50436210632324, 34.69225597381592], 'avgDuration': 38.14925060272217, 'rank': 10, 'error': 1.0615910235969408}, {'regParam': 0.5, 'durations': [36.90130400657654, 35.52209711074829, 46.03167796134949, 48.22877597808838, 34.691269874572754], 'avgDuration': 40.27502498626709, 'rank': 11, 'error': 1.0599071663952697}, {'regParam': 0.5, 'durations': [37.30120491981506, 37.28495216369629, 39.19371509552002, 36.720272064208984, 36.192981004714966], 'avgDuration': 37.338625049591066, 'rank': 12, 'error': 1.060727002395333}, {'regParam': 0.3, 'durations': [33.8825249671936, 34.29401898384094, 32.0565299987793, 32.917362213134766, 33.20589780807495], 'avgDuration': 33.27126679420471, 'rank': 8, 'error': 1.0126312526258054}, {'regParam': 0.3, 'durations': [34.555712938308716, 33.02296710014343, 33.36643314361572, 34.69293689727783, 36.80426216125488], 'avgDuration': 34.48846244812012, 'rank': 9, 'error': 1.0136808730900195}, {'regParam': 0.3, 'durations': [97.66613578796387, 104.21938014030457, 47.73914408683777, 34.20307993888855, 98.09312295913696], 'avgDuration': 76.38417258262635, 'rank': 10, 'error': 1.0118410472686057}, {'regParam': 0.3, 'durations': [38.85944104194641, 37.06703209877014, 36.569177865982056, 36.079416036605835, 37.96514916419983], 'avgDuration': 37.308043241500854, 'rank': 11, 'error': 1.0094529827509475}, {'regParam': 0.3, 'durations': [35.37493085861206, 35.529764890670776, 36.0109338760376, 37.24548006057739, 36.71477699279785], 'avgDuration': 36.175177335739136, 'rank': 12, 'error': 1.0106059167022554}, {'regParam': 0.1, 'durations': [31.943068981170654, 34.17848610877991, 34.07354187965393, 33.39654994010925, 47.8437020778656], 'avgDuration': 36.28706979751587, 'rank': 8, 'error': 1.146320530504729}, {'regParam': 0.1, 'durations': [33.1464569568634, 33.61918807029724, 32.40067219734192, 33.44556212425232, 33.31796979904175], 'avgDuration': 33.185969829559326, 'rank': 9, 'error': 1.128135892512113}, {'regParam': 0.1, 'durations': [33.00710988044739, 34.71892714500427, 33.539387941360474, 34.04915499687195, 35.79251289367676], 'avgDuration': 34.22141857147217, 'rank': 10, 'error': 1.13294058111402}, {'regParam': 0.1, 'durations': [36.42662215232849, 36.001179933547974, 37.901947021484375, 37.20575499534607, 36.32979607582092], 'avgDuration': 36.77306003570557, 'rank': 11, 'error': 1.1080604641972822}, {'regParam': 0.1, 'durations': [37.348905086517334, 36.93867111206055, 36.17887616157532, 38.64650201797485, 38.60742712020874], 'avgDuration': 37.54407629966736, 'rank': 12, 'error': 1.107211933370861}, {'regParam': 0.05, 'durations': [33.562127113342285, 32.788208961486816, 34.555376052856445, 33.90340304374695, 35.40863108634949], 'avgDuration': 34.0435492515564, 'rank': 8, 'error': 1.5562782194589175}, {'regParam': 0.05, 'durations': [35.81029391288757, 43.64779710769653, 140.89167308807373, 56.54567503929138, 38.90817594528198], 'avgDuration': 63.16072301864624, 'rank': 9, 'error': 1.4549486760825026}, {'regParam': 0.05, 'durations': [40.87470602989197, 180.16276597976685, 104.89376306533813, 51.55047011375427, 55.352210998535156], 'avgDuration': 86.56678323745727, 'rank': 10, 'error': 1.4944022100655403}, {'regParam': 0.05, 'durations': [41.60296988487244, 45.95779204368591, 46.89685392379761, 46.132251024246216, 59.31430697441101], 'avgDuration': 47.980834770202634, 'rank': 11, 'error': 1.390626220667131}, {'regParam': 0.05, 'durations': [52.52171206474304, 41.30989909172058, 70.61736178398132, 67.79461908340454, 77.2807970046997], 'avgDuration': 61.90487780570984, 'rank': 12, 'error': 1.3866168389127873}, {'regParam': 0.01, 'durations': [43.14728879928589, 42.91088390350342, 43.59098196029663, 69.92471694946289, 75.35859203338623], 'avgDuration': 54.98649272918701, 'rank': 8, 'error': 3.653058013032669}, {'regParam': 0.01, 'durations': [37.39614796638489, 48.56079602241516, 41.564960956573486, 43.18789005279541, 47.08396506309509], 'avgDuration': 43.55875201225281, 'rank': 9, 'error': 3.3986463877874393}, {'regParam': 0.01, 'durations': [48.68684792518616, 44.58919405937195, 63.126070976257324, 34.27534794807434, 35.29286599159241], 'avgDuration': 45.19406538009643, 'rank': 10, 'error': 3.447593235500553}, {'regParam': 0.01, 'durations': [36.70055413246155, 36.300585985183716, 35.778634786605835, 138.66716885566711, 37.15303707122803], 'avgDuration': 56.91999616622925, 'rank': 11, 'error': 3.093274271413025}, {'regParam': 0.01, 'durations': [37.25367498397827, 44.96364998817444, 94.21205997467041, 110.95095205307007, 51.0137197971344], 'avgDuration': 67.67881135940551, 'rank': 12, 'error': 3.1342055148356445}]

avgDurationsDf = sc.parallelize(avgDurations).toDF()
avgDurationsSortedDf = (avgDurationsDf
                        .sort(avgDurationsDf.avgDuration)
                        .withColumn("avg(s)", round(col("avgDuration"), 2))
                        .withColumn("rmse", round(col("error"), 3))
                        .select("avg(s)", "rmse", "rank", "regParam"))
avgDurationsSortedDf.show()

# COMMAND ----------

display(avgDurationsSortedDf)

# COMMAND ----------

display(avgDurationsSortedDf)

# COMMAND ----------

# MAGIC %md 
# MAGIC The above two graphs show how the amount of time taken to train our model using different combinations hyper parameters. There is a clear trade-off between model training time and accuracy. Based on our best "business" judgement we chose to set our regularization parameter to 0.3 and rank as 8.  

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Conclusions
# MAGIC 
# MAGIC We were able to show both anecdotally and mathematically the impressive performance of our ALS model. 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Collaborative Filtering with Stratified Stochastic Gradient Descent  
# MAGIC 
# MAGIC For a fully online or real-time model, we explored SSGD as an alternative method for deriving U and V for our matrix factorization model. In his [paper](http://cs229.stanford.edu/proj2014/Christopher%20Aberger,%20Recommender.pdf), Christopher R. Aberger show that stochastic gradient descent is generally faster and more accurate than ALS except in situations of sparse data in which ALS tends to performs better. Motivated by this research, we decided to try our hand at this model. The following is a brief summary of the method.     
# MAGIC 
# MAGIC #### SSGD
# MAGIC 
# MAGIC The Stratified Stochastic Gradient Descent is a particular implementation of the Stochastic Gradient Descent made for distributed environments and it is particularly suitable for map/reduce implementations. The stratification consists on dividing the ratings matrix and the two matrices H and W produced by the factorization algorithm into blocks. The blocks will form a new matrix and the block is the unit of computation, meaning that the number of blocks should be higher than the number of cores in your cluster. At the same time, the algorithm will perform better with fewer blocks, which means we need to find the smallest value of blocks that optimizes the computation between error and speed. My implementation uses a parallelization rank. I calculated the rank doing the ceiling of the square root of the number of cores of the cluster. Each stratus is composed by diagonal blocks to make sure that the same block of H and W is not present in two blocks inside the same stratus. A stratus has access to a complete H and a complete W during every iteration. At the stratus level, we generally compute only the regularization term which requires access to the full H and W. However, this implementation doesn't include the regularization parameter because I didn't have enough time to get to the end of the implementation. Each stratus is merged at the end through a weighted sum of the H and W calculated. However, most of our dataset contains few ratings per user and they should be well distributed in all blocks because of the way how we preprocessed our data. We decided to skip the implementation of weighting and we used a simple average, filtering the entries of H and W in the blocks that do not contain any rating for the user/product.
# MAGIC 
# MAGIC This is just a proof of concept. A production implementation would require all those aspects to be covered. We don't expect good results from this model given the variability of the data. However, it is still interesting to see how it looks like an algorithm that could potentially learn online.
# MAGIC 
# MAGIC Due to the lack of any existing implementation of SGD in Spark, we had to write the full implementation by hand as follows.   

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
    def __init__(self, maxIter=10, parallelizationRank=12, nFactors=8, sigma=0.1, lambda_input=0.3):
        self.iter = maxIter
        self.nExecutors = parallelizationRank
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

(split_70_df, split_b_30_df) = ratingsDF_complete.randomSplit([0.7, 0.3], seed)

training_df = split_70_df.cache()
test_df = split_b_30_df.cache()
ssgd = SSGD()

(productFeatures,userFeatures) = ssgd.trainSSGD(training_df)

# COMMAND ----------

#loading model previously generated and saved in parquet files
(split_70_df, split_b_30_df) = ratingsDF_complete.randomSplit([0.7, 0.3], seed)

training_df = split_70_df.cache()
test_df = split_b_30_df.cache()

from pyspark.mllib.linalg import Vectors
productFeaturesDF = sqlContext.read.parquet("/mnt/%s/parquetDataset/results-productFeatures-full.parquet" % MOUNT_NAME).coalesce(128)
userFeaturesDF = sqlContext.read.parquet("/mnt/%s/parquetDataset/results-userFeatures-full.parquet" % MOUNT_NAME).coalesce(128)
productFeatures = productFeaturesDF.rdd.map(lambda x: (x[0], Vectors.dense([x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]])))
userFeatures = userFeaturesDF.rdd.map(lambda x: (x[0], Vectors.dense([x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]])))


# COMMAND ----------

print productFeatures.take(1)
print userFeatures.take(1)
userFeaturesDF = sqlContext.createDataFrame(userFeatures)
display(userFeaturesDF)

# COMMAND ----------

def predict(entry):
  """
  Inputs: 
    
    user (integer): userId
    
    item (integer): itemId
    
    userFactors (np.array): Features array
                              
    itemFactors (np.array): Features array
    
  Outputs: 
    
    user_item_rating (float): Predicted rating for item by user
    
    
  """
  (item, user, itemFactors, userFactors) = entry
  # extract features for user
  user_features = np.asarray(userFactors, dtype=np.float64
                            )
  
  # extract features for item
  item_features = np.asarray(itemFactors, dtype=np.float64
                            ) 
  if (np.count_nonzero(~np.isnan(user_features)) == 0 or np.count_nonzero(~np.isnan(item_features)) == 0):
    #in case I have no data to make the prediction I will choose an equidistant number to minimize the error
    return (user, item, float(3))
  
  # dot product of user_features and item_features
  user_item_rating = user_features.dot(item_features)
  
  return (user, item, float(user_item_rating))

# COMMAND ----------

#generate predictions and prepare dataset for evaluation
userFeaturesDF = sqlContext.createDataFrame(userFeatures).select(col("_1").alias("userId"), col("_2").alias("userFeatures"))
productFeaturesDF = sqlContext.createDataFrame(productFeatures).select(col("_1").alias("productId"), col("_2").alias("productFeatures"))
test_prediction_df = test_df.join(userFeaturesDF, test_df.userId == userFeaturesDF.userId, "left").join(productFeaturesDF, test_df.productId == productFeaturesDF.productId, "left").select(test_df.productId,test_df.userId,col("productFeatures"),col("userFeatures")).rdd.map(lambda x: (x[0], x[1], x[2], x[3])).map(predict).toDF().select(col("_1").alias("productId"), col("_2").alias("userId"),col("_3").alias("prediction"))

test_complete_df = test_prediction_df.join(test_df, (test_prediction_df.userId == test_df.userId) & (test_prediction_df.productId == test_df.productId)).select(test_df.userId, test_df.productId, test_df.rating, test_prediction_df.prediction)

# COMMAND ----------

display(test_prediction_df)

# COMMAND ----------

#rmse evaluation
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="rmse")
error = reg_eval.evaluate(test_complete_df)
print error

# COMMAND ----------

# MAGIC %md
# MAGIC ###Comparison with ALS

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
model = als.fit(training_df)
ALSpredictionsDF = model.transform(test_df)


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Final Conclusions 
# MAGIC 
# MAGIC While recommending products using the Book ratings and reviews datasets, we found that providing recommendations based on the reviews dataset alone yields a lower RMSE compared to the ratings dataset alone, or the rating+reviews dataset. This is likely due to written reviews leading to a more deliberate consideration of the product's rating. However, we chose to combine the two datasets to maximise the data points available for the algorithm since the two datasets are already sparse. 
# MAGIC 
# MAGIC Collaborative Filtering and Content Based Recommendations are the two pieces of our recommendation system. Collaborative filtering is a great tool for discovering new products that the user would otherwise be unaware of. This is because it considers the product ratings provided by the users in conjunction with the product ratings provided by other similar users of the platform.  
# MAGIC 
# MAGIC The Content Based Recommendation system, unlike the collaborative model, which considers users' ratings for products, Content Based filtering starts at the products and recommends similar products based on the description and common features of the products. So while the Collaborative recommendation works by taking the user's ratings and products as a whole, content based recommendation works by starting from the product selected by the user. In other words, Content Based Filtering is the second piece that completes our recommendation system. 
# MAGIC 
# MAGIC ![Amazon_home_page_mock](https://lh3.googleusercontent.com/mPRNHoFv0Fn8TpvSMO2tig-4y1hLqCbjzF_BR-AP1XnClt5it43aX38JhxrnI12x32wOzluW_r2kR8c=w1366-h659
# MAGIC  "image description")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Further areas of improvement 
# MAGIC 
# MAGIC - Rewrite in scala to improve performances
# MAGIC - Improve the implementation of algorithms 
# MAGIC - Implement Spark Streaming for online prediction and training
# MAGIC - Add autotests to make it more robust
# MAGIC - Tune up the models
# MAGIC - Semantic understanding for Content Based
# MAGIC - ngrams for Content Based
# MAGIC - Graph Network model for related products
# MAGIC 
# MAGIC #### Future areas for research
# MAGIC 
# MAGIC Due to time and resource constraints, we were unable to explore all possible areas of research. Some of our ideas are documented here and we encourage others to take these forward. 
# MAGIC 
# MAGIC #### Graph Network model
# MAGIC 
# MAGIC We were provided data on how different products are bought/viewed together. This information naturally lends itself to graph network based approaches to building a basic reccommendations system. A natural advantage of this approach is better reccommendations for complementary products.      
# MAGIC 
# MAGIC #### Visual features model
# MAGIC 
# MAGIC We were also provided with data pertaining to the visual features of each product. We believe it is possible to use these as features for finding similar products. Our hypothesis is that despite the old adagae, consumers will often judge a book by its cover and that therefore there may be predictive information to be mined from visual features for making reccommendations.       
# MAGIC 
# MAGIC #### User Bias
# MAGIC 
# MAGIC Most recommender systems perform better if user and item biases are taken into account. Suppose we have a ratings system that allows each user to rate each item on a scale of 1 to 5 stars. Suppose we have two users: Alice, who rates items with an average of 4 stars, and Bob, whose average rating is 1.5 stars. If Bob rates some new item with 3 stars, it means something very different than if Alice rates the same item with 3 stars (Bob really liked the new item, Alice didn't). The difference is what we call user bias. Modelling for this bias can naturally improve recommendations.  
# MAGIC 
# MAGIC #### Ensemble model
# MAGIC 
# MAGIC Its likely that each of these models would do well in certain areas and worse in others. Since most of the models suggested are sufficiently uncorrelated (as they all use different data), an ensemble approach would be a valid modelling technique and likely would achieve better results.   

# COMMAND ----------

