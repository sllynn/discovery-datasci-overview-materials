# Databricks notebook source
# MAGIC %md  
# MAGIC # DSML Overview session
# MAGIC 
# MAGIC ### Replicating a traditional ML workflow in Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC # Reading data into Spark
# MAGIC 
# MAGIC 1. Using Spark's native file readers for csv, json, parquet and Delta
# MAGIC 2. Different paths to accessing a dataset

# COMMAND ----------

# DBTITLE 1,Read from .csv file (schema unknown)
display(spark.read.csv("/databricks-datasets/samples/population-vs-price/data_geo.csv", header=True, inferSchema=True))

# COMMAND ----------

# DBTITLE 1,Read from .csv file (known schema)
display(
  spark.read
  .schema("`2014 rank` int,`City` string,`State` string,`State Code` string,`2014 Population estimate` long,`2015 median sales price` float")
  .csv("/databricks-datasets/samples/population-vs-price/data_geo.csv", header=True)
)

# COMMAND ----------

# MAGIC %fs head /stuart/demo/samples/people/people.json

# COMMAND ----------

# DBTITLE 1,Read a .json file (row per observation)
display(
  spark.read
  .json("/stuart/demo/samples/people/", multiLine=True)
)

# COMMAND ----------

# DBTITLE 1,Read delta table directly from Azure Data Lake Storage
display(
  spark.read
  .format("delta")
  .load("abfss://covid@shareddatalake.dfs.core.windows.net/timeseries")
)

# COMMAND ----------

# DBTITLE 1,Register table in workspace catalogue and access using spark.table()
# MAGIC %sql
# MAGIC drop table if exists covid.timeseries;
# MAGIC create table covid.timeseries 
# MAGIC using delta
# MAGIC location 'abfss://covid@shareddatalake.dfs.core.windows.net/timeseries'

# COMMAND ----------

display(
  spark.table("covid.timeseries")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from covid.timeseries
# MAGIC where Country_Region = "UK"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Machine Learning workflow on Databricks
# MAGIC ### A 'hello world' example

# COMMAND ----------

# DBTITLE 1,Read parquet into Spark DataFrame
df = (
  spark.read
  .format("parquet")
  .load("/databricks-datasets/samples/lending_club/parquet") 
)
display(df)

# COMMAND ----------

# DBTITLE 1,Clean data using PySpark SQL API
from pyspark.sql.functions import *

# subset columns with DataFrame.select()
df = df.select(
  "purpose", "loan_status", "int_rate", "revol_util", "issue_d", 
  "earliest_cr_line", "emp_length", "verification_status", "total_pymnt", 
  "loan_amnt", "grade", "annual_inc", "dti", "addr_state", "term", 
  "home_ownership",  "application_type", "delinq_2yrs", "total_acc"
)

# subset rows with DataFrame.filter() or .where()
# refer to columns using col() function or SparkDataFrame.<<column_name>> notation
print("------------------------------------------------------------------------------------------------")
print("Create bad loan label, this will include charged off, defaulted, and late repayments on loans...")
df = (
  df
  .filter(col("loan_status").isin(["Default", "Charged Off", "Fully Paid"]))
  .withColumn("bad_loan", (~(df.loan_status == "Fully Paid")).cast("string"))
)


# create new derived columns with DataFrame.withColumn()
print("------------------------------------------------------------------------------------------------")
print("Turning string interest rate and revoling util columns into numeric columns...")
df = (
  df.withColumn('int_rate', regexp_replace('int_rate', '%', '').cast('float'))
  .withColumn('revol_util', regexp_replace('revol_util', '%', '').cast('float'))
  .withColumn('issue_year',  substring(df.issue_d, 5, 4).cast('double') )
  .withColumn('earliest_year', substring(df.earliest_cr_line, 5, 4).cast('double'))
  .withColumn('credit_length_in_years', col("issue_year") - col("earliest_year"))
)

print("------------------------------------------------------------------------------------------------")
print("Converting emp_length column into numeric...")
df = df.withColumn('emp_length', trim(regexp_replace(df.emp_length, "([ ]*+[a-zA-Z].*)|(n/a)", "") ))
df = df.withColumn('emp_length', trim(regexp_replace(df.emp_length, "< 1", "0") ))
df = df.withColumn('emp_length', trim(regexp_replace(df.emp_length, "10\\+", "10") ).cast('float'))

print("------------------------------------------------------------------------------------------------")
print("Map multiple levels into one factor level for verification_status...")
df = df.withColumn('verification_status', trim(regexp_replace(df.verification_status, 'Source Verified', 'Verified')))

print("------------------------------------------------------------------------------------------------")
print("Calculate the total amount of money earned or lost per loan...")
df = df.withColumn('net', round( df.total_pymnt - df.loan_amnt, 2))

display(df)

# COMMAND ----------

# DBTITLE 1,Save this cleaned dataframe as a new Delta table on ADLS and register in metastore
df.createOrReplaceTempView("lending_club_cleaned")

# COMMAND ----------

# MAGIC %sql
# MAGIC create database if not exists lending_club;
# MAGIC use lending_club;

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from lending_club_cleaned
# MAGIC limit 5

# COMMAND ----------

(
  df.write
  .format("delta")
  .mode("overwrite")
  .partitionBy("issue_d")
  .saveAsTable(name="lending_club.cleaned", path="abfss://mldemo@shareddatalake.dfs.core.windows.net/lending_club/cleaned")
)

df = spark.table("lending_club.cleaned")

# COMMAND ----------

# DBTITLE 1,Visualise distribution of loan amount
display(df)

# COMMAND ----------

# DBTITLE 1,Assign target and predictor columns
predictors = [
  "purpose", "term", "home_ownership", "addr_state", "verification_status",
  "application_type", "loan_amnt", "emp_length", "annual_inc", "dti", 
  "delinq_2yrs", "revol_util", "total_acc", "credit_length_in_years", 
  "int_rate", "net", "issue_year"
]

target = 'bad_loan'

# COMMAND ----------

# DBTITLE 1,Prepare training and test sets
from sklearn.model_selection import train_test_split

# collect Spark DataFrame to a Pandas DataFrame local to driver node using SparkDataFrame.toPandas()
pdDf = df.toPandas()

for col in pdDf.columns:
  if pdDf.dtypes[col]=='object':
    pdDf[col] =  pdDf[col].astype('category').cat.codes
  pdDf[col] = pdDf[col].fillna(0)
    
X_train, X_test, Y_train, Y_test = train_test_split(pdDf[predictors], pdDf[target], test_size=0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train model

# COMMAND ----------

# DBTITLE 1,Train RandomForest
from sklearn.ensemble import RandomForestClassifier

params = {
  "n_estimators": 5,
  "max_depth": 5,
  "random_state": 42
}
rf = RandomForestClassifier(**params)
rf.fit(X_train, Y_train)

# COMMAND ----------

# DBTITLE 1,Evaluate on holdout
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score

predictions = rf.predict(X_test)
acc = accuracy_score(Y_test, predictions)
roc = roc_auc_score(Y_test, predictions)
mse = mean_squared_error(Y_test, predictions)
mae = mean_absolute_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)

print("  acc: {}".format(acc))
print("  roc: {}".format(roc))
print("  mse: {}".format(mse))
print("  mae: {}".format(mae))
print("  R2 : {}".format(r2))

# COMMAND ----------

# DBTITLE 1,Extract feature importance
import pandas as pd
importance = pd.DataFrame(list(zip(df.columns, rf.feature_importances_)), 
                            columns=["Feature", "Importance"]
                          ).sort_values("Importance", ascending=False)
print(importance)

# COMMAND ----------

# DBTITLE 1,(for later use) Save the modified training data to a delta table
(
  spark.createDataFrame(pdDf).write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(name="lending_club.model_test", path="abfss://mldemo@shareddatalake.dfs.core.windows.net/lending_club/model_test")
)

# COMMAND ----------

