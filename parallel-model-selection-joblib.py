# Databricks notebook source
# MAGIC %md  
# MAGIC # Parallel model selection
# MAGIC ## Joblib and the Spark backend for sklearn.model_selection

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Key take aways for this demo:
# MAGIC 
# MAGIC * Showcase how to leverage cloud infrastructure for parallel hyperparameter optimisation 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare data

# COMMAND ----------

# DBTITLE 1,Import needed packages
import os

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from pyspark.sql.functions import *

import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# DBTITLE 1,Read dataset into Spark DataFrame
source_table = "lending_club.cleaned"
df = spark.table(source_table)

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

# MAGIC %md ## List and compare models from tracking server

# COMMAND ----------

# DBTITLE 1,Get MLflow Experiment ID
from mlflow.tracking import MlflowClient

path = "/Shared/lending_club"

client = MlflowClient()
experimentID = [e.experiment_id for e in client.list_experiments() if e.name==path][0]
experimentID

# COMMAND ----------

# DBTITLE 1,Get all runs for our experiment
runs = spark.read.format("mlflow-experiment").load(experimentID)

display(runs)

# COMMAND ----------

# DBTITLE 1,Pick run with top ROC
best_run_id = runs.orderBy(desc("metrics.roc")).limit(1).select("run_id").collect()[0].run_id
best_run_id

# COMMAND ----------

# DBTITLE 1,Retrieve model as scikit-learn model and score on Pandas DataFrame
import mlflow.sklearn
model = mlflow.sklearn.load_model(model_uri=f"runs:/{best_run_id}/random-forest-model")
model

# COMMAND ----------

pdDf = df.toPandas()

for col in pdDf.columns:
  if pdDf.dtypes[col]=='object':
    pdDf[col] =  pdDf[col].astype('category').cat.codes
  pdDf[col] = pdDf[col].fillna(0)
    
X_test, Y_test = pdDf[predictors], pdDf[target]

# COMMAND ----------

predictions = model.predict(X_test)
predictions[:20]

# COMMAND ----------

spark.createDataFrame(pdDf).createOrReplaceTempView("model_test")

# COMMAND ----------

# DBTITLE 1,Retrieve model with mlflow.pyfunc.spark_udf and push into Spark pipeline
import mlflow.pyfunc
spark_model = mlflow.pyfunc.spark_udf(spark, model_uri=f"runs:/{best_run_id}/random-forest-model")
spark_model

# COMMAND ----------

predictions_df = spark.table("model_test").withColumn("prediction", spark_model(*predictors))
display(predictions_df)

# COMMAND ----------

# DBTITLE 1,Use the model in a Spark Structured Streaming pipeline?
# write our dataset out first to allow us to simulate streaming
(
  spark.table("model_test")
  .repartition(200)
  .write
  .mode("overwrite")
  .csv("abfss://mldemo@shareddatalake.dfs.core.windows.net/lending_club/model_test/", header=True)
)

# COMMAND ----------

model_test_schema = spark.table("model_test").schema

streaming_df = (
  spark.readStream
  .format("csv")
  .schema(model_test_schema)
  .option("maxFilesPerTrigger", 1)
  .option("header", True)
  .load("abfss://mldemo@shareddatalake.dfs.core.windows.net/lending_club/model_test/")
)

scored_stream_df = streaming_df.withColumn("prediction", spark_model(*predictors))

display(scored_stream_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Parallel model selection with sklearn.model_selection and joblibspark
# MAGIC 
# MAGIC - Install joblibspark

# COMMAND ----------

# DBTITLE 1,Run random search to find best hyperparameter combination
# from spark_sklearn import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import parallel_backend
from joblibspark import register_spark

register_spark()

with mlflow.start_run(run_name="Random Search - RandomForest") as run:

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 20, stop = 100, num = 20)]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 100, num = 20)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf_random = RandomizedSearchCV(
      estimator = RandomForestClassifier(), 
      param_distributions = random_grid, 
      n_iter = 40, cv = 5, 
      verbose=2, random_state=42, n_jobs = -1
    )
    
    # Fit the random search model
    with parallel_backend('spark', n_jobs=3):
      rf_random.fit(X_train, Y_train)
    # log metrics
    eval_and_log_metrics(rf_random.best_estimator_, X_test, Y_test)
    # log best model
    mlflow.sklearn.log_model(rf_random.best_estimator_, "random-forest-model-best")
    # log best parameters
    mlflow.log_param("best_set_of_parameters", rf_random.cv_results_['params'][rf_random.best_index_])

# COMMAND ----------

# DBTITLE 1,Best combination of parameters
best_set_of_parameters = rf_random.cv_results_['params'][rf_random.best_index_]
best_set_of_parameters

# COMMAND ----------

# MAGIC %md
# MAGIC # Registering a model

# COMMAND ----------

# DBTITLE 1,Get ID of run with best 'roc auc' score & register the model in Model Registry
run_id = runs.orderBy(desc("metrics.roc")).limit(1).select("run_id").collect()[0].run_id
model_name = "random-forest-model-best"

result = mlflow.register_model(
    f"runs:/{run_id}/{model_name}",
    model_name
)

# COMMAND ----------

result

# COMMAND ----------

# DBTITLE 1,Promote this version to 'deployment ready' status
client.transition_model_version_stage(
    name=result.name,
    version=result.version,
    stage="Production"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploying the model into a Spark pipeline

# COMMAND ----------

current_prod_model = [mv.source for mv in client.search_model_versions(f"name='{model_name}'") if mv.current_stage == "Production"][0]
print(current_prod_model)

# COMMAND ----------

mlflow.pyfunc.spark_udf(spark, current_prod_model)

# COMMAND ----------

# DBTITLE 1,Get ID of run with best 'roc auc' score & load the model from MLflow
run_id = runs.toPandas().sort_values("roc", ascending=False)["run_id"][0]
print(run_id)

path="random-forest-model-best"
model_uri = "runs:/" + run_id + "/" + path

import mlflow.sklearn

model = mlflow.sklearn.load_model(model_uri=model_uri)
model.predict(X_test)

# COMMAND ----------

print(model_uri)

# COMMAND ----------

display(spark.read.format("mlflow-experiment").load(experimentID))

# COMMAND ----------

spark.read.format("mlflow-experiment").load(experimentID).createOrReplaceTempView("my_mlflow_exp")

# COMMAND ----------

# MAGIC %sql
# MAGIC select run_id, params.n_estimators, metrics.acc, metrics.roc
# MAGIC from my_mlflow_exp
# MAGIC where metrics.roc>0.7

# COMMAND ----------

# MAGIC %md ## Model deployment

# COMMAND ----------

# DBTITLE 1,Create Spark DataFrame with test data
spark_df = spark.createDataFrame(pd.DataFrame(X_test))
display(spark_df)

# COMMAND ----------

# DBTITLE 1,Create pandas UDF for our selected model/run
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# COMMAND ----------

# DBTITLE 1,Predict using pandas UDF and DataFrame API
df = spark_df.withColumn("prediction", pyfunc_udf(*spark_df.columns))  
display(df)

# COMMAND ----------

# DBTITLE 1,Register our UDF to be used in SQL
spark_df.registerTempTable("sql_table_example")
spark.udf.register("model", pyfunc_udf)

# COMMAND ----------

# DBTITLE 1,Run prediction from SQL with Spark SQL
# MAGIC %sql
# MAGIC select *, model(term, home_ownership, purpose, addr_state, verification_status, application_type, loan_amnt, emp_length, annual_inc,dti, delinq_2yrs, revol_util, total_acc, credit_length_in_years, int_rate, net, issue_year) as predictions from sql_table_example

# COMMAND ----------

streamDF = spark.readStream.format("delta").load("/ml/tmp/loans.delta")
streamDF.createOrReplaceTempView("loans_stream")

# COMMAND ----------

df.write.format("delta").mode("overwrite").save("/ml/tmp/loans.delta")

# COMMAND ----------

# MAGIC %sql
# MAGIC select *, model(term, home_ownership, purpose, addr_state, verification_status, application_type, loan_amnt, emp_length, annual_inc,dti, delinq_2yrs, revol_util, total_acc, credit_length_in_years, int_rate, net, issue_year) as predictions from loans_stream

# COMMAND ----------

