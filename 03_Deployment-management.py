# Databricks notebook source
# MAGIC %md  
# MAGIC # DSML Overview session
# MAGIC ### Model Deployment & Management

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Key take aways for this demo:
# MAGIC 
# MAGIC * Show how the models can be for batch and streaming inference with MLflow
# MAGIC * Show how to manage incremental versions of models with MLflow model registry

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare data

# COMMAND ----------

# DBTITLE 1,Import needed packages
from pyspark.sql.functions import *

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
model_name = "random-forest-model"
model = mlflow.sklearn.load_model(model_uri=f"runs:/{best_run_id}/{model_name}")
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

# DBTITLE 1,Retrieve model with mlflow.pyfunc.spark_udf and push into Spark pipeline
import mlflow.pyfunc
spark_model = mlflow.pyfunc.spark_udf(spark, model_uri=f"runs:/{best_run_id}/{model_name}")
spark_model

# COMMAND ----------

predictions_df = spark.table("lending_club.model_test").withColumn("prediction", spark_model(*predictors))
display(predictions_df)

# COMMAND ----------

# DBTITLE 1,Use the model in a Spark Structured Streaming pipeline?
streaming_df = (
  spark.readStream
  .format("delta")
  .option("maxFilesPerTrigger", 1)
  .load("abfss://mldemo@shareddatalake.dfs.core.windows.net/lending_club/model_test/")
)

scored_stream_df = streaming_df.withColumn("prediction", spark_model(*predictors))

display(scored_stream_df)

# COMMAND ----------

# DBTITLE 1,Finally, we all know the world runs on SQL
spark.udf.register("debt_model", spark_model)

# COMMAND ----------

# MAGIC %sql
# MAGIC select *, debt_model(
# MAGIC   purpose, term, home_ownership, addr_state
# MAGIC   , verification_status, application_type, loan_amnt
# MAGIC   , emp_length, annual_inc, dti, 
# MAGIC   delinq_2yrs, revol_util, total_acc
# MAGIC   , credit_length_in_years, int_rate, net, issue_year) as prediction
# MAGIC from lending_club.model_test

# COMMAND ----------

# MAGIC %md
# MAGIC # Registering a model

# COMMAND ----------

result = mlflow.register_model(f"runs:/{best_run_id}/{model_name}", model_name)
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
# MAGIC # Deploying the latest best model

# COMMAND ----------

client.search_model_versions(f"name='{model_name}'")

# COMMAND ----------

current_prod_model = [mv.source for mv in client.search_model_versions(f"name='{model_name}'") if mv.current_stage == "Production"][0]
spark_model = mlflow.pyfunc.spark_udf(spark, current_prod_model)
predictions_df = spark.table("lending_club.model_test").withColumn("prediction", spark_model(*predictors))
display(predictions_df)

# COMMAND ----------

