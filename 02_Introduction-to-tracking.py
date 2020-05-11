# Databricks notebook source
# MAGIC %md  
# MAGIC # DSML Overview session
# MAGIC ### Introduction to MLflow tracking

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare data

# COMMAND ----------

# DBTITLE 1,Import needed packages
import pandas as pd
from pyspark.sql.functions import *

# COMMAND ----------

# DBTITLE 1,Read dataset into Spark DataFrame
table_name = "lending_club.cleaned"
df = spark.table(table_name)

# COMMAND ----------

# DBTITLE 1,Delta table history
table_history = spark.sql(f"describe history {table_name}")
display(table_history)

# COMMAND ----------

latest_table_version = table_history.orderBy(desc("version")).limit(1).collect()[0].version
latest_table_version

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

# DBTITLE 1,A helper function to evaluate metrics for a given model
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score

def eval_metrics(estimator, X, Y):
  predictions = estimator.predict(X)

  # Calc metrics
  metrics = dict(
    acc = accuracy_score(Y, predictions),
    roc = roc_auc_score(Y, predictions),
    mse = mean_squared_error(Y, predictions),
    mae = mean_absolute_error(Y, predictions),
    r2 = r2_score(Y_test, predictions)
  )
  
  print(metrics)
  
  return metrics

# COMMAND ----------

# DBTITLE 1,Train RandomForest and log model, parameters and metrics to MLflow
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier

def train(run_name, params, X_train, X_test, Y_train, Y_test):

  with mlflow.start_run(run_name=run_name) as run:
  
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_test)

    # Log params
    mlflow.log_params(params)
    
    # Add source data info
    mlflow.set_tag("source_table", table_name)
    mlflow.set_tag("source_table_history_version", latest_table_version)
    
    # Log metrics
    metrics = eval_metrics(rf, X_test, Y_test)
    mlflow.log_metrics(metrics)

    # Log model
    mlflow.sklearn.log_model(rf, "random-forest-model")
    
    # log some other artifacts
    importance = pd.DataFrame(list(zip(df.columns, rf.feature_importances_)), 
                                columns=["Feature", "Importance"]
                              ).sort_values("Importance", ascending=False)
    print(importance)
    csvPath = "/tmp/feature-importance.csv"
    importance.to_csv(csvPath, index=False)
    
    mlflow.log_artifact(csvPath)

  return run.info.run_uuid

# COMMAND ----------

# DBTITLE 1,Specify hyper-parameters and train a model
params = {
  "n_estimators": 5,
  "max_depth": 5,
  "random_state": 42
}

train("sklearn-random-forest-run-1", params, X_train, X_test, Y_train, Y_test)

# COMMAND ----------

# DBTITLE 1,Train another model
params = {
  "n_estimators": 8,
  "max_depth": 5,
  "random_state": 42
}

train("sklearn-random-forest-run-2", params, X_train, X_test, Y_train, Y_test)

# COMMAND ----------

# DBTITLE 1,Train one more model
params = {
  "n_estimators": 10,
  "max_depth": 7,
  "random_state": 66
}

train("sklearn-random-forest-run-3", params, X_train, X_test, Y_train, Y_test)

# COMMAND ----------

# MAGIC %md ## List and compare models from tracking server

# COMMAND ----------

# DBTITLE 1,Get MLflow Experiment ID
from mlflow.tracking import MlflowClient

path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().getOrElse(None)

client = MlflowClient()
experimentID = [e.experiment_id for e in client.list_experiments() if e.name==path][0]
experimentID

# COMMAND ----------

# DBTITLE 1,Get all runs for our experiment
runs = spark.read.format("mlflow-experiment").load(experimentID)

display(runs)

# COMMAND ----------

# DBTITLE 1,Get only runs with ROC>0.7
display(runs.where("metrics.roc>0.975").select("run_id",  "artifact_uri", "metrics.roc", "metrics.acc"))

# COMMAND ----------

dbutils.notebook.exit("0")

# COMMAND ----------

# DBTITLE 1,Reset demo
import mlflow
mlflow.set_experiment("/Shared/lending_club")

# COMMAND ----------

