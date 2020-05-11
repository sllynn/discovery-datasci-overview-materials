# Databricks notebook source
from mlflow.tracking import MlflowClient
client = MlflowClient()

# COMMAND ----------

path = "/Users/stuart.lynn@databricks.com/customers/rolls-royce/02_Introduction-to-tracking"
model_name = "random-forest-model"

# COMMAND ----------

for mv in client.search_model_versions(f"name='{model_name}'"):
  client.transition_model_version_stage(name=mv.name, version=mv.version, stage="None")
  client.delete_model_version(name=mv.name, version=mv.version)

# COMMAND ----------

experimentID = [e.experiment_id for e in client.list_experiments() if e.name==path][0]
runs = spark.read.format("mlflow-experiment").load(experimentID)

# COMMAND ----------

for run in runs.collect():
  client.delete_run(run.run_id)

# COMMAND ----------

# dbutils.fs.cp("/databricks-datasets/samples/population-vs-price/data_geo.csv", "/stuart/demo/samples/population-vs-price/data_geo.csv")
# dbutils.fs.cp("/databricks-datasets/samples/people/people.json", "/stuart/demo/samples/people/people.json")
# dbutils.fs.cp("/databricks-datasets/samples/lending_club/parquet", "/stuart/demo/samples/lending_club/parquet", True)

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists covid.timeseries;
# MAGIC drop table if exists lending_club.cleaned;
# MAGIC drop table if exists lending_club.model_test;

# COMMAND ----------

dbutils.fs.rm("abfss://mldemo@shareddatalake.dfs.core.windows.net/lending_club/cleaned", True)
dbutils.fs.rm("abfss://mldemo@shareddatalake.dfs.core.windows.net/lending_club/model_test", True)

# COMMAND ----------

