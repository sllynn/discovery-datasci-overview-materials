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

import mlflow

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

pdDf = df.toPandas()

for col in pdDf.columns:
  if pdDf.dtypes[col]=='object':
    pdDf[col] =  pdDf[col].astype('category').cat.codes
  pdDf[col] = pdDf[col].fillna(0)
    
X_train, X_test, Y_train, Y_test = train_test_split(pdDf[predictors], pdDf[target], test_size=0.2)

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

