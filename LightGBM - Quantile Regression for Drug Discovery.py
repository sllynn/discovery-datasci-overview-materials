# Databricks notebook source
# MAGIC %md ## 106 - Quantile Regression with LightGBM
# MAGIC 
# MAGIC We will demonstrate how to use the LightGBM quantile regressor with
# MAGIC TrainRegressor and ComputeModelStatistics on the Triazines dataset.
# MAGIC 
# MAGIC 
# MAGIC This sample demonstrates how to use the following APIs:
# MAGIC - [`TrainRegressor`
# MAGIC   ](http://mmlspark.azureedge.net/docs/pyspark/TrainRegressor.html)
# MAGIC - [`LightGBMRegressor`
# MAGIC   ](http://mmlspark.azureedge.net/docs/pyspark/LightGBMRegressor.html)
# MAGIC - [`ComputeModelStatistics`
# MAGIC   ](http://mmlspark.azureedge.net/docs/pyspark/ComputeModelStatistics.html)

# COMMAND ----------

triazines = spark.read.format("libsvm")\
    .load("wasbs://publicwasb@mmlspark.blob.core.windows.net/triazines.scale.svmlight")

# COMMAND ----------

# print some basic info
print("records read: " + str(triazines.count()))
print("Schema: ")
triazines.printSchema()

# COMMAND ----------

display(triazines)

# COMMAND ----------

# MAGIC %md Split the dataset into train and test

# COMMAND ----------

train, test = triazines.randomSplit([0.85, 0.15], seed=1)

# COMMAND ----------

# MAGIC %md Train the quantile regressor on the training data.

# COMMAND ----------

from mmlspark.lightgbm import LightGBMRegressor
model = LightGBMRegressor(objective='quantile',
                          alpha=0.2,
                          learningRate=0.3,
                          numLeaves=31).fit(train)

# COMMAND ----------

# MAGIC %md We can save and load LightGBM to a file using the LightGBM native representation

# COMMAND ----------

from mmlspark.lightgbm import LightGBMRegressionModel
model.saveNativeModel("mymodel")
model = LightGBMRegressionModel.loadNativeModelFromFile("mymodel")

# COMMAND ----------

# MAGIC %md View the feature importances of the trained model.

# COMMAND ----------

print(model.getFeatureImportances())

# COMMAND ----------

# MAGIC %md Score the regressor on the test data.

# COMMAND ----------

scoredData = model.transform(test)
display(scoredData)

# COMMAND ----------

# MAGIC %md Compute metrics using ComputeModelStatistics

# COMMAND ----------

from mmlspark.train import ComputeModelStatistics
metrics = ComputeModelStatistics(evaluationMetric='regression',
                                 labelCol='label',
                                 scoresCol='prediction') \
            .transform(scoredData)
display(metrics)