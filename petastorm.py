# Databricks notebook source
# MAGIC %md # Use Spark and Petastorm to prepare data for deep learning
# MAGIC 
# MAGIC This notebook demonstrates the following workflow on Databricks:
# MAGIC 1. Use Spark to load and preprocess data.
# MAGIC 2. Save data using Parquet under `dbfs:/ml`.
# MAGIC 3. Load data using Petastorm via the optimized FUSE mount `file:/dbfs/ml`.
# MAGIC 4. Feed data into a DL framework for training or inference.
# MAGIC 
# MAGIC Environment tested: 
# MAGIC * Databricks Runtime 5.4 ML with one worker.
# MAGIC * `petastorm==0.7.4` installed as a PyPI library.
# MAGIC 
# MAGIC We recommend using Databricks Runtime 6.0 ML or above which provides high-performance I/O for deep learning workloads for all storage mounted to `/dbfs`.
# MAGIC 
# MAGIC If you are using Databricks Runtime 5.4 ML or Databricks Runtime 5.5 ML, save training data under `dbfs:/ml`, which maps to `file:/dbfs/ml` on driver and worker nodes. In these versions only `dbfs:/ml` is accelerated.

# COMMAND ----------

import os
import subprocess
import uuid

# COMMAND ----------

# Set a unique working directory for this notebook.
work_dir = os.path.join("/ml/tmp/petastorm", str(uuid.uuid4()))
dbutils.fs.mkdirs(work_dir)

def get_local_path(dbfs_path):
  return os.path.join("/dbfs", dbfs_path.lstrip("/"))

# COMMAND ----------

# MAGIC %md ##Load, preprocess, and save data using Spark
# MAGIC 
# MAGIC Spark can load data from many sources.
# MAGIC Let us download the MNIST dataset in LIBSVM format and load it using Spark's built-in LIBSVM data source.

# COMMAND ----------

data_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2"
libsvm_path = os.path.join(work_dir, "mnist.bz2")
subprocess.check_output(["wget", data_url, "-O", get_local_path(libsvm_path)])

# COMMAND ----------

df = spark.read.format("libsvm") \
  .option("numFeatures", "784") \
  .load(libsvm_path)

# COMMAND ----------

# MAGIC %md Petastorm supports scalar and array columns in Spark DataFrame.
# MAGIC MLlib vector is a user-defined type (UDT), which requires special handling.
# MAGIC Register a user-defined function (UDF) that converts MLlib vectors into dense arrays.

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC import org.apache.spark.ml.linalg.Vector
# MAGIC 
# MAGIC val toArray = udf { v: Vector => v.toArray }
# MAGIC spark.sqlContext.udf.register("toArray", toArray)

# COMMAND ----------

# Convert sparse vectors to dense arrays and write the data as Parquet.
# Petastorm will sample Parquet row groups into batches.
# Batch size is important for the utilization of both I/O and compute.
# You can use parquet.block.size to control the size.
parquet_path = os.path.join(work_dir, "parquet")
df.selectExpr("toArray(features) AS features", "int(label) AS label") \
  .repartition(10) \
  .write.mode("overwrite") \
  .option("parquet.block.size", 1024 * 1024) \
  .parquet(parquet_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load data using Petastorm and feed data into a DL framework
# MAGIC 
# MAGIC Use Petastorm to load the Parquet data and create a `tf.data.Dataset`.
# MAGIC Then fit a simple neural network model using `tf.Keras`.

# COMMAND ----------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers

from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset

# COMMAND ----------

def get_model():
  model = models.Sequential()
  model.add(layers.Conv2D(32, kernel_size=(3, 3),
                          activation='relu',
                          input_shape=(28, 28, 1)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Dropout(0.25))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(10, activation='softmax'))
  return model

# COMMAND ----------

# MAGIC %md Until [ARROW-4723](https://issues.apache.org/jira/browse/ARROW-4723) is resolved we need to whitelist `_*` files that Databricks Runtime creates when saving data as Parquet.

# COMMAND ----------

import pyarrow.parquet as pq

underscore_files = [f for f in os.listdir(get_local_path(parquet_path)) if f.startswith("_")]
pq.EXCLUDED_PARQUET_PATHS.update(underscore_files)

# COMMAND ----------

# We use make_batch_reader to load Parquet row groups into batches.
# HINT: Use cur_shard and shard_count params to shard data in distributed training.
petastorm_dataset_url = "file://" + get_local_path(parquet_path)
with make_batch_reader(petastorm_dataset_url, num_epochs=100) as reader:
  dataset = make_petastorm_dataset(reader) \
    .map(lambda x: (tf.reshape(x.features, [-1, 28, 28, 1]), tf.one_hot(x.label, 10)))
  model = get_model()
  optimizer = keras.optimizers.Adadelta()
  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(dataset, steps_per_epoch=10, epochs=10)

# COMMAND ----------

# Clean up the working directory.
dbutils.fs.rm(work_dir, recurse=True)

# COMMAND ----------

