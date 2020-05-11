# Databricks notebook source
# MAGIC %md
# MAGIC # Distributed deep learning training using TensorFlow and Keras with `HorovodRunner` for MNIST
# MAGIC 
# MAGIC This notebook demonstrates how to train a simple model for MNIST dataset using
# MAGIC  `tensorFlow.keras` api. We will first show how to do so on a single node and then adapt
# MAGIC  the code to distribute the training on Databricks with `HorovodRunner`.
# MAGIC 
# MAGIC This guide consists of the following sections:
# MAGIC 
# MAGIC * Set up checkpoint location
# MAGIC * Run training on single node
# MAGIC * Migrate to HorovodRunner
# MAGIC 
# MAGIC **Note:**
# MAGIC * The notebook runs on CPU or GPU-enabled Apache Spark clusters.
# MAGIC * To run the notebook, create a cluster with
# MAGIC  - Two **workers**
# MAGIC  - Databricks Runtime 6.3 ML or above

# COMMAND ----------

# MAGIC %md ## Preparing Deep Learning Storage
# MAGIC 
# MAGIC We recommend using Databricks Runtime 6.0 ML or above which provides high-performance I/O for deep learning workloads for all of `/dbfs`.
# MAGIC 
# MAGIC If you are using Databricks Runtime 5.4 ML or Databricks Runtime 5.5 ML, save training data under `dbfs:/ml`, which maps to `file:/dbfs/ml` on driver and worker nodes. In these versions only `dbfs:/ml` is accelerated.

# COMMAND ----------

import os
import time

checkpoint_dir = '/dbfs/ml/MNISTDemo/train/{}/'.format(time.time())

os.makedirs(checkpoint_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Training on Single Node
# MAGIC 
# MAGIC First we shall develop a single-node training code using `tensorflow.keras`. This code is adapted from the [Keras MNIST Example](https://github.com/keras-team/keras/blob/master/examples/mnist_dataset_api.py).

# COMMAND ----------

# MAGIC %md
# MAGIC Define a function that generates the data for training. This function downloads the data using keras's mnist dataset, shards it based on the rank and size of the worker, and converts it to shapes and types suitable for training.

# COMMAND ----------

def get_dataset(num_classes, rank=0, size=1):
  from tensorflow import keras
  
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('MNIST-data-%d' % rank)
  x_train = x_train[rank::size]
  y_train = y_train[rank::size]
  x_test = x_test[rank::size]
  y_test = y_test[rank::size]
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  return (x_train, y_train), (x_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC This function defines the model to be trained using `tensorflow.keras` API. We define a simple model, that consists of 2 convolutional layers, a max-pooling layer, and a final dense layer. We also add some dropoit layers in between.

# COMMAND ----------

def get_model(num_classes):
  from tensorflow.keras import models
  from tensorflow.keras import layers
  
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
  model.add(layers.Dense(num_classes, activation='softmax'))
  return model

# COMMAND ----------

# MAGIC %md
# MAGIC We define the training function that takes in the dataset, the model definition, adds a compiler and sets some hyperparameters for tuning.

# COMMAND ----------

batch_size = 128
epochs = 5
num_classes = 10

# COMMAND ----------

def train(learning_rate=1.0):
  from tensorflow import keras
  
  (x_train, y_train), (x_test, y_test) = get_dataset(num_classes)
  model = get_model(num_classes)

  optimizer = keras.optimizers.Adadelta(lr=learning_rate)

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test, y_test))

# COMMAND ----------

# MAGIC %md
# MAGIC We can run this function and train a model on the driver itself. You will notice that the training goes on for 5 epochs during which we observe improving validation accuracy.

# COMMAND ----------

train(learning_rate=0.1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Migrate to `HorovodRunner`
# MAGIC 
# MAGIC Below, we show how to modify your single-machine code to use Horovod. For an additional reference, also check out the [Horovod usage guide](https://github.com/uber/horovod#usage).

# COMMAND ----------

def train_hvd(learning_rate=1.0):
  # Tensorflow has given up on pickling. We need to explicitly import its modules inside workers
  from tensorflow.keras import backend as K
  from tensorflow.keras.models import Sequential
  import tensorflow as tf
  from tensorflow import keras
  import horovod.tensorflow.keras as hvd
  
  # Horovod: initialize Horovod.
  hvd.init()

  # Horovod: pin GPU to be used to process local rank (one GPU per process)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = str(hvd.local_rank())
  K.set_session(tf.Session(config=config))

  (x_train, y_train), (x_test, y_test) = get_dataset(num_classes, hvd.rank(), hvd.size())
  model = get_model(num_classes)

  # Horovod: adjust learning rate based on number of GPUs.
  optimizer = keras.optimizers.Adadelta(lr=learning_rate * hvd.size())

  # Horovod: add Horovod Distributed Optimizer.
  optimizer = hvd.DistributedOptimizer(optimizer)

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  callbacks = [
      # Horovod: broadcast initial variable states from rank 0 to all other processes.
      # This is necessary to ensure consistent initialization of all workers when
      # training is started with random weights or restored from a checkpoint.
      hvd.callbacks.BroadcastGlobalVariablesCallback(0),
  ]

  # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
  if hvd.rank() == 0:
      callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_dir + '/checkpoint-{epoch}.ckpt', save_weights_only = True))

  model.fit(x_train, y_train,
            batch_size=batch_size,
            callbacks=callbacks,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test, y_test))

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have a training function with Horovod, we can use `HorovodRunner` to run distributed. To run this example on a cluster with 2 workers, each with a single GPU, initialize `HorovodRunner` with `np=2`:

# COMMAND ----------

from sparkdl import HorovodRunner

hr = HorovodRunner(np=2)
hr.run(train_hvd, learning_rate=0.1)

# COMMAND ----------

# MAGIC %md 
# MAGIC Under the hood, HorovodRunner takes a Python method that contains deep learning training code with Horovod hooks. This method gets pickled on the driver and sent to Spark workers. A Horovod MPI job is embedded as a Spark job using the barrier execution mode. The first executor collects the IP addresses of all task executors using BarrierTaskContext and triggers a Horovod job using `mpirun`. Each Python MPI process loads the pickled user program back, deserializes it, and runs it.
# MAGIC 
# MAGIC For further information on HorovodRunner API, please refer to the [documentation](https://databricks.github.io/spark-deep-learning/docs/_site/api/python/index.html#sparkdl.HorovodRunner). Note that you can use `np=-1` to spawn a subprocess on the driver node for quicker development cycle.
# MAGIC ```
# MAGIC hr = HorovodRunner(np=-1)
# MAGIC hr.run(run_training)
# MAGIC ```