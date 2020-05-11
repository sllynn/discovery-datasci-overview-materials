# Databricks notebook source
# MAGIC %md 
# MAGIC ## Hyperopt + Apache Spark + MLflow integration
# MAGIC 
# MAGIC ### Hyperparameter tuning using PyTorch for MNIST

# COMMAND ----------

# Imports:

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from hyperopt import fmin, rand, tpe, hp, Trials, exceptions, space_eval, STATUS_FAIL, STATUS_OK
from hyperopt import SparkTrials

from plotly.offline import plot
import plotly.graph_objs as go
import pandas as pd

import mlflow

def get_result_dict(trial):
  result = {'loss': trial['result']['loss']}
  for k,v in trial['misc']['vals'].items():
    if k is 'loss':
      continue
    result[k] = v[0]
  return result

def add_color_col(df):
  """Add color column based on loss"""
  min_loss, max_loss = min(df.loss), max(df.loss)
  min_color, max_color = -4000.0, -100
  df['color'] = df['loss'].apply(lambda loss: ((loss - min_loss) / (max_loss - min_loss)) * (max_color - min_color) + min_color)
  return df

def get_dimensions(plotly_df):
  dims = []
  dims.append(dict(
    range = [min(plotly_df.loss), max(plotly_df.loss)],
    label = 'Loss',
    values = plotly_df['loss']
  ))
  for c in plotly_df.columns:
    if c is 'loss' or c is 'color':
      continue
    dims.append(dict(
      range = [min(plotly_df[c]), max(plotly_df[c])],
      label = c,
      values = plotly_df[c]
    ))
  return dims

def plot_trials(spark_trials):
  data = [get_result_dict(t) for t in spark_trials.trials if t['result']['status'] == 'ok' and not np.isnan(t['result']['loss'])]
  df = pd.DataFrame(data = data)
  df = add_color_col(df)
  plotly_data = [
    go.Parcoords(
        line = dict(color = df['color'],
                   colorscale = 'Jet',
                   showscale = True,
                   reversescale = True,
                   cmin = -4000,
                   cmax = -100),
        dimensions = get_dimensions(df)
    )
  ]
  displayHTML(plot(plotly_data, output_type='div'))

# COMMAND ----------

# MAGIC %md ### Regular (single-machine) Hyperopt workflow

# COMMAND ----------

# MAGIC %md
# MAGIC **Define a function to minimize**
# MAGIC 
# MAGIC * Inputs: hyperparameters
# MAGIC * Internally: Read data, fit a model, evaluate.
# MAGIC * Output: loss

# COMMAND ----------

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# COMMAND ----------

def train_one_epoch(model, device, train_dataset, batch_size, learning_rate, momentum):
    model = model.to(device)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    model.train()
    
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))

    return loss

# COMMAND ----------

def train(params):
  """
  This method will be passed to `hyperopt.fmin()`.  It fits and evaluates the model using the given hyperparameters.
  
  :param params: This dict of parameters specifies hyperparameter values to test.
  :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
  """
  batch_size = int(params['batch_size'])
  learning_rate = params['learning_rate']
  momentum = params['momentum']
    
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train_dataset = datasets.MNIST(
    'data', 
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
  
  model = Net().to(device)
  loss = train_one_epoch(model, device, train_dataset, batch_size, learning_rate, momentum)
  return {'loss': loss.item(), 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md 
# MAGIC **Define the search space over hyperparameters**

# COMMAND ----------

search_space = {
  'batch_size': hp.uniform('batch_size', 10, 200),
  'learning_rate': hp.loguniform('learning_rate', -2.0, 0.),
  'momentum': hp.uniform('momentum', 0.1, 0.5),
}

# COMMAND ----------

# MAGIC %md
# MAGIC **Select a search algorithm**

# COMMAND ----------

algo=tpe.suggest  # Tree of Parzen Estimators (a "Bayesian" method)

# COMMAND ----------

# MAGIC %md 
# MAGIC **Run model tuning with Hyperopt fmin()**

# COMMAND ----------

argmin = fmin(
  fn=train,
  space=search_space,
  algo=algo,
  max_evals=12,
  show_progressbar=False)

# COMMAND ----------

argmin

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribute tuning using Spark

# COMMAND ----------

spark_trials = SparkTrials(parallelism=4)

# COMMAND ----------

# Best practice: Active MLflow run management via `with mlflow.start_run():`
with mlflow.start_run():
  argmin = fmin(
    fn=train,
    space=search_space,
    algo=algo,
    max_evals=12,
    show_progressbar=False,
    trials=spark_trials)

# COMMAND ----------

argmin

# COMMAND ----------

