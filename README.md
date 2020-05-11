# Rolls-Royce // Databricks

## Data Science & Machine Learning: platform overview

The materials in this repository accompany the session delivered on 7th May 2020.



### Environment set-up

1. Install the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html) and configure for access to your workspace

2. Clone the repository to your local environment and navigate to that folder on the command line

3. Create the demo cluster using the command
    `databricks clusters create --json-file cluster-spec.json`
    
4. Upload the mmlspark python library to the cluster
    `databricks fs cp -r ./libraries dbfs:/FileStore/jars/datasci-overview`
    
5. Install all of the libraries on your cluster using the cluster UI:
    | library name (or maven coords) | version | type | repository / location |
    | --------- | ------- | ---- | --------------------- |
    | `fbprophet` | latest  | PyPI | default               |
    | `joblibspark` | latest | PyPI  |default |
    | `plotly` | latest | PyPI |default |
    |`scikit-learn==0.21.3` |0.21.3 | PyPI |default |
    | `petastorm==0.7.2` | 0.7.2 | PyPI | default |
    | `com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1` | 1.0.0-rc1 | Maven | https://mmlspark.azureedge.net/maven |
    | `mmlspark-0.17` | 0.17 | DBFS, Python Whl | dbfs:/FileStore/jars/datasci-overview/mmlspark-0.17-py2.py3-none-any.whl |
    
6. Upload the repository to your workspace using e.g. 
    `databricks workspace import_dir . /Users/email@company.com/datasci-overview`
    where `email@company.com` is the address you use to log into your workspace
    
7. Attach a notebook to your newly created cluster and try running it.



### Notebook descriptions

#### `01_Replicate-traditional-workflow`

First example of accessing data, training and evaluating a simple model in a Databricks notebook.

#### `02_Introduction-to-tracking`

Building on the first notebook, showing how to log parameters, metrics, models etc. to the MLflow tracking server.

#### `03_Deployment-management`

Demonstrating different ways of using MLflow to deploy the models trained in the previous notebook.

#### `00_Reset`

Resets some of the MLflow components (for demo purposes)

#### `Fine Grained Demand Forecasting`

Example of using pandas_udf to create forecasts for subgroups within a large dataset.

#### `Hyperopt + Spark demo`

Simple example of hyperparameter tuning on a single-node or cluster environment using the Hyperopt package.

#### `LightGBM - Quantile Regression for Drug Discovery`

Example of data distributed training of a statistical machine learning model using the Microsoft LightGBM implementation.

#### `mnist-tensorflow-keras`

Example of data distributed training of a deep neural network by pairing Keras and HorovodRunner.

#### `parallel-model-selection-joblib`

Parallel model selection on scikit-learn models using the joblibspark backend.

#### `petastorm`

Batch-at-a-time data loading for training of neural networks on datasets whose size exceeds available memory.



### Other material

Slides for the session are located [here](https://docs.google.com/presentation/d/1fLWMs5jkDGoVVkhqlbUe8qbK5N9cmtaD1sr_B-Ym-WA/edit?usp=sharing) and should be accessible immediately after the session.



### Questions?

Please contact me at stuart@databricks.com.