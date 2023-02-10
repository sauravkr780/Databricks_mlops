# Databricks notebook source
# MAGIC %md # Get started with machine learning in Databricks using scikit-learn, MLflow autologging, and Hyperopt
# MAGIC 
# MAGIC This tutorial is designed as an introduction to machine learning in Databricks. It uses algorithms from the popular machine learning package scikit-learn along with MLflow for tracking the model development process and Hyperopt to automate hyperparameter tuning. 
# MAGIC 
# MAGIC It includes the following steps:
# MAGIC - Load and preprocess a small dataset
# MAGIC - Part 1. Create a random forest model with scikit-learn
# MAGIC - Part 2. Perform automated hyperparameter tuning with Hyperopt and MLflow
# MAGIC - Part 3. Register the model in Model Registry and use the registered model to make predictions
# MAGIC 
# MAGIC ### Requirements
# MAGIC - Databricks Runtime 7.3 LTS ML or above

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# COMMAND ----------

# MAGIC %md ## Load and preprocess data

# COMMAND ----------

# MAGIC %md Import the dataset from scikit-learn and create the training and test datasets. 

# COMMAND ----------

cal_housing = fetch_california_housing()

# split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                    cal_housing.target,
                                                    test_size=0.2)

# COMMAND ----------

# MAGIC %md Scale the data

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md ## Part 1. Create a random forest model

# COMMAND ----------

# MAGIC %md ### Use `mlflow.sklearn.autolog()` to automatically log parameters, metrics, and the model.

# COMMAND ----------

# Enable autolog()
# mlflow.sklearn.autolog() requires mlflow 1.11.0 or above.
mlflow.sklearn.autolog()

# COMMAND ----------

# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run():
  
  # Set the model parameters. 
  n_estimators = 100
  max_depth = 6
  max_features = 3
  
  # Create and train model.
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
  
  # Use the model to make predictions on the test dataset.
  predictions = rf.predict(X_test) 
  

# COMMAND ----------

# MAGIC %md ### View results in MLflow
# MAGIC To view the results, click **Experiment** at the upper right of this page. The Experiments sidebar appears. This sidebar displays the parameters and metrics for each run of this notebook. Click the circular arrows icon to refresh the display to include the latest runs. 
# MAGIC 
# MAGIC When you click the square icon with the arrow to the right of the date and time of the run, the Runs page opens in a new tab. This page shows all of the information that was logged from the run. Scroll down to the **Artifacts** section and click on the **model** folder. The right panel provides code snippets showing how to use the model for predictions on Spark and Pandas DataFrames. 
# MAGIC 
# MAGIC For more information, see View results ([AWS](https://docs.databricks.com/applications/mlflow/quick-start-python.html#view-results)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/quick-start-python#view-results)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/quick-start-python.html#view-results)).

# COMMAND ----------

# MAGIC %md ## Part 2. Perform automated hyperparameter tuning with Hyperopt and MLflow
# MAGIC [Hyperopt](https://github.com/hyperopt/hyperopt) is a Python library for hyperparameter tuning. Databricks Runtime for Machine Learning includes an optimized and enhanced version of Hyperopt, including automated MLflow tracking.  
# MAGIC For more information about using Hyperopt in Databricks, see ([AWS](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/automl-hyperparam-tuning/index#hyperparameter-tuning-with-hyperopt)|[GCP](https://docs.gcp.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt)).  
# MAGIC For general information about Hyperopt, see the [Hyperopt documentation](https://github.com/hyperopt/hyperopt/wiki/FMin).

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK

search_space = {
  'max_depth': hp.quniform('max_depth', 2, 10, 1),
  'n_estimators': hp.quniform('n_estimators', 200, 1000, 100),
  'max_features': hp.quniform('max_features', 3, 8, 1),
}

def train_model(params):
   
  # Create and train model.
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
  
  predictions = rf.predict(X_test)
  
  # Evaluate the model
  mse = mean_squared_error(y_test, predictions)
  
  return {"loss": mse, "status": STATUS_OK}
  
  
spark_trials = SparkTrials()

with mlflow.start_run() as run:
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=32,
    trials=spark_trials)

# COMMAND ----------

# MAGIC %md ### Review the feature importances determined by the model

# COMMAND ----------

feature_importances = pd.DataFrame(rf.feature_importances_, index=cal_housing.feature_names, columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md ### Build the final model

# COMMAND ----------

# MAGIC %md Use `hyperopt.space_eval` to display the results of the hyperparameter search. 

# COMMAND ----------

import hyperopt

print(hyperopt.space_eval(search_space, best_params))

# COMMAND ----------

max_depth = int(hyperopt.space_eval(search_space, best_params)["max_depth"])
max_features = int(hyperopt.space_eval(search_space, best_params)["max_features"])
n_estimators = int(hyperopt.space_eval(search_space, best_params)["n_estimators"]) 

# COMMAND ----------

# MAGIC %md Because you use all of the data to build the final model, you must scale the entire dataset.

# COMMAND ----------

X_all_train = scaler.fit_transform(cal_housing.data)
y_all_train = cal_housing.target

# COMMAND ----------

# MAGIC %md Start a new MLflow run to create the final model

# COMMAND ----------

with mlflow.start_run() as run:
  
  rf_new = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf_new.fit(X_all_train, y_all_train)
  
  # Save the run information to register the model later
  rf_uri = run.info.artifact_uri
  
  # Plot predicted vs known values for a quick visual check of the model and log the plot as an artifact
  rf_pred = rf_new.predict(X_all_train)
  plt.plot(y_all_train, rf_pred, "o", markersize=2)
  plt.xlabel("observed value")
  plt.ylabel("predicted value")
  plt.savefig("rfplot.png")
  mlflow.log_artifact("rfplot.png") 

# COMMAND ----------

# MAGIC %md ## Part 3. Register the final model in Model Registry

# COMMAND ----------

import time

model_name = "rf_cal_housing"
model_uri = rf_uri+"/model"
new_model_version = mlflow.register_model(model_uri, model_name)

# Registering the model takes a few seconds, so add a delay before continuing with the next cell
time.sleep(5)

# COMMAND ----------

# MAGIC %md ### Load the model to make predictions on new data

# COMMAND ----------

new_data = [[ 2.2 , -0.9,  1.05, -0.08, -0.34, 0.01,  0.74, -1.1],
            [ -0.9 , 2.6,  -1.4, -0.54, -0.86, 0.77,  0.35, -.08] ]

rf_model = mlflow.sklearn.load_model(f"models:/{model_name}/{new_model_version.version}")
preds = rf_model.predict(new_data)
preds
