# Databricks notebook source
import pandas as pd
import re
import json
import os
import sys
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from collections import defaultdict
import mlflow
from mlflow.entities.run import Run
from mlflow.tracking.client import MlflowClient, ModelVersion
from mlflow.utils import mlflow_tags
pd.set_option("display.max_columns" , 50)

# COMMAND ----------

def run_zs_experiment(taxonomy_type):
     # set up experiment
    mlflow_client = MlflowClient()
    exp_name ="/Users/vpeng@rockfound.org/zero_shot_scores"
    exp = mlflow_client.get_experiment_by_name(exp_name)
    mlflow.set_experiment(exp_name)
    run_name = "zero_shot_run"
    parent_run = mlflow.start_run(run_name = run_name)

    # read in data
    df = pd.read_pickle("./model_training_data.pkl")
    df = df[df["split"]== "labeled"]
    df.dropna(subset=["themeIds", "themeIdsReviewed"], inplace = True)
    df.dropna(subset=["themeIdsParent", "themeIdsReviewedParent"], inplace = True)

    # get themes
    with open('theme_dict_parent.json', 'r') as f:
        theme_dict = json.load(f)

    # create labels list
    all_labels = list(theme_dict.keys())

    # transform data
    mlb = MultiLabelBinarizer()
    mlb.fit([all_labels])

    y_true = mlb.transform(df["themeIdsReviewedParent"])
    y_pred = mlb.transform(df["themeIdsParent"])

    # get scores
    macro_score = f1_score(y_true, y_pred, average='macro')
    micro_score = f1_score(y_true, y_pred, average='micro')

    # run experiment
    mlflow.log_param("taxonomy_type", taxonomy_type)
    mlflow.log_metrics({"macro_f1": macro_score, "micro_f1": micro_score})    
    # end run
    mlflow.end_run()


# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

run_zs_experiment("zero_shot_parent")

# COMMAND ----------


