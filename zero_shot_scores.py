# Databricks notebook source
import pandas as pd
import re
import json
import os
import sys
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict
import mlflow
from mlflow.entities.run import Run
from mlflow.tracking.client import MlflowClient, ModelVersion
from mlflow.utils import mlflow_tags
pd.set_option("display.max_columns" , 50)

# COMMAND ----------

def mlb_transform(label_list, true_col, pred_col, num_labels):
    # read in data
    df = pd.read_pickle("./model_training_data.pkl")
    df = df[df["split"]== "labeled"]
    df.dropna(subset=["themeIds", "themeIdsReviewed"], inplace = True)
    df.dropna(subset=["themeIdsParent", "themeIdsReviewedParent"], inplace = True)

    mlb = MultiLabelBinarizer()
    mlb.fit([label_list])
    if num_labels ==2:
        pred_col=pred_col+str(num_labels)
        
    y_true = mlb.transform(df[true_col])
    y_pred = mlb.transform(df[pred_col])

    # get scores
    macro_score = f1_score(y_true, y_pred, average='macro')
    micro_score = f1_score(y_true, y_pred, average='micro')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    return macro_score, micro_score, precision, recall

# COMMAND ----------

def run_zs_experiment(num_labels):
    # set up experiment
    mlflow_client = MlflowClient()
    exp_name ="/Users/vpeng@rockfound.org/zero_shot_scores"
    exp = mlflow_client.get_experiment_by_name(exp_name)
    mlflow.set_experiment(exp_name)
    run_name = "zero_shot_run"
    parent_run = mlflow.start_run(run_name = run_name, nested = True)

    # get themes
    for taxonomy_type in ["zero_shot_parent", "zero_shot_child"]:
        if taxonomy_type == "zero_shot_parent":
            with open("theme_dict_parent.json", 'r') as f:
                theme_dict = json.load(f)
            # create labels list
            all_labels_parent = list(theme_dict.keys())
            # get scores
            macro_score_parent, micro_score_parent, precision_parent, recall_parent = mlb_transform(all_labels_parent, "themeIdsReviewedParent", "themeIdsParent", 2)

        else:
            with open("theme_dict.json", 'r') as f:
                theme_dict = json.load(f)
            # create labels list
            all_labels = list(theme_dict.keys())

            # get scores
            macro_score, micro_score, precision, recall = mlb_transform(all_labels, "themeIdsReviewed", "themeIds", 2)

    # run experiment
    mlflow.log_params({"taxonomy_type": taxonomy_type, "number_of_labels": num_labels})
    mlflow.log_metrics({"macro_f1_parent": macro_score_parent, "micro_f1_parent": micro_score_parent, "macro_f1_child": macro_score, "micro_f1_child": micro_score, "precision_parent": precision_parent, "recall_parent": recall_parent, "precision_child": precision, "recall_child": recall})    
    
    # end run
    mlflow.end_run()


# COMMAND ----------

run_zs_experiment(2)

# COMMAND ----------

mlflow.end_run()
