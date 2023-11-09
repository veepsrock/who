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
    parent_run = mlflow.start_run(run_name = run_name, nested = True)

    # read in data
    df = pd.read_pickle("./model_training_data.pkl")
    df = df[df["split"]== "labeled"]
    df.dropna(subset=["themeIds", "themeIdsReviewed"], inplace = True)
    df.dropna(subset=["themeIdsParent", "themeIdsReviewedParent"], inplace = True)

    # get themes
    if taxonomy_type == "zero_shot_parent":
        json_file = "theme_dict_parent.json"
        true_col = "themeIdsReviewedParent"
        pred_col = "themeIdsParent"
    else:
        json_file = "theme_dict.json"
        true_col = "themeIdsReviewed"
        pred_col = "themeIds"
        
    with open(json_file, 'r') as f:
        theme_dict = json.load(f)

    # create labels list
    all_labels = list(theme_dict.keys())

    # transform data
    mlb = MultiLabelBinarizer()
    mlb.fit([all_labels])

    y_true = mlb.transform(df[true_col])
    y_pred = mlb.transform(df[pred_col])

    # get scores
    macro_score = f1_score(y_true, y_pred, average='macro')
    micro_score = f1_score(y_true, y_pred, average='micro')

    # run experiment
    mlflow.log_param("taxonomy_type", taxonomy_type)
    mlflow.log_metrics({"macro_f1": macro_score, "micro_f1": micro_score})    
    # end run
    mlflow.end_run()


# COMMAND ----------

def mlb_transform(label_list, true_col, pred_col):
    # read in data
    df = pd.read_pickle("./model_training_data.pkl")
    df = df[df["split"]== "labeled"]
    df.dropna(subset=["themeIds", "themeIdsReviewed"], inplace = True)
    df.dropna(subset=["themeIdsParent", "themeIdsReviewedParent"], inplace = True)

    mlb = MultiLabelBinarizer()
    mlb.fit([label_list])
    y_true = mlb.transform(df[true_col])
    y_pred = mlb.transform(df[pred_col])

    # get scores
    macro_score = f1_score(y_true, y_pred, average='macro')
    micro_score = f1_score(y_true, y_pred, average='micro')

    return macro_score, micro_score

# COMMAND ----------

def run_zs_experiment():
     # set up experiment
    mlflow_client = MlflowClient()
    exp_name ="/Users/vpeng@rockfound.org/zero_shot_scores"
    exp = mlflow_client.get_experiment_by_name(exp_name)
    mlflow.set_experiment(exp_name)
    run_name = "zero_shot_run"
    parent_run = mlflow.start_run(run_name = run_name, nested = True)

    # read in data
    df = pd.read_pickle("./model_training_data.pkl")
    df = df[df["split"]== "labeled"]
    df.dropna(subset=["themeIds", "themeIdsReviewed"], inplace = True)
    df.dropna(subset=["themeIdsParent", "themeIdsReviewedParent"], inplace = True)

    # get themes
    for taxonomy_type in ["zero_shot_parent", "zero_shot_child"]:
        if taxonomy_type == "zero_shot_parent":
            with open("theme_dict_parent.json", 'r') as f:
                theme_dict = json.load(f)
            # create labels list
            all_labels_parent = list(theme_dict.keys())
            # get scores
            macro_score_parent, micro_score_parent = mlb_transform(all_labels_parent, "themeIdsReviewedParent", "themeIdsParent")

        else:
            with open("theme_dict.json", 'r') as f:
                theme_dict = json.load(f)
            # create labels list
            all_labels = list(theme_dict.keys())

            # get scores
            macro_score, micro_score = mlb_transform(all_labels, "themeIdsReviewed", "themeIds")

    # run experiment
    mlflow.log_param("taxonomy_type", taxonomy_type)
    mlflow.log_metrics({"macro_f1_parent": macro_score_parent, "micro_f1_parent": micro_score_parent, "macro_f1_child": macro_score, "micro_f1_child": micro_score})    
    
    # end run
    mlflow.end_run()


# COMMAND ----------

run_zs_experiment()

# COMMAND ----------

def run_zs_experiment():
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

            # transform data
            mlb_parent = MultiLabelBinarizer()
            mlb_parent.fit([all_labels_parent])
            y_true_parent = mlb_parent.transform(df["themeIdsReviewedParent"])
            y_pred_parent = mlb_parent.transform(df["themeIdsParent"])

            # get scores
            macro_score_parent = f1_score(y_true_parent, y_pred_parent, average='macro')
            micro_score_parent = f1_score(y_true_parent, y_pred_parent, average='micro')
        else:
            with open("theme_dict.json", 'r') as f:
                theme_dict = json.load(f)
            # create labels list
            all_labels = list(theme_dict.keys())

            # transform data
            mlb = MultiLabelBinarizer()
            mlb.fit([all_labels])
            y_true = mlb.transform(df["themeIdsReviewed"])
            y_pred = mlb.transform(df["themeIds"])

            # get scores
            macro_score = f1_score(y_true, y_pred, average='macro')
            micro_score = f1_score(y_true, y_pred, average='micro')

    # run experiment
    mlflow.log_param("taxonomy_type", taxonomy_type)
    mlflow.log_metrics({"macro_f1_parent": macro_score_parent, "micro_f1_parent": micro_score_parent, "macro_f1_child": macro_score, "micro_f1_child": micro_score})    
    
    # end run
    mlflow.end_run()


# COMMAND ----------

mlflow.end_run()

# COMMAND ----------



# COMMAND ----------

run_zs_experiment("zero_shot_parent")

# COMMAND ----------

run_zs_experiment("zero_shot_child")

# COMMAND ----------


