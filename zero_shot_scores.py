# Databricks notebook source
import pandas as pd
import re
import json
import os
import sys
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
from collections import defaultdict
import mlflow
from mlflow.entities.run import Run
from mlflow.tracking.client import MlflowClient, ModelVersion
from mlflow.utils import mlflow_tags
import warnings
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
    class_names = mlb.classes_
    jaccard_scores = jaccard_score(y_true,y_pred, average =None)

    return macro_score, micro_score, precision, recall, class_names, jaccard_scores

# COMMAND ----------

df = pd.read_pickle("./model_training_data.pkl")
df = df[df["split"]== "labeled"]
df.dropna(subset=["themeIds", "themeIdsReviewed"], inplace = True)
df.dropna(subset=["themeIdsParent", "themeIdsReviewedParent"], inplace = True)

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
            macro_score_parent, micro_score_parent, precision_parent, recall_parent, class_names, jaccard_scores = mlb_transform(all_labels_parent, "themeIdsReviewedParent", "themeIdsParent", 2)
            class_names = ["p-" + class_name for class_name in class_names]
            jaccard_dict_p = dict(zip(class_names, jaccard_scores))
            true_values = df["themeIdsReviewedParent"].explode().value_counts(normalize = True).to_dict()
            true_dict_p = {'p-' + key + "-true": value for key, value in true_values.items()}

        else:
            with open("theme_dict.json", 'r') as f:
                theme_dict = json.load(f)
            # create labels list
            all_labels = list(theme_dict.keys())

            # get scores
            macro_score, micro_score, precision, recall, class_names, jaccard_scores = mlb_transform(all_labels, "themeIdsReviewed", "themeIds", num_labels)
            class_names = ["c-" + class_name for class_name in class_names]
            jaccard_dict = dict(zip(class_names, jaccard_scores))
            jaccard_dict_p.update(jaccard_dict)
            true_values = df["themeIdsReviewed"].explode().value_counts(normalize = True).to_dict()
            true_dict_c = {'c-' + key + "-true": value for key, value in true_values.items()}

    # run experiment
    metrics = {"p-macro_f1": macro_score_parent, "p-micro_f1": micro_score_parent, "c-macro_f1": macro_score, "c-micro_f1": micro_score, "p-precision": precision_parent, "p-recall": recall_parent, "c-precision": precision, "c-recall": recall}
    metrics.update(jaccard_dict_p, **true_dict_p, **true_dict_c)
    mlflow.log_params({"taxonomy_type": taxonomy_type, "number_of_labels": num_labels})
    mlflow.log_metrics(metrics)    
    
    # end run
    mlflow.end_run()


# COMMAND ----------

run_zs_experiment("all")

# COMMAND ----------

# MAGIC %md
# MAGIC # Threshold Experiment

# COMMAND ----------

# get theme threshold
def filter_threshold(df, threshold):
    for index, row in df.iterrows():
        confidences = row["themeConfidence"]
        theme_ids = row["themeIds"]

        # Ensure lengths are the same
        min_len = min(len(confidences), len(theme_ids))
        filtered_confidences = [confidences[i] if confidences[i] > threshold else None for i in range(min_len)]
        filtered_theme_ids = [theme_ids[i] if confidences[i] > threshold else None for i in range(min_len)]
        df.at[index, "themeConfidence"] = filtered_confidences
        df.at[index, "themeIds"] = filtered_theme_ids

    return df

# COMMAND ----------

def run_zs_threshold():
    with open("theme_dict.json", 'r') as f:
        theme_dict = json.load(f)
    # create labels list
    all_labels = list(theme_dict.keys())

    # set up experiment
    mlflow_client = MlflowClient()
    exp_name ="/Users/vpeng@rockfound.org/zero_shot_threshold"
    exp = mlflow_client.get_experiment_by_name(exp_name)
    mlflow.set_experiment(exp_name)
    run_name = "zero_shot_threshold"
    parent_run = mlflow.start_run(run_name = run_name)

    
    for prob_threshold in [i/100 for i in range(75, 96, 5)]:
        # read in data
        df = pd.read_pickle("./model_training_data.pkl")
        df = df[df["split"]== "labeled"]
        df.dropna(subset=["themeIds", "themeIdsReviewed", "themeConfidence"], inplace = True)
        df = filter_threshold(df, prob_threshold)

        # transform data
        mlb = MultiLabelBinarizer()
        mlb.fit([all_labels])
            
        y_true = mlb.transform(df["themeIdsReviewed"])
        y_pred = mlb.transform(df["themeIds"])

        # get scores
        macro_score = f1_score(y_true, y_pred, average='macro')
        micro_score = f1_score(y_true, y_pred, average='micro')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)

        # run experiment
        #mlflow.log_params({"taxonomy_type": taxonomy_type})
        mlflow.log_metrics({"macro_f1": macro_score, "micro_f1": micro_score, "precision": precision, "recall": recall}, step= int(prob_threshold*100))    
        
    # end run
    mlflow.end_run()

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

run_zs_threshold()

# COMMAND ----------


