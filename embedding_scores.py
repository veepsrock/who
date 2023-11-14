# Databricks notebook source
import pandas as pd
import re
import json
import os
import sys
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_scor
e, jaccard_score
from collections import defaultdict
import mlflow
from mlflow.entities.run import Run
from mlflow.tracking.client import MlflowClient, ModelVersion
from mlflow.utils import mlflow_tags
import warnings
pd.set_option("display.max_columns" , 50)

# COMMAND ----------

def mlb_transform(label_list, true_col, pred_col):
    # read in data
    df = pd.read_pickle("./embedding_predictions.pkl")
    df.dropna(subset=["embeddingPredictions", "themeIdsReviewed"], inplace = True)

    mlb = MultiLabelBinarizer()
    mlb.fit([label_list])
    

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

df = pd.read_pickle("./embedding_predictions.pkl")

# COMMAND ----------

def run_embedding_experiment():
    # set up experiment
    mlflow_client = MlflowClient()
    exp_name ="/Users/vpeng@rockfound.org/embedding_scores"
    exp = mlflow_client.get_experiment_by_name(exp_name)
    mlflow.set_experiment(exp_name)
    run_name = "embedding_run"
    parent_run = mlflow.start_run(run_name = run_name)

    # get themes
    with open("theme_dict.json", 'r') as f:
        theme_dict = json.load(f)
    
    # create labels list
    all_labels = list(theme_dict.keys())

    # get scores
    macro_score, micro_score, precision, recall, class_names, jaccard_scores = mlb_transform(all_labels, "themeIdsReviewed", "embeddingPredictions")
    jaccard_dict = dict(zip(class_names, jaccard_scores))

    # run experiment
    metrics = {"macro_f1": macro_score, "micro_f1": micro_score, "precision": precision, "recall": recall}
    metrics.update(jaccard_dict)
    mlflow.log_metrics(metrics)    
    
    # end run
    mlflow.end_run()


# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

run_embedding_experiment()

# COMMAND ----------

# MAGIC %md
# MAGIC # Threshold Experiment

# COMMAND ----------

# get theme threshold
def filter_threshold(df, threshold):
    for index, row in df.iterrows():
        confidences = row["embeddingScores"]
        theme_ids = row["embeddingPredictions"]

        # Ensure lengths are the same
        min_len = min(len(confidences), len(theme_ids))
        filtered_confidences = [confidences[i] if confidences[i] > threshold else None for i in range(min_len)]
        filtered_theme_ids = [theme_ids[i] if confidences[i] > threshold else None for i in range(min_len)]
        df.at[index, "themeConfidence"] = filtered_confidences
        df.at[index, "themeIds"] = filtered_theme_ids

    return df

# COMMAND ----------

test_df = filter_threshold(df, 80)

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


