# Databricks notebook source
import pandas as pd
import re
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from collections import defaultdict
pd.set_option("display.max_columns" , 50)

# COMMAND ----------

# MAGIC %md
# MAGIC # Read in data

# COMMAND ----------

df = pd.read_pickle("./model_training_data.pkl")

# COMMAND ----------

df = df[df["split"]== "labeled"]

# COMMAND ----------

df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # Check for undefined themes

# COMMAND ----------

df[df["themeIdsReviewedParent"].isna()]

# COMMAND ----------

df[df['themeIdsReviewedParent'].apply(len) == 0]

# COMMAND ----------

df.dropna(subset=["themeIds", "themeIdsReviewed"], inplace = True)

# COMMAND ----------

df.dropna(subset=["themeIdsParent", "themeIdsReviewedParent"], inplace = True)

# COMMAND ----------

theme_dict = {
    "conspiracy-corruption": ["bioweapon", "conspiracy", "corruption", "media-bias", "medical-exploitation"],
    "illness-cause": ["stigmatization", "case-reporting", "symptoms-severity", "variants"],
    "intervention-capacity":["capacity"],
    "prevention-treatment-alternative": ["alternative-cures"],
    "prevention-treatment-approved": ["alternative-cures","prevention-collective", "prevention-individual", "treatment", "vaccine-efficacy", "vaccine-side-effects"]
}

# COMMAND ----------

all_labels = list(theme_dict.keys())

# COMMAND ----------

# MAGIC %md
# MAGIC # Transform

# COMMAND ----------

mlb = MultiLabelBinarizer()
mlb.fit([all_labels])

# COMMAND ----------

y_true = mlb.transform(df["themeIdsReviewedParent"])
y_pred = mlb.transform(df["themeIdsParent"])

# COMMAND ----------

# MAGIC %md 
# MAGIC # Perform for original

# COMMAND ----------

theme_dict_child= {
    "bioweapon":["bioligical weapon", "chemical agent"],
    "conspiracy": ["conspiracy", "nefarious plots"],
    "corruption": ["corruption", "economic exploitation", "profiteering","extortion"],
    "media-bias": ["media slant and bias", "fake news"],
    "medical-exploitation": ["medical exploitation", "experimental treatments", "expired medicine", "guinea pigs"],
    "variants": ["disease variants", "disease genetic modifications"],
    "alternative-cures": ["alternative cures", "herbal remedies", "home remedies", "healers and healing"],
    "prevention-collective":["collective prevention", "lockdowns", "travel bans", "travel restrictions"],
    "prevention-individual":["individual prevention", "non-pharmaceutical interventions", "quarantine", "face masks", "hand washing"],
    "capacity":["capacity of public health system (hospitals, doctors, governments, aid)"],
    "religious-practices":["religious belief", "religious leaders", "cultural practices"],
    "treatment":["pharmaceutical treatment", "clinical treatment", "pills"],
    "vaccine-efficacy":["vaccine efficacy", "vaccines"],
    'case-reporting':[],
    'stigmatization':[],
    'symptoms-severity': [], 
    'vaccine-side-effects':[]
}

# COMMAND ----------

all_labels_child = list(theme_dict_child.keys())

# COMMAND ----------

mlb_child = MultiLabelBinarizer()
mlb_child.fit([all_labels_child])

# COMMAND ----------

df.dropna(subset=["themeIds", "themeIdsReviewed"], inplace = True)

# COMMAND ----------



# COMMAND ----------

y_true_child = mlb_child.transform(df["themeIdsReviewed"])
y_pred_child = mlb_child.transform(df["themeIds"])

# COMMAND ----------

macro_scores, micro_scores = defaultdict(list), defaultdict(list)


# COMMAND ----------

macro_scores["Zero Shot Parent"] = [f1_score(y_true, y_pred, average='macro')]
micro_scores["Zero Shot Parent"] = [f1_score(y_true, y_pred, average='micro')]

# COMMAND ----------

macro_scores["Zero Shot Child"] = [f1_score(y_true_child, y_pred_child, average='macro')]
micro_scores["Zero Shot Child"] = [f1_score(y_true_child, y_pred_child, average='micro')]

# COMMAND ----------

pd.DataFrame(macro_scores).T

# COMMAND ----------

pd.DataFrame(micro_scores).T

# COMMAND ----------


