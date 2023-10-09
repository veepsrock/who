# Databricks notebook source
import pandas as pd
import re
import json
import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer

pd.set_option("display.max_columns" , 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read in data

# COMMAND ----------

# read in audit data
audit = pd.read_pickle("audit_model_training_data.pkl")

# read in eval data
df= pd.read_pickle("./model_training_data.pkl")  


# COMMAND ----------

# MAGIC %md
# MAGIC ## Find which labels are most common

# COMMAND ----------

df_counts = df["themeIdsReviewed"].explode().value_counts()

# COMMAND ----------

df_counts.to_frame().head(8).T

# COMMAND ----------

# MAGIC %md
# MAGIC ### drop duplicates

# COMMAND ----------

len_before = len(df)
df = df.drop_duplicates(subset = "text")
print(f"Removed {(len_before - len(df))/len_before:.2%} duplicates.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create training sets

# COMMAND ----------

from sklearn.preprocessing import MultiLabelBinarizer

# COMMAND ----------

# MAGIC %md
# MAGIC MultiLabelBinarizer takes a list of label names and creates a vector with zeros for absent labels and ones for present labels. We can test this by fitting MultiLabelBinarizer on all_labels to learn the mapping from label name to ID

# COMMAND ----------

theme_dict= {
    "bioweapon":["bioligical weapon", "chemical agent"],
    "conspiracy": ["conspiracy", "nefarious plots"],
    "corruption": ["corruption", "economic exploitation", "profiteering","extortion"],
    "media-bias": ["media slant and bias", "fake news"],
    "medical-exploitation": ["medical exploitation", "experimental treatments", "expired medicine", "guinea pigs"],
    "rfi": ["request for help or information", "request for medical explanation"],
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

all_labels = list(theme_dict.keys())

# COMMAND ----------

mlb = MultiLabelBinarizer()
mlb.fit([all_labels])
# mlb.transform([["Bio-weapon", "Vaccine Side Effects"], ["Home Remedies"]])

# COMMAND ----------

audit["themeIdsReviewed"] = audit["themeIdsReviewed"].fillna("")

# COMMAND ----------

audit["label_ids"] = mlb.transform(list(audit["themeIdsReviewed"])).tolist()
audit["pred_label_ids"] = mlb.transform(list(audit["themeIdsSystem"])).tolist()

# COMMAND ----------

from datasets import Dataset, DatasetDict

# Create an instance of the custom dataset
ds_zero_shot = Dataset.from_pandas(audit[["id", "textTranslated.en", "label_ids", "pred_label_ids"]].reset_index(drop=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate Jaccard SCore

# COMMAND ----------

y_true = mlb.transform(list(audit["themeIdsReviewed"]))
y_pred = mlb.transform(list(audit["themeIdsSystem"]))

# COMMAND ----------

jaccard_scores = jaccard_score(y_true,y_pred, average =None)

# COMMAND ----------

jaccard_scores

# COMMAND ----------

from sklearn.metrics import multilabel_confusion_matrix
# Create confusion matrix
confusion_matrix = multilabel_confusion_matrix(y_true,y_pred)

# COMMAND ----------

# Plot confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix[0], annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Label Bioweapon')
plt.show()


# COMMAND ----------

# Extract and split long class names by "-"
class_names = mlb.classes_
split_class_names = [name.split('-') for name in class_names]

# Plot Jaccard Index for each label
plt.figure(figsize=(8, 4))
plt.bar(mlb.classes_, jaccard_scores, color='skyblue')
plt.xlabel('Labels')
plt.ylabel('Jaccard Index')
plt.title('Jaccard Index for Each Label')
plt.xticks(range(len(class_names)), ['\n'.join(name) for name in split_class_names], rotation=90)
plt.show()

