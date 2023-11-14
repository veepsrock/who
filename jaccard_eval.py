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
df= pd.read_pickle("./embedding_predictions")  
df["text"]= df["textTranslated.en"]


# COMMAND ----------

df.head()

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

df["themeIdsReviewed"] = df["themeIdsReviewed"].fillna("")
df["themeIds"] = df["themeIds"].fillna("")

# COMMAND ----------

df["label_ids"] = mlb.transform(list(df["themeIdsReviewed"])).tolist()
df["pred_label_ids"] = mlb.transform(list(df["themeIds"])).tolist()

# COMMAND ----------

from datasets import Dataset, DatasetDict

# Create an instance of the custom dataset
ds_zero_shot = Dataset.from_pandas(audit[["id", "textTranslated.en", "label_ids", "pred_label_ids"]].reset_index(drop=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate Jaccard SCore

# COMMAND ----------

y_true = mlb.transform(list(df["themeIdsReviewed"]))
y_pred = mlb.transform(list(df["themeIds"]))

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

class_names = mlb.classes_

# COMMAND ----------

class_names

# COMMAND ----------

scores = dict(zip(class_names, jaccard_scores))

# COMMAND ----------

new_scores = {"recall": 2}

# COMMAND ----------

scores.update(new_scores)

# COMMAND ----------

scores.update({"macro_f1": 1}, {"macro_f2": 3})

# COMMAND ----------

scores

# COMMAND ----------

["c-" + class_name for class_name in class_names]

# COMMAND ----------

# Extract and split long class names by "-"
class_names = mlb.classes_
split_class_names = [name.split('-') for name in class_names]

# Plot Jaccard Index for each label
plt.figure(figsize=(8, 4))
plt.bar(mlb.classes_, jaccard_scores, color='skyblue')
plt.xlabel('Labels')
plt.ylabel('Jaccard Index')
plt.ylim([0,1])
plt.title('Jaccard Index for Each Label')
plt.xticks(range(len(class_names)), ['\n'.join(name) for name in split_class_names], rotation=90)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Breakdown for each theme

# COMMAND ----------

trues = df["themeIdsReviewed"].explode().value_counts().to_frame().reset_index()
negatives = df["themeIdsSystemFalseNegatives"].explode().value_counts().to_frame().reset_index()
positives = df["themeIdsSystemFalsePositives"].explode().value_counts().to_frame().reset_index()

# COMMAND ----------

count_table = pd.merge(trues,positives).merge(negatives).rename(columns={"index": "theme"})
count_table = count_table.melt(id_vars="theme", var_name="predictionType", value_name = "count")

# COMMAND ----------

count_table.shape

# COMMAND ----------

count_table = count_table[count_table["theme"]!= "rfi"]
count_table = count_table[count_table["theme"]!= "case-reporting"]

# COMMAND ----------

count_table.shape

# COMMAND ----------

# Create a bar chart using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(data=count_table, x="count", y="theme", hue = "predictionType", palette=["#008753", "#df4e83", "#aad3df"])
plt.xlabel("Count")
plt.ylabel("Theme")
plt.title("False Positives v. False Negatives")
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability

# Show the plot
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Run jaccard for embeddings

# COMMAND ----------

df.head()

# COMMAND ----------

y_true_embs = mlb.transform(list(df["themeIdsReviewed"]))
y_pred_embs = mlb.transform(list(df["embeddingPredictions"]))

# COMMAND ----------

jaccard_scores_embs = jaccard_score(y_true_embs,y_pred_embs, average =None)

# COMMAND ----------

# Extract and split long class names by "-"
class_names = mlb.classes_
split_class_names = [name.split('-') for name in class_names]

# Plot Jaccard Index for each label
plt.figure(figsize=(8, 4))
plt.bar(mlb.classes_, jaccard_scores_embs, color='skyblue')
plt.xlabel('Labels')
plt.ylabel('Jaccard Index')
plt.title('Jaccard Index for Each Label')
plt.ylim([0,1])
plt.xticks(range(len(class_names)), ['\n'.join(name) for name in split_class_names], rotation=90)
plt.show()


# COMMAND ----------


