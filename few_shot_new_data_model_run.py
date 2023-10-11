# Databricks notebook source
import pandas as pd
import re
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
# load functions
with open('project_config.json','r') as fp: 
    project_config = json.load(fp)
 
module_path = os.path.join(project_config['project_module_relative_path'])
sys.path.append(module_path)
 
from data_processing import *

pd.set_option("display.max_columns" , 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read in data

# COMMAND ----------

df = pd.read_pickle("./model_training_data.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find which labels are most common

# COMMAND ----------

df_counts = df["themeIdsReviewed"].explode().value_counts()

# COMMAND ----------

df_counts.to_frame().head(8).T

# COMMAND ----------

# MAGIC %md
# MAGIC ## create an identifier to split if row has label or not

# COMMAND ----------

df["themeIdsReviewed"] = df["themeIdsReviewed"].fillna("")

# COMMAND ----------

df["split"] = "unlabeled"
mask = df["themeIdsReviewed"].apply(lambda x: len(x)) > 0
df.loc[mask, "split"] = "labeled"

# COMMAND ----------

df["split"].value_counts()

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

theme_dict.keys()

# COMMAND ----------

all_labels = list(theme_dict.keys())

# COMMAND ----------

mlb = MultiLabelBinarizer()
mlb.fit([all_labels])
# mlb.transform([["Bio-weapon", "Vaccine Side Effects"], ["Home Remedies"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create splits iteratively

# COMMAND ----------

from skmultilearn.model_selection import iterative_train_test_split

def balanced_split(df, test_size=0.5):
    ind = np.expand_dims(np.arange(len(df)), axis=1)
    labels = mlb.transform(df["themeIdsReviewed"])
    ind_train, _, ind_test, _ = iterative_train_test_split(ind, labels,
                                                           test_size)
    return df.iloc[ind_train[:, 0]], df.iloc[ind_test[:,0]]

# COMMAND ----------

from sklearn.model_selection import train_test_split

df_clean = df[["text", "themeIdsReviewed", "split"]].reset_index(drop=True).copy()

# unsupervised set
df_unsup = df_clean.loc[df_clean["split"] == "unlabeled", ["text", "themeIdsReviewed"]]

# supervised set
df_sup = df_clean.loc[df_clean["split"] == "labeled", ["text", "themeIdsReviewed"]]

np.random.seed(0)
df_train, df_tmp = balanced_split(df_sup, test_size=0.5)
df_valid, df_test = balanced_split(df_tmp, test_size=0.5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a dataset so all these splits are in one

# COMMAND ----------

from datasets import Dataset, DatasetDict

ds = DatasetDict({
    "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
    "valid": Dataset.from_pandas(df_valid.reset_index(drop=True)),
    "test": Dataset.from_pandas(df_test.reset_index(drop=True)),
    "unsup": Dataset.from_pandas(df_unsup.reset_index(drop=True))})


# COMMAND ----------

ds

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create training slices to investigate what's the right balance of supervised to unsupervised data needed

# COMMAND ----------

np.random.seed(0)
all_indices = np.expand_dims(list(range(len(ds["train"]))), axis=1)
indices_pool = all_indices
labels = mlb.transform(ds["train"]["themeIdsReviewed"])
train_samples = [8, 16, 32]
train_slices, last_k = [], 0

for i, k in enumerate(train_samples):
    # Split off samples necessary to fill the gap to the next split size
    indices_pool, labels, new_slice, _ = iterative_train_test_split(
        indices_pool, labels, (k-last_k)/len(labels))
    last_k = k
    if i==0: train_slices.append(new_slice)
    else: train_slices.append(np.concatenate((train_slices[-1], new_slice)))


# COMMAND ----------

# Add full dataset as last slice
train_slices.append(all_indices), train_samples.append(len(ds["train"]))


# COMMAND ----------

train_slices = [np.squeeze(train_slice) for train_slice in train_slices]


# COMMAND ----------

print("Target split sizes:")
print(train_samples)
print("Actual split sizes:")
print([len(x) for x in train_slices])

# COMMAND ----------

# MAGIC %md
# MAGIC # Use embeddings as a lookup table

# COMMAND ----------

import torch
from transformers import AutoTokenizer, AutoModel

# COMMAND ----------

model_ckpt = "miguelvictor/python-gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

def mean_pooling(model_output, attention_mask):
    # Extract the token embeddings
    token_embeddings = model_output[0]
    # Compute the attention mask
    input_mask_expanded = (attention_mask
                           .unsqueeze(-1)
                           .expand(token_embeddings.size())
                           .float())
    # Sum the embeddings, but ignore masked tokens
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # Return the average as a single vector
    return sum_embeddings / sum_mask

def embed_text(examples):
    inputs = tokenizer(examples["text"], padding=True, truncation=True,
                       max_length=128, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    pooled_embeds = mean_pooling(model_output, inputs["attention_mask"])
    return {"embedding": pooled_embeds.cpu().numpy()}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get embedding for each split

# COMMAND ----------

tokenizer.pad_token = tokenizer.eos_token


# COMMAND ----------

embs_train = ds["train"].map(embed_text, batched=True, batch_size=16)
embs_valid = ds["valid"].map(embed_text, batched=True, batch_size=16)
embs_test = ds["test"].map(embed_text, batched=True, batch_size=16)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to pickle to save

# COMMAND ----------

import pickle

# COMMAND ----------

train_file = open("amp_embs_train", "ab")
valid_file = open("amp_embs_valid", "ab")
test_file = open("amp_embs_test", "ab")

# COMMAND ----------

pickle.dump(embs_train, train_file)
pickle.dump(embs_valid, valid_file)
pickle.dump(embs_test, test_file)

# COMMAND ----------

embs_train

# COMMAND ----------

# MAGIC %md
# MAGIC ### For when we want to load data back in

# COMMAND ----------

train_file = open("amp_embs_train", "rb")
valid_file = open("amp_embs_valid", "rb")
test_file = open("amp_embs_test", "rb")

# COMMAND ----------

embs_train = pickle.load(train_file)
embs_valid = pickle.load(valid_file)
embs_test = pickle.load(test_file)

# COMMAND ----------

embs_train

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Faiss

# COMMAND ----------

import faiss

# COMMAND ----------

# MAGIC %md
# MAGIC # Add embedding index

# COMMAND ----------

embs_train.add_faiss_index("embedding")


# COMMAND ----------

embs_valid

# COMMAND ----------

# MAGIC %md
# MAGIC ## Remove the fiass embedding so we can compare to NB

# COMMAND ----------


