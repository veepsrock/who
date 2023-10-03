# Databricks notebook source
import pandas as pd
import re
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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

# Create a list for each row
json_data =[]

# Read in json by each lind
with open('./evidence-10-02-23-split.json') as f:
    for line in f:
        json_data.append(json.loads(line))

# Convert to dataframe
json_df = pd.DataFrame(json_data)

# COMMAND ----------

# Unpack the evidence from its json structure, so we get the text and themeIds, issueIds, id, and status all as its own column
evidence_list = []
for index, row in json_df.iterrows():
    evidence={}
    evidence["id"] = row["_id"]
    evidence["status"]= row["_source"].get("status", "")
    text_translated = row["_source"].get("textTranslated", {})
    evidence["text"] = text_translated.get("en", "")
    evidence["themeIds"] = row["_source"].get("themeIds", np.NaN)
    evidence["issueIds"] = row["_source"].get("issueIds", "")
    evidence_list.append(evidence)

# convert into dataframe
evidence=pd.DataFrame(evidence_list)

# COMMAND ----------

# read in audit data
audit = pd.read_csv("amp_audit.csv")

# clean audit columns
cols_to_clean = ["themeIdsReviewed", "themeIdsSystemFalseNegatives", "themeIdsSystemFalsePositives"]

for column in cols_to_clean:
    audit = string_to_list_column(audit, column)

# COMMAND ----------

# merge with new data
df = pd.merge(evidence, audit, how = "left", on = "id")

# create labels column so I can reuse code from previous
df["labels"] = df["themeIdsReviewed"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find which labels are most common

# COMMAND ----------

df_counts = df["labels"].explode().value_counts()

# COMMAND ----------

df_counts.to_frame().head(8).T

# COMMAND ----------

# MAGIC %md
# MAGIC ## create an identifier to split if row has label or not

# COMMAND ----------

df["labels"] = df["labels"].fillna("")

# COMMAND ----------

df["split"] = "unlabeled"
mask = df["labels"].apply(lambda x: len(x)) > 0
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
    "vaccine-efficacy":["vaccine efficacy", "vaccines"]
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

df_clean = df[["text", "labels", "split"]].reset_index(drop=True).copy()

# unsupervised set
df_unsup = df_clean.loc[df_clean["split"] == "unlabeled", ["text", "labels"]]

# supervised set
df_sup = df_clean.loc[df_clean["split"] == "labeled", ["text", "labels"]]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a dataset so all these splits are in one

# COMMAND ----------

from datasets import Dataset, DatasetDict

ds = DatasetDict({
    "sup": Dataset.from_pandas(df_sup.reset_index(drop=True)),
    "unsup": Dataset.from_pandas(df_unsup.reset_index(drop=True))})


# COMMAND ----------

ds

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create training slides to investigate what's the right balance of supervised to unsupervised data needed

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
# MAGIC ## Figuring out the type for variables

# COMMAND ----------

inputs = tokenizer(ds["sup"]["text"], padding=True, truncation=True,
                       max_length=128, return_tensors="pt")

# COMMAND ----------

type(inputs["attention_mask"])

# COMMAND ----------

with torch.no_grad():
    model_output = model(**inputs)

# COMMAND ----------

type(model_output[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get embedding for each split

# COMMAND ----------

tokenizer.pad_token = tokenizer.eos_token


# COMMAND ----------

embs_train = ds["sup"].map(embed_text, batched=True, batch_size=16)

# COMMAND ----------

embs_test = ds["unsup"].map(embed_text, batched=True, batch_size=16)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to pickle to save

# COMMAND ----------

import pickle

# COMMAND ----------

embs_train_file = open("amp_embs_labels", "ab")
embs_test_file = open("amp_embs_test", "ab")

# COMMAND ----------

embs_train

# COMMAND ----------

# MAGIC %md
# MAGIC ### For when we want to load data back in

# COMMAND ----------

embs_train_file = open("amp_embs_labels", "rb")
embs_test_file = open("amp_embs_test", "rb")

# COMMAND ----------

embs_train = pickle.load(embs_train_file)
embs_test = pickle.load(embs_test_file)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Faiss

# COMMAND ----------

pip install faiss-gpu

# COMMAND ----------

import faiss

# COMMAND ----------

# MAGIC %md
# MAGIC ## Remove the fiass embedding so we can compare to NB

# COMMAND ----------

test_queries = np.array(embs_test["embedding"], dtype=np.float32)

# COMMAND ----------

embs_train.add_faiss_index("embedding")

# COMMAND ----------

_, samples = embs_train.get_nearest_examples_batch("embedding", test_queries, k = 4)

# COMMAND ----------

len(samples)

# COMMAND ----------

len(y_pred)

# COMMAND ----------

samples[0]["text"]

# COMMAND ----------

samples[0]["labels"]

# COMMAND ----------

def get_sample_preds(sample):
    return sample["labels"][0:2]

# COMMAND ----------

y_pred = [get_sample_preds(s) for s in samples]

# COMMAND ----------

y_pred[64]

# COMMAND ----------

predictions = pd.DataFrame({"text": embs_test["text"],
             "themeName": y_pred})

# COMMAND ----------

predictions

# COMMAND ----------

predictions.to_csv("few_shot_predictions.csv", index = False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Try getting prediction for single input

# COMMAND ----------

sample_text = "CBD oil is a cure for COVID-19."

# COMMAND ----------

def embed_single_text(text):
    inputs = tokenizer(text, padding=True, truncation=True,
                       max_length=128, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    pooled_embeds = mean_pooling(model_output, inputs["attention_mask"])
    return {"embedding": pooled_embeds.cpu().numpy()}

# COMMAND ----------

embs_sample = embed_single_text(sample_text)

# COMMAND ----------

scores, sample = embs_train.get_nearest_examples_batch("embedding", embs_sample["embedding"], k = 4)

# COMMAND ----------

sample[0]["labels"]
