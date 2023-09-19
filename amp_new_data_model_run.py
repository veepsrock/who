# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
pd.set_option("display.max_columns" , 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read in data

# COMMAND ----------

df = pd.read_csv('amp.csv')

# COMMAND ----------

df['labels'] = df['themeIds']

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

all_labels = list(df["themeName"].unique())

# COMMAND ----------

for theme in list(df["manual_themeName"].unique()):
    if theme not in all_labels:
        all_labels.append(theme)

# COMMAND ----------

all_labels = [x for x in all_labels if str(x) != 'nan']

# COMMAND ----------

mlb = MultiLabelBinarizer()
mlb.fit([all_labels])
mlb.transform([["Bio-weapon", "Vaccine Side Effects"], ["Home Remedies"]])

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
