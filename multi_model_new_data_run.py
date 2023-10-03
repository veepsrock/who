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
    labels = mlb.transform(df["labels"])
    ind_train, _, ind_test, _ = iterative_train_test_split(ind, labels,
                                                           test_size)
    return df.iloc[ind_train[:, 0]], df.iloc[ind_test[:,0]]

# COMMAND ----------

df_clean = df[["text", "labels", "split"]].reset_index(drop=True).copy()

# unsupervised set
df_unsup = df_clean.loc[df_clean["split"] == "unlabeled", ["text", "labels"]]

# supervised set
df_sup = df_clean.loc[df_clean["split"] == "labeled", ["text", "labels"]]

np.random.seed(28)
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
# MAGIC ### Create training slides to investigate what's the right balance of supervised to unsupervised data needed

# COMMAND ----------

np.random.seed(0)
all_indices = np.expand_dims(list(range(len(ds["train"]))), axis=1)
indices_pool = all_indices
labels = mlb.transform(ds["train"]["labels"])
train_samples = [2, 4, 8, 16]
#train_samples = [8, 16, 32, 64, 128]
train_slices, last_k = [], 0

for i, k in enumerate(train_samples):
    # Split off samples necessary to fill the gap to the next split size
    indices_pool, labels, new_slice, _ = iterative_train_test_split(
        indices_pool, labels, (k-last_k)/len(labels))
    last_k = k
    if i==0: train_slices.append(new_slice)
    else: train_slices.append(np.concatenate((train_slices[-1], new_slice)))

# Add full dataset as last slice
train_slices.append(all_indices), train_samples.append(len(ds["train"]))
train_slices = [np.squeeze(train_slice) for train_slice in train_slices]

# COMMAND ----------

print("Target split sizes:")
print(train_samples)
print("Actual split sizes:")
print([len(x) for x in train_slices])

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Naive Bayesline

# COMMAND ----------

def prepare_labels(batch):
    batch["label_ids"] = mlb.transform(batch["labels"])
    return batch

ds = ds.map(prepare_labels, batched=True)

# COMMAND ----------

from collections import defaultdict

macro_scores, micro_scores = defaultdict(list), defaultdict(list)


# COMMAND ----------

## Train baseline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.feature_extraction.text import CountVectorizer

for train_slice in train_slices:
    # Get training slice and test data
    ds_train_sample = ds["train"].select(train_slice)
    y_train = np.array(ds_train_sample["label_ids"])
    y_test = np.array(ds["test"]["label_ids"])
    # Use a simple count vectorizer to encode our texts as token counts
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(ds_train_sample["text"])
    X_test_counts = count_vect.transform(ds["test"]["text"])
    # Create and train our model!
    classifier = BinaryRelevance(classifier=MultinomialNB())
    classifier.fit(X_train_counts, y_train)
    # Generate predictions and evaluate
    y_pred_test = classifier.predict(X_test_counts)
    clf_report = classification_report(
        y_test, y_pred_test, target_names=mlb.classes_, zero_division=0,
        output_dict=True)
    # Store metrics
    macro_scores["Naive Bayes"].append(clf_report["macro avg"]["f1-score"])
    micro_scores["Naive Bayes"].append(clf_report["micro avg"]["f1-score"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot baseline results

# COMMAND ----------

import matplotlib.pyplot as plt

def plot_metrics(micro_scores, macro_scores, sample_sizes, current_model):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for run in micro_scores.keys():
        if run == current_model:
            ax0.plot(sample_sizes, micro_scores[run], label=run, linewidth=2)
            ax1.plot(sample_sizes, macro_scores[run], label=run, linewidth=2)
        else:
            ax0.plot(sample_sizes, micro_scores[run], label=run,
                     linestyle="dashed")
            ax1.plot(sample_sizes, macro_scores[run], label=run,
                     linestyle="dashed")

    ax0.set_title("Micro F1 scores")
    ax1.set_title("Macro F1 scores")
    ax0.set_ylabel("Test set F1 score")
    ax0.legend(loc="lower right")
    for ax in [ax0, ax1]:
        ax.set_xlabel("Number of training samples")
        ax.set_xscale("log")
        ax.set_xticks(sample_sizes)
        ax.set_xticklabels(sample_sizes)
        ax.minorticks_off()
    plt.tight_layout()
    plt.show()

plot_metrics(micro_scores, macro_scores, train_samples, "Naive Bayes")


# COMMAND ----------

# MAGIC %md
# MAGIC # Zero Shot Classification

# COMMAND ----------

# from datasets import Dataset, DatasetDict

# Create an instance of the custom dataset
# ds_zero_shot = Dataset.from_pandas(df[["id", "text", "split", "labels"]].reset_index(drop=True))

# mlb.transform(list(df["themeIds"]))

# COMMAND ----------

from transformers import pipeline

pipe = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Function to predict labels

# COMMAND ----------

def zero_shot_pipeline(example):
    output = pipe(example["text"], all_labels, multi_label=True)
    example["predicted_labels"] = output["labels"]
    example["scores"] = output["scores"]
    return example

ds_zero_shot = ds["valid"].map(zero_shot_pipeline)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get predictions

# COMMAND ----------

def get_preds(example, threshold=.8, topk=None):
    preds = []
    if threshold:
        for label, score in zip(example["predicted_labels"], example["scores"]):
            if score >= threshold:
                preds.append(label)
    elif topk:
        for i in range(topk):
            preds.append(example["predicted_labels"][i])
    else:
        raise ValueError("Set either `threshold` or `topk`.")
    return {"pred_label_ids": list(np.squeeze(mlb.transform([preds])))}

def get_clf_report(ds):
    y_true = np.array(ds["label_ids"])
    y_pred = np.array(ds["pred_label_ids"])
    return classification_report(
        y_true, y_pred, target_names=mlb.classes_, zero_division=0,
        output_dict=True)

# COMMAND ----------

macros, micros = [], []
topks = [1, 2, 3, 4]
for topk in topks:
    ds_zero_shot = ds_zero_shot.map(get_preds, batched=False,
                                    fn_kwargs={'topk': topk})
    clf_report = get_clf_report(ds_zero_shot)
    micros.append(clf_report['micro avg']['f1-score'])
    macros.append(clf_report['macro avg']['f1-score'])

# COMMAND ----------

plt.plot(topks, micros, label='Micro F1')
plt.plot(topks, macros, label='Macro F1')
plt.xlabel("Top-k")
plt.ylabel("F1-score")
plt.legend(loc='best')
plt.show()

# COMMAND ----------

macros, micros = [], []
thresholds = np.linspace(0.01, 1, 100)
for threshold in thresholds:
    ds_zero_shot = ds_zero_shot.map(get_preds,
                                    fn_kwargs={"threshold": threshold})
    clf_report = get_clf_report(ds_zero_shot)
    micros.append(clf_report["micro avg"]["f1-score"])
    macros.append(clf_report["macro avg"]["f1-score"])

# COMMAND ----------

plt.plot(thresholds, micros, label="Micro F1")
plt.plot(thresholds, macros, label="Macro F1")
plt.xlabel("Threshold")
plt.ylabel("F1-score")
plt.legend(loc="best")
plt.show()

# COMMAND ----------

best_t, best_micro = thresholds[np.argmax(micros)], np.max(micros)
print(f'Best threshold (micro): {best_t} with F1-score {best_micro:.2f}.')
best_t, best_macro = thresholds[np.argmax(macros)], np.max(macros)
print(f'Best threshold (micro): {best_t} with F1-score {best_macro:.2f}.')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compare to how this performed to NB

# COMMAND ----------

ds_zero_shot = ds['test'].map(zero_shot_pipeline)
ds_zero_shot = ds_zero_shot.map(get_preds, fn_kwargs={'topk': 1})
clf_report = get_clf_report(ds_zero_shot)
for train_slice in train_slices:
    macro_scores['Zero Shot'].append(clf_report['macro avg']['f1-score'])
    micro_scores['Zero Shot'].append(clf_report['micro avg']['f1-score'])

# COMMAND ----------

plot_metrics(micro_scores, macro_scores, train_samples, "Zero Shot")


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
