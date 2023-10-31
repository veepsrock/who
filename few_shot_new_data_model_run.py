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

df["text"] = df["text"].fillna(df["textTranslated.en"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Try taking the first two labels from model predictions and dropping rfi

# COMMAND ----------

df["themeIds"] = df['themeIds'].apply(lambda themes: [theme for theme in themes if theme != 'rfi'] if isinstance(themes, list) else themes)

# COMMAND ----------

df["themeIds"]=df["themeIds"].apply(lambda x: x[:2] if isinstance(x, list) else x)

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

   # "rfi": ["request for help or information", "request for medical explanation"],

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

# MAGIC %md
# MAGIC ### Create training slices to investigate what's the right balance of supervised to unsupervised data needed

# COMMAND ----------

np.random.seed(0)
all_indices = np.expand_dims(list(range(len(ds["train"]))), axis=1)
indices_pool = all_indices
labels = mlb.transform(ds["train"]["themeIdsReviewed"])
train_samples = [8, 16, 32, 64, 128]
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
# MAGIC # Run Naive Bayes baseline
# MAGIC

# COMMAND ----------

from collections import defaultdict

macro_scores, micro_scores = defaultdict(list), defaultdict(list)

def prepare_labels(batch):
    batch["label_ids"] = mlb.transform(batch["themeIdsReviewed"])
    return batch

ds = ds.map(prepare_labels, batched=True)

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

# write to pickle to save
ds_file = open("model_dataset", "ab")
pickle.dump(ds, ds_file)

# COMMAND ----------

# MAGIC %md
# MAGIC # Use embeddings as a lookup table

# COMMAND ----------

import torch
from transformers import AutoTokenizer, AutoModel

# COMMAND ----------

model_ckpt = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
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
# MAGIC
# MAGIC train_file = open("amp_embs_train", "rb")
# MAGIC valid_file = open("amp_embs_valid", "rb")
# MAGIC test_file = open("amp_embs_test", "rb")
# MAGIC
# MAGIC embs_train = pickle.load(train_file)
# MAGIC embs_valid = pickle.load(valid_file)
# MAGIC embs_test = pickle.load(test_file)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Faiss

# COMMAND ----------

import faiss

# COMMAND ----------

embs_train.add_faiss_index("embedding")


# COMMAND ----------

def get_sample_preds(sample, m):
    return (np.sum(sample["label_ids"], axis=0) >= m).astype(int)

def find_best_k_m(ds_train, valid_queries, valid_labels, max_k=17):
    max_k = min(len(ds_train), max_k)
    perf_micro = np.zeros((max_k, max_k))
    perf_macro = np.zeros((max_k, max_k))
    for k in range(1, max_k):
        for m in range(1, k + 1):
            _, samples = ds_train.get_nearest_examples_batch("embedding",
                                                             valid_queries, k=k)
            y_pred = np.array([get_sample_preds(s, m) for s in samples])
            clf_report = classification_report(valid_labels, y_pred,
                target_names=mlb.classes_, zero_division=0, output_dict=True)
            perf_micro[k, m] = clf_report["micro avg"]["f1-score"]
            perf_macro[k, m] = clf_report["macro avg"]["f1-score"]
    return perf_micro, perf_macro

# COMMAND ----------

valid_labels = np.array(embs_valid["label_ids"])
valid_queries = np.array(embs_valid["embedding"], dtype=np.float32)
perf_micro, perf_macro = find_best_k_m(embs_train, valid_queries, valid_labels)

# COMMAND ----------

embs_train.drop_index("embedding")
test_labels = np.array(embs_test["label_ids"])
test_queries = np.array(embs_test["embedding"], dtype=np.float32)

for train_slice in train_slices:
    # Create a Faiss index from training slice
    embs_train_tmp = embs_train.select(train_slice)
    embs_train_tmp.add_faiss_index("embedding")

    # Get best k, m values with validation set
    perf_micro, _ = find_best_k_m(embs_train_tmp, valid_queries, valid_labels)
    k, m = np.unravel_index(perf_micro.argmax(), perf_micro.shape)

    # Get predictions on test set
    _, samples = embs_train_tmp.get_nearest_examples_batch("embedding", test_queries, k = int(k))
    y_pred = np.array([get_sample_preds(s, m) for s in samples])
    # Evaluate predictions
    clf_report = classification_report(test_labels, y_pred,
        target_names=mlb.classes_, zero_division=0, output_dict=True,)
    macro_scores["Embedding"].append(clf_report["macro avg"]["f1-score"])
    micro_scores["Embedding"].append(clf_report["micro avg"]["f1-score"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Plot Performance

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

# COMMAND ----------

plot_metrics(micro_scores, macro_scores, train_samples, "Naive Bayes")


# COMMAND ----------

# MAGIC %md
# MAGIC F1 score:
# MAGIC The F1 score combines both precision and recall into a single metric. It's like looking at both precision and recall at the same time. A high F1 score indicates that the model is both accurate in its predictions and able to capture most of the positive cases.
# MAGIC Micro F1 score:
# MAGIC
# MAGIC This considers the overall performance of the model across all the classes. A low Micro F1 score means the model is not doing well in predicting any of the classes.
# MAGIC Macro F1 score:
# MAGIC
# MAGIC This takes the average of the F1 scores of each class. If the Macro F1 score is low, it means the model is struggling to predict each category correctly.

# COMMAND ----------

macro_scores["Embedding"] = macro_scores["Embedding"][6:]

# COMMAND ----------

micro_scores["Embedding"] = micro_scores["Embedding"][6:]

# COMMAND ----------

# MAGIC %md
# MAGIC # Get scores for zeroshot values

# COMMAND ----------

from sklearn.metrics import f1_score

# COMMAND ----------

audit = pd.read_pickle("./audit_model_training_data.pkl")

# COMMAND ----------

audit.dropna(subset=["themeIds", "themeIdsReviewed"], inplace = True)

# COMMAND ----------

audit.head()

# COMMAND ----------

y_true = mlb.transform(audit["themeIdsReviewed"])
y_pred = mlb.transform(audit["themeIds"])

# COMMAND ----------

macro_scores["Zero Shot"] = [f1_score(y_true, y_pred, average='macro')]*6
micro_scores["Zero Shot"] = [f1_score(y_true, y_pred, average='micro')]*6

# COMMAND ----------

plot_metrics(micro_scores, macro_scores, train_samples, "Naive Bayes")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Run embeddings model on all audit dataset so we can use for model validation later

# COMMAND ----------

embs_train.add_faiss_index("embedding")

# COMMAND ----------

def embed_single_text(text):
    inputs = tokenizer(text, padding=True, truncation=True,
                       max_length=128, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    pooled_embeds = mean_pooling(model_output, inputs["attention_mask"])
    return {"embedding": pooled_embeds.cpu().numpy()}

# COMMAND ----------

def get_embedding_predictions(df):
    result_list=[]
    for i, row in df.iterrows():
        embs_sample = embed_single_text(row["textTranslated.en"])
        scores, sample = embs_train.get_nearest_examples_batch("embedding", embs_sample["embedding"], k = 4)
        embs_sample["embeddingScores"] = scores
        embs_sample["embeddingPredictions"] = sample[0]["themeIdsReviewed"][0]
        embs_sample["textTranslated.en"] = row["textTranslated.en"]
        result_list.append(embs_sample)
    result_df = pd.DataFrame(result_list)
    result_df = pd.merge(df, result_df, on = "textTranslated.en", how = "left")
    return result_df

# COMMAND ----------

embedding_predictions = get_embedding_predictions(audit)

# COMMAND ----------

# write to save
embedding_file = open("embedding_predictions", "ab")
pickle.dump(embedding_predictions, embedding_file)

# COMMAND ----------


