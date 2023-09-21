# Databricks notebook source
import pandas as pd
pd.set_option('display.max_columns', None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in data

# COMMAND ----------

df = pd.read_csv("amp_audit.csv")

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Calculate false negatives and false positives

# COMMAND ----------

df['falseNegatives'] = df['themeIdsSystemFalseNegatives'].str.split(',')
df['falsePositives'] = df['themeIdsSystemFalsePositives'].str.split(',')

# COMMAND ----------

negatives = df["falseNegatives"].explode().value_counts().to_frame().T
positives = df["falsePositives"].explode().value_counts().to_frame().T

# COMMAND ----------

negatives

# COMMAND ----------

positives

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get scores (we don't have to do this once we have scores in the future)

# COMMAND ----------

from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write functoin to classify text

# COMMAND ----------

theme_labels = ["bioligical weapon", "chemical agent",  "conspiracy",  "nefarious plots",  "corruption",  "economic exploitation",  "profiteering",  "extortion",  "media slant and bias",  "fake news",  "medical exploitation",  "experimental treatments",  "expired medicine",  "guinea pigs",  "request for help or information",  "request for medical explanation",  "disease variants",  "disease genetic modifications",  "alternative cures",  "herbal remedies",  "home remedies",  "healers and healing",  "collective prevention",  "lockdowns",  "travel bans",  "travel restrictions",  "individual prevention",  "non-pharmaceutical interventions",  "quarantine",  "face masks",  "hand washing",  "capacity of public health system (hospitals, doctors, governments, aid)",  "religious belief",  "religious leaders",  "cultural practices",  "pharmaceutical treatment",  "clinical treatment",  "pills",  "vaccine efficacy",  "vaccines",  "vaccine-side-effects", "stigmatization",  "case-reporting",  "symptoms-severity"]

# COMMAND ----------

def classify_themes(df):
    result_list = []
    for index, row in df.iterrows():
        text_sequence = row['textTranslated.en']
        result = classifier(text_sequence, theme_labels, multi_label = True)
        result['predictedThemes'] = [label for label, score in zip(result['labels'], result['scores']) if score > 0.8]
        result['themeScores'] = [score for score in result['scores'] if score > 0.8]
        result_list.append(result)
    result_df = pd.DataFrame(result_list)[['sequence', 'predictedThemes', 'themeScores']]
    return result_df

# COMMAND ----------

# predict themes
themes_df = classify_themes(df)

# COMMAND ----------

themes_df['id'] = df['id']

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

reverse_theme_dict = {value: key for key, values in theme_dict.items() for value in values}


# COMMAND ----------

# Function to replace themes based on the reverse_theme_dict
def replace_themes(themes_list):
    return [reverse_theme_dict.get(theme, theme) for theme in themes_list]

# COMMAND ----------

themes_df

# COMMAND ----------

# Apply the replacement function to the "themes" column
themes_df['modelThemes'] = themes_df['predictedThemes'].apply(replace_themes)


# COMMAND ----------

audit = pd.merge(df, themes_df, how = "left", on = "id")

# COMMAND ----------

audit.columns

# COMMAND ----------

audit_df = audit[["id", "textTranslated.en",  'themeIdsReviewed', "modelThemes",  "themeScores"]]

# COMMAND ----------

audit_df["themeIdsReviewed"] = audit_df['themeIdsReviewed'].str.split(',')

# COMMAND ----------

# Function to calculate false positives
def calculate_false_positives(row):
    reviewed_themes = set(row['themeIdsReviewed']) if isinstance(row['themeIdsReviewed'], list) else set()
    model_themes = set(row['modelThemes']) if isinstance(row['modelThemes'], list) else set()
    false_positives = list(model_themes - reviewed_themes)
    return false_positives

# COMMAND ----------

audit_df['falsePositives'] = audit_df.apply(calculate_false_positives, axis=1)


# COMMAND ----------

audit_df.head()

# COMMAND ----------

# Function to calculate falseScores
def calculate_false_scores(row):
    model_themes = row['modelThemes']
    false_positives = row['falsePositives']
    scores = row['themeScores']

    false_scores = []

    for false_theme in false_positives:
        if false_theme in model_themes:
            index = model_themes.index(false_theme)
            false_scores.append(scores[index])
        else:
            false_scores.append(np.nan)  # Handle cases where the theme is not found

    return false_scores

# COMMAND ----------

audit_df['falseScores'] = audit_df.apply(calculate_false_scores, axis=1)


# COMMAND ----------

positives = audit_df["falsePositives"].explode().value_counts().to_frame().T

# COMMAND ----------

# Create a list to store all "falsePositives" and corresponding "falseScores"
false_data = []

for _, row in audit_df.iterrows():
    false_positives = row['falsePositives']
    false_scores = row['falseScores']

    for i, false_theme in enumerate(false_positives):
        false_data.append((false_theme, false_scores[i]))

# COMMAND ----------

# Create a summary DataFrame from the collected data
summary_df = pd.DataFrame(false_data, columns=['falsePositives', 'falseScores'])

# Calculate the total count and average falseScores for each unique "falsePositives"
summary_table = summary_df.groupby('falsePositives').agg({'falseScores': ['count', 'mean']}).reset_index()

# Rename the columns for clarity
summary_table.columns = ['falsePositives', 'Count', 'Average falseScores']


# COMMAND ----------

summary_table.sort_values(by='Average falseScores', ascending=False)

# COMMAND ----------

audit_df.to_csv("amp_audit_viv.csv", index=False)

# COMMAND ----------


