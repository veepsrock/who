# Databricks notebook source
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in data

# COMMAND ----------

df = pd.read_csv("amp_audit.csv")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Clean data types

# COMMAND ----------

df['falseNegatives'] = df['themeIdsSystemFalseNegatives'].str.split(',')
df['falsePositives'] = df['themeIdsSystemFalsePositives'].str.split(',')


# COMMAND ----------

negatives = df["falseNegatives"].explode().value_counts().to_frame().T
positives = df["falsePositives"].explode().value_counts().to_frame().T

# COMMAND ----------

negatives.T

# COMMAND ----------

positives.T

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
summary_df["falseScores"] = summary_df["falseScores"].astype(float)

# Calculate the total count and average falseScores for each unique "falsePositives"
summary_table = summary_df.groupby('falsePositives').agg({'falseScores': ['count', 'mean']}).reset_index()

# Rename the columns for clarity
summary_table.columns = ['falsePositives', 'Count', 'Average falseScores']


# COMMAND ----------

summary_table.sort_values(by='Average falseScores', ascending=False)

# COMMAND ----------

# Create the scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=summary_table, x="Count", y="Average falseScores")

# Label each point with falsePositives value
for line in range(0, summary_table.shape[0]):
    plt.text(
        summary_table["Count"][line] + 0.5,
        summary_table["Average falseScores"][line],
        summary_table["falsePositives"][line],
        horizontalalignment="left",
        size="medium",
        color="black",
    )

# Set labels and title
plt.xlabel("Count")
plt.ylabel("Average falseScores")
plt.title("Scatterplot with falsePositives Labels")

# Show the plot
plt.show()

# COMMAND ----------

# Sort the DataFrame by the "Count" column in ascending order
summary_table = summary_table.sort_values(by="Count", ascending=False)

# Create a bar chart using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_table, x="Count", y="falsePositives", palette="Blues_d")
plt.xlabel("Count")
plt.ylabel("False Positives")
plt.title("Total False Positives for Each Theme)")
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability

# Show the plot
plt.tight_layout()
plt.show()

# COMMAND ----------

audit_df.to_csv("amp_audit_viv.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read in audit for EDA

# COMMAND ----------

audit_df = pd.read_csv("amp_audit_viv.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean data types

# COMMAND ----------

# Functoin to clean columns

def string_to_list_column(df, column):
    df[column] = df[column].apply(lambda x: x.strip('"').split(','))
    df[column] = df[column].apply(lambda x: [item.strip(']') for item in x])
    df[column] = df[column].apply(lambda x: [item.strip('[') for item in x])
    df[column] = df[column].apply(lambda x: [item.strip("'") for item in x])
    df[column] = df[column].apply(lambda x: [item.strip("  '") for item in x])
    if df[column].dtype == "object":
        df[column] = df[column].apply(lambda x: [str(i) for i in x])
    return df

# COMMAND ----------

audit_df.dropna(subset=["themeIdsReviewed"], inplace = True)

# COMMAND ----------

cols_to_clean = ["themeIdsReviewed", "falsePositives", "falseScores"]

for column in cols_to_clean:
    audit_df = string_to_list_column(audit_df, column)

# COMMAND ----------

# get the first element of verified theme, fales positive, and scores so that we can make a corre plot
audit_df["themeIdsReviewed_1"] = audit_df["themeIdsReviewed"].apply(lambda x: x[0])
audit_df["falsePositives_1"] = audit_df["falsePositives"].apply(lambda x: x[0])
audit_df["falseScores_1"] = audit_df["falseScores"].apply(lambda x: x[0]).astype(float)


# COMMAND ----------

corr_plot = audit_df[["themeIdsReviewed_1", "falsePositives_1", "falseScores_1"]]

# COMMAND ----------

df_heatmap = corr_plot.pivot_table(values='falseScores_1',index='themeIdsReviewed_1',columns='falsePositives_1',aggfunc=np.mean)
sns.heatmap(df_heatmap,annot=True, cmap = sns.cm.rocket_r)
plt.show()

# COMMAND ----------

counts = corr_plot.groupby(['themeIdsReviewed_1', 'falsePositives_1']).size().reset_index(name = 'count')
counts = counts.pivot(index = 'themeIdsReviewed_1', columns = 'falsePositives_1', values = 'count')

# COMMAND ----------

sns.heatmap(counts,  annot = True, fmt = '.0f', cmap = sns.cm.rocket_r)

# COMMAND ----------

corr_plot["falsePositives_1"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Looking at character length for posts

# COMMAND ----------

# get character length for text
audit_df["length"] = audit_df["textTranslated.en"].str.len()

# COMMAND ----------

# calculate the total number of false positives per post
audit_df["totalFalsePositives"] = audit_df["falsePositives"].apply(lambda x: len(x))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explore relationship with character length and total false positives

# COMMAND ----------

audit_df["length"].value_counts(bins=10, sort= False)

# COMMAND ----------

sns.scatterplot(data=audit_df, x="length", y = "totalFalsePositives")
