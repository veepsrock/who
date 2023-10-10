# Databricks notebook source
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
pd.set_option('display.max_columns', None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in data

# COMMAND ----------

df = pd.read_pickle("audit_model_training_data.pkl")

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review RFI Tags

# COMMAND ----------

df.dropna(subset="themeIdsReviewed", inplace = True)

# COMMAND ----------

rfi = df[df["themeIdsReviewed"].str.contains("rfi")]
cr = df[df["themeIdsReviewed"].str.contains("case-reporting")]

# COMMAND ----------

rfi[["textTranslated.en", "themeIdsReviewed", "themeIdsSystemFalseNegatives", "themeIdsSystemFalsePositives"]].to_csv("review_audit_rfi.csv", index=False)
cr[["textTranslated.en", "themeIdsReviewed", "themeIdsSystemFalseNegatives", "themeIdsSystemFalsePositives"]].to_csv("review_audit_cr.csv", index=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Clean data types

# COMMAND ----------

trues = df["themeIdsReviewed"].explode().value_counts().to_frame().reset_index()
negatives = df["themeIdsSystemFalseNegatives"].explode().value_counts().to_frame().reset_index()
positives = df["themeIdsSystemFalsePositives"].explode().value_counts().to_frame().reset_index()

# COMMAND ----------

count_table = pd.merge(trues,positives).merge(negatives).rename(columns={"index": "theme"})
count_table = count_table.melt(id_vars="theme", var_name="predictionType", value_name = "count")

# COMMAND ----------

# Create a bar chart using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(data=count_table, x="count", y="theme", hue = "predictionType", palette=["#008753", "#df4e83", "#aad3df"])
plt.xlabel("Count")
plt.ylabel("Theme")
plt.title("Themes Identifed v. False Predictions")
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability

# Show the plot
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Word count analysis

# COMMAND ----------

# get character length for text
df["wordCount"] = df["textTranslated.en"].apply(lambda x: len(str.split(x)))

# COMMAND ----------

def custom_len(x):
    if isinstance(x, list):
        return len(x)
    return 0

# calculate the total number of false positives per post
df["totalFalsePositives"] = df["themeIdsSystemFalsePositives"].apply(custom_len)
df["totalFalseNegatives"] = df["themeIdsSystemFalseNegatives"].apply(custom_len)
df["totalTrues"] = df['themeIdsReviewed'].apply(custom_len)

# COMMAND ----------

totals = df[["id", "totalTrues", "totalFalsePositives", "totalFalseNegatives"]].melt(id_vars="id", var_name="predictionType", value_name = "totals")

# COMMAND ----------

totals

# COMMAND ----------

sns.displot(totals[totals["totals"]>0], x="totals", hue="predictionType", kind="kde", fill=True, multiple="stack", palette=["#008753", "#df4e83", "#aad3df"])

# COMMAND ----------

totals2 = df[["totalTrues", "totalFalsePositives", "totalFalseNegatives"]]

# COMMAND ----------

totals2

# COMMAND ----------

fig, axs = plt.subplots(ncols=1, nrows=3)

for i, column in enumerate(totals2.columns):
    sns.kdeplot(data =totals2, x=column, ax=axs[i], fill = True)
    median_val = totals2[column].median()
    axs[i].axvline(median_val, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_val}')
    
    # Set x-axis limits
    axs[i].set_ylim(0, 1)
    axs[i].set_xlim(0, 15)
    
    # Add labels and legend
    axs[i].set_xlabel(column)
    axs[i].legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

sns.scatterplot(data=df, x="wordCount", y = "totalFalsePositives")

# COMMAND ----------

sns.scatterplot(data=df, x="wordCount", y = "totalFalseNegatives")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate false scores

# COMMAND ----------

audit_df = df.dropna(subset=['themeConfidence', 'themeIdsSystemFalsePositives'])

# COMMAND ----------

audit_df = audit_df[audit_df["themeConfidence"].apply(lambda x: len(x) > 0)]

# COMMAND ----------

# Function to calculate falseScores
def calculate_false_scores(row):
    model_themes = row['themeIds']
    false_positives = row['themeIdsSystemFalsePositives']
    scores = row['themeConfidence']

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

# Create a list to store all "falsePositives" and corresponding "falseScores"
false_data = []

for _, row in audit_df.iterrows():
    false_positives = row['themeIdsSystemFalsePositives']
    false_scores = row['themeConfidence']

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
plt.title("Total False Positives for Each Theme")
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability

# Show the plot
plt.tight_layout()
plt.show()

# COMMAND ----------

audit_df.to_csv("amp_audit_viv.csv", index=False)

# COMMAND ----------

audit_df = audit_df[audit_df["themeIdsSystemFalsePositives"].apply(lambda x: len(x) > 0)]

# COMMAND ----------

audit_df.head()

# COMMAND ----------

# get the first element of verified theme, fales positive, and scores so that we can make a corre plot
audit_df["themeIdsReviewed_1"] = audit_df["themeIdsReviewed"].apply(lambda x: x[0])

# COMMAND ----------

audit_df["themeIdsSystemFalsePositives_1"] = audit_df["themeIdsSystemFalsePositives"].apply(lambda x: x[0])
audit_df["falseScores_1"] = audit_df["falseScores"].apply(lambda x: x[0]).astype(float)

# COMMAND ----------

corr_plot = audit_df[["themeIdsReviewed_1", "themeIdsSystemFalsePositives_1", "falseScores_1"]]

# COMMAND ----------

df_heatmap = corr_plot.pivot_table(values='falseScores_1',index='themeIdsReviewed_1',columns='themeIdsSystemFalsePositives_1',aggfunc=np.mean)
sns.heatmap(df_heatmap,annot=True, cmap = sns.cm.rocket_r)
plt.show()

# COMMAND ----------

counts = corr_plot.groupby(['themeIdsReviewed_1', 'themeIdsSystemFalsePositives_1']).size().reset_index(name = 'count')
counts = counts.pivot(index = 'themeIdsReviewed_1', columns = 'themeIdsSystemFalsePositives_1', values = 'count')

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

# get character length for text
audit_df["length"] = audit_df["textTranslated.en"].str.len()

# calculate the total number of false positives per post
audit_df["totalFalsePositives"] = audit_df["falsePositives"].apply(lambda x: len(x))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explore relationship with character length and total false positives

# COMMAND ----------

audit_df["totalFalsePositives"].value_counts(sort = False)

# COMMAND ----------

audit_df[audit_df["totalFalsePositives"] == 12]

# COMMAND ----------

audit_df["length"].value_counts(bins=10, sort= False)

# COMMAND ----------

sns.scatterplot(data=audit_df, x="length", y = "totalFalsePositives")
