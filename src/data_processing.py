import pandas as pd
import re


# Functoin to clean columns
def string_to_list_column(df, column):
    df[column] = df[column].apply(lambda x: x if not isinstance(x, str) else x.strip('"').split(','))
    df[column] = df[column].apply(lambda x: [item.strip(']') for item in x] if isinstance(x, list) else x)
    df[column] = df[column].apply(lambda x: [item.strip('[') for item in x] if isinstance(x, list) else x)
    df[column] = df[column].apply(lambda x: [item.strip("'") for item in x] if isinstance(x, list) else x)
    df[column] = df[column].apply(lambda x: [item.strip("  '") for item in x] if isinstance(x, list) else x)
    if df[column].dtype == "object":
        df[column] = df[column].apply(lambda x: [str(i) for i in x] if isinstance(x, list) else x)
    return df

# function to clean text
def clean_text(df, text):
    """
    Clean text column
    df = dataframe
    text (string) = column name containing text
    """
    # lowercase text
    df[text] = df[text].str.lower()
 
    # remove URLs
    df[text] = df[text].map(lambda x: re.sub('http[s]?:\/\/[^\s]*', ' ', x))
 
    # remove URL cutoffs
    df[text] = df[text].map(lambda x: re.sub('\\[^\s]*', ' ', x))
 
    # remove spaces
    df[text] = df[text].map(lambda x: re.sub('\n', ' ', x))
 
    # remove picture URLs
    df[text] = df[text].map(lambda x: re.sub('pic.twitter.com\/[^\s]*', ' ', x))
 
    # remove blog/map type
    df[text] = df[text].map(lambda x: re.sub('blog\/maps\/info\/[^\s]*', ' ', x))
 
    # remove hashtags =
    df[text] = df[text].map(lambda x: re.sub("\#[\w]*", "", x))
 
    # remove and signs
    df[text] = df[text].map(lambda x: re.sub("\&amp;", "", x))
 
    # remove single quotations
    df[text] = df[text].map(lambda x: re.sub("'", "", x))
    df[text] = df[text].map(lambda x: re.sub("'", "", x))
 
    # remove characters that are not word characters or digits
    df[text] = df[text].map(lambda x: re.sub("[^\w\d]", " ", x))
 
    # remove all characters that are not letters
    #df[text] = df[text].map(lambda x: re.sub("[^a-zA-Z]", " ", x))
 
    # remove multiple spaces
    df[text] = df[text].map(lambda x: re.sub("\s{2,6}", " ", x))
 
    # drop duplicate rows
    #df.drop_duplicates(subset='text', keep='first', inplace=True)
 
    # remove multiple spaces
    df[text] = df[text].map(lambda x: re.sub("\s{3,20}", "", x))
 
    return df

# function to map to parent themes
def map_themes(themes):
    if isinstance(themes, list):
        big_themes = []
        for theme in themes:
            for key, values in theme_dict.items():
                if theme in values:
                    big_themes.append(key)
                    break
        return big_themes if big_themes else None
    return None