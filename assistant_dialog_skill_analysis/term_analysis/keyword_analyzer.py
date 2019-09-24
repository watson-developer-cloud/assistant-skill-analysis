from collections import Counter
import pandas as pd
import numpy as np
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from ..utils import skills_util

def _preprocess_for_heat_map(workspace_df, label_for_display=30,
                             max_token_display=30, class_list=None):
    '''
    Preprocess dataframe for heat map visualization
    :param workspace_df:
    :param label_for_display:
    :param max_token_display:
    :param class_list:
    '''
    label_frequency_dict = dict(Counter(workspace_df['intent']).most_common())
    if class_list:
        workspace_subsampled = workspace_df[workspace_df['intent'].isin(class_list)]
        counts = _get_counts_per_label(workspace_subsampled, unigrams_col_name="unigrams")
    else:
        if len(label_frequency_dict) > label_for_display:
            top_30_labels = list(label_frequency_dict.keys())[:label_for_display]
            workspace_subsampled = workspace_df[workspace_df['intent'].isin(top_30_labels)]
            counts = _get_counts_per_label(workspace_subsampled, unigrams_col_name="unigrams")
        else:
            counts = _get_counts_per_label(workspace_df, unigrams_col_name="unigrams")

    max_n = np.int(np.ceil(max_token_display / len(counts.index.get_level_values(0).unique())))
    top_counts = _get_top_n(counts['n_w'], top_n=max_n)
    return counts, top_counts

def _get_counts_per_label(training_data, unigrams_col_name="unigrams"):
    '''
    Create a new dataframe to store unigram counts for each label
    :param training_data: pandas df
    :param unigrams_col_name: name of unigrams column name
    :return counts: dataframe that contains the counts for all unigrams per label
    '''
    training_data[unigrams_col_name] = training_data['utterance'].apply(nltk.word_tokenize)
    rows = list()
    stopword_list = skills_util.STOP_WORDS
    for row in training_data[['intent', unigrams_col_name]].iterrows():
        r = row[1]
        for word in r.unigrams:
            rows.append((r.intent, word))

    words = pd.DataFrame(rows, columns=['intent', 'word'])
    # delete all empty words and chars
    words = words[words.word.str.len() > 1]
    # delete stopwords
    words = words.loc[~words["word"].isin(stopword_list)]
    # get counts per word
    counts = words.groupby('intent')\
        .word.value_counts()\
        .to_frame()\
        .rename(columns={'word':'n_w'})
    return counts

def _get_top_n(series, top_n=5, index_level=0):
    '''
    Get most frequent words per label
    :param series: product of a call to get_counts_per_label
    :param top_n: integer signifying the number of most frequent tokens per class
    :param index_level: index to group by
    :return df: dataframe that contains the top_n unigrams per label
    '''
    return series\
    .groupby(level=index_level)\
    .nlargest(top_n)\
    .reset_index(level=index_level, drop=True)

def seaborn_heatmap(workspace_df, label_for_display=30, max_token_display=30, class_list=None):
    '''
    Create heat map of word frequencies per intent
    :param workspace_df:
    :param label_for_display:
    :param max_token_display:
    :param class_list:
    '''
    counts, top_counts = _preprocess_for_heat_map(
        workspace_df,
        label_for_display,
        max_token_display,
        class_list)
    reset_groupby = counts.reset_index()
    most_frequent_words = top_counts.reset_index()['word'].unique()
    table_format = reset_groupby.pivot(index="word", columns="intent", values="n_w")
    table_format = table_format[
        table_format.index.isin(most_frequent_words)].fillna(0).astype("int32")
    display(Markdown('## <p style="text-align: center;"> Token Frequency per Intent </p>'))
    fig, ax = plt.subplots(figsize=(20, 20))

    sns.heatmap(table_format, annot=True, fmt='d', linewidths=.1, cmap="PuBu", ax=ax)
    plt.ylabel('Token', fontdict=skills_util.LABEL_FONT)
    plt.xlabel('Intent', fontdict=skills_util.LABEL_FONT)
