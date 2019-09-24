import re
from collections import Counter
import pandas as pd
import numpy as np
from IPython.display import display, Markdown, HTML
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from ..utils import skills_util


def strip_punctuations(utterance: str):
    """
    function to strip punctuations from the utternace
    :param utterance:
    :return:
    """
    normalization_pattern = '\'s'
    utterance = re.sub(normalization_pattern, ' is', utterance)
    puncuation_pattern = '|'.join(skills_util.PUNCTUATION)
    utterance = re.sub(puncuation_pattern, ' ', utterance)
    return utterance

def _preprocess_chi2(workspace_pd):
    """
    Preprocess dataframe for chi2 analysis
    :param workspace_pd: Preprocess dataframe for chi2
    :return labels: intents processed
    :return count_vectorizer: vectorizer instance
    :return features: features from transform
    """
    stopword_list = skills_util.STOP_WORDS

    workspace_pd['utterance_punc_stripped'] = \
        workspace_pd['utterance'].apply(strip_punctuations)

    count_vectorizer = CountVectorizer(
        min_df=1,
        encoding='utf-8',
        ngram_range=(1, 2),
        stop_words=stopword_list,
        tokenizer=word_tokenize)
    features = count_vectorizer.fit_transform(workspace_pd['utterance_punc_stripped']).toarray()
    labels = workspace_pd['intent']
    return labels, count_vectorizer, features

def _compute_chi2_top_feature(features, labels, vectorizer, cls, significance_level=.05):
    """
    Perform chi2 analysis, punctuation filtering and deduplication
    :param features: count vectorizer features
    :param labels: intents processed
    :param vectorizer: count vectorizer instances
    :param cls: classes for chi square
    :param significance_level: specify an alpha
    :return deduplicated_unigram:
    :return deduplicated_bigram:
    """
    features_chi2, pval = chi2(features, labels == cls)

    feature_names = np.array(vectorizer.get_feature_names())

    features_chi2 = features_chi2[pval < significance_level]
    feature_names = feature_names[pval < significance_level]

    indices = np.argsort(features_chi2)
    feature_names = feature_names[indices]

    unigrams = [v.strip() for v in feature_names if len(v.strip().split()) == 1]
    deduplicated_unigram = list()

    for unigram in unigrams:
        if unigram not in deduplicated_unigram:
            deduplicated_unigram.append(unigram)

    bigrams = [v.strip() for v in feature_names if len(v.strip().split()) == 2]

    deduplicated_bigram = list()
    for bigram in bigrams:
        if bigram not in deduplicated_bigram:
            deduplicated_bigram.append(bigram)

    return deduplicated_unigram, deduplicated_bigram

def get_chi2_analysis(workspace_pd, significance_level=.05):
    """
    find correlated unigram and bigram of each intent with Chi2 analysis
    :param workspace_pd: dataframe, workspace data
    :param signficance_level: float, significance value to reject the null hypothesis
    :return unigram_intent_dict:
    :return bigram_intent_dict:
    """
    labels, vectorizer, features = _preprocess_chi2(workspace_pd)

    label_frequency_dict = dict(Counter(workspace_pd['intent']).most_common())
    N = 5

    # keys are the set of unigrams/bigrams and value will be the intent
    # maps one-to-many relationship between unigram and intent,
    unigram_intent_dict = dict()
    # maps one-to-many relationship between bigram and intent
    bigram_intent_dict = dict()

    classes = list()
    chi_unigrams = list()
    chi_bigrams = list()
    for cls in label_frequency_dict.keys():

        unigrams, bigrams = _compute_chi2_top_feature(
            features,
            labels,
            vectorizer,
            cls,
            significance_level)
        classes.append(cls)

        if unigrams:
            chi_unigrams.append(', '.join(unigrams[-N:]))
        else:
            chi_unigrams.append('None')

        if bigrams:
            chi_bigrams.append(', '.join(bigrams[-N:]))
        else:
            chi_bigrams.append('None')

        if unigrams:
            if frozenset(unigrams[-N:]) in unigram_intent_dict:
                unigram_intent_dict[frozenset(unigrams[-N:])].append(cls)
            else:
                unigram_intent_dict[frozenset(unigrams[-N:])] = list()
                unigram_intent_dict[frozenset(unigrams[-N:])].append(cls)

        if bigrams:
            if frozenset(bigrams[-N:]) in bigram_intent_dict:
                bigram_intent_dict[frozenset(bigrams[-N:])].append(cls)
            else:
                bigram_intent_dict[frozenset(bigrams[-N:])] = list()
                bigram_intent_dict[frozenset(bigrams[-N:])].append(cls)

    chi_df = pd.DataFrame(data={'Intent': classes})
    chi_df['Correlated Unigrams'] = chi_unigrams
    chi_df['Correlated Bigrams'] = chi_bigrams

    display(Markdown(("## Chi-squared Analysis")))
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None, 'display.max_colwidth', 100):
        chi_df.index = np.arange(1, len(chi_df) + 1)
        display(chi_df)
    return unigram_intent_dict, bigram_intent_dict

def get_confusing_key_terms(keyterm_intent_map):
    """
    Greedy search for overlapping intents
    :param keyterm_intent_map: correlated terms
    :return df: ambiguous terms data frame
    """
    ambiguous_intents = list()
    ambiguous_keywords = list()
    intents_seen = list()

    for i in range(len(keyterm_intent_map)):
        correlated_unigrams = list(keyterm_intent_map.keys())[i]
        current_label = keyterm_intent_map[correlated_unigrams]
        intents_seen.append(current_label)

        if len(keyterm_intent_map[correlated_unigrams]) > 1:
            print(keyterm_intent_map[correlated_unigrams])
            print(correlated_unigrams)

        for other_correlated_unigrams in keyterm_intent_map.keys():
            if keyterm_intent_map[other_correlated_unigrams] in intents_seen:
                continue
            overlap = correlated_unigrams.intersection(other_correlated_unigrams)
            if overlap:
                for keyword in overlap:
                    ambiguous_intents.append("<" + current_label[0] + ", " +
                                             keyterm_intent_map[other_correlated_unigrams][0] + ">")
                    ambiguous_keywords.append(keyword)

    df = pd.DataFrame(data={'Intent Pairs': ambiguous_intents, 'Terms': ambiguous_keywords})

    if not ambiguous_intents:
        display(Markdown("There is no ambiguity based on top 5 key terms in chi2 analysis"))
    else:
        display_size = 10
        if not df.empty:
            if len(df) < display_size:
                display_size = len(df)
            display(HTML(df.sample(n=display_size).to_html(index=False)))

    return df

def chi2_overlap_check(ambiguous_unigram_df, ambiguous_bigram_df, intent1, intent2):
    """
    looks for intent overlap for specific intent or intent pairs
    :param ambiguous_unigram_df:
    :param ambiguous_bigram_df:
    :param intent1:
    :param intent2:
    """
    intent = intent1 + ", " + intent2 + "|" + intent2 + ", " + intent1
    part1 = None
    part2 = None
    if not ambiguous_unigram_df.empty:
        part1 = ambiguous_unigram_df[ambiguous_unigram_df['Intent Pairs'].str.contains(intent)]

    if not ambiguous_bigram_df.empty:
        part2 = ambiguous_bigram_df[ambiguous_bigram_df['Intent Pairs'].str.contains(intent)]

    if part1 is not None and part2 is not None:
        display(HTML(pd.concat([part1, part2]).to_html(index=False)))
    elif part1 is not None:
        display(HTML(part1.to_html(index=False)))
    elif part2 is not None:
        display(HTML(part2.to_html(index=False)))
