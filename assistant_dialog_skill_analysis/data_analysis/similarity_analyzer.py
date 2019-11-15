import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from IPython.display import display, Markdown, HTML


def ambiguous_examples_analysis(workspace_pd, threshold=0.7):
    """
    Analyze the test workspace and find out similar utterances that belongs to different intent
    :param workspace_pd: pandas dataframe in format of [utterance,label]
    :param threshold: cut off for similarity score
    :return: pands dataframe in format of ['Intent1', 'Utterance1', 'Intent2', 'Utterance2',
                                           'similarity score']
    """
    # first create the feature matrix
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    workspace_bow = vectorizer.fit_transform(workspace_pd["utterance"]).todense()
    cos_sim_score_matrix = _calculate_cosine_similarity(workspace_bow)

    # remove the lower triangle of the matrix and apply threshold
    similar_utterance_index = np.argwhere(
        (cos_sim_score_matrix - np.tril(cos_sim_score_matrix)) > threshold
    )
    similar_utterance_pd = pd.DataFrame(
        columns=["Intent1", "Utterance1", "Intent2", "Utterance2", "similarity score"]
    )

    for index in similar_utterance_index:
        if (
            workspace_pd["intent"].iloc[index[0]]
            != workspace_pd["intent"].iloc[index[1]]
        ):
            intent1 = workspace_pd["intent"].iloc[index[0]]
            utterance1 = workspace_pd["utterance"].iloc[index[0]]
            intent2 = workspace_pd["intent"].iloc[index[1]]
            utterance2 = workspace_pd["utterance"].iloc[index[1]]
            score = cos_sim_score_matrix[index[0], index[1]]
            temp_pd = pd.DataFrame(
                {
                    "Intent1": [intent1],
                    "Utterance1": [utterance1],
                    "Intent2": [intent2],
                    "Utterance2": [utterance2],
                    "similarity score": [score],
                }
            )
            similar_utterance_pd = similar_utterance_pd.append(
                temp_pd, ignore_index=True
            )

    if not similar_utterance_pd.empty:
        with pd.option_context("max_colwidth", 250):
            display(
                HTML(
                    similar_utterance_pd.sort_values(
                        by=["similarity score"], ascending=False
                    ).to_html(index=False)
                )
            )
    else:
        display(Markdown("### There are no similar utterances within different Intent"))

    return similar_utterance_pd


def _calculate_cosine_similarity(workspace_bow):
    """
    Given bow representation of the workspace utterance, calculate cosine similarity score
    :param workspace_bow: dense representation of BOW of workspace utterances
    :return: cosine_similarity_matrix
    """
    # normalized and calculate cosine similarity
    workspace_bow = workspace_bow / np.linalg.norm(workspace_bow, axis=1, keepdims=True)
    cosine_similarity_matrix = workspace_bow.dot(np.transpose(workspace_bow))
    return cosine_similarity_matrix
