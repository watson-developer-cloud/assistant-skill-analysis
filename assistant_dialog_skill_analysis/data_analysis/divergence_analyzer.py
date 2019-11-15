from collections import Counter
from IPython.display import Markdown, display
import numpy as np
import pandas as pd
from scipy.spatial import distance
from nltk import word_tokenize


def _label_percentage(data_frame):
    """
    Calculate the percentage of each labels in the data frame
    :param data_frame: dataframe for train or test
    :return: label_percentage_dict: dictionary maps label : % of labels
    """
    total_examples = len(data_frame)
    label_frequency_dict = dict(Counter(data_frame["intent"]).most_common())
    percentage_list = np.array(list(label_frequency_dict.values())) / total_examples
    label_percentage_dict = dict(
        zip(list(label_frequency_dict.keys()), percentage_list)
    )
    return label_percentage_dict


def _train_test_coloring(val):
    """
    color scheme for train test difference statistics
    :param val:
    :return:
    """
    if val > 25:
        color = "red"
    elif val > 10:
        color = "DarkBlue"
    else:
        color = "green"
    return "color: %s" % color


def _train_test_label_difference(
    workspace_label_percentage_dict, test_label_percentage_dict
):
    """
    analyze the difference between training set and test set
    :param workspace_label_percentage_dict:
    :param test_label_percentage_dict:
    :return:
    missing_label: list of labels that are missing in the test set
    difference_dict: dictionary that maps intent:percentage difference
    js_distance: jensen-shannon distance between train and test label percentages
    """
    difference_dict = dict()
    missing_label = list()
    distribution1 = list()
    distribution2 = list()

    for key in workspace_label_percentage_dict:
        workspace_percentage = workspace_label_percentage_dict[key]
        distribution1.append(workspace_percentage)
        if key in test_label_percentage_dict:

            test_percentage = test_label_percentage_dict[key]

            distribution2.append(test_percentage)
        else:
            missing_label.append(key)
            test_percentage = 0
            distribution2.append(test_percentage)

        # L1 dist
        current_difference = np.abs(test_percentage - workspace_percentage)

        if key in test_label_percentage_dict:
            difference_dict[key] = [
                workspace_percentage * 100,
                test_percentage * 100,
                current_difference * 100,
            ]

    js_distance = distance.jensenshannon(distribution1, distribution2, 2.0)

    return missing_label, difference_dict, js_distance


def _train_test_vocab_difference(train_set_pd, test_set_pd):
    """
    Analyze the training set and test set and retrieve the vocabulary of each set
    :param train_set_pd:
    :param test_set_pd:
    :return:
    train vocab: the set that contains the vocabulary of training set
    test vocab: the set that contains the vocabulary of test set
    """
    train_vocab = set()
    test_vocab = set()
    train_set_tokens = train_set_pd["utterance"].apply(word_tokenize)
    test_set_tokens = test_set_pd["utterance"].apply(word_tokenize)

    for tokens in train_set_tokens.tolist():
        train_vocab.update(tokens)

    for tokens in test_set_tokens.tolist():
        test_vocab.update(tokens)

    return train_vocab, test_vocab


def _train_test_utterance_length_difference(train_set_pd, test_set_pd):
    """
    Analyze difference in length of utterance of training set and test set per label
    :param train_set_pd:
    :param test_set_pd:
    :return:
    train_test_legnth_comparison: pandas dataframe [Intent, Absolute Difference]
    """
    train_pd_temp = train_set_pd.copy()
    train_pd_temp["tokens"] = train_set_pd["utterance"].apply(word_tokenize)
    train_pd_temp["Train"] = train_pd_temp["tokens"].apply(len)
    train_avg_len_by_label = train_pd_temp[["intent", "Train"]].groupby("intent").mean()

    test_pd_temp = test_set_pd.copy()
    test_pd_temp["tokens"] = test_set_pd["utterance"].apply(word_tokenize)
    test_pd_temp["Test"] = test_pd_temp["tokens"].apply(len)
    test_avg_len_by_label = test_pd_temp[["intent", "Test"]].groupby("intent").mean()

    train_test_length_comparison = pd.merge(
        train_avg_len_by_label, test_avg_len_by_label, on="intent"
    )
    train_test_length_comparison["Absolute Difference"] = np.abs(
        train_test_length_comparison["Train"] - train_test_length_comparison["Test"]
    )
    train_test_length_comparison = train_test_length_comparison.sort_values(
        by=["Absolute Difference"], ascending=False
    )
    train_test_length_comparison = train_test_length_comparison.reset_index()
    train_test_length_comparison.rename(columns={"intent": "Intent"}, inplace=True)
    return train_test_length_comparison


def _get_metrics(results):
    """
    compute the metrics of precision, recall and f1 per label
    :param results: inference results of the test set
    :return:
    precision_dict: maps the {intent: precision}
    recall_dict: maps the {intent: recall}
    f1_dict: maps the {intent:f1}
    """
    groundtruth = results["correct_intent"].values.tolist()
    top_intent = results["top_intent"].values.tolist()
    gt_cnt_dict = dict()
    pred_cnt_dict = dict()
    true_positive_dict = dict()
    for gt, pred in zip(groundtruth, top_intent):
        gt_cnt_dict[gt] = gt_cnt_dict.get(gt, 0) + 1
        pred_cnt_dict[pred] = pred_cnt_dict.get(pred, 0) + 1
        if gt == pred:
            true_positive_dict[pred] = true_positive_dict.get(pred, 0) + 1
    precision_dict = dict()
    recall_dict = dict()
    f1_dict = dict()
    for lb in true_positive_dict:

        recall_dict[lb] = (
            true_positive_dict[lb] / gt_cnt_dict[lb] if lb in gt_cnt_dict else 0
        )

        precision_dict[lb] = (
            true_positive_dict[lb] / pred_cnt_dict[lb] if lb in pred_cnt_dict else 0
        )

        f1_dict[lb] = (
            0.0
            if recall_dict[lb] == 0 and precision_dict[lb] == 0
            else 2.0
            * recall_dict[lb]
            * precision_dict[lb]
            / (recall_dict[lb] + precision_dict[lb])
        )
    return precision_dict, recall_dict, f1_dict


def analyze_train_test_diff(train_set_pd, test_set_pd, results):
    """
    analyze the difference between training set and test set and generate visualizations
    :param train_set_pd:
    :param test_set_pd:
    :param results:
    """
    workspace_label_percentage_dict = _label_percentage(train_set_pd)
    test_label_percentage_dict = _label_percentage(test_set_pd)

    missing_label, difference_dict, js = _train_test_label_difference(
        workspace_label_percentage_dict, test_label_percentage_dict
    )
    train_vocab, test_vocab = _train_test_vocab_difference(train_set_pd, test_set_pd)

    train_test_length_comparison_pd = _train_test_utterance_length_difference(
        train_set_pd, test_set_pd
    )

    display(Markdown("## Test Data Evaluation"))

    if difference_dict:

        label = list(difference_dict.keys())
        diff = np.round(list(difference_dict.values()), 2)
        precision_dict, recall_dict, f1_dict = _get_metrics(results)
        precision = np.round(
            [precision_dict[l] * 100.0 if l in precision_dict else 0.0 for l in label],
            2,
        )

        recall = np.round(
            [recall_dict[l] * 100.0 if l in recall_dict else 0.0 for l in label], 2
        )

        f1 = np.round([f1_dict[l] * 100.0 if l in f1_dict else 0.0 for l in label], 2)

        train_count_dict = dict(Counter(train_set_pd["intent"]))
        test_count_dict = dict(Counter(test_set_pd["intent"]))
        tr_cnt = [train_count_dict[l] if l in train_count_dict else 0.0 for l in label]
        te_cnt = [test_count_dict[l] if l in test_count_dict else 0.0 for l in label]

        difference_pd = pd.DataFrame(
            {
                "Intent": label,
                "% of Train": diff[:, 0],
                "% of Test": diff[:, 1],
                "Absolute Difference %": diff[:, 2],
                "Train Examples": tr_cnt,
                "Test Examples": te_cnt,
                "Test Precision %": precision,
                "Test Recall %": recall,
                "Test F1 %": f1,
            }
        )

        if not difference_pd[difference_pd["Absolute Difference %"] > 0.001].empty:
            table_for_display = difference_pd[
                difference_pd["Absolute Difference %"] > 0.001
            ].sort_values(by=["Absolute Difference %"], ascending=False)
            table_for_display = table_for_display.style.applymap(
                _train_test_coloring, subset=pd.IndexSlice[:, ["Absolute Difference %"]]
            )
            display(table_for_display)
            display(Markdown("\n"))
            display(Markdown("Distribution Mismatch Color Code"))
            display(Markdown("<font color = 'red'>      Red - Severe </font>"))
            display(Markdown("<font color = 'blue'>     Blue - Caution </font>"))
            display(Markdown("<font color = 'green'>    Green - Good </font>"))

    if js >= 0:
        js = np.round(js, 2) * 100
        display(
            Markdown(
                "### Data Distribution Divergence Test vs Train \
        <font color='blue'>{}%</font>".format(
                    js
                )
            )
        )
        display(Markdown("**Note** Metric used is Jensen Shannon Distance"))

    if missing_label:
        display(Markdown("### Missing Intents in Test Data"))
        missing_label_pd = pd.DataFrame(
            missing_label, columns=["Missing Intents in Test Set "]
        )
        missing_label_pd.index = np.arange(1, len(missing_label_pd) + 1)
        display(missing_label_pd)

    display(Markdown("### Test Data Example Length"))
    condition1 = (
        train_test_length_comparison_pd["Absolute Difference"]
        / train_test_length_comparison_pd["Train"]
        > 0.3
    )
    condition2 = train_test_length_comparison_pd["Absolute Difference"] > 3

    length_comparison_pd = train_test_length_comparison_pd[condition1 & condition2]

    if not length_comparison_pd.empty:
        display(
            Markdown(
                "Divergence found in average length of user examples in test vs training data"
            )
        )
        length_comparison_pd.index = np.arange(1, len(length_comparison_pd) + 1)
        display(length_comparison_pd.round(2))
    else:
        display(Markdown("Average length of user examples is comparable"))

    if train_vocab and test_vocab:
        display(Markdown("### Vocabulary Size Test vs Train"))
        oov_vocab_percentage = (
            (len(test_vocab) - len(train_vocab.intersection(test_vocab)))
            / len(test_vocab)
            * 100
        )

        vocab_df = pd.DataFrame(
            data={
                "Train Vocabulary Size": [len(train_vocab)],
                "Test Vocabulary Size": [len(test_vocab)],
                "% Test Set Vocabulary not found in Train": [oov_vocab_percentage],
            }
        )
        vocab_df.index = np.arange(1, len(vocab_df) + 1)
        display(vocab_df.round(2))

    display(Markdown("   "))
