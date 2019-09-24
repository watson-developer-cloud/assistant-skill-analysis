import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from ..utils.skills_util import OFFTOPIC_LABEL

OFFTOPIC_CNT_THRESHOLD_FOR_DISPLAY = 5


def abnormal_conf(full_results, correct_thresh, incorrect_thresh):
    """
    perform abnormal confidence analysis on prediction results on the test set
    :param full_results:
    :param correct_thresh:
    :param incorrect_thresh:
    :return:
    """
    test_pd = pd.DataFrame(full_results)
    test_pd = test_pd.loc[~(test_pd['correct_intent'] == 'SYSTEM_OUT_OF_DOMAIN')]
    correct = test_pd.loc[test_pd['correct_intent'] == test_pd['top_intent']]

    correct_low_conf = correct.loc[correct['top_confidence'] < correct_thresh]
    correct_low_conf = correct_low_conf[
        ['correct_intent',	'utterance', 'top_confidence', 'top_intent']]

    incorrect = test_pd.loc[~(test_pd['correct_intent'] == test_pd['top_intent'])]
    incorrect_high_conf = incorrect.loc[incorrect['top_confidence'] > incorrect_thresh]

    top1 = list()
    top2 = list()
    top3 = list()

    for i in range(len(incorrect_high_conf)):
        possible_range = len(incorrect_high_conf.iloc[i, :]['top_predicts'])

        for j in range(3):
            if j == 0:
                if possible_range >= 1:
                    top1.append(incorrect_high_conf.iloc[i, :]['top_predicts'][j]['intent'] + ' ' +
                                '(' + str(np.round(incorrect_high_conf.iloc[i, :]['top_predicts'][j]
                                                   ['confidence'], 3)) + ')')
                else:
                    top1.append('NA')
            if j == 1:
                if possible_range >= 2:
                    top2.append(incorrect_high_conf.iloc[i, :]['top_predicts'][j]['intent'] + ' ' +
                                '(' + str(np.round(incorrect_high_conf.iloc[i, :]['top_predicts'][j]
                                                   ['confidence'], 3)) + ')')
                else:
                    top2.append('NA')
            if j == 2:
                if possible_range >= 3:
                    top3.append(incorrect_high_conf.iloc[i, :]['top_predicts'][j]['intent'] + ' ' +
                                '(' + str(np.round(incorrect_high_conf.iloc[i, :]['top_predicts'][j]
                                                   ['confidence'], 3)) + ')')
                else:
                    top3.append('NA')

    incorrect_high_conf['top1_prediction'] = top1
    incorrect_high_conf['top2_prediction'] = top2
    incorrect_high_conf['top3_prediction'] = top3
    incorrect_high_conf = incorrect_high_conf[
        ['correct_intent', 'utterance', 'top1_prediction', 'top2_prediction', 'top3_prediction']]

    return correct_low_conf, incorrect_high_conf


def analysis(results, intent_list=None):
    """
    perform confidence analysis at the overall level or per intent basis
    :param results:
    :param intent_list:
    :return:
    """

    if not intent_list:
        _display_analysis_metrics(True)
        analysis_df = analysis_pipeline(results)
        return analysis_df

    if len(intent_list) == 1 and intent_list[0] == 'ALL_INTENTS':
        intent_list = list(results['correct_intent'].unique())
        if OFFTOPIC_LABEL in intent_list:
            intent_list.remove(OFFTOPIC_LABEL)
    analysis_df_list = list()
    for intent_name in intent_list:
        display(Markdown('### Threshold Analysis for Intent: {}'.format(intent_name)))
        analysis_df = analysis_pipeline(results, intent_name)
        if all(analysis_df):
            analysis_df.index = np.arange(1, len(analysis_df) + 1)
            display(analysis_df)
        analysis_df_list.append(analysis_df)

    return analysis_df_list

def _display_analysis_metrics(display_far):
    """display the explanation for analysis metrics"""
    display(Markdown("### Threshold Metrics"))
    display(Markdown(
        "We calculate metrics for responses where the top intent has a confidence above the \
        threshold specified on the x-axis.  "))

    display(Markdown(
        "We consider examples which are within the scope of the chatbot's problem formulation as \
         on topic or in domain and those examples which are outside the scope of the problem to be \
         out of domain or irrelevant"))

    display(Markdown("#### 1) Thresholded On Topic Accuracy (TOA)"))
    display(Markdown(
        "x-axis: Confidence threshold used || " +
        "y-axis: Intent Detection Accuracy for On Topic utterances"))

    display(Markdown("#### 2)  Bot Coverage %"))
    display(Markdown(
        "x-axis: Confidence threshold used || " +
        "y-axis: Fraction of All utterances above the threshold"))

    if display_far:
        display(Markdown("#### 3) False Acceptance Rate for Out of Domain Examples (FAR)"))
        display(Markdown(
            "x-axis: Confidence threshold used || " +
            "y-axis: Fraction of Out of Domain utterances falsely considered on topic"))

    display(Markdown(
        "#### Note: Default acceptance threshold for Watson Assistant is set at 0.2.\
        Utterances with top intent confidence < 0.2 will be considered irrelevant"))


def generate_unique_thresholds(sorted_results_tuples):
    """
    generate list of unique thresholds based off changes in confidence
    and sorted list of unique confidences
    :return: unique_thresholds
    """
    sort_uniq_confs = list(sorted(set([info[2] for info in sorted_results_tuples])))
    thresholds = [0]
    thresholds.extend([(sort_uniq_confs[idx] + sort_uniq_confs[idx + 1]) / 2
                       for idx in range(len(sort_uniq_confs) - 1)])
    return thresholds, sort_uniq_confs


def _find_threshold(t, thresholds):
    """
    find the appropriate cut-off
    :param t:
    :param thresholds:
    :return:
    """
    for index in range(len(thresholds) - 1):
        if thresholds[index] <= t < thresholds[index + 1]:
            return index

    return len(thresholds) - 1


def _get_ontopic_accuracy_list(sorted_infos, thresholds):
    """
    generate the list of on-topic accuracy and on-topic counts
    based on the list of thresholds
    :param sorted_infos:
    :param thresholds:
    :return:
    """
    ontopic_infos = [info for info in sorted_infos if info[0] != OFFTOPIC_LABEL]
    cor = len([info for info in ontopic_infos if info[0] == info[1]])
    tol = len(ontopic_infos)
    accuracy_list = list()
    count_list = list()
    current_step = 0
    for t in thresholds:
        while current_step < len(ontopic_infos):

            if ontopic_infos[current_step][2] < t:
                tol -= 1
                if ontopic_infos[current_step][0] == ontopic_infos[current_step][1]:
                    cor -= 1
            else:
                break
            current_step += 1
        accuracy_list.append(cor / tol)
        count_list.append(cor)

    return accuracy_list, count_list


def _get_bot_coverage_list(sorted_infos, thresholds):
    """
    generate the list of bot coverage ratio and bot coverage counts
    based on the list of thresholds
    :param sorted_infos:
    :param thresholds:
    :return:
    """
    tol = len(sorted_infos)
    cur_bot_coverage = tol
    bot_coverage_count_list = list()
    bot_coverage_list = list()
    current_step = 0
    for t in thresholds:
        while sorted_infos[current_step][2] < t:
            cur_bot_coverage -= 1
            current_step += 1
        bot_coverage_count_list.append(cur_bot_coverage)
        bot_coverage_list.append(cur_bot_coverage/tol)
    return bot_coverage_list, bot_coverage_count_list


def _get_far_list(sorted_infos, thresholds):
    """
    find the list of false acceptance rates and false acceptance counts
    :param sorted_infos:
    :param thresholds:
    :return:
    """
    offtopic_infos = [info for info in sorted_infos if info[0] == OFFTOPIC_LABEL]
    cur_fa_count = len(offtopic_infos)
    tol = len(offtopic_infos)
    far_list = list()
    far_count = list()
    current_step = 0
    for t in thresholds:
        while current_step < len(offtopic_infos):
            if offtopic_infos[current_step][2] < t:
                cur_fa_count -= 1
                current_step += 1
            else:
                break
        far_list.append(cur_fa_count/tol)
        far_count.append(cur_fa_count)
    return far_list, far_count


def _convert_data_format(results, intent_name=None):
    """
    convert the dataframe format to tuples of (ground_truth, prediction, confidence)
    :param results: results dataframe
    :param intent_name: optional parameter to allow different definition of offtopic label in per
    intent cases
    :return: result_list: list of tuples of (ground_truth, prediction, confidence) sorted by conf
    """
    if intent_name:
        results = results[(results['correct_intent'] == intent_name) |
                          (results['top_intent'] == intent_name)].copy()

        results['correct_intent'] = np.where((results['correct_intent'] !=
                                              results['top_intent']) &
                                             (results['top_intent'] == intent_name),
                                             OFFTOPIC_LABEL,
                                             results['correct_intent'])

        results_list = [(gt, pred, conf) for gt, pred, conf in
                        zip(results['correct_intent'],
                            results['top_intent'],
                            results['top_confidence'])]

        results_list = sorted(results_list, key=lambda x: x[2])

    else:
        results_list = [(truth, prediction, confidence) for truth, prediction, confidence
                        in zip(results['correct_intent'],
                               results['top_intent'],
                               results['top_confidence'])]
        results_list = sorted(results_list, key=lambda x: x[2])

    return results_list


def extract_by_topic(sorted_results):
    """
    extract information by topics
    :param sorted_results:
    :return:
    ontopic_infos, list
    """
    offtopic_infos = [prediction for prediction in sorted_results
                      if prediction[0] == OFFTOPIC_LABEL]

    ontopic_infos = [prediction for prediction in sorted_results
                     if prediction[0] != OFFTOPIC_LABEL]

    return ontopic_infos, offtopic_infos


def analysis_pipeline(results, intent_name=None):
    """
    perform the operation of extraction of table analysis and produce threshold graph
    :param results: list of tuples of (ground_truth, prediction, confidence) sorted by confidence
    :param intent_name:
    :return: analysis_df
    """
    sorted_results = _convert_data_format(results, intent_name=intent_name)

    ontopic_infos, offtopic_infos = extract_by_topic(sorted_results)

    # if ontopic counts or sorted results are less than 3, the graph will show almost no variation
    # if all confidence of the predicted result are the same, there will be no variation
    if len(ontopic_infos) < 3 or len(sorted_results) < 3 \
            or all(ele[2] == sorted_results[0][2] for ele in sorted_results):
        display(Markdown('**Inadequate Data Points**: No analysis will be conducted'))
        analysis_df = pd.DataFrame()
        return analysis_df

    analysis_df, toa_list, bot_coverage_list, far_list, thresholds = \
        extract_table_analysis(sorted_results,
                               ontopic_infos,
                               offtopic_infos)

    if not intent_name and not analysis_df.empty:
        line_graph_data = pd.DataFrame(data={'Thresholded On Topic Accuracy': toa_list,
                                             'Bot Coverage %': bot_coverage_list,
                                             'False Acceptance Rate (FAR) for Out of Domain Examples':
                                                 far_list},
                                       index=thresholds)

        create_threshold_graph(line_graph_data)

    return analysis_df


def extract_table_analysis(sorted_results, ontopic_infos, offtopic_infos):
    """
    extract informations for table analysis
    :param sorted_results:
    :return:
        analysis_df: pandas dataframe of the table for dispaly
        toa_list: list of sorted on-topic accuracy
        bot_coverage_list: list of sorted bot coverage ratio
        far_list: list of sorted false acceptance rate
        thresholds: list of sorted & unique thresholds
    """
    thresholds, sort_uniq_confs = generate_unique_thresholds(sorted_results)

    toa_list, toa_count = _get_ontopic_accuracy_list(sorted_results, thresholds)
    bot_coverage_list, bot_coverage_count = _get_bot_coverage_list(sorted_results, thresholds)

    if len(offtopic_infos) >= OFFTOPIC_CNT_THRESHOLD_FOR_DISPLAY:

        far_list, _ = _get_far_list(sorted_results, thresholds)
    else:
        display(Markdown(
            'Out of Domain examples fewer than **%d** thus \
            no False Acceptance Rate (FAR) calculated'
            % OFFTOPIC_CNT_THRESHOLD_FOR_DISPLAY))
        far_list = [-1]*len(thresholds)

    analysis_df = create_display_table(toa_list,
                                       bot_coverage_list,
                                       bot_coverage_count,
                                       sorted_results,
                                       thresholds,
                                       offtopic_infos,
                                       far_list)

    return analysis_df, toa_list, bot_coverage_list, far_list, thresholds


def create_threshold_graph(data):
    """
    display threshold analysis graph
    :param data:
    :return: None
    """
    sns.set(rc={'figure.figsize': (20.7, 10.27)})
    plt.ylim(0, 1.1)
    plt.axvline(.2, 0, 1)
    plot = sns.lineplot(data=data, palette="tab10", linewidth=3.5)
    plt.setp(plot.legend().get_texts(), fontsize='22')
    plot.set_xlabel('Threshold T', fontsize=18)
    plot.set_ylabel('Metrics mentioned above', fontsize=18)

def create_display_table(toa_list,
                         bot_coverage_list,
                         bot_coverage_count,
                         sorted_results,
                         thresholds,
                         offtopic_infos,
                         far_list):
    """
    create table for display purpose
    :param toa_list:
    :param bot_coverage_list:
    :param bot_coverage_count:
    :param sorted_results:
    :param thresholds:
    :param offtopic_infos:
    :param far_list:
    :return: analysis_df, pandas dataframe containing metrics at intervals of 10%
    """
    # produce the threhold quantiles for extraction of relevant information
    display_thresholds = [t/100 for t in range(0, 100, 10)]
    display_indexes = [_find_threshold(t, thresholds) for t in display_thresholds]

    analysis_data = dict()
    analysis_data['Threshold (T)'] = display_thresholds
    analysis_data['Ontopic Accuracy (TOA)'] = [toa_list[idx]*100 for idx in display_indexes]
    analysis_data['Bot Coverage %'] = [bot_coverage_list[idx]*100 for idx in display_indexes]
    analysis_data['Bot Coverage Counts'] = [str(np.round(bot_coverage_count[idx], decimals=0))
                                            + ' / ' + str(len(sorted_results))
                                            for idx in display_indexes]

    if len(offtopic_infos) >= OFFTOPIC_CNT_THRESHOLD_FOR_DISPLAY:
        analysis_data['False Acceptance Rate (FAR)'] = [far_list[idx]*100 for
                                                        idx in display_indexes]

    analysis_df = pd.DataFrame(data=analysis_data)
    return analysis_df
