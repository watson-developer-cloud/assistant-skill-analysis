import os
import math
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils import skills_util
from ..inferencing import inferencer

MAX_TOKEN_LENGTH = 20
NGRAM_RANGE = [1]


def get_highlights_in_batch_multi_thread(
    conversation,
    workspace_id,
    full_results,
    output_folder,
    confidence_threshold,
    show_worst_k,
):
    """
    Given the prediction result, rank prediction results from worst to best
    & analyze the top k worst results.
    Term level highlighting on the worst results shows the sensitivity of terms in utterance
    :param conversation: conversation object produced by watson api
    :param workspace_id: workspace id
    :param full_results: prediction result showing the ranked list of intents by confidence scores
    :param output_folder: the output folder where the highlighting images will be saved
    :param confidence_threshold: the confidence threshold for offtopic detection
    :param show_worst_k: the top worst k results based on heuristics
    :return:
    """
    wrong_examples_sorted = _filter_results(full_results, confidence_threshold)
    display(
        Markdown(
            "### Identified {} problematic utterances ".format(
                len(wrong_examples_sorted)
            )
        )
    )
    display(Markdown("  "))

    wrong_examples_sorted = wrong_examples_sorted[:show_worst_k]

    (
        adversarial_results,
        adversarial_span_dict,
    ) = _adversarial_examples_multi_thread_inference(
        wrong_examples_sorted, conversation, workspace_id
    )

    if not adversarial_results.empty:

        display(Markdown("{} examples are shown below:".format(show_worst_k)))
        for original_example in wrong_examples_sorted:
            if not original_example[2]:
                label = skills_util.OFFTOPIC_LABEL
            else:
                label = original_example[2]
            label_idx = label + "\t" + str(original_example[0])
            adversarial_result_subset = adversarial_results[
                adversarial_results["correct_intent"] == label_idx
            ]
            highlight = _highlight_scoring(
                original_example, adversarial_result_subset, adversarial_span_dict
            )
            _plot_highlight(highlight, original_example, output_folder)


def _filter_results(full_results, confidence_threshold):
    """
    Given the full predicted results and confidence threshold,
    this function returns a ranked list of the mis-classified examples
    :param full_results:
    :param confidence_threshold:
    :return highlighting_candidates_sorted
    """
    highlighting_candidates = list()
    for idx in range(len(full_results)):
        item = full_results.iloc[idx]
        results_intent_list = [predict["intent"] for predict in item["top_predicts"]]
        result_dict = dict(item["top_predicts"])
        if item["correct_intent"] in results_intent_list:
            reference_position = results_intent_list.index(item["correct_intent"])
        else:
            reference_position = len(results_intent_list)

        rank_score = 0
        # for off-topic examples, rank score = off-topic confidence score - confidence threshold
        if item["correct_intent"] == skills_util.OFFTOPIC_LABEL:
            if item["top_confidence"] > confidence_threshold:
                rank_score = item["top_confidence"] - confidence_threshold

                highlighting_candidates.append(
                    (
                        idx,
                        item["utterance"],
                        None,
                        item["top_intent"],
                        item["top_confidence"],
                        rank_score,
                        reference_position,
                    )
                )
        else:
            if (item["top_intent"] != item["correct_intent"]) or (
                item["top_confidence"] <= confidence_threshold
            ):
                if item["top_intent"] != item["correct_intent"]:
                    # for incorrectly predicted examples, if the correct intent is not in top 10
                    # rank score = confidence of the predicted intent
                    if item["correct_intent"] not in result_dict:
                        rank_score = item["top_confidence"]
                    else:
                        # for incorrectly predicted examples, if the correct intent is in top 10,
                        # rank score = confidence of predicted intent - confidence of correct intent
                        rank_score = (
                            item["top_confidence"] - result_dict[item["correct_intent"]]
                        )
                elif item["top_confidence"] <= confidence_threshold:
                    # for correctly predicted examples, if the predicted confidence is less than
                    # confidence threshold, rank score = confidence threshold - predicted confidence
                    rank_score = confidence_threshold - item["top_confidence"]
                highlighting_candidates.append(
                    (
                        idx,
                        item["utterance"],
                        item["correct_intent"],
                        item["top_intent"],
                        item["top_confidence"],
                        rank_score,
                        reference_position,
                    )
                )

    highlighting_candidates_sorted = sorted(
        highlighting_candidates, key=lambda x: x[5], reverse=True
    )
    highlighting_candidates_sorted = [
        candidate
        for candidate in highlighting_candidates_sorted
        if len(nltk.word_tokenize(candidate[1])) < MAX_TOKEN_LENGTH
    ]

    return highlighting_candidates_sorted


def _plot_highlight(highlight, original_example, output_folder):
    """
    Plot the highlighting score into a plot and store the plot in the output folder
    :param highlight:
    :param original_example:
    :param output_folder:
    """
    if not original_example[2]:
        label = skills_util.OFFTOPIC_LABEL
    else:
        label = original_example[2]
    fig, ax = plt.subplots(figsize=(2, 5))
    ax = sns.heatmap(
        [[i] for i in highlight.tolist()],
        yticklabels=nltk.word_tokenize(original_example[1]),
        xticklabels=["Sensitivity to intent: " + '"' + label + '"'],
        cbar_kws={"orientation": "vertical"},
        linewidths=0,
        square=False,
        cmap="Blues",
    )

    if output_folder:
        conf_str = "%.3f" % (original_example[4])
        if original_example[2]:
            filename = (
                str(original_example[0])
                + "_groundtruth_"
                + original_example[2]
                + "_prediction_"
                + original_example[3]
                + "_confidence_"
                + conf_str
                + ".png"
            )
        else:
            filename = (
                str(original_example[0])
                + "_groundtruth_offtopic_prediction_"
                + original_example[3]
                + "_confidence_"
                + conf_str
                + ".png"
            )

        save_path = os.path.join(output_folder, filename)
        plt.savefig(os.path.join(save_path), bbox_inches="tight")

    table = list()
    table.append(["Test Set Index", original_example[0]])
    table.append(["Utterance", original_example[1]])
    table.append(
        [
            "Actual Intent",
            original_example[2]
            if (original_example[2])
            else skills_util.OFFTOPIC_LABEL,
        ]
    )
    table.append(["Predicted Intent", original_example[3]])
    table.append(["Confidence", original_example[4]])
    with pd.option_context("max_colwidth", 250):
        df = pd.DataFrame(data=table, columns=["Characteristic", "Value"])
        df.index = np.arange(1, len(df) + 1)
        display(df)
    plt.show()


def _adversarial_examples_multi_thread_inference(
    wrong_examples_sorted, conversation, workspace_id
):
    """
    Perform multi threaded inference on all the adversarial examples
    :param wrong_examples_sorted:
    :param conversation:
    :param workspace_id:
    """
    all_adversarial_examples = list()
    # the adversarial labels will be label\tidx for later regrouping purposes
    all_adversarial_label_idx = list()
    # map the adversarial example: span of adversarial
    adversarial_span_dict = dict()
    for original_example in wrong_examples_sorted:

        adversarial_examples, adversarial_span = _generate_adversarial_examples(
            original_example[1], original_example[0]
        )

        if not original_example[2]:
            label = skills_util.OFFTOPIC_LABEL
        else:
            label = original_example[2]
        adversarial_label = label + "\t" + str(original_example[0])

        all_adversarial_examples.extend(adversarial_examples)
        all_adversarial_label_idx.extend(
            [adversarial_label] * len(adversarial_examples)
        )
        adversarial_span_dict.update(adversarial_span)

    adversarial_test_data_frame = pd.DataFrame(
        {"utterance": all_adversarial_examples, "intent": all_adversarial_label_idx}
    )
    adversarial_results = inferencer.inference(
        conversation,
        workspace_id,
        adversarial_test_data_frame,
        max_retries=10,
        max_thread=5,
        verbose=False,
    )
    display(Markdown("   "))
    return adversarial_results, adversarial_span_dict


def _generate_adversarial_examples(utt, original_idx):
    """
    Generate adversarial examples by removing single tokens
    :param utt: string, utterance for generation of adversarial examples
    :param original_idx: the idx of the example in the original input data
    :returns
    adversarial_examples: list of strings, list of adversarial examples
    adversarial_span: dictionary of adversarial examples and the token span of the removed token
    """
    adversarial_examples = []
    adversarial_span = dict()
    tokens = utt.split()
    for idx in range(len(tokens)):
        for ngram in NGRAM_RANGE:
            new_sent = " ".join(tokens[:idx] + tokens[idx + ngram :])
            adversarial_examples.append(new_sent)
            adversarial_span[new_sent + "_" + str(original_idx)] = (idx, idx + ngram)
    return adversarial_examples, adversarial_span


def _highlight_scoring(
    original_example, subset_adversarial_result, adversarial_span_dict
):
    """
    Calculate the highlighting score using classification results of adversarial examples
    :param original_example:
    :param subset_adversarial_result:
    :param adversarial_span_dict:
    """
    original_utterance = " ".join(nltk.word_tokenize(original_example[1]))
    original_idx = original_example[0]
    original_intent = original_example[3]
    original_confidence = original_example[4]
    original_position = original_example[6]
    tokens = original_utterance.split(" ")
    highlight = np.zeros(len(tokens), dtype="float32")
    for idx in range(len(subset_adversarial_result)):
        adversarial_example = subset_adversarial_result.iloc[idx]
        if not adversarial_example["top_predicts"]:
            continue

        predict_dict = dict()
        predict_intent_list = list()
        for prediction in adversarial_example["top_predicts"]:
            predict_dict[prediction["intent"]] = prediction["confidence"]
            predict_intent_list.append(prediction["intent"])

        if original_intent in predict_dict:
            adversarial_position = list(predict_dict.keys()).index(original_intent)
            adversarial_confidence = predict_dict[original_intent]
        else:
            adversarial_position = len(list(predict_dict.keys()))
            adversarial_confidence = 0

        start, end = adversarial_span_dict[
            adversarial_example["utterance"] + "_" + str(original_idx)
        ]

        highlight = _scoring_function(
            highlight,
            original_position,
            adversarial_position,
            original_confidence,
            adversarial_confidence,
            start,
            end,
        )

    return highlight


def _scoring_function(
    highlight,
    original_position,
    adversarial_position,
    original_confidence,
    adversarial_confidence,
    start_idx,
    end_idx,
):
    """
    scoring function for highlighting of the interval start_idx:end_idx
    :param highlight: np.array of shape (n_tokens)
    :param original_position: ranking position of the target intent for the original sentence
    :param adversarial_position: ranking position of the target intent for the adversarial sentence
    :param original_confidence: confidence of the target intent for the original sentence
    :param adversarial_confidence: confidence of the target intent for the adversarial sentence
    :param start_idx: starting index of the adversarial mask
    :param end_idx: ending index of the adversarial mask
    :return: highlight: np.array of shape (n_tokens)
    """
    # position difference accounts for the change in the position of the target intent among
    # the top 10 intents return by the message api
    position_difference = (1 / float(original_position + 1.0)) - (
        1 / float(adversarial_position + 1.0)
    )

    # confidence difference accounts for the change in the confidence
    confidence_difference = original_confidence - adversarial_confidence

    ngram_size = end_idx - start_idx
    weight = math.pow(1.0 / ngram_size, 2.0)

    # highlight score for the interval of start_idx:end_idx is a weighted average of
    # the position difference and confidence difference
    weighted_difference = (
        weight
        * ((0.2 * confidence_difference) + (0.8 * position_difference))
        / ngram_size
    )

    highlight[start_idx:end_idx] += weighted_difference

    return highlight
