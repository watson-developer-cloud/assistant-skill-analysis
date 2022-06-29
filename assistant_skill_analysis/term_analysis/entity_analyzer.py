import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2

N = 5


def _derive_entity_label_matrix(train_full_results, entities):
    """
    Derive entity feature matrix for chi2 anaylsis using entity annotations from message api
    :param train_full_results: pandas data frame outputed by inference
    :param entities: list of entities that is defined in the workspace
    :return entity_feature_matrix: numpy matrix of examples with entities x number of entities
    :return labels: numpy array: number of labels correspond to number of examples
    :return entity_average_confidence_dict: dict entity --> average confidence for entity
    """
    entity_feature_matrix = list()
    labels = list()
    entity_conf_dict = dict()
    entity_count_dict = dict()
    entity_average_confidence_dict = dict()
    for i in range(len(train_full_results)):
        current_result = train_full_results.iloc[i]
        if current_result["entities"]:
            # create empty feature vector
            current_feature = [0] * len(entities)
            for entity_reference in current_result["entities"]:
                e_ref = entity_reference["entity"]
                e_conf = entity_reference["confidence"]

                entity_idx = entities.index(e_ref)
                current_feature[entity_idx] += 1
                entity_conf_dict[e_ref] = entity_conf_dict.get(e_ref, 0) + e_conf
                entity_count_dict[e_ref] = entity_count_dict.get(e_ref, 0) + 1

            entity_feature_matrix.append(current_feature)
            labels.append(current_result["correct_intent"])

    entity_feature_matrix = np.array(entity_feature_matrix)
    labels = np.array(labels)
    for key in entity_conf_dict:
        entity_average_confidence_dict[key] = (
            entity_conf_dict[key] / entity_count_dict[key]
        )

    return entity_feature_matrix, labels, entity_average_confidence_dict


def entity_label_correlation_analysis(train_full_results, entities_list, p_value=0.05):
    """
    Apply chi2 analysis on entities of the training set
    :param train_full_results: pandas data frame output by inference
    :param entities_list: the list of entities that is defined in the workspace
    :param p_value: threshold for chi2 analysis
    :return entity_label_df: pandas df with col 1 being intents and col 2 entities
    """
    (
        entity_feature_matrix,
        labels,
        entity_average_confidence_dict,
    ) = _derive_entity_label_matrix(train_full_results, entities_list)
    entities_list = np.array(entities_list)
    unique_labels = list(set(labels))
    final_labels = list()
    final_entities = list()

    for label in unique_labels:
        chi2_statistics, pval = chi2(entity_feature_matrix, labels == label)
        temp_entities_list = entities_list[pval < p_value]
        chi2_statistics = chi2_statistics[pval < p_value]
        ordered_entities = temp_entities_list[np.argsort(chi2_statistics)]
        if len(ordered_entities) == 0:
            continue

        final_labels.append(label)
        final_entities.append(", ".join(ordered_entities[-N:]))

    entity_label_df = pd.DataFrame(
        {"Intent": final_labels, "Correlated Entities": final_entities}
    )

    return entity_label_df
