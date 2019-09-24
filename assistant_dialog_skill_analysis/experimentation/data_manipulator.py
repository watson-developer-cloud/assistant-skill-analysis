from collections import Counter
import random
import numpy as np

def under_sampling(workspace, workspace_pd, quantile=None):
    """
    Under sample data
    :param workspace: json format outputed by assistant api
    :param workspace_pd: workspace dataframe
    :param quantile: threshold to sample from
    :return train_workspace_data: list of intent json
    """
    label_frequency_dict = dict(Counter(workspace_pd['intent']).most_common())
    train_workspace_data = list()

    if not quantile:
        quantile = .75
    sampling_threshold = int(np.quantile(a=list(label_frequency_dict.values()), q=[quantile])[0])

    for i in range(len(workspace['intents'])):

        if not workspace['intents'][i]['examples']:
            continue

        if label_frequency_dict[workspace['intents'][i]['intent']] > sampling_threshold:
            intent = workspace['intents'][i]
            sampling_index = list(np.arange(len(workspace['intents'][i]['examples'])))
            random.shuffle(sampling_index)
            train_examples = [intent['examples'][index] for index in
                              sampling_index[:sampling_threshold]]
            train_workspace_data.append({'intent': workspace['intents'][i]['intent']})
            train_workspace_data[-1].update({'description': 'string'})
            train_workspace_data[-1].update({'examples': train_examples})
        else:
            train_workspace_data.append({'intent': workspace['intents'][i]['intent']})
            train_workspace_data[-1].update({'description': 'string'})
            train_workspace_data[-1].update({'examples': [example for example in
                                                          workspace['intents'][i]['examples']]})

    return train_workspace_data
