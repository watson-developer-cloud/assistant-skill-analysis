import os
import random
import copy
import argparse
import json
import numpy as np

OFFTOPIC_LABEL = 'SYSTEM_OUT_OF_DOMAIN'

def stratified_sampling(workspace, sampling_percentage=.8):
    """
    Perform stratified sampling on the workspace json
    :param workspace: json acceptable by watson assistant
    :param sampling_percentage: percentage of total to use for train
    """

    train_workspace_data = copy.deepcopy(workspace) # copy everything except the intent list
    train_workspace_data['name'] = workspace['name'] + '_train'
    train_workspace_data.pop('intents')

    train_workspace_intent_list = list()
    test_workspace_data = list()

    for i in range(len(workspace['intents'])):
        intent = workspace['intents'][i]

        cutoff, sampling_index = find_split_cut_off(
            intent['examples'], sampling_percentage)

        # train set
        train_examples = [
            intent['examples'][index] for index in sampling_index[:cutoff]]
        train_workspace_intent_list.append({'intent': workspace['intents'][i]['intent']})
        train_workspace_intent_list[i].update({"description": "string"})
        train_workspace_intent_list[i].update({"examples": train_examples})

        # test set
        test_examples = [
            intent['examples'][index] for index in sampling_index[cutoff:]]
        test_workspace_data.extend(
            [utterances['text'] + '\t' +
             workspace['intents'][i]['intent'] for utterances in test_examples])
    train_workspace_data['intents'] = train_workspace_intent_list
    
    # counter examples
    if len(workspace['counterexamples']) > 0:
        train_workspace_data.pop('counterexamples')
        # train
        cutoff, sampling_index = find_split_cut_off(
            workspace['counterexamples'], sampling_percentage)
        train_workspace_data['counterexamples'] = [
            workspace['counterexamples'][index] for index in sampling_index[:cutoff]]
        # test
        test_workspace_data.extend(
            [workspace['counterexamples'][index]['text'] + '\t' +
             OFFTOPIC_LABEL for index in sampling_index[cutoff:]])

    return train_workspace_data, test_workspace_data

def find_split_cut_off(enumerable, sampling_percentage):
    """
    Find split cutoff point
    :param enumerable:
    :param sampling_percentage:
    """
    sampling_index = list(np.arange(len(enumerable)))
    random.shuffle(sampling_index)

    if len(enumerable) * (1 - sampling_percentage) < 1:
        cutoff = -1
    else:
        cutoff = int(np.ceil(sampling_percentage * len(sampling_index)))

    return cutoff, sampling_index

def main(args):
    workspace_data = json.load(open(args.input_data, 'r'))
    train_workspace_data, test_workspace_data = stratified_sampling(workspace_data, args.percentage)
    output_name = os.path.basename(args.input_data).replace('.json','')
    with open(os.path.join(args.output_folder, output_name+'_train.json'),'w',encoding='utf-8') as file:
        json.dump(train_workspace_data, file)
    with open(os.path.join(args.output_folder, output_name+'_test.tsv'), 'w', encoding='utf-8') as file:
        file.writelines([line +'\n' for line in test_workspace_data])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script will split workspace json ')
    parser.add_argument('-p','--percentage', help='how much percentage of the data to keep in train', default=.8, type=float)
    parser.add_argument('-input', '--input_data', help='the location of the workspace json',required=True)
    parser.add_argument('-output','--output_folder', help='the location of the train.json and test.tsv to be saved',required=True)
    args = parser.parse_args()
    main(args)

