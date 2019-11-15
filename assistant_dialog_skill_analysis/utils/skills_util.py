import os
import json
import random
import csv
import re
import getpass
import nbformat
import pandas as pd
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor
import nltk
import ibm_watson
from ibm_cloud_sdk_core.authenticators import \
    IAMAuthenticator, BasicAuthenticator, NoAuthAuthenticator

DEFAULT_API_VERSION = '2019-02-28'
DEFAULT_PROD_URL = 'https://gateway.watsonplatform.net/assistant/api'
DEFAULT_USERNAME = 'apikey'
STAGE_IAM_URL = 'https://iam.stage1.bluemix.net/identity/token'

OFFTOPIC_LABEL = 'SYSTEM_OUT_OF_DOMAIN'

LABEL_FONT = {'family': 'normal',
              'weight': 'bold',
              'size': 17}

TITLE_FONT = {'family': 'normal',
              'weight': 'bold',
              'size': 25}

PUNCTUATION = [";", ":", ",", "\.", "\"", "\'",
               "\?", "\(", "\)", "!", "？", "！",
               "；", "：", "。", "、", "《", "》",
               "，", "¿", "¡", "؟", "،"]

STOP_WORDS = ['an', 'a', 'in', 'on', 'be', 'or', 'of',
              'and', 'can', 'is', 'to', 'the', 'i']


def stratified_sampling(workspace, sampling_percentage=.8):
    '''
    Create a stratified sample of the workspace json
    & return a intent json acceptable in Assistant API

    :param workspace: json format output defined by Assistant API
    :param sampling_percentage: percentage of original to sample
    :return train_workspace_data: list of intents for train
    :return test_workspace_data: list of utterance,intent pairs for test
    '''
    train_workspace_data = list()
    test_workspace_data = list()
    for i in range(len(workspace['intents'])):
        intent = workspace['intents'][i]
        sampling_index = list(np.arange(len(intent['examples'])))
        random.shuffle(sampling_index)
        # training set
        train_test_split_cutoff = int(sampling_percentage * len(sampling_index))
        train_examples = [
            intent['examples'][index] for index in sampling_index[:train_test_split_cutoff]
            ]
        train_workspace_data.append({'intent': workspace['intents'][i]['intent']})
        train_workspace_data[i].update({"description": "string"})
        train_workspace_data[i].update({"examples": train_examples})
        # test set
        test_examples = [
            intent['examples'][index] for index in sampling_index[train_test_split_cutoff:]
            ]
        test_workspace_data.extend([
            utterances['text'] + '\t' +
            workspace['intents'][i]['intent'] for utterances in test_examples
            ])

    return train_workspace_data, test_workspace_data

def create_workspace(conversation, intent_json=None):
    '''
    Create a workspace for testing purpose
    :param conversation: conversation object created by Watson Assistant api
    :param intent_json: nested json of utternance and intent pairs
    :return response: the workspace id and other metadata related to the new workspace
    '''
    response = conversation.create_workspace(
        name='test_workspace',
        description='',
        language='en',
        intents=intent_json,
        entities=[],
        counterexamples=[],
        metadata={}).get_result()
    return response

def input_credentials():
    '''
    Prompt user to enter apikey and workspace id
    '''
    apikey = getpass.getpass("Please enter apikey: ")
    workspace_id = getpass.getpass("Please enter workspace-id: ")
    return apikey, workspace_id

def retrieve_workspace(iam_apikey=None,
                       workspace_id=None,
                       url=DEFAULT_PROD_URL,
                       api_version=DEFAULT_API_VERSION,
                       username=DEFAULT_USERNAME,
                       password=None,
                       export_flag=True):
    """
    Retrieve workspace from Assistant instance
    :param iam_apikey:
    :param workspace_id:
    :param url:
    :param api_version:
    :param username:
    :param password:
    :param export_flag:
    :return workspace: workspace json
    """

    if iam_apikey:
        authenticator = IAMAuthenticator(apikey=iam_apikey)
    elif username and password:
        authenticator = BasicAuthenticator(username=username, password=password)
    else:
        authenticator = NoAuthAuthenticator()

    conversation = ibm_watson.AssistantV1(authenticator=authenticator,
                                          version=api_version)
    conversation.set_service_url(url)

    if export_flag:
        ws_json = conversation.get_workspace(workspace_id, export=export_flag)
        return conversation, ws_json.get_result()
    return conversation


def extract_workspace_data(workspace):
    '''
    Extract relevant data and vocabulary
    :param workspace:
    :return relevant_data:
    :return vocabulary:
    '''
    relevant_data = {'utterance': list(), 'intent': list()}
    vocabulary = set()
    for i in range(len(workspace['intents'])):
        current_intent = workspace['intents'][i]['intent']
        for j in range(len(workspace['intents'][i]['examples'])):
            current_example = workspace['intents'][i]['examples'][j]['text']
            relevant_data['utterance'].append(current_example)
            relevant_data['intent'].append(current_intent)
            vocabulary.update(nltk.word_tokenize(current_example))
    return relevant_data, vocabulary

def process_test_set(test_set_filename, delim='\t'):
    '''
    Process test set given the path to the test file
    :param test_set_filename: link to the test file in tsv format
    :return test_df: test set stored in pandas dataframe
    '''
    user_inputs = list()
    intents = list()
    with open(test_set_filename, "r", encoding="utf-8") as ts:
        reader = csv.reader(ts, delimiter=delim)
        for row in reader:
            if len(row) == 2:
                user_inputs.append(row[0])
                intents.append(row[1])
            elif len(row) == 1:
                user_inputs.append(row[0])
                intents.append(OFFTOPIC_LABEL)
            else:
                continue
    test_df = pd.DataFrame(data={'utterance':user_inputs, 'intent':intents})
    return test_df

def export_workspace(conversation, experiment_workspace_id, export_path):
    """
    Export the workspace to target path
    :param conversation: conversation object output by assistant api
    :param experiment_workspace_id: id of the experimental workspace
    :param export_path: the path where the exported workspace will be saved
    """
    response = conversation.get_workspace(
        workspace_id=experiment_workspace_id, export=True).get_result()
    with open(export_path, 'w+', encoding='utf-8') as outfile:
        json.dump(response, outfile)

def load_stopword_list(path):
    '''
    :param path: path to stopwords list
    :return stopword_list:
    '''
    stopword_list = list()
    with open(path, 'r', encoding='utf-8') as filehandle:
        for line in filehandle:
            stopword_list.append(line.strip())
    return stopword_list

def run_notebook(notebook_path, iam_apikey, wksp_id, test_file, output_path):
    '''
    Run notebook for end to end test
    :param notebook_path:
    :param uname:
    :param pwd:
    :param wksp_id:
    :param test_file:
    :param output_path:
    '''
    notebook_name, _ = os.path.splitext(os.path.basename(notebook_path))

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    nb, old_cred_text = _replace_nb_input(nb, iam_apikey, wksp_id, test_file)
    #nb = _remove_experimentation(nb)

    proc = ExecutePreprocessor(timeout=60 * 60, kernel_name='python3')
    proc.allow_errors = True

    proc.preprocess(nb, {'metadata': {'path': os.getcwd()}})
    errors = []
    for cell in nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'error':
                    errors.append(output)
        if 'source' in cell and 'iam_apikey = ' in cell['source']:
            cell['source'] = old_cred_text

    with open(output_path + '.ipynb', mode='wt') as f:
        nbformat.write(nb, f)
    return nb, errors

def _replace_nb_input(nb, apikey, wksp_id, test_file):
    '''
    Replace notebook interactive input for tests
    :param nb:
    :param uname:
    :param pwd:
    :param wksp_id:
    :param test_file:
    '''
    apikey_patt = 'iam_apikey = '
    wksp_id_patt = 'workspace_id = '
    test_file_name_patt = 'test_set_path = '
    old_cred_text = ''
    for cell in nb.cells:
        if 'source' in cell and apikey_patt in cell['source']:
            old_cred_text = cell['source']
            text = re.sub('(.*)\niam_apikey, (.*)',
                          (r'\1\n#iam_apikey, \2'),
                          cell['source']) # comment out input_credentials

            text = re.sub('(.*)#' + apikey_patt + '\'###\'(.*)',
                          r'\1' + apikey_patt + '\'' + apikey + '\'' + r'\2',
                          text) # replace pwd
            text = re.sub('(.*)#' + wksp_id_patt + '\'###\'(.*)',
                          r'\1' + wksp_id_patt + '\'' + wksp_id + '\'' + r'\2',
                          text) # replace wksp_id
            cell['source'] = text
        elif 'source' in cell and test_file_name_patt in cell['source']:
            text = re.sub('(.*)\n' + test_file_name_patt + '\'test_set.tsv\'(.*)',
                          r'\1\n' + test_file_name_patt + '\'' + test_file + '\'' + r'\2',
                          cell['source']) # replace test file
            cell['source'] = text
    return nb, old_cred_text

def _remove_experimentation(nb):
    '''
    Remove the experimentation session from end-to-end test
    :param nb:
    '''
    exp_patt = 'Part 3: Experimentation'
    new_nb_cells = []
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and 'source' in cell and exp_patt in cell['source']:
            break
        else:
            new_nb_cells.append(cell)
    nb.cells = new_nb_cells
    return nb


def retrieve_classifier_response(conversation, workspace_id, text_input, alternate_intents=False):
    """
    retrieve classifier response
    :param conversation: instance
    :param workspace_id: workspace or skill id
    :param text_input: the input utterance
    :param alternate_intents:
    :return response:
    """
    response = conversation.message(
        input={
            'message_type': 'text',
            'text': text_input
        },
        workspace_id=workspace_id,
        alternate_intents=alternate_intents,
    ).get_result()
    return response