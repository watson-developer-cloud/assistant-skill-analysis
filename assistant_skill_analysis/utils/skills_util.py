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
import ibm_watson
import codecs
from ibm_cloud_sdk_core.authenticators import (
    IAMAuthenticator,
    BasicAuthenticator,
    NoAuthAuthenticator,
)


DEV_DATACENTER = (
    "https://api.us-south.assistant.dev.watson.cloud.ibm.com",
    "https://iam.test.cloud.ibm.com/identity/token",
)
DEFAULT_V1_API_VERSION = "2019-02-28"
DEFAULT_V2_API_VERSION = "2021-11-27"
DEFAULT_PROD_URL = "https://gateway.watsonplatform.net/assistant/api"
DEFAULT_USERNAME = "apikey"
STAGE_IAM_URL = "https://iam.stage1.bluemix.net/identity/token"
DEFAULT_AUTHENTICATOR_URL = "https://iam.cloud.ibm.com/identity/token"

OFFTOPIC_LABEL = "SYSTEM_OUT_OF_DOMAIN"

OFFTOPIC_CONF_THRESHOLD = 0.2

LABEL_FONT = {"family": "normal", "weight": "bold", "size": 17}

TITLE_FONT = {"family": "normal", "weight": "bold", "size": 25}


def stratified_sampling(workspace, sampling_percentage=0.8):
    """
    Create a stratified sample of the workspace json
    & return a intent json acceptable in Assistant API

    :param workspace: json format output defined by Assistant API
    :param sampling_percentage: percentage of original to sample
    :return train_workspace_data: list of intents for train
    :return test_workspace_data: list of utterance,intent pairs for test
    """
    train_workspace_data = list()
    test_workspace_data = list()
    for i in range(len(workspace["intents"])):
        intent = workspace["intents"][i]
        sampling_index = list(np.arange(len(intent["examples"])))
        random.shuffle(sampling_index)
        # training set
        train_test_split_cutoff = int(sampling_percentage * len(sampling_index))
        train_examples = [
            intent["examples"][index]
            for index in sampling_index[:train_test_split_cutoff]
        ]
        train_workspace_data.append({"intent": workspace["intents"][i]["intent"]})
        train_workspace_data[i].update({"description": "string"})
        train_workspace_data[i].update({"examples": train_examples})
        # test set
        test_examples = [
            intent["examples"][index]
            for index in sampling_index[train_test_split_cutoff:]
        ]
        test_workspace_data.extend(
            [
                utterances["text"] + "\t" + workspace["intents"][i]["intent"]
                for utterances in test_examples
            ]
        )

    return train_workspace_data, test_workspace_data


def create_workspace(conversation, intent_json=None):
    """
    Create a workspace for testing purpose
    :param conversation: conversation object created by Watson Assistant api
    :param intent_json: nested json of utternance and intent pairs
    :return response: the workspace id and other metadata related to the new workspace
    """
    response = conversation.create_workspace(
        name="test_workspace",
        description="",
        language="en",
        intents=intent_json,
        entities=[],
        counterexamples=[],
        metadata={},
    ).get_result()
    return response


def input_credentials(
    input_apikey=True, input_workspace_id=True, input_assistant_id=False
):
    """
    Prompt user to enter apikey and workspace id
    """
    apikey, workspace_id, assistant_id = None, None, None
    if input_apikey:
        apikey = getpass.getpass("Please enter apikey: ")
    if input_workspace_id:
        workspace_id = getpass.getpass("Please enter workspace-id: ")
    if input_assistant_id:
        assistant_id = getpass.getpass("Please enter assistant-id: ")
    return apikey, workspace_id, assistant_id


def retrieve_conversation(
    iam_apikey=None,
    url=DEFAULT_PROD_URL,
    api_version=None,
    username=DEFAULT_USERNAME,
    password=None,
    authenticator_url=DEFAULT_AUTHENTICATOR_URL,
    sdk_version="V1",
):
    """
    Retrieve workspace from Assistant instance
    :param iam_apikey:
    :param url:
    :param api_version:
    :param username:
    :param password:
    :param sdk_version: V2 is needed for action workspaces
    :return workspace: workspace json
    """
    assert sdk_version in ["V1", "V2"]
    if api_version is None:
        if sdk_version == "V1":
            api_version = DEFAULT_V1_API_VERSION
        else:
            api_version = DEFAULT_V2_API_VERSION
    if iam_apikey:
        authenticator = IAMAuthenticator(apikey=iam_apikey, url=authenticator_url)
    elif username and password:
        authenticator = BasicAuthenticator(username=username, password=password)
    else:
        authenticator = NoAuthAuthenticator()

    if sdk_version == "V1":
        conversation = ibm_watson.AssistantV1(
            authenticator=authenticator, version=api_version
        )
    else:
        conversation = ibm_watson.AssistantV2(
            authenticator=authenticator, version=api_version
        )

    conversation.set_service_url(url)

    return conversation


def retrieve_workspace(workspace_id, conversation, export_flag=True):
    """
    retrieve the workspace based on the workspace id
    :param workspace_id:
    :param conversation:
    :param export_flag:
    :return: workspace_dictionary
    """
    ws_json = conversation.get_workspace(workspace_id, export=export_flag)
    return ws_json.get_result()


def parse_workspace_json(workspace_json):
    """
    Parse workspace json and returns list of utterances, list of intents, and list of entities, and intent to action title mapping
    """
    ws_type = workspace_json.get("type", "dialog")
    dialog_setting_action_flag = bool(
        workspace_json.get("dialog_settings", {}).get("actions", False)
    )
    if dialog_setting_action_flag:
        ws_type = "action"
    utterances, intents, entities = [], [], []
    raw_intent_name_to_action_title_mapping = None

    if ws_type == "dialog":

        for intent in workspace_json["intents"]:
            for example in intent["examples"]:
                utterances.append(example["text"])
                intents.append(intent["intent"])
        entities = workspace_json["entities"]

    else:
        # intent name to action title mapping for readability
        raw_intent_name_to_action_title_mapping = {
            action["condition"]["intent"]: action["title"]
            for action in workspace_json["workspace"]["actions"]
            if action.get("condition", {}).get("intent")
        }
        for intent in workspace_json["workspace"]["intents"]:
            action_title = raw_intent_name_to_action_title_mapping[intent["intent"]]
            for example in intent["examples"]:
                utterances.append(example["text"])
                intents.append(action_title)
        entities = workspace_json["workspace"]["entities"]
    return utterances, intents, entities, raw_intent_name_to_action_title_mapping


def extract_workspace_data(workspace, language_util):
    """
    Extract relevant data and vocabulary
    :param workspace: workspace json, could be from an action skill or a dialog skill
    :param language_util:
    :return: workspace_pd, vocabulary, entities, intent name to action title mapping
    """
    relevant_data = {"utterance": list(), "intent": list(), "tokens": list()}
    vocabulary = set()
    (
        utterances,
        intents,
        entities,
        raw_intent_name_to_action_title_mapping,
    ) = parse_workspace_json(workspace)

    for utterance, intent in zip(utterances, intents):
        # preprocess utterance
        utterance = language_util.preprocess(utterance)
        tokens = language_util.tokenize(utterance)

        relevant_data["utterance"].append(utterance)
        relevant_data["intent"].append(intent)
        relevant_data["tokens"].append(tokens)

        vocabulary.update(tokens)

    workspace_pd = pd.DataFrame(relevant_data)
    return workspace_pd, vocabulary, entities, raw_intent_name_to_action_title_mapping


def process_test_set(test_set, lang_util, delim="\t", cos=False):
    """
    Process test set given the path to the test fil
    :param test_set: path to the test set on the local computer or cos object body of test csv
    :param lang_util: language utility
    :param delim: delimiter, use "," for cos instance
    :param cos: cos flag to indicate whether this is a path from local system or stream body from cos
    :return:
    """
    user_inputs = list()
    intents = list()
    tokens_list = list()
    file_handle = None
    if not cos:
        file_handle = open(test_set, "r", encoding="utf-8")
        reader = csv.reader(file_handle, delimiter=delim)
    else:
        reader = csv.reader(codecs.getreader("utf-8")(test_set), delimiter=delim)

    for row in reader:
        if len(row) == 0:
            continue
        cur_example = lang_util.preprocess(row[0])
        tokens = lang_util.tokenize(cur_example)
        user_inputs.append(cur_example)
        tokens_list.append(tokens)
        if len(row) == 2:
            intents.append(row[1].strip())
        elif len(row) == 1:
            intents.append(OFFTOPIC_LABEL)
    if file_handle:
        file_handle.close()

    test_df = pd.DataFrame(
        data={"utterance": user_inputs, "intent": intents, "tokens": tokens_list}
    )
    return test_df


def export_workspace(conversation, experiment_workspace_id, export_path):
    """
    Export the workspace to target path
    :param conversation: conversation object output by assistant api
    :param experiment_workspace_id: id of the experimental workspace
    :param export_path: the path where the exported workspace will be saved
    """
    response = conversation.get_workspace(
        workspace_id=experiment_workspace_id, export=True
    ).get_result()
    with open(export_path, "w+", encoding="utf-8") as outfile:
        json.dump(response, outfile)


def run_notebook(
    notebook_path,
    iam_apikey,
    test_file,
    output_path,
    wksp_id=None,
    assistant_id=None,
    action_wksp_json_path=None,
):
    """
    Run notebook for end to end test
    :param notebook_path:
    :param uname:
    :param pwd:
    :param wksp_id:
    :param assistant_id:
    :param test_file:
    :param action_wksp_json_path:
    :param output_path:
    """
    notebook_name, _ = os.path.splitext(os.path.basename(notebook_path))

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    nb, old_cred_text = _replace_nb_input(
        nb, iam_apikey, test_file, wksp_id, assistant_id, action_wksp_json_path
    )
    # nb = _remove_experimentation(nb)

    proc = ExecutePreprocessor(timeout=60 * 60, kernel_name="python3")
    proc.allow_errors = True

    proc.preprocess(nb, {"metadata": {"path": os.getcwd()}})
    errors = []
    for cell in nb.cells:
        if "outputs" in cell:
            for output in cell["outputs"]:
                if output.output_type == "error":
                    errors.append(output)
        if "source" in cell and "iam_apikey = " in cell["source"]:
            cell["source"] = old_cred_text

    with open(output_path + ".ipynb", mode="wt") as f:
        nbformat.write(nb, f)
    return nb, errors


def _replace_nb_input(
    nb, apikey, test_file, wksp_id=None, assistant_id=None, action_wksp_json_path=None
):
    """
    Replace notebook interactive input for tests
    :param nb:
    :param uname:
    :param pwd:
    :param wksp_id:
    :param assistant_id:
    :param action_wksp_json_path:
    :param test_file:
    """
    apikey_patt = "iam_apikey = "
    wksp_id_patt = "workspace_id = "
    assistant_id_patt = "ASSISTANT_ID = "
    action_wksp_json_patt = "ACTION_SKILL_FILENAME = "
    test_file_name_patt = "test_set_path = "
    old_cred_text = ""
    test_urls = '"' + DEV_DATACENTER[0] + '",' + '"' + DEV_DATACENTER[1] + '"'
    for cell in nb.cells:
        if "source" in cell and apikey_patt in cell["source"]:
            old_cred_text = cell["source"]
            text = re.sub(
                "(.*)\niam_apikey, (.*)", (r"\1\n#iam_apikey, \2"), cell["source"]
            )  # comment out input_credentials
            text = re.sub("datacenters\['dallas'\]", test_urls, text)

            text = re.sub(
                "(.*)#" + apikey_patt + "'###'(.*)",
                r"\1" + apikey_patt + "'" + apikey + "'" + r"\2",
                text,
            )  # replace pwd
            if wksp_id:
                text = re.sub(
                    "(.*)#" + wksp_id_patt + "'###'(.*)",
                    r"\1" + wksp_id_patt + "'" + wksp_id + "'" + r"\2",
                    text,
                )  # replace wksp_id
            if assistant_id:
                text = re.sub(
                    "(.*)#" + assistant_id_patt + "'###'(.*)",
                    r"\1" + assistant_id_patt + "'" + assistant_id + "'" + r"\2",
                    text,
                )  # replace assistant_id
            if action_wksp_json_path:
                text = re.sub(
                    "(.*)" + action_wksp_json_patt + "'###'(.*)",
                    r"\1"
                    + action_wksp_json_patt
                    + "'"
                    + action_wksp_json_path
                    + "'"
                    + r"\2",
                    text,
                )  # replace action workspace json path
            cell["source"] = text
        elif "source" in cell and test_file_name_patt in cell["source"]:
            text = re.sub(
                "(.*)\n" + test_file_name_patt + '"./test.tsv"(.*)',
                r"\1\n" + test_file_name_patt + "'" + test_file + "'" + r"\2",
                cell["source"],
            )  # replace test file
            cell["source"] = text
    return nb, old_cred_text


def _remove_experimentation(nb):
    """
    Remove the experimentation session from end-to-end test
    :param nb:
    """
    exp_patt = "Part 3: Experimentation"
    new_nb_cells = []
    for cell in nb.cells:
        if (
            cell.cell_type == "markdown"
            and "source" in cell
            and exp_patt in cell["source"]
        ):
            break
        else:
            new_nb_cells.append(cell)
    nb.cells = new_nb_cells
    return nb


def retrieve_classifier_response(
    conversation,
    text_input,
    alternate_intents=False,
    user_id="256",
    assistant_id=None,
    workspace_id=None,
):
    """
    retrieve classifier response
    :param conversation: instance
    :param workspace_id: workspace or skill id
    :param text_input: the input utterance
    :param alternate_intents:
    :param user_id:
    :param assistant_id:
    :return response:
    """
    if isinstance(conversation, ibm_watson.AssistantV1):
        assert workspace_id is not None
        response = conversation.message(
            input={"message_type": "text", "text": text_input},
            context={"metadata": {"user_id": user_id}},
            workspace_id=workspace_id,
            alternate_intents=alternate_intents,
        ).get_result()
    else:
        assert assistant_id is not None
        response = conversation.message_stateless(
            input={
                "message_type": "text",
                "text": text_input,
                "options": {"alternate_intents": alternate_intents},
            },
            context={"metadata": {"user_id": user_id}},
            assistant_id=assistant_id,
        ).get_result()
    return response
