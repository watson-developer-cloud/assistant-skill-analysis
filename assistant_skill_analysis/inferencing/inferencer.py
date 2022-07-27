import time
import pandas as pd
import numpy as np
import ibm_watson
from ..utils import skills_util
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

MAX_RETRY = 5


def inference(
    conversation,
    test_data,
    max_thread=5,
    user_id="256",
    assistant_id=None,
    workspace_id=None,
    intent_to_action_mapping=None,
):
    """
    query the message api to generate results on the test data
    :parameter: conversation: the conversation object produced by AssistantV1 api
    :parameter: workspace_id: the workspace id of the
    :parameter: test_data: the data that will be sent to the classifier
    :parameter: max_thread: the max number of threads to use for multi-threaded inference
    :parameter: verbose: flag indicates verbosity of outputs during mutli-threaded inference
    :parameter: assistant_id:
    :parameter: intent_to_action_mapping:
    :return result_df: results dataframe
    """
    skd_version = "V1"
    if isinstance(conversation, ibm_watson.AssistantV1):
        assert workspace_id is not None
    else:
        assert assistant_id is not None
        assert intent_to_action_mapping is not None
        skd_version = "V2"

    if max_thread == 1:
        reach_max_retry = False
        responses = []
        for test_example, ground_truth in zip(
            test_data["utterance"], test_data["intent"]
        ):
            attempt = 1
            while attempt <= MAX_RETRY:
                try:
                    prediction_json = skills_util.retrieve_classifier_response(
                        conversation=conversation,
                        text_input=test_example,
                        alternate_intents=True,
                        user_id=user_id,
                        assistant_id=assistant_id,
                        workspace_id=workspace_id,
                    )
                    time.sleep(0.3)

                    success_flag = True
                except Exception:
                    continue
                if success_flag:
                    break
                attempt += 1

                if attempt > MAX_RETRY:
                    reach_max_retry = True

            if reach_max_retry:
                raise Exception("Maximum attempt of {} has reached".format(MAX_RETRY))

            if skd_version == "V2":
                prediction_json = prediction_json["output"]
                if len(prediction_json["intents"]) > 0:
                    # v2 api returns all intent predictions
                    if (
                        prediction_json["intents"][0]["confidence"]
                        < skills_util.OFFTOPIC_CONF_THRESHOLD
                    ):
                        prediction_json["intents"] = []

                for intents_prediction in prediction_json["intents"]:
                    intents_prediction["intent"] = intent_to_action_mapping[
                        intents_prediction["intent"]
                    ]

            if not prediction_json["intents"]:
                responses.append(
                    {
                        "top_intent": skills_util.OFFTOPIC_LABEL,
                        "top_confidence": 0.0,
                        "correct_intent": ground_truth,
                        "utterance": test_example,
                        "top_predicts": [],
                        "entities": [],
                    }
                )
            else:
                responses.append(
                    {
                        "top_intent": prediction_json["intents"][0]["intent"],
                        "top_confidence": prediction_json["intents"][0]["confidence"],
                        "correct_intent": ground_truth,
                        "utterance": test_example,
                        "top_predicts": prediction_json["intents"],
                        "entities": prediction_json["entities"],
                    }
                )
        result_df = pd.DataFrame(data=responses)
    else:
        result_df = thread_inference(
            conversation=conversation,
            test_data=test_data,
            max_thread=max_thread,
            user_id=user_id,
            workspace_id=workspace_id,
            assistant_id=assistant_id,
            intent_to_action_mapping=intent_to_action_mapping,
        )
    return result_df


def thread_inference(
    conversation,
    test_data,
    max_thread=5,
    user_id="256",
    assistant_id=None,
    workspace_id=None,
    intent_to_action_mapping=None,
):
    """
    Perform multi thread inference for faster inference time
    :param conversation:
    :param workspace_id: Assistant workspace id
    :param test_data: data to test on
    :param max_thread: max threads to use
    :param verbose: verbosity of output
    :param user_id: user_id for billing purpose
    :param assistant_id:
    :parameter: intent_to_action_mapping:
    :return result_df: results dataframe
    """
    if isinstance(conversation, ibm_watson.AssistantV1):
        assert workspace_id is not None
        sdk_version = "V1"
    else:
        assert assistant_id is not None
        sdk_version = "V2"
    count = 0
    response = None
    while count < MAX_RETRY and not response:
        try:
            response = skills_util.retrieve_classifier_response(
                conversation=conversation,
                text_input="ping",
                alternate_intents=True,
                user_id=user_id,
                assistant_id=assistant_id,
                workspace_id=workspace_id,
            )
        except Exception:
            count += 1
            time.sleep(0.5)

    executor = ThreadPoolExecutor(max_workers=max_thread)
    futures = {}
    result = []
    for test_example, ground_truth in zip(test_data["utterance"], test_data["intent"]):
        future = executor.submit(
            get_intent_confidence_retry,
            conversation=conversation,
            text_input=test_example,
            alternative_intents=True,
            user_id=user_id,
            assistant_id=assistant_id,
            workspace_id=workspace_id,
            retry=0,
        )
        futures[future] = (test_example, ground_truth)

    for future in tqdm(futures):
        res = future.result(timeout=1)
        test_example, ground_truth = futures[future]
        result.append(
            process_result(
                test_example,
                ground_truth,
                res,
                intent_to_action_mapping,
                sdk_version=sdk_version,
            )
        )

    result_df = pd.DataFrame(data=result)
    return result_df


def process_result(
    utterance, ground_truth, response, intent_to_action_mapping, sdk_version
):
    if sdk_version == "V2":
        response = response["output"]
        # v2 api returns all intent predictions
        if response["intents"][0]["confidence"] < skills_util.OFFTOPIC_CONF_THRESHOLD:
            response["intents"] = []
        for intents_prediction in response["intents"]:
            intents_prediction["intent"] = intent_to_action_mapping[
                intents_prediction["intent"]
            ]
    if response["intents"]:
        top_predicts = response["intents"]
        top_intent = response["intents"][0]["intent"]
        top_confidence = response["intents"][0]["confidence"]
    else:
        top_predicts = []
        top_intent = skills_util.OFFTOPIC_LABEL
        top_confidence = 0

    if response["entities"]:
        entities = response["entities"]
    else:
        entities = []

    new_dict = {
        "utterance": utterance,
        "correct_intent": ground_truth,
        "top_intent": top_intent,
        "top_confidence": top_confidence,
        "top_predicts": top_predicts,
        "entities": entities,
    }
    return new_dict


def get_intent_confidence_retry(
    conversation,
    text_input,
    alternative_intents,
    user_id,
    assistant_id,
    workspace_id,
    retry=0,
):
    try:
        return skills_util.retrieve_classifier_response(
            conversation=conversation,
            text_input=text_input,
            alternate_intents=True,
            user_id=user_id,
            assistant_id=assistant_id,
            workspace_id=workspace_id,
        )
    except Exception as e:
        if retry < MAX_RETRY:
            return get_intent_confidence_retry(
                conversation,
                text_input,
                alternative_intents,
                user_id,
                assistant_id,
                workspace_id,
                retry=retry + 1,
            )
        else:
            raise e


def calculate_mistakes(results):
    """
    retrieve the data frame of miss-classified examples
    :param results: results after tersting
    :return wrongs_df: data frame of mistakes
    """
    wrongs = list()
    for idx, row in results.iterrows():
        if row["correct_intent"] != row["top_intent"]:
            wrongs.append(row)
    wrongs_df = pd.DataFrame(data=wrongs)
    wrongs_df.index.name = "Test Example Index"
    return wrongs_df


def calculate_accuracy(results):
    """
    calculate the accuracy on the test set
    :param results: the results of testing
    :return accuracy: get accuracy on test set
    """
    correct = 0
    for i in range(0, len(results["correct_intent"])):
        correct += 1 if results["top_intent"][i] == results["correct_intent"][i] else 0
    accuracy = np.around((correct / len(results["correct_intent"])) * 100, 2)
    return accuracy
