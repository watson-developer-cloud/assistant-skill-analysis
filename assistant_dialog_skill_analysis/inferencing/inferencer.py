import time
import queue
import pandas as pd
import numpy as np
from ..utils import skills_util
from ..inferencing.multi_thread_inference import InferenceThread

def inference(conversation, workspace_id, test_data, max_retries=10, max_thread=5, verbose=False):
    """
    query the message api to generate results on the test data
    :parameter: conversation: the conversation object produced by AssistantV1 api
    :parameter: workspace_id: the workspace id of the
    :parameter: test_data: the data that will be sent to the classifier
    :parameter: max_retries: the maximum number of retries
    :parameter: max_thread: the max number of threads to use for multi-threaded inference
    :parameter: verbose: flag indicates verbosity of outputs during mutli-threaded inference
    :return result_df: results dataframe
    """

    if max_thread == 1:
        reach_max_retry = False
        responses = []
        for test_example, ground_truth in zip(test_data['utterance'], test_data['intent']):
            attempt = 1
            while attempt <= max_retries:
                try:
                    prediction_json = skills_util.retrieve_classifier_response(
                        conversation, workspace_id, test_example, True)
                    time.sleep(.3)

                    success_flag = True
                except Exception:
                    continue
                if success_flag:
                    break
                attempt += 1

                if attempt > max_retries:
                    reach_max_retry = True

            if reach_max_retry:
                raise Exception('Maximum attempt of {} has reached'.format(max_retries))

            if not prediction_json['intents']:
                responses.append({'top_intent' : skills_util.OFFTOPIC_LABEL,
                                  'top_confidence' : 0.0,
                                  'correct_intent' : ground_truth,
                                  'utterance' : test_example,
                                  'top_predicts' : [],
                                  'entities' : []})
            else:
                responses.append({'top_intent' : prediction_json['intents'][0]['intent'],
                                  'top_confidence' : prediction_json['intents'][0]['confidence'],
                                  'correct_intent' : ground_truth,
                                  'utterance' : test_example,
                                  'top_predicts' : prediction_json['intents'],
                                  'entities' : prediction_json['entities']})
        result_df = pd.DataFrame(data=responses)
    else:
        result_df = thread_inference(conversation,
                                     workspace_id,
                                     test_data,
                                     max_retries,
                                     max_thread,
                                     verbose)
    return result_df

def thread_inference(conversation, workspace_id, test_data,
                     max_retries=10, max_thread=5, verbose=False):
    """
    Perform multi thread inference for faster inference time
    :param conversation:
    :param workspace_id: Assistant workspace id
    :param test_data: data to test on
    :param max_retries: max retries for each call
    :param max_thread: max threads to use
    :param verbose: verbosity of output
    :return result_df: results dataframe
    """
    if max_thread > 5:
        print('only maximum of 5 threads are allowed')
    thread_list = ["Thread-1", "Thread-2", "Thread-3", "Thread-4", "Thread-5"]
    thread_list = thread_list[:max_thread]

    query_queue = queue.Queue(0)
    threads = []
    thread_id = 1
    result = list()

    start_time = time.time()

    for i in range(len(test_data)):
        data_point = [test_data['utterance'].iloc[i], test_data['intent'].iloc[i]]
        query_queue.put(data_point)

    # Create new threads
    for thread_name in thread_list:
        thread = InferenceThread(
            thread_id,
            thread_name,
            query_queue,
            conversation,
            workspace_id,
            result,
            max_retries,
            verbose)
        thread.start()
        threads.append(thread)
        thread_id += 1

    while len(result) != len(test_data):
        pass

    for thread in threads:
        thread.join()

    print("--- Total time: {} seconds ---".format(time.time() - start_time))
    result_df = pd.DataFrame(data=result)
    return result_df

def get_intents_confidences(conversation, workspace_id, text_input):
    """
    Retrieve a list of confidence for analysis purpose
    :param conversation: conversation instance
    :param workspace_id: workspace id
    :param text_input: input utterance
    :return intent_conf: intent confidences
    """
    response_info = skills_util.retrieve_classifier_response(
        conversation, workspace_id, text_input, True)['intents']
    intent_conf = [(r['intent'], r['confidence']) for r in response_info]
    return intent_conf

def calculate_mistakes(results):
    """
    retrieve the data frame of miss-classified examples
    :param results: results after tersting
    :return wrongs_df: data frame of mistakes
    """
    wrongs = list()
    for idx, row in results.iterrows():
        if row['correct_intent'] != row['top_intent']:
            wrongs.append(row)
    wrongs_df = pd.DataFrame(data=wrongs)
    wrongs_df.index.name = 'Test Example Index'
    return wrongs_df

def calculate_accuracy(results):
    """
    calculate the accuracy on the test set
    :param results: the results of testing
    :return accuracy: get accuracy on test set
    """
    correct = 0
    for i in range(0, len(results['correct_intent'])):
        correct += 1 if results['top_intent'][i] == results['correct_intent'][i] else 0
    accuracy = np.around((correct / len(results['correct_intent']))*100, 2)
    return accuracy
