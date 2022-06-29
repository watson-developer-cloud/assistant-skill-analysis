import sys
import time
import threading
import _thread
import traceback
import ibm_watson

from ..utils import skills_util


class InferenceThread(threading.Thread):
    """
    InferenceThread class is used for multi-thread inferencing for faster inference speed
    """

    def __init__(
        self,
        thread_id,
        name,
        que,
        conversation,
        result,
        max_retries=10,
        verbose=False,
        user_id="256",
        assistant_id=None,
        workspace_id=None,
        intent_to_action_mapping=None,
    ):
        """
        Initialize inferencer
        :param thread_id:
        :param name:
        :param que:
        :param conversation:
        :param workspace_id:
        :param result:
        :param max_retries:
        :param verbose:
        :param user_id:
        :param assistant_id:
        :param intent_to_action_mapping:
        """
        if isinstance(conversation, ibm_watson.AssistantV1):
            assert workspace_id is not None
            self.sdk_version = "V1"
        else:
            assert assistant_id is not None
            self.sdk_version = "V2"

        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.que = que
        self.conversation = conversation
        self.result = result
        self.workspace_id = workspace_id
        self.max_retries = max_retries
        self.verbose = verbose
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.intent_to_action_mapping = intent_to_action_mapping
        self.exitflag = 0

    def run(self):
        """
        Start thread
        """
        print("Starting " + self.name)
        self.thread_inference()
        print("Exiting " + self.name)

    def thread_inference(self):
        """
        Define thread run logic
        """
        while not self.exitflag:
            if not self.que.empty():
                attempt = 1
                success_flag = False
                query_data = self.que.get()
                query_question = query_data[0]
                for i in range(self.max_retries):
                    if self.verbose:
                        print("{} processing {}".format(self.name, query_question))
                    if success_flag:
                        break
                    attempt += 1
                    try:
                        response = skills_util.retrieve_classifier_response(
                            conversation=self.conversation,
                            text_input=query_question,
                            alternate_intents=True,
                            user_id=self.user_id,
                            assistant_id=self.assistant_id,
                            workspace_id=self.workspace_id,
                        )
                        time.sleep(0.2)
                        if self.sdk_version == "V2":
                            response = response["output"]
                            # v2 api returns all intent predictions
                            if (
                                response["intents"][0]["confidence"]
                                < skills_util.OFFTOPIC_CONF_THRESHOLD
                            ):
                                response["intents"] = []
                            for intents_prediction in response["intents"]:
                                intents_prediction[
                                    "intent"
                                ] = self.intent_to_action_mapping[
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
                            "utterance": query_question,
                            "correct_intent": query_data[1],
                            "top_intent": top_intent,
                            "top_confidence": top_confidence,
                            "top_predicts": top_predicts,
                            "entities": entities,
                        }
                        self.result.append(new_dict)
                        success_flag = True
                    except KeyboardInterrupt as e:
                        self.join()
                        sys.exit(e)
                    except Exception:
                        if self.verbose:
                            traceback.print_exc()
                            print(
                                "{} process {} fail attempt {}".format(
                                    self.name, query_question, i
                                )
                            )
                        time.sleep(0.1)

                if attempt >= self.max_retries:
                    print(
                        "Maximum attempt of {} has reached for query {}".format(
                            self.max_retries, query_question
                        )
                    )
                    _thread.interrupt_main()
                    self.exit()
            else:
                self.exit()

    def exit(self):
        """
        Exit thread
        """
        self.exitflag = 1
