import unittest
from assistant_skill_analysis.utils.skills_util import (
    retrieve_workspace,
    retrieve_conversation,
    DEV_DATACENTER
)

CONFIG_FILE = "./wa_config.txt"
CONFIG_FILE_ACTION = "./wa_config_action.txt"


@unittest.skip("skip")
class TestWorkspaceCredential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONFIG_FILE) as fi:
            cls.apikey = fi.readline().strip()
            cls.wksp_id = fi.readline().strip()

        with open(CONFIG_FILE_ACTION) as fi:
            _ = fi.readline().strip()
            cls.assistant_id = fi.readline().strip()

    def test_workspace_credentials(self):
        conversation = retrieve_conversation(iam_apikey=self.apikey,
                                             url=DEV_DATACENTER[0],
                                             authenticator_url=DEV_DATACENTER[1],
                                             )
        ws_json = retrieve_workspace(
            workspace_id=self.wksp_id, conversation=conversation
        )
        self.assertTrue(len(ws_json["intents"]) == 9)

    def test_action_credentials(self):
        conversation = retrieve_conversation(iam_apikey=self.apikey,
                                             url=DEV_DATACENTER[0],
                                             authenticator_url=DEV_DATACENTER[1],
                                             sdk_version="V2"
                                             )
        result = conversation.message_stateless(
            input={
                "message_type": "text",
                "text": "thank you",
                "options": {"alternate_intents": True},
            },
            context={"metadata": {"user_id": "123"}},
            assistant_id=self.assistant_id,
        ).get_result()

        self.assertAlmostEqual(1, result["output"]["intents"][0]["confidence"], delta=1e-6)


if __name__ == "__main__":
    unittest.main()
