import unittest
from assistant_dialog_skill_analysis.utils.skills_util import (
    retrieve_workspace,
    retrieve_conversation,
)

CONFIG_FILE = "./wa_config.txt"


class TestWorkspaceCredential(unittest.TestCase):
    def setUp(self):
        with open(CONFIG_FILE) as fi:
            self.apikey = fi.readline().strip()
            self.wksp_id = fi.readline().strip()

    def tearDown(self):
        pass

    def test_workspace_credentials(self):
        conversation = retrieve_conversation(iam_apikey=self.apikey)
        ws_json = retrieve_workspace(
            workspace_id=self.wksp_id, conversation=conversation
        )
        self.assertTrue(len(ws_json["intents"]) == 9)


if __name__ == "__main__":
    unittest.main()
