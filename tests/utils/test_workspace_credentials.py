import unittest
from assistant_dialog_skill_analysis.utils.skills_util import retrieve_workspace

CONFIG_FILE = './wa_config.txt'

class TestWorkspaceCredential(unittest.TestCase):

    def setUp(self):
        with open(CONFIG_FILE) as fi:
            self.apikey = fi.readline().strip()
            self.wksp_id = fi.readline().strip()

    def tearDown(self):
        pass

    def test_workspace_credentials(self):
        _, ws_json = retrieve_workspace(iam_apikey=self.apikey, workspace_id=self.wksp_id)
        self.assertTrue(len(ws_json['intents']) == 9)

if __name__ == "__main__":
    unittest.main()
