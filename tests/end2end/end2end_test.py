import unittest
from assistant_skill_analysis.utils import skills_util
import json

class TestNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # points to dev010_Haode-Qi
        CONFIG_FILE = "./wa_config.txt"
        with open(CONFIG_FILE) as fi:
            cls.apikey = fi.read()#fi.readline().strip()

        with open(
            "tests/resources/test_workspaces/skill-Customer-Care-Sample.json",
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        URL, authenticator_url = skills_util.DEV_DATACENTER
        cls.conversation = skills_util.retrieve_conversation(
            iam_apikey=cls.apikey,
            url=URL,
            authenticator_url=authenticator_url,
            api_version=skills_util.DEFAULT_V1_API_VERSION,
        )
        raise ValueError("api:"+str(cls.apikey)+",url:"+str(URL)+",auth:"+str(authenticator_url))
        cls.wksp_id = skills_util.get_test_workspace(
            conversation=cls.conversation, workspace_json=data
        )

        # points to dev010_Haode-Qi
        CONFIG_FILE = "./wa_config_action.txt"
        with open(CONFIG_FILE) as fi:
            _ = fi.readline().strip()
            cls.assistant_id = fi.readline().strip()

    def test_notebook(self):
        test_file = "tests/resources/test_workspaces/customer_care_skill_test.tsv"
        nb, errors = skills_util.run_notebook(
            notebook_path="classic_dialog_skill_analysis.ipynb",
            iam_apikey=self.apikey,
            wksp_id=self.wksp_id,
            test_file=test_file,
            output_path="notebook_output",
        )
        self.assertEqual(errors, [])

    def test_action_notebook(self):
        test_file = "tests/resources/test_workspaces/test_set_action.tsv"
        wksp_json = (
            "tests/resources/test_workspaces/customer_care_sample_action_skill.json"
        )
        nb, errors = skills_util.run_notebook(
            notebook_path="new_experience_skill_analysis.ipynb",
            iam_apikey=self.apikey,
            test_file=test_file,
            output_path="notebook_output",
            assistant_id=self.assistant_id,
            action_wksp_json_path=wksp_json,
        )
        self.assertEqual(errors, [])

    @classmethod
    def tearDownClass(cls):
        unittest.TestCase.tearDown(cls)
        cls.conversation.delete_workspace(workspace_id=cls.wksp_id)


if __name__ == "__main__":
    unittest.main()
