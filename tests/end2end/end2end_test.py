import unittest
from assistant_skill_analysis.utils import skills_util


class TestNotebook(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        CONFIG_FILE = "./wa_config.txt"
        with open(CONFIG_FILE) as fi:
            self.apikey = fi.readline().strip()
            self.wksp_id = fi.readline().strip()

        CONFIG_FILE = "./wa_config_action.txt"
        with open(CONFIG_FILE) as fi:
            _ = fi.readline().strip()
            self.assistant_id = fi.readline().strip()

    def test_notebook(self):
        test_file = "tests/resources/test_workspaces/customer_care_skill_test.tsv"
        nb, errors = skills_util.run_notebook(
            notebook_path="dialog_skill_analysis.ipynb",
            iam_apikey=self.apikey,
            wksp_id=self.wksp_id,
            test_file=test_file,
            output_path="notebook_output",
        )
        self.assertEqual(errors, [])

    def test_action_notebook(self):
        test_file = "tests/resources/test_workspaces/test_set_action.tsv"
        wksp_json = "tests/resources/test_workspaces/customer_care_sample_action_skill.json"
        nb, errors = skills_util.run_notebook(
            notebook_path="action_skill_analysis.ipynb",
            iam_apikey=self.apikey,
            test_file=test_file,
            output_path="notebook_output",
            assistant_id=self.assistant_id,
            action_wksp_json_path=wksp_json,
        )
        self.assertEqual(errors, [])

    def tearDown(self):
        unittest.TestCase.tearDown(self)


if __name__ == "__main__":
    unittest.main()
