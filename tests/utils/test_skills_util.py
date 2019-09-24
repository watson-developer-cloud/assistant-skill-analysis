import sys
import unittest
import json
import pandas as pd

from assistant_dialog_skill_analysis.utils import skills_util

class TestSkillsUtil(unittest.TestCase):
    """Test for skills utils module"""

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.skill_file = open(
            "tests/resources/test_workspaces/skill-Customer-Care-Sample.json",
            "r")

    def test_extract_workspace_data(self):
        skill_json = json.load(self.skill_file)
        workspace_data, workspace_vocabulary = skills_util.extract_workspace_data(skill_json)
        workspace_df = pd.DataFrame(workspace_data)
        self.assertNotEqual(workspace_data, None, "Extract workspace failed")
        self.assertEqual(len(workspace_df["intent"].unique()), 9, "Extract workspace failed")

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.skill_file.close()

if __name__ == '__main__':
    unittest.main()
