import unittest
import json
import pandas as pd

from assistant_dialog_skill_analysis.utils import skills_util, lang_utils


class TestSkillsUtil(unittest.TestCase):
    """Test for skills utils module"""

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.skill_file = open(
            "tests/resources/test_workspaces/skill-Customer-Care-Sample.json", "r"
        )
        self.lang_util = lang_utils.LanguageUtility("en")

    def test_extract_workspace_data(self):
        skill_json = json.load(self.skill_file)
        workspace_pd, workspace_vocabulary = skills_util.extract_workspace_data(
            skill_json, self.lang_util
        )
        self.assertTrue(workspace_pd is not None, "Extract workspace failed")
        self.assertEqual(
            len(workspace_pd["intent"].unique()), 9, "Extract workspace failed"
        )

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.skill_file.close()


if __name__ == "__main__":
    unittest.main()
