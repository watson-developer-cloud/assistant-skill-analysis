import unittest
import json
import pandas as pd

from assistant_dialog_skill_analysis.data_analysis import summary_generator
from assistant_dialog_skill_analysis.utils import skills_util


class TestSummaryGenerator(unittest.TestCase):
    """Test for summary generator module"""

    def setUp(self):
        unittest.TestCase.setUp(self)
        with open("tests/resources/test_workspaces/skill-Customer-Care-Sample.json", "r") \
                as skill_file:
            workspace_data, workspace_vocabulary = \
                skills_util.extract_workspace_data(json.load(skill_file))
            self.workspace_df = pd.DataFrame(workspace_data)

    def test_class_imbalance(self):

        is_imbalanced = summary_generator.class_imbalance_analysis(self.workspace_df)
        self.assertEqual(is_imbalanced, True, "Test class imbalance detection failed")

    def tearDown(self):
        unittest.TestCase.tearDown(self)


if __name__ == '__main__':
    unittest.main()
