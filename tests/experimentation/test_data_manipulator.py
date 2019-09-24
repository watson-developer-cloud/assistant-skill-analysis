from collections import Counter
import sys
import unittest
import json
import numpy as np
import pandas as pd

from assistant_dialog_skill_analysis.utils import skills_util
from assistant_dialog_skill_analysis.experimentation import data_manipulator


class TestDataManipulator(unittest.TestCase):
    """Test for Data manipulator module"""

    def setUp(self):
        unittest.TestCase.setUp(self)
        with open("tests/resources/test_workspaces/skill-Customer-Care-Sample.json", "r") \
                as skill_file:
            self.workspace = json.load(skill_file)
            workspace_data, workspace_vocabulary = \
                skills_util.extract_workspace_data(self.workspace)
            self.workspace_df = pd.DataFrame(workspace_data)

    def test_undersampling(self):
        quantile = .6
        train_workspace_data = data_manipulator.under_sampling(self.workspace,
                                                               self.workspace_df,
                                                               quantile)
        label_frequency_dict = dict(Counter(self.workspace_df['intent']).most_common())
        sampling_threshold = int(np.quantile(a=list(label_frequency_dict.values()),
                                             q=[quantile])[0])
        example_length = np.array([len(train_workspace_data[i]['examples'])
                                   for i in range(len(train_workspace_data))])
        self.assertEqual(np.sum(example_length <= sampling_threshold),
                         len(example_length),
                         'Data manipulator test fail')

    def tearDown(self):
        unittest.TestCase.tearDown(self)


if __name__ == '__main__':
    unittest.main()
