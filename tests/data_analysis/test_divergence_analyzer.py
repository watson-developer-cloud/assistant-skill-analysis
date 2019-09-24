import unittest
import json
import pandas as pd
import numpy as np
from assistant_dialog_skill_analysis.utils import skills_util
from assistant_dialog_skill_analysis.data_analysis import divergence_analyzer


class TestDivergenceAnalyzer(unittest.TestCase):
    """Test for Divergence Analyzer module"""

    def setUp(self):
        unittest.TestCase.setUp(self)
        with open("tests/resources/test_workspaces/skill-Customer-Care-Sample.json",
                  "r") as skill_file:
            workspace_data, workspace_vocabulary = \
                skills_util.extract_workspace_data(json.load(skill_file))

            self.workspace_df = pd.DataFrame(workspace_data)
            self.train_set_pd = pd.DataFrame({'utterance': ['boston is close to new york'],
                                              'intent': ['Boston_New_York']})
            self.test_set_pd = pd.DataFrame(
                {'utterance': ['both boston and new york are on east coast',
                               'boston is close to new york'],
                 'intent': ['Boston_New_York', 'Boston_New_York']})

    def test_label_percentage(self):
        label_percentage_dict = divergence_analyzer._label_percentage(self.workspace_df)
        label_percentage_vec = np.array(list(label_percentage_dict.values()))
        self.assertEqual(np.all(label_percentage_vec > 0), True, "label percentage test fail")
        self.assertEqual(np.sum(label_percentage_vec), 1, "label percentage test fail")

    def test_train_test_vocab_difference(self):
        train_vocab, test_vocab = \
            divergence_analyzer._train_test_vocab_difference(self.train_set_pd, self.test_set_pd)

        self.assertEqual(train_vocab, set(['boston', 'is', 'close', 'to', 'new', 'york']),
                         "train test vocab difference test fail")

    def test_train_test_uttterance_length_difference(self):
        temp_df = divergence_analyzer._train_test_utterance_length_difference(self.train_set_pd,
                                                                              self.test_set_pd)

        self.assertEqual(temp_df.iloc[0]['Absolute Difference'],
                         1.5, 'train test utterance length differene test fail')

    def test_train_test_label_difference(self):
        # Test 1
        percentage_dict1 = {'Intent1': .5, 'Intent2': .5}
        percentage_dict2 = {'Intent1': .5, 'Intent2': .5}

        missing_labels, difference_dict, js_distance = \
            divergence_analyzer._train_test_label_difference(percentage_dict1, percentage_dict2)
        self.assertEqual(js_distance, 0, "train test difference test fail")
        self.assertEqual(missing_labels, [], "train test difference test fail")
        self.assertEqual(difference_dict['Intent1'], [50, 50, 0], "train test difference test fail")

        # Test 2
        percentage_dict1 = {'Intent1': 1, 'Intent2': 0}
        percentage_dict2 = {'Intent1': 1}

        missing_labels, difference_dict, js_distance = \
            divergence_analyzer._train_test_label_difference(percentage_dict1, percentage_dict2)
        self.assertEqual(js_distance, 0, "train test difference test fail")
        self.assertEqual(missing_labels, ['Intent2'], "train test difference test fail")
        self.assertEqual(difference_dict['Intent1'],
                         [100, 100, 0],
                         "train test difference test fail")

        # Test 3
        percentage_dict1 = {'Intent1': 1, 'Intent2': 0}
        percentage_dict2 = {'Intent1': 0, 'Intent2': 1}
        missing_labels, difference_dict, js_distance = \
            divergence_analyzer._train_test_label_difference(percentage_dict1, percentage_dict2)
        self.assertEqual(js_distance, 1, "train test difference test fail")
        self.assertEqual(difference_dict['Intent1'],
                         [100, 0, 100],
                         "train test difference test fail")
        self.assertEqual(difference_dict['Intent2'],
                         [0, 100, 100],
                         "train test difference test fail")
        self.assertEqual(len(missing_labels), 0, "train test difference test fail")

        # Test 4
        percentage_dict1 = {'Intent1': 1}
        percentage_dict2 = {'Intent2': 1}
        missing_labels, difference_dict, js_distance = \
            divergence_analyzer._train_test_label_difference(percentage_dict1, percentage_dict2)
        self.assertEqual(str(js_distance), 'nan', "train test difference test fail")
        self.assertEqual(missing_labels, ['Intent1'], "train test difference test fail")
        self.assertEqual(len(difference_dict), 0, "train test difference test fail")

    def tearDown(self):
        unittest.TestCase.tearDown(self)


if __name__ == '__main__':
    unittest.main()
