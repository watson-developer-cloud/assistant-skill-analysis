import sys
import unittest
import json
import numpy as np
import pandas as pd

from assistant_dialog_skill_analysis.utils import skills_util
from assistant_dialog_skill_analysis.term_analysis import keyword_analyzer

class TestKeywordAnalyzer(unittest.TestCase):
    '''
    Test for Key Word Analyzer module
    '''
    def setUp(self):
        unittest.TestCase.setUp(self)
        test_skill_file = "tests/resources/test_workspaces/skill-Customer-Care-Sample.json"
        with open(test_skill_file, "r") as skill_file:
            workspace_data, workspace_vocabulary = skills_util.extract_workspace_data(
                json.load(skill_file))
            self.workspace_df = pd.DataFrame(workspace_data)
            self.test_data = pd.DataFrame(
                {'utterance': ['Boston is the capital city of massachusetts ',
                               'Boston Celtics is a famous NBA team',
                               'new york is a big city in the east coast'],
                 'intent': ['boston', 'boston', 'nyc']})

    def test_get_counts_per_label(self):
        counts = keyword_analyzer._get_counts_per_label(self.test_data)
        self.assertEqual(('boston', 'Celtics') in counts.index.tolist(), True,
                         'Key word analyzer test fails')
        self.assertEqual(('nyc', 'coast') in counts.index.tolist(), True,
                         'Key word analyzer test fails')
        self.assertEqual(('boston', 'is') in counts.index.tolist(), False,
                         'Key word analyzer test fails')

    def test_get_top_n(self):
        counts = keyword_analyzer._get_counts_per_label(self.test_data)
        top_n = keyword_analyzer._get_top_n(counts['n_w'], top_n=4)
        labels = [item for (item, _) in top_n.index.tolist() if item == 'boston']
        self.assertEqual(len(labels), 4, 'Key word analyzer test fails')

    def test_preprocess_for_heat_map(self):
        counts, top_counts = keyword_analyzer._preprocess_for_heat_map(
            self.workspace_df, label_for_display=30, max_token_display=30, class_list=None)
        unique_counts = len(counts.index.get_level_values(0).unique())
        actual_labels_shown = np.int(np.ceil(30 / unique_counts)) * unique_counts
        self.assertEqual(len(top_counts) == actual_labels_shown, True,
                         'Key word analyzer test fails')

    def tearDown(self):
        unittest.TestCase.tearDown(self)

if __name__ == '__main__':
    unittest.main()
