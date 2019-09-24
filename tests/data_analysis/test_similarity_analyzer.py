import unittest
import json
import pandas as pd
import numpy as np

from assistant_dialog_skill_analysis.data_analysis import similarity_analyzer
from assistant_dialog_skill_analysis.utils import skills_util

TOLERANCE = .0000001

class TestSimilarityAnalzyer(unittest.TestCase):
    """Test for Similarity Analyzer module"""
    def setUp(self):
        with open("tests/resources/test_workspaces/skill-Customer-Care-Sample.json", "r") \
                as skill_file:
            workspace_data, workspace_vocabulary = \
                skills_util.extract_workspace_data(json.load(skill_file))
            self.workspace_df = pd.DataFrame(workspace_data)

    def test_calculate_cosine_similarity(self):
        feature_matrix1 = np.array([[1, 2, 0], [0, 0, 1], [1, 2, 0]])
        cos_sim_score1 = similarity_analyzer._calculate_cosine_similarity(feature_matrix1)
        self.assertEqual(np.abs(np.sum(np.diag(cos_sim_score1) - np.array([1, 1, 1]))) < TOLERANCE,
                         True,
                         "Similarity Analyzer Test fail")

        self.assertEqual(np.abs(cos_sim_score1[0, 1]) < TOLERANCE,
                         True,
                         "Similarity Analyzer Test fail")

        self.assertEqual(np.abs(cos_sim_score1[0, 2] - 1) < TOLERANCE,
                         True,
                         "Similarity Analyzer Test fail")

    def test_ambiguous_examples_analysis(self):
        ambiguous_dataframe = similarity_analyzer.ambiguous_examples_analysis(self.workspace_df,
                                                                              threshold=.85)
        self.assertEqual(len(ambiguous_dataframe[ambiguous_dataframe['similarity score'] < .85]),
                         0,
                         "Similarity Analyzer Test fail")

        self.assertEqual(len(np.intersect1d(ambiguous_dataframe['Intent1'],
                                            ambiguous_dataframe['Intent2'])),
                         0,
                         "Similarity Analyzer Test fail")

    def tearDown(self):
        unittest.TestCase.tearDown(self)


if __name__ == '__main__':
    unittest.main()
