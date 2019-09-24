import sys
import unittest
import json
import pandas as pd

from assistant_dialog_skill_analysis.utils import skills_util
from assistant_dialog_skill_analysis.term_analysis import chi2_analyzer

class TestChi2Analyzer(unittest.TestCase):
    """Test for Chi2 Analyzer module"""

    def setUp(self):
        unittest.TestCase.setUp(self)
        test_skill_file = "tests/resources/test_workspaces/skill-Customer-Care-Sample.json"
        with open(test_skill_file, "r") as skill_file:
            workspace_data, workspace_vocabulary = skills_util.extract_workspace_data(
                json.load(skill_file))
            self.workspace_df = pd.DataFrame(workspace_data)

    def test_strip_punctuations(self):
        test_utterance = 'Boston\'s located in Mass!'
        self.assertEqual(chi2_analyzer.strip_punctuations(test_utterance).strip(),
                         'Boston is located in Mass',
                         'test for strip punctuations fail'
                         )

    def test_preprocess_chi2(self):
        test_data = pd.DataFrame({'utterance':['This is boston'], 'intent': 'label1'})
        labels, convec, features = chi2_analyzer._preprocess_chi2(test_data)
        self.assertEqual(set(convec.get_feature_names()),
                         set(['this', 'boston', 'this boston']),
                         'Test for chi2 analyzer fail')

        labels, convec, features = chi2_analyzer._preprocess_chi2(self.workspace_df)
        max_len = 0
        for ngram in convec.get_feature_names():
            if len(ngram.split(' ')) > max_len:
                max_len = len(ngram.split(' '))
        assert max_len <= 2

    def test_compute_chi2_top_feature(self):
        # test case 1, mini dataset
        test_data = pd.DataFrame({'utterance':['Boston is the capital city of massachusetts ',
                                               'Boston Celtics is a famous NBA team',
                                               'new york is a big city in the east coast'],
                                  'intent': ['boston', 'boston', 'nyc']})

        labels, con_vec, features = chi2_analyzer._preprocess_chi2(test_data)
        unigrams, bigrams = chi2_analyzer._compute_chi2_top_feature(
            features, labels, con_vec, 'boston', .05)
        self.assertEqual(len(unigrams), 0, 'chi2 analyzer fail')
        self.assertEqual(len(bigrams), 0, 'chi2 analyzer fail')

        # test case 2 with punctuation
        test_data = pd.DataFrame({'utterance':['Boston is the capital city of massachusetts! ',
                                               'Boston Celtics is a famous NBA team!',
                                               'new york is a big city in the east coast'],
                                  'intent': ['boston', 'boston', 'nyc']})
        labels, con_vec, features = chi2_analyzer._preprocess_chi2(test_data)
        unigrams, bigrams = chi2_analyzer._compute_chi2_top_feature(
            features, labels, con_vec, 'boston', 1)
        self.assertEqual('!' not in unigrams, True, 'chi2 analyzer fail')

        # test case 3 , medium size dataset
        labels, con_vec, features = chi2_analyzer._preprocess_chi2(self.workspace_df)
        unigrams, bigrams = chi2_analyzer._compute_chi2_top_feature(
            features, labels, con_vec, 'Help')
        self.assertEqual(unigrams, ['need', 'me', 'assist', 'assistance', 'decide', 'help'],
                         'chi2 analyzer fail')
        test_bigrams = ['assist me', 'you assist', 'me decide',
                        'need assistance', 'you help', 'help me']
        self.assertEqual(bigrams, test_bigrams, 'chi2 analyzer fail')

    def test_get_chi2_analysis(self):
        test_data = pd.DataFrame(
            {'utterance': ['Boston is the capital city of massachusetts ',
                           'Boston Celtics is a famous NBA team',
                           'new york is a big city in the east coast'],
             'intent': ['boston', 'boston', 'nyc']})
        unigram_intent_dict, bigram_intent_dict = chi2_analyzer.get_chi2_analysis(
            test_data, significance_level=.05)
        self.assertEqual(len(unigram_intent_dict), 0, 'chi2 analyzer fail')

        unigram_intent_dict, bigram_intent_dict = chi2_analyzer.get_chi2_analysis(
            self.workspace_df, significance_level=.05)
        list_of_intent_list = list(unigram_intent_dict.values())
        one_bigram_set = list(bigram_intent_dict.keys())[0]
        self.assertEqual(all(len(intents) >= 1 for intents in list_of_intent_list), True,
                         'chi2 analyzer fail')
        self.assertEqual(all(len(item.split(' ')) == 2 for item in one_bigram_set), True,
                         'chi2 analyzer fail')

    def test_get_confusing_keyterms(self):
        unigram_intent_dict = {
            frozenset(['a', 'b', 'c']) : ['intent1'],
            frozenset(['a', 'b']) : ['intent2']}
        ambiguous_data_frame = chi2_analyzer.get_confusing_key_terms(unigram_intent_dict)
        self.assertTrue(str(ambiguous_data_frame.iloc[0, 0]) == '<intent1, intent2>' \
                        or str(ambiguous_data_frame.iloc[0, 0]) == '<intent2, intent1>',
                         'chi2 analyzer fail')
        self.assertEqual('a' in list(ambiguous_data_frame['Terms']), True,
                         'chi2 analyzer fail')

    def tearDown(self):
        unittest.TestCase.tearDown(self)

if __name__ == '__main__':
    unittest.main()
