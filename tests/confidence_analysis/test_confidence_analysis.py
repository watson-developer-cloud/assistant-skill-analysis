import unittest
import math
import pandas as pd
from assistant_dialog_skill_analysis.confidence_analysis.confidence_analyzer import \
    _get_far_list, _get_ontopic_accuracy_list, _find_threshold, abnormal_conf, \
    generate_unique_thresholds, _get_bot_coverage_list, analysis, \
    extract_table_analysis, _convert_data_format, analysis_pipeline, extract_by_topic, \
    create_display_table
from assistant_dialog_skill_analysis.utils.skills_util import OFFTOPIC_LABEL


class TestThresholdAnalysis(unittest.TestCase):
    """Test for summary generator module"""

    def setUp(self):

        self.sorted_list = [(OFFTOPIC_LABEL, 'B', 0.1),
                            ('A', 'A', 0.2),
                            (OFFTOPIC_LABEL, 'A', 0.2),
                            ('A', 'A', 0.2),
                            ('A', 'B', 0.3),
                            ('A', 'A', 0.4),
                            (OFFTOPIC_LABEL, 'A', 0.8),
                            ('A', 'A', 0.8),
                            ('B', 'B', 1.0)]

        self.results = pd.DataFrame(self.sorted_list,
                                    columns=['correct_intent',
                                             'top_intent',
                                             'top_confidence'])

        self.thresholds = [0.15, 0.25, 0.35, 0.6, 0.9]

    def test_abnormal_conf(self):
        test_dataframe = pd.DataFrame({'correct_intent': ['A', 'A', 'A'],
                                       'top_intent': ['A', 'B', 'B'],
                                       'top_confidence': [.2, .9, .9],
                                       'utterance': ['a', 'a', 'a'],
                                       'top_predicts': [[], [],
                                                        [{'intent': 'B', 'confidence': .9},
                                                         {'intent': 'C', 'confidence': .1}]
                                                        ]})

        correct_low_conf, incorrect_high_conf = abnormal_conf(test_dataframe, .3, .7)
        self.assertEqual(len(correct_low_conf), 1, "test_find_threshold failed: first")
        self.assertEqual(len(incorrect_high_conf), 2, "test_find_threshold failed: first")
        self.assertEqual(incorrect_high_conf.iloc[0, :]['top2_prediction'],
                         'NA',
                         "test_find_threshold failed: first")

    def test_get_ontopic_accuracy_list(self):

        res, _ = _get_ontopic_accuracy_list(self.sorted_list, self.thresholds)
        gt = [5/6, 0.75, 1.0, 1.0, 1.0]
        for r, g in zip(res, gt):
            self.assertEqual(math.fabs(r - g) < 0.0001, True, "FAR values changed")

    def test_find_threshold(self):
        a = 0.0
        b = [0, 0.1, 0.2, 0.3, 0.5]
        pos = _find_threshold(a, b)
        self.assertEqual(pos, 0, "test_find_threshold failed: first")
        a = 0.1
        b = [0, 0.1, 0.2, 0.3, 0.5]
        pos = _find_threshold(a, b)
        self.assertEqual(pos, 1, "test_find_threshold failed: second")
        a = 0.5
        b = [0, 0.1, 0.2, 0.3, 0.5]
        pos = _find_threshold(a, b)
        self.assertEqual(pos, 4, "test_find_threshold failed: third")

    def test_get_far_list(self):

        res, _ = _get_far_list(self.sorted_list, self.thresholds)
        gt = [2/3, 1/3, 1/3, 1/3, 0.0]
        for r, g in zip(res, gt):
            self.assertEqual(math.fabs(r - g) < 0.0001, True, "FAR values changed")

    def test_get_bot_coverage_list(self):

        res, _ = _get_bot_coverage_list(self.sorted_list, self.thresholds)
        gt = [0.888, 0.555, 0.444, 0.333, 0.111]
        for r, g in zip(res, gt):
            self.assertEqual(math.isclose(r, g, abs_tol=.01, rel_tol=.0001), True,
                             "bot coverage value change")

    def test_analysis(self):
        analysis_df1 = analysis(self.results)
        self.assertEqual(analysis_df1['Bot Coverage Counts'].iloc[9], '1 / 9', 'analysis fail')
        analysis_df_list = analysis(self.results, ['A'])
        self.assertEqual(analysis_df_list[0]['Bot Coverage Counts'].iloc[9], '2 / 7', 'analysis fail')

    def test_convert_data_format(self):

        test1 = _convert_data_format(self.results)
        for element1, element2 in zip(test1, self.sorted_list):
            for ele1, ele2 in zip(element1, element2):
                self.assertEqual(ele1, ele2, 'test for covert data format fail')

    def test_analysis_pipeline(self):
        analysis_df = analysis_pipeline(self.results)
        self.assertEqual(analysis_df['Bot Coverage Counts'].iloc[9], '1 / 9', 'analysis fail')

    def test_extract_table_analysis(self):
        sorted_results = _convert_data_format(self.results)
        ontopic_infos, offtopics_infos = extract_by_topic(sorted_results)
        analysis_df, toa_list, bot_coverage_list, far_list, thresholds = \
            extract_table_analysis(sorted_results, ontopic_infos, offtopics_infos)
        self.assertEqual(math.isclose(toa_list[2], .75, abs_tol=.01, rel_tol=.0001),
                         True,
                         'extract table analysis fail')

    def test_create_display_table(self):
        sorted_results = _convert_data_format(self.results)
        thresholds, sort_uniq_confs = generate_unique_thresholds(sorted_results)
        toa_list, toa_count = _get_ontopic_accuracy_list(sorted_results, thresholds)
        bot_coverage_list, bot_coverage_count = _get_bot_coverage_list(sorted_results, thresholds)
        ontopic_infos, offtopic_infos = extract_by_topic(sorted_results)
        far_list, _ = _get_far_list(sorted_results, thresholds)
        analysis_df = create_display_table(toa_list,
                                           bot_coverage_list,
                                           bot_coverage_count,
                                           sorted_results,
                                           thresholds,
                                           offtopic_infos,
                                           far_list)

        self.assertEqual(analysis_df['Bot Coverage Counts'].iloc[-1], '1 / 9', 'create display \
                         table test fail')

    def test_generate_unique_thresholds(self):
        thresholds, unique_confidence = generate_unique_thresholds(self.sorted_list)
        self.assertEqual(math.isclose(thresholds[1], .15, abs_tol=.01, rel_tol=.0001), True, "test \
                         generate unique threshold fail")

        self.assertEqual(math.isclose(unique_confidence[5], 1, abs_tol=.01, rel_tol=.0001), True,
                         "test generate unique threshold fail")

    def tearDown(self):
        unittest.TestCase.tearDown(self)


if __name__ == '__main__':
    unittest.main()
