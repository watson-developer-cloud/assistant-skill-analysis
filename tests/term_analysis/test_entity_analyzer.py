import unittest
import json
import pandas as pd

from assistant_dialog_skill_analysis.utils import skills_util
from assistant_dialog_skill_analysis.term_analysis import entity_analyzer

class TestChi2Analyzer(unittest.TestCase):
    '''
    Test for Chi2 Analyzer module
    '''
    def setUp(self):
        unittest.TestCase.setUp(self)
        test_skill_file = "tests/resources/test_workspaces/skill-Customer-Care-Sample.json"
        with open(test_skill_file, "r") as skill_file:
            workspace_data, workspace_vocabulary = skills_util.extract_workspace_data(
                json.load(skill_file))
            self.workspace_df = pd.DataFrame(workspace_data)
            self.mock_test_result = pd.DataFrame(
                {'correct_intent':['intent1', 'intent2'],
                 'entities':[
                     [{'entity' : 'entity1', 'confidence':1},
                      {'entity' : 'entity2', 'confidence':1}],
                     [{'entity' : 'entity1', 'confidence':.5}]]
                 })

    def test_derive_entity_label_matrix(self):
        entity_feat_mat, labels, entity_avg_conf = entity_analyzer._derive_entity_label_matrix(
            self.mock_test_result, ['entity1', 'entity2'])
        self.assertEqual(entity_feat_mat[1][1], 0,
                         'test for entity analyzer fail')
        self.assertEqual(entity_avg_conf['entity1'], .75,
                         'test for entity analyzer fail')

    def test_entity_label_correlation_analysis(self):
        entity = {'entities' : [{'entity' : 'entity1'}, {'entity' : 'entity2'}]}
        entities_list = [item['entity'] for item in entity['entities']]
        entity_label_df = entity_analyzer.entity_label_correlation_analysis(
            self.mock_test_result, entities_list, p_value=1)
        self.assertEqual(entity_label_df.iloc[0]['Correlated Entities'], 'entity2',
                         'test for entity analyzer fail')

    def tearDown(self):
        unittest.TestCase.tearDown(self)

if __name__ == '__main__':
    unittest.main()
