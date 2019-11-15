import unittest
import sys

from assistant_dialog_skill_analysis.utils import skills_util

class TestNotebook(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        CONFIG_FILE = './wa_config.txt'
        with open(CONFIG_FILE) as fi:
            self.apikey = fi.readline().strip()
            self.wksp_id = fi.readline().strip()

    def test_notebook(self):
        test_file = 'tests/resources/test_workspaces/customer_care_skill_test.tsv'
        nb, errors = skills_util.run_notebook('skill_analysis.ipynb', self.apikey, self.wksp_id, test_file, 'notebook_output')
        self.assertEqual(errors, [])

    def tearDown(self):
        unittest.TestCase.tearDown(self)

if __name__ == '__main__':
    unittest.main()