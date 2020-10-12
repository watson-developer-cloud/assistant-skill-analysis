import unittest
from assistant_dialog_skill_analysis.utils.lang_utils import LanguageUtility


class TestLangUtils(unittest.TestCase):
    """Test for lang utils module"""

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.skill_file = open(
            "tests/resources/test_workspaces/skill-Customer-Care-Sample.json", "r"
        )

    def test_en(self):
        util = LanguageUtility("en")
        sent = util.preprocess("boston resided on the east coast of us!")
        self.assertTrue("!" not in sent)
        sent = util.tokenize(sent)
        self.assertTrue("resid" in sent)

    def test_fr(self):
        util = LanguageUtility("fr")
        sent = util.preprocess("ils se donnèrent")
        self.assertTrue("è" not in sent)
        sent = util.tokenize(sent)
        self.assertTrue(len(sent) == 3)

    def test_es(self):
        util = LanguageUtility("es")
        sent = util.preprocess("ils se donnèrent")
        self.assertTrue("è" not in sent)
        sent = util.tokenize(sent)
        self.assertTrue(len(sent) == 3)

    def test_cs(self):
        util = LanguageUtility("cs")
        sent = util.preprocess("ils se donnèrent")
        self.assertTrue("è" not in sent)
        sent = util.tokenize(sent)
        self.assertTrue(len(sent) == 3)

    def test_de(self):
        util = LanguageUtility("cs")
        sent = util.preprocess("ils se donnèrent")
        self.assertTrue("è" not in sent)
        sent = util.tokenize(sent)
        self.assertTrue(len(sent) == 3)

    def test_pt(self):
        util = LanguageUtility("cs")
        sent = util.preprocess("ils se donnèrent")
        self.assertTrue("è" not in sent)
        sent = util.tokenize(sent)
        self.assertTrue(len(sent) == 3)

    def test_it(self):
        util = LanguageUtility("cs")
        sent = util.preprocess("ils se donnèrent")
        self.assertTrue("è" not in sent)
        sent = util.tokenize(sent)
        self.assertTrue(len(sent) == 3)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.skill_file.close()


if __name__ == "__main__":
    unittest.main()
