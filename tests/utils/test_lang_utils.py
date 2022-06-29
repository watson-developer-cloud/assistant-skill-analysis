import unittest
from assistant_skill_analysis.utils.lang_utils import LanguageUtility


class TestLangUtils(unittest.TestCase):
    """Test for lang utils module"""

    @classmethod
    def setUpClass(cls):
        cls.skill_file = open(
            "tests/resources/test_workspaces/skill-Customer-Care-Sample.json", "r"
        )

    def test_en(self):
        util = LanguageUtility("en")
        sent = util.preprocess("boston resided on the east coast of us!")
        self.assertEqual(sent, "boston resided on the east coast of us ")
        sent = util.tokenize(sent)
        self.assertEqual(
            sent, ["boston", "resid", "on", "the", "east", "coast", "of", "us"]
        )

    def test_fr(self):
        util = LanguageUtility("fr")
        sent = util.preprocess("ils se donnèrent")
        self.assertEqual(sent, "ils se donnerent")
        sent = util.tokenize(sent)
        self.assertEqual(sent, ["il", "se", "donnerent"])

    def test_cs(self):
        util = LanguageUtility("cs")
        sent = util.preprocess("ils se donnèrent")
        sent = util.tokenize(sent)
        self.assertEqual(sent, ["ils", "se", "donnerent"])

    def test_pt(self):
        util = LanguageUtility("pt")
        sent = util.preprocess("ils se donnèrent")
        self.assertEqual(sent, "ils se donnerent")
        sent = util.tokenize(sent)
        self.assertEqual(sent, ["ils", "se", "donnerent"])

    def test_it(self):
        util = LanguageUtility("it")
        sent = util.preprocess("pronuncerà")
        self.assertEqual(sent, "pronuncerà")
        sent = util.tokenize(sent)
        self.assertEqual(sent, ["pronunc"])

    def test_es(self):
        util = LanguageUtility("es")
        sent = util.preprocess("toreándolo")
        self.assertEqual(sent, "toreandolo")
        sent = util.tokenize(sent)
        self.assertEqual(sent, ["tor"])

    def test_de(self):
        util = LanguageUtility("de")
        sent = util.preprocess("Autobahnen")
        self.assertEqual(sent, "autobahnen")
        sent = util.tokenize(sent)
        self.assertEqual(sent, ["autobahn"])

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.skill_file.close()


if __name__ == "__main__":
    unittest.main()
