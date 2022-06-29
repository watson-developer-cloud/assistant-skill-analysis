import unittest
import json

from assistant_skill_analysis.utils import skills_util, lang_utils


class TestSkillsUtil(unittest.TestCase):
    """Test for skills utils module"""

    @classmethod
    def setUpClass(cls):
        cls.skill_file = open(
            "tests/resources/test_workspaces/skill-Customer-Care-Sample.json", "r"
        )
        cls.action_skill_file = open(
            "tests/resources/test_workspaces/customer_care_sample_action_skill.json", "r"
        )
        cls.lang_util = lang_utils.LanguageUtility("en")

    def test_extract_action_workspace_data(self):
        skill_json = json.load(self.action_skill_file)
        workspace_pd, workspace_vocabulary, entities, intent_action_map = skills_util.extract_workspace_data(
            skill_json, self.lang_util
        )

        self.assertTrue(workspace_pd is not None, "Extract workspace failed")
        self.assertEqual(
            len(workspace_pd["intent"].unique()), 7, "Extract workspace failed"
        )

        # check correct number of entities parsed
        self.assertEqual(7, len(entities))

        # check intent to action mapping working expectedly
        self.assertEqual('Where are you located?', intent_action_map['action_11419_intent_44259'])
        self.assertEqual('Thank you', intent_action_map['action_12038_intent_13364'])
        self.assertEqual('Goodbye', intent_action_map['action_22890_intent_48257'])
        self.assertEqual('Schedule An Appointment', intent_action_map['action_27164_intent_22860'])
        self.assertEqual('What are your hours?', intent_action_map['action_33190_intent_33203'])
        self.assertEqual('What can I do?', intent_action_map['action_5042_intent_38841'])
        self.assertEqual('Fallback', intent_action_map['fallback_connect_to_agent'])

    def test_extract_workspace_data(self):
        skill_json = json.load(self.skill_file)
        workspace_pd, workspace_vocabulary, _, _ = skills_util.extract_workspace_data(
            skill_json, self.lang_util
        )
        self.assertTrue(workspace_pd is not None, "Extract workspace failed")
        self.assertEqual(
            len(workspace_pd["intent"].unique()), 9, "Extract workspace failed"
        )

    @classmethod
    def tearDownClass(cls):
        cls.skill_file.close()
        cls.action_skill_file.close()


if __name__ == "__main__":
    unittest.main()
