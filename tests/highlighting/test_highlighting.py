import unittest
import math
import os
import shutil
import numpy as np
import pandas as pd

from assistant_skill_analysis.highlighting import highlighter
from assistant_skill_analysis.inferencing import inferencer
from assistant_skill_analysis.utils import skills_util, lang_utils


CONFIG_FILE = "./wa_config.txt"
THREAD_NUM = 5
TOLERANCE = 0.3


def compare(a, b):
    if len(a) != len(b):
        return False
    for i, j in zip(a, b):
        if math.fabs(i - j) > 0.01:
            return False
    return True


@unittest.skip("takes too long to run in github action")
class TestHighLighting(unittest.TestCase):
    """
    Test for summary generator module
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpfolder = "tests/resources/highlight_temp_folder/"
        cls.tmpbatchfolder = "tests/resources/highlight_temp_folder/batch"
        cls.input_file = "tests/resources/test_workspaces/customer_care_skill_test.tsv"
        cls.lang_util = lang_utils.LanguageUtility("en")
        unittest.TestCase.setUp(cls)
        with open(CONFIG_FILE) as fi:
            cls.apikey = fi.readline().strip()
            cls.wksp_id = fi.readline().strip()

        cls.conversation = skills_util.retrieve_conversation(iam_apikey=cls.apikey,
                                                             url=skills_util.DEV_DATACENTER[0],
                                                             authenticator_url=skills_util.DEV_DATACENTER[1],
                                                             )

        if not os.path.exists(cls.tmpfolder):
            os.makedirs(cls.tmpfolder)
            os.makedirs(cls.tmpbatchfolder)

        test_df = skills_util.process_test_set(cls.input_file, cls.lang_util)
        cls.results = inferencer.inference(
            cls.conversation,
            test_df,
            max_thread=THREAD_NUM,
            workspace_id=cls.wksp_id,
        )

    def test_filter_results(self):
        wrong_examples_sorted = highlighter._filter_results(
            self.results, 0.4, self.lang_util
        )
        ground_truth = (
            47,
            "no he is an arrogant self serving immature idiot get it right",
            None,
            "General_Connect_to_Agent",
            0.6697888851165772,
            0.09834358692169187,
            9,
        )

        self.assertEqual(
            wrong_examples_sorted[0][2], None, "Test for filter results fail"
        )
        self.assertEqual(
            math.isclose(
                wrong_examples_sorted[0][4], ground_truth[4], rel_tol=0.01, abs_tol=0.1
            ),
            True,
            "Test for filter results fail",
        )

    def test_generate_adversarial_examples(self):
        test_utterance = "winter is coming"
        (
            adversarial_examples,
            adversarial_span,
        ) = highlighter._generate_adversarial_examples(test_utterance, 1)
        self.assertEqual(
            "winter coming" in adversarial_examples,
            True,
            "Test for generate adversarial example fail",
        )
        self.assertEqual(
            adversarial_span["winter coming_1"],
            (1, 2),
            "Test for generate adversarial example fail",
        )

    def test_adversarial_examples_multi_thread_inference(self):
        long_example1 = (
                "um taking a shot here um lets say three "
                + "separate people whos wills are to each other"
        )
        wrong_examples_sorted = [
            (
                1,
                "see ya",
                "Goodbye",
                "General_Greetings",
                0.5005551099777221,
                0.5005551099777221,
                1,
            ),
            (
                42,
                long_example1,
                None,
                "General_Connect_to_Agent",
                0.6537539958953857,
                0.2537539958953857,
                9,
            ),
        ]

        (
            adv_results,
            adv_dict_span,
        ) = highlighter._adversarial_examples_multi_thread_inference(
            wrong_examples_sorted, self.conversation, self.wksp_id
        )
        result = adv_results[adv_results["utterance"].str.match("see")]

        self.assertEqual(
            np.abs(result["top_confidence"].values[0] - 0.478708) < TOLERANCE,
            True,
            "Test for adversarial examples inference fail",
        )

        self.assertEqual(
            adv_dict_span["see_1"],
            (1, 2),
            "test for adversarial example inference: adversarial span dict mismatch",
        )

    def test_scoring_function(self):
        highlight = np.zeros(3, dtype="float32")
        highlight = highlighter._scoring_function(
            highlight=highlight,
            original_position=0,
            adversarial_position=1,
            original_confidence=0.7,
            adversarial_confidence=0.5,
            start_idx=2,
            end_idx=3,
        )
        self.assertEqual(
            math.isclose(highlight[2], 0.44, rel_tol=0.0001, abs_tol=0.01),
            True,
            "Test for adversarial scoring fail",
        )

    def test_highlight_scoring(self):
        original_example = (
            1,
            "see ya",
            "Goodbye",
            "General_Greetings",
            0.5005551099777221,
            0.5005551099777221,
            1,
        )
        subset_adversarial_result = pd.DataFrame(
            data={
                "utterance": ["see"],
                "top_predicts": [
                    [
                        {"intent": "General_Greetings", "confidence": 0.6},
                        {"intent": "Goodbye", "confidence": 0.5},
                    ]
                ],
            }
        )
        adversarial_span_dict = {"see_1": (1, 2)}

        highlight = highlighter._highlight_scoring(
            original_example,
            subset_adversarial_result,
            adversarial_span_dict,
            self.lang_util,
        )
        self.assertEqual(
            compare(highlight, [0, -0.41988897]),
            True,
            "Test highlight scoring function fail",
        )

    def test_get_highlights_in_batch_multi_thread(self):
        highlighter.get_highlights_in_batch_multi_thread(
            conversation=self.conversation,
            full_results=self.results,
            output_folder=self.tmpbatchfolder,
            confidence_threshold=0.4,
            show_worst_k=3,
            lang_util=self.lang_util,
            workspace_id=self.wksp_id,
        )
        self.assertEqual(
            len(os.listdir(self.tmpbatchfolder)),
            3,
            "# of batch highlighting files is mismatched.",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpfolder)
        unittest.TestCase.tearDown(cls)


if __name__ == "__main__":
    unittest.main()
