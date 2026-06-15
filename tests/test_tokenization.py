import unittest

from llm import detokenize, filter_known_tokens, format_unknown_token_warning, tokenize


class TokenizationTest(unittest.TestCase):
    def test_keeps_apostrophe_contractions(self):
        self.assertEqual(
            tokenize("I'm sure you're there. They've gone!"),
            ["i'm", "sure", "you're", "there", ".", "they've", "gone", "!"],
        )

    def test_normalizes_curly_apostrophes(self):
        self.assertEqual(
            tokenize("Who\u2019s asleep? I\u2019m awake."),
            ["who's", "asleep", "?", "i'm", "awake", "."],
        )

    def test_detokenizes_contractions(self):
        self.assertEqual(
            detokenize(["i'm", "sure", "you're", "there", "."]),
            "I'm sure you're there.",
        )

    def test_filter_known_tokens_reports_unknown_prompt_tokens(self):
        token_to_index = {"<unk>": 0, "how": 1, "do": 2, "we": 3, "the": 4, "of": 5}
        known, unknown = filter_known_tokens(
            tokenize("how do we seize the means of production"),
            token_to_index,
        )

        self.assertEqual(known, ["how", "do", "we", "the", "of"])
        self.assertEqual(unknown, ["seize", "means", "production"])

    def test_unknown_token_warning_deduplicates_in_order(self):
        self.assertEqual(
            format_unknown_token_warning(["seize", "means", "seize"]),
            "warning: ignoring unknown prompt tokens: seize, means",
        )


if __name__ == "__main__":
    unittest.main()
