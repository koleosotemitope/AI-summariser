import unittest
from unittest.mock import patch

import summariser


class SummariserTests(unittest.TestCase):
    def test_summarise_text_returns_empty_for_empty_input(self):
        self.assertEqual(summariser.summarise_text("", max_sentences=2), "")

    def test_summarise_text_limits_sentence_count(self):
        text = (
            "AI can read text from images. "
            "AI can summarise the extracted text quickly. "
            "This sentence is less important."
        )
        summary = summariser.summarise_text(text, max_sentences=2)
        self.assertIn("AI can read text from images.", summary)
        self.assertIn("AI can summarise the extracted text quickly.", summary)
        self.assertNotIn("This sentence is less important.", summary)

    @patch("summariser.extract_text_from_image", return_value="First sentence. Second sentence.")
    def test_summarise_image_uses_extracted_text(self, _mock_extract):
        summary = summariser.summarise_image("fake-image.png", max_sentences=1)
        _mock_extract.assert_called_once_with("fake-image.png")
        self.assertEqual(summary, "First sentence.")


if __name__ == "__main__":
    unittest.main()
