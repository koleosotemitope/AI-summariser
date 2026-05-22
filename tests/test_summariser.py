import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
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

    def test_summarise_text_handles_stopwords_only(self):
        text = "The and is. To the and."
        self.assertEqual(summariser.summarise_text(text, max_sentences=1), "The and is.")

    def test_summarise_text_handles_no_sentence_punctuation(self):
        text = "ai summariser extracts text and returns concise output"
        self.assertEqual(
            summariser.summarise_text(text, max_sentences=3),
            "ai summariser extracts text and returns concise output",
        )

    def test_summarise_text_max_sentences_above_available(self):
        text = "First sentence. Second sentence."
        self.assertEqual(summariser.summarise_text(text, max_sentences=5), text)

    @patch("summariser.extract_text_from_image", return_value="First sentence. Second sentence.")
    def test_summarise_image_uses_extracted_text(self, mock_extract):
        summary = summariser.summarise_image("fake-image.png", max_sentences=1)
        mock_extract.assert_called_once_with("fake-image.png")
        self.assertEqual(summary, "First sentence.")

    @patch("summariser.extract_text_from_image", side_effect=RuntimeError("bad image"))
    def test_summarise_image_propagates_ocr_errors(self, _mock_extract):
        with self.assertRaisesRegex(RuntimeError, "bad image"):
            summariser.summarise_image("fake-image.png", max_sentences=1)

    def test_main_rejects_missing_image(self):
        with patch("sys.argv", ["summariser.py", "missing-file.png"]):
            with self.assertRaisesRegex(SystemExit, "Image not found: missing-file.png"):
                summariser.main()

    @patch("summariser.summarise_image", return_value="ok")
    def test_main_rejects_invalid_sentence_count(self, _mock_summary):
        with NamedTemporaryFile(suffix=".png") as tmp_file:
            with patch("sys.argv", ["summariser.py", tmp_file.name, "--sentences", "0"]):
                with self.assertRaisesRegex(SystemExit, "--sentences must be at least 1"):
                    summariser.main()

    @patch("builtins.print")
    @patch("summariser.summarise_image", return_value="final summary")
    def test_main_prints_summary(self, mock_summarise_image, mock_print):
        with NamedTemporaryFile(suffix=".png") as tmp_file:
            image_path = Path(tmp_file.name)
            with patch("sys.argv", ["summariser.py", str(image_path), "--sentences", "2"]):
                summariser.main()
        mock_summarise_image.assert_called_once_with(str(image_path), max_sentences=2)
        mock_print.assert_called_once_with("final summary")


if __name__ == "__main__":
    unittest.main()
