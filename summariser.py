import argparse
import re
from collections import Counter
from pathlib import Path


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def extract_text_from_image(image_path: str) -> str:
    try:
        from PIL import Image, UnidentifiedImageError
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "OCR dependencies are missing. Install requirements.txt first."
        ) from exc

    try:
        image = Image.open(image_path)
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Unable to open image file: {image_path}") from exc
    return pytesseract.image_to_string(image).strip()


def summarise_text(text: str, max_sentences: int = 3) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if sentence.strip()
    ]
    if not sentences:
        return cleaned

    words = re.findall(r"\b[a-zA-Z']+\b", cleaned.lower())
    freq = Counter(word for word in words if word not in STOPWORDS)
    if not freq:
        return " ".join(sentences[:max_sentences])

    sentence_scores = []
    for idx, sentence in enumerate(sentences):
        sentence_words = re.findall(r"\b[a-zA-Z']+\b", sentence.lower())
        score = sum(freq[word] for word in sentence_words if word in freq)
        sentence_scores.append((score, idx, sentence))

    selected = sorted(
        sorted(sentence_scores, key=lambda item: item[0], reverse=True)[:max_sentences],
        key=lambda item: item[1],
    )
    return " ".join(sentence for _, _, sentence in selected)


def summarise_image(image_path: str, max_sentences: int = 3) -> str:
    return summarise_text(extract_text_from_image(image_path), max_sentences=max_sentences)


def main() -> None:
    parser = argparse.ArgumentParser(description="Image-to-text summariser")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--sentences", type=int, default=3, help="Summary sentence count")
    args = parser.parse_args()

    path = Path(args.image_path)
    if not path.exists() or not path.is_file():
        raise SystemExit(f"Image not found: {path}")
    if args.sentences < 1:
        raise SystemExit("--sentences must be at least 1")

    summary = summarise_image(str(path), max_sentences=args.sentences)
    print(summary)


if __name__ == "__main__":
    main()
