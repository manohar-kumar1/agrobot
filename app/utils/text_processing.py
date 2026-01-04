import re
from typing import List


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    return text.strip()


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    words = text.lower().split()
    return list(set(words))[:max_keywords]


def truncate_text(text: str, max_length: int = 500) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."
