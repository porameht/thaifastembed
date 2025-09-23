import os
from typing import Optional
from pythainlp.tokenize import Trie


def load_custom_dict(path: str = "words.txt") -> Optional[Trie]:
    """Load custom dictionary from words.txt using PyThaiNLP's Trie function."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    words_file = os.path.join(os.path.dirname(current_dir), path)

    if not os.path.exists(words_file):
        return None

    try:
        with open(words_file, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        
        return Trie(words)
    except Exception:
        return None
