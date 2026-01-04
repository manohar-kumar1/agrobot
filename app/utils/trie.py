from typing import List, Set, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    is_end_of_word: bool = False
    word: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self._word_count = 0

    def insert(self, word: str, metadata: Dict[str, Any] | None = None) -> None:
        word_lower = word.lower()
        node = self.root

        for char in word_lower:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        if not node.is_end_of_word:
            self._word_count += 1

        node.is_end_of_word = True
        node.word = word_lower
        if metadata:
            node.metadata = metadata

    def search(self, word: str) -> bool:
        node = self._find_node(word.lower())
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix: str) -> List[str]:
        prefix_lower = prefix.lower()
        node = self._find_node(prefix_lower)

        if node is None:
            return []

        results: List[str] = []
        self._collect_words(node, results)
        return results

    def search_in_text(self, text: str) -> Set[str]:
        text_lower = text.lower()
        words = set(text_lower.split())
        matched: Set[str] = set()

        for word in words:
            clean_word = "".join(c for c in word if c.isalnum())
            if clean_word and self.search(clean_word):
                matched.add(clean_word)

        return matched

    def search_prefixes_in_text(self, text: str) -> Set[str]:
        text_lower = text.lower()
        words = text_lower.split()
        matched: Set[str] = set()

        for word in words:
            clean_word = "".join(c for c in word if c.isalnum())
            if not clean_word:
                continue

            node = self.root
            for i, char in enumerate(clean_word):
                if char not in node.children:
                    break
                node = node.children[char]
                if node.is_end_of_word:
                    matched.add(node.word)

        return matched

    def bulk_insert(self, words: List[str], category: str | None = None) -> None:
        """Insert multiple words efficiently."""
        for word in words:
            metadata = {"category": category} if category else None
            self.insert(word, metadata)

    def get_word_count(self) -> int:
        """Return number of words in trie."""
        return self._word_count

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Traverse to node for given prefix. O(m) time."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def _collect_words(self, node: TrieNode, results: List[str]) -> None:
        if node.is_end_of_word and node.word:
            results.append(node.word)

        for child in node.children.values():
            self._collect_words(child, results)


class KeywordMatcher:
    def __init__(self):
        self.disease_trie = Trie()
        self.scheme_trie = Trie()
        self._initialized = False

    def initialize(
        self, disease_keywords: List[str], scheme_keywords: List[str]
    ) -> None:
        self.disease_trie.bulk_insert(disease_keywords, category="disease")
        self.scheme_trie.bulk_insert(scheme_keywords, category="scheme")
        self._initialized = True

    def match_disease_keywords(self, text: str) -> Set[str]:
        return self.disease_trie.search_prefixes_in_text(text)

    def match_scheme_keywords(self, text: str) -> Set[str]:
        return self.scheme_trie.search_prefixes_in_text(text)

    def match_all_keywords(self, text: str) -> Dict[str, Set[str]]:
        return {
            "disease": self.match_disease_keywords(text),
            "scheme": self.match_scheme_keywords(text),
        }

    def get_keyword_overlap(
        self, query_words: Set[str], content_words: Set[str], category: str = "both"
    ) -> int:
        trie = (
            self.disease_trie
            if category == "disease"
            else self.scheme_trie if category == "scheme" else None
        )

        if trie is None:
            disease_matches = sum(
                1
                for w in query_words
                if self.disease_trie.search(w) and w in content_words
            )
            scheme_matches = sum(
                1
                for w in query_words
                if self.scheme_trie.search(w) and w in content_words
            )
            return disease_matches + scheme_matches

        return sum(1 for w in query_words if trie.search(w) and w in content_words)

    @property
    def is_initialized(self) -> bool:
        return self._initialized


_keyword_matcher: KeywordMatcher | None = None


def get_keyword_matcher() -> KeywordMatcher:
    global _keyword_matcher
    if _keyword_matcher is None:
        _keyword_matcher = KeywordMatcher()
    return _keyword_matcher
