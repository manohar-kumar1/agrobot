from typing import List
import math
import hashlib


class BloomFilter:
    def __init__(
        self, expected_elements: int = 10000, false_positive_rate: float = 0.01
    ):
        self.expected_elements = expected_elements
        self.false_positive_rate = false_positive_rate

        self.size = self._optimal_size(expected_elements, false_positive_rate)

        self.hash_count = self._optimal_hash_count(self.size, expected_elements)

        self._bit_array = bytearray(math.ceil(self.size / 8))
        self._count = 0

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        m = -n * math.log(p) / (math.log(2) ** 2)
        return int(math.ceil(m))

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        k = (m / n) * math.log(2)
        return max(1, int(round(k)))

    def _get_hash_values(self, item: str) -> List[int]:
        item_bytes = item.encode("utf-8")
        h1 = int(hashlib.md5(item_bytes).hexdigest(), 16)
        h2 = int(hashlib.sha1(item_bytes).hexdigest(), 16)

        return [(h1 + i * h2) % self.size for i in range(self.hash_count)]

    def _set_bit(self, index: int) -> None:
        byte_index = index // 8
        bit_index = index % 8
        self._bit_array[byte_index] |= 1 << bit_index

    def _get_bit(self, index: int) -> bool:
        byte_index = index // 8
        bit_index = index % 8
        return bool(self._bit_array[byte_index] & (1 << bit_index))

    def add(self, item: str) -> None:
        for index in self._get_hash_values(item):
            self._set_bit(index)
        self._count += 1

    def __contains__(self, item: str) -> bool:
        return all(self._get_bit(index) for index in self._get_hash_values(item))

    def contains(self, item: str) -> bool:
        return item in self

    def add_if_new(self, item: str) -> bool:
        if item in self:
            return False
        self.add(item)
        return True

    @property
    def count(self) -> int:
        return self._count

    @property
    def fill_ratio(self) -> float:
        set_bits = sum(bin(byte).count("1") for byte in self._bit_array)
        return set_bits / self.size

    @property
    def current_fp_rate(self) -> float:
        return self.fill_ratio**self.hash_count

    def get_stats(self) -> dict:
        return {
            "size_bits": self.size,
            "size_bytes": len(self._bit_array),
            "hash_count": self.hash_count,
            "elements_added": self._count,
            "fill_ratio": round(self.fill_ratio, 4),
            "estimated_fp_rate": round(self.current_fp_rate, 6),
            "target_fp_rate": self.false_positive_rate,
        }

    def clear(self) -> None:
        self._bit_array = bytearray(math.ceil(self.size / 8))
        self._count = 0


class ScalableBloomFilter:
    def __init__(
        self,
        initial_capacity: int = 1000,
        false_positive_rate: float = 0.01,
        growth_factor: int = 2,
        tightening_ratio: float = 0.5,
    ):
        self.initial_capacity = initial_capacity
        self.false_positive_rate = false_positive_rate
        self.growth_factor = growth_factor
        self.tightening_ratio = tightening_ratio

        self._filters: List[BloomFilter] = []
        self._add_filter()

    def _add_filter(self) -> None:
        """Add a new filter with tightened FP rate."""
        # Each successive filter gets tighter FP rate
        filter_index = len(self._filters)
        fp_rate = self.false_positive_rate * (self.tightening_ratio**filter_index)
        capacity = self.initial_capacity * (self.growth_factor**filter_index)

        self._filters.append(BloomFilter(capacity, fp_rate))

    def add(self, item: str) -> None:
        """Add item, growing if necessary."""
        current_filter = self._filters[-1]

        # Check if current filter is full
        if current_filter.count >= current_filter.expected_elements:
            self._add_filter()
            current_filter = self._filters[-1]

        current_filter.add(item)

    def __contains__(self, item: str) -> bool:
        """Check across all filters."""
        return any(item in f for f in self._filters)

    def contains(self, item: str) -> bool:
        return item in self

    def add_if_new(self, item: str) -> bool:
        """Add only if not present in any filter."""
        if item in self:
            return False
        self.add(item)
        return True

    @property
    def count(self) -> int:
        return sum(f.count for f in self._filters)

    def get_stats(self) -> dict:
        return {
            "num_filters": len(self._filters),
            "total_elements": self.count,
            "filters": [f.get_stats() for f in self._filters],
        }


class DocumentDeduplicator:
    def __init__(
        self,
        expected_docs: int = 5000,
        false_positive_rate: float = 0.001,
    ):
        self.bloom = BloomFilter(expected_docs, false_positive_rate)
        self._seen_ids: set = set()

    def _generate_content_hash(self, content: str, prefix_len: int = 200) -> str:
        normalized = " ".join(content.lower().split())
        prefix = normalized[:prefix_len]
        return hashlib.md5(prefix.encode()).hexdigest()

    def _generate_doc_id(
        self, content: str, source: str | None = None, page: int | None = None
    ) -> str:
        content_hash = self._generate_content_hash(content)
        if source and page is not None:
            return f"{source}_p{page}_{content_hash[:8]}"
        return content_hash

    def is_duplicate(
        self, content: str, source: str | None = None, page: int | None = None
    ) -> bool:
        doc_id = self._generate_doc_id(content, source, page)
        return doc_id in self.bloom

    def add_document(
        self, content: str, source: str | None = None, page: int | None = None
    ) -> bool:
        doc_id = self._generate_doc_id(content, source, page)
        return self.bloom.add_if_new(doc_id)

    def deduplicate_batch(self, documents: List[dict]) -> List[dict]:
        unique_docs = []
        for doc in documents:
            content = doc.get("content", "")
            source = doc.get("source")
            page = doc.get("page")

            if self.add_document(content, source, page):
                unique_docs.append(doc)

        return unique_docs

    def get_stats(self) -> dict:
        return {
            "bloom_stats": self.bloom.get_stats(),
            "exact_ids_tracked": len(self._seen_ids),
        }

    def clear(self) -> None:
        self.bloom.clear()
        self._seen_ids.clear()
_document_deduplicator: DocumentDeduplicator | None = None


def get_document_deduplicator() -> DocumentDeduplicator:
    global _document_deduplicator
    if _document_deduplicator is None:
        _document_deduplicator = DocumentDeduplicator()
    return _document_deduplicator
