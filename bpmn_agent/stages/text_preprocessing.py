"""
Stage 1: Text Pre-Processing

Prepares raw input text for entity extraction by:
- Cleaning and normalizing the text
- Segmenting into sentences
- Chunking large descriptions for LLM token limits
- Validating and sanitizing input
- Detecting domain and complexity with KB integration
- Augmenting metadata with pattern hints

This is the first stage in the BPMN extraction pipeline.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from bpmn_agent.knowledge import DomainClassifier, PatternRecognizer

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of preprocessed text with metadata."""

    content: str
    """The cleaned text content"""

    chunk_id: int
    """Chunk index (0-based)"""

    source_range: Tuple[int, int]
    """Start and end character positions in original text"""

    sentence_count: int = 0
    """Number of sentences in this chunk"""

    word_count: int = 0
    """Number of words in this chunk"""

    def __post_init__(self):
        """Calculate metrics after initialization."""
        self.word_count = len(self.content.split())
        self.sentence_count = len([s for s in self.content.split(".") if s.strip()])


@dataclass
class PreprocessedText:
    """Result of text preprocessing with KB integration."""

    original_text: str
    """Original unprocessed input"""

    cleaned_text: str
    """Full cleaned/normalized text"""

    sentences: List[str]
    """Segmented sentences"""

    chunks: List[TextChunk]
    """Text chunks for processing"""

    metadata: dict = field(default_factory=dict)
    """Processing metadata including KB insights"""

    def __post_init__(self):
        """Calculate basic statistics and KB metadata."""
        self.metadata["original_length"] = len(self.original_text)
        self.metadata["cleaned_length"] = len(self.cleaned_text)
        self.metadata["sentence_count"] = len(self.sentences)
        self.metadata["chunk_count"] = len(self.chunks)
        self.metadata["total_words"] = sum(c.word_count for c in self.chunks)

        # KB metadata (will be populated by preprocessor)
        if "detected_domain" not in self.metadata:
            self.metadata["detected_domain"] = None
        if "detected_complexity" not in self.metadata:
            self.metadata["detected_complexity"] = None
        if "recognized_patterns" not in self.metadata:
            self.metadata["recognized_patterns"] = []
        if "domain_confidence" not in self.metadata:
            self.metadata["domain_confidence"] = 0.0
        if "pattern_hints" not in self.metadata:
            self.metadata["pattern_hints"] = []


class TextPreprocessor:
    """Preprocesses raw text for the extraction pipeline with KB integration."""

    # Configuration constants
    MIN_INPUT_LENGTH = 10
    """Minimum characters required for valid input"""

    MAX_INPUT_LENGTH = 1_000_000
    """Maximum characters allowed (1MB)"""

    CHUNK_SIZE = 4000
    """Target size per chunk (characters, approximate)"""

    CHUNK_OVERLAP = 200
    """Overlap between chunks to preserve context"""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_length: int = MIN_INPUT_LENGTH,
        max_length: int = MAX_INPUT_LENGTH,
        enable_kb: bool = True,
    ):
        """
        Initialize the text preprocessor with optional KB integration.

        Args:
            chunk_size: Target characters per chunk
            chunk_overlap: Characters to overlap between chunks
            min_length: Minimum valid input length
            max_length: Maximum valid input length
            enable_kb: Whether to enable KB domain/complexity analysis
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_length = min_length
        self.max_length = max_length
        self.enable_kb = enable_kb

        # Initialize KB components (lazy loading)
        self._domain_classifier: Optional[DomainClassifier] = None
        self._pattern_recognizer: Optional[PatternRecognizer] = None

    def _ensure_kb_initialized(self) -> None:
        """Initialize KB components if enabled and not already done."""
        if not self.enable_kb:
            return

        if self._domain_classifier is None:
            try:
                self._domain_classifier = DomainClassifier()
                self._pattern_recognizer = PatternRecognizer()
                logger.debug("KB components initialized for text preprocessing")
            except Exception as e:
                logger.warning(f"Failed to initialize KB components: {e}")
                self.enable_kb = False

    def preprocess(self, text: str) -> PreprocessedText:
        """
        Execute full preprocessing pipeline with KB integration.

        Args:
            text: Raw input text

        Returns:
            PreprocessedText with cleaned text, sentences, chunks, and KB insights

        Raises:
            ValueError: If text is invalid or outside length constraints
        """
        # 1. Validate input
        self._validate_input(text)

        # 2. Clean and normalize
        cleaned = self._clean_and_normalize(text)

        # 3. Segment into sentences
        sentences = self._segment_sentences(cleaned)

        # 4. Create chunks
        chunks = self._create_chunks(cleaned, sentences)

        # Create base preprocessing result
        result = PreprocessedText(
            original_text=text,
            cleaned_text=cleaned,
            sentences=sentences,
            chunks=chunks,
        )

        # 5. Perform KB analysis if enabled
        if self.enable_kb:
            self._augment_with_kb_insights(result, cleaned)

        return result

    def _validate_input(self, text: str) -> None:
        """
        Validate input text for safety and format.

        Args:
            text: Input to validate

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(text, str):
            raise ValueError(f"Input must be string, got {type(text).__name__}")

        # Check length constraints
        if len(text) < self.min_length:
            raise ValueError(f"Input too short: {len(text)} chars (minimum: {self.min_length})")

        if len(text) > self.max_length:
            raise ValueError(f"Input too long: {len(text)} chars (maximum: {self.max_length})")

        # Check for reasonable character distribution
        # Reject if >80% non-ASCII (likely corrupted)
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        if len(text) > 0:
            non_ascii_ratio = non_ascii_count / len(text)
            if non_ascii_ratio > 0.8:
                raise ValueError(f"Input has too many non-ASCII characters ({non_ascii_ratio:.1%})")

        # Check for suspicious patterns (likely spam/injection)
        if self._is_suspicious_content(text):
            raise ValueError("Input detected as suspicious content (potential spam/injection)")

    def _is_suspicious_content(self, text: str) -> bool:
        """
        Detect suspicious/spam content patterns.

        Args:
            text: Text to check

        Returns:
            True if content appears suspicious
        """
        text_lower = text.lower()

        # Repeated characters (spam indicator)
        if re.search(r"(.)\1{20,}", text):
            return True

        # Excessive URLs (spam indicator)
        url_count = len(re.findall(r"https?://", text_lower))
        if url_count > len(text) / 500:  # More than ~1 URL per 500 chars
            return True

        # Code-like injection patterns
        if re.search(r"<script|<iframe|javascript:|on\w+\s*=", text_lower):
            return True

        # Excessive punctuation
        punct_count = sum(1 for c in text if c in "!@#$%^&*(){}[]")
        if punct_count > len(text) / 20:  # More than 5% punctuation
            return True

        return False

    def _clean_and_normalize(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            Cleaned and normalized text
        """
        # 1. Normalize Unicode using NFKD (compatibility decomposition)
        # This decomposes composed characters (e.g., é → e + combining accent)
        text = unicodedata.normalize("NFKD", text)

        # 2. Remove combining marks (accents, diacritics) to get ASCII equivalents
        # Category 'Mn' = Mark, nonspacing (combining marks)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")

        # 3. Normalize line endings first (before general whitespace handling)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 4. Normalize multiple spaces on same line (but preserve newlines)
        lines = text.split("\n")
        lines = [re.sub(r" +", " ", line) for line in lines]
        text = "\n".join(lines)

        # 5. Remove common control characters (but preserve \n and \t)
        text = "".join(c for c in text if unicodedata.category(c) != "Cc" or c in "\n\t")

        # 6. Clean up multiple blank lines (keep max 2 newlines for single blank line)
        text = re.sub(r"\n\n\n+", "\n\n", text)

        # 7. Remove excessive leading/trailing whitespace
        text = text.strip()

        return text

    def _segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences.

        Simple rule-based approach using common sentence boundaries.
        More sophisticated approaches (spaCy, NLTK) can be integrated.

        Args:
            text: Cleaned text

        Returns:
            List of sentences
        """
        # Split on sentence boundaries
        # Handles: period, question mark, exclamation
        # Preserves boundaries between abbreviations

        sentences = []

        # Split on common sentence endings
        # Use lookahead to keep delimiters info if needed
        parts = re.split(r"(?<=[.!?])\s+", text)

        for part in parts:
            if part.strip():
                sentences.append(part.strip())

        return sentences

    def _create_chunks(
        self,
        cleaned_text: str,
        sentences: List[str],
    ) -> List[TextChunk]:
        """
        Create text chunks preserving sentence boundaries.

        Uses sliding window approach with overlap to maintain context.
        Respects sentence boundaries to avoid splitting mid-sentence.

        Args:
            cleaned_text: Full cleaned text
            sentences: Segmented sentences

        Returns:
            List of text chunks
        """
        if not sentences:
            return []

        chunks: List[TextChunk] = []
        current_chunk_sentences: List[str] = []
        current_chunk_size = 0
        chunk_start_char = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_size = len(sentence) + 1  # +1 for space

            # Check if adding this sentence would exceed chunk size
            if current_chunk_size + sentence_size > self.chunk_size and current_chunk_sentences:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunk_end_char = chunk_start_char + len(chunk_text)

                chunks.append(
                    TextChunk(
                        content=chunk_text,
                        chunk_id=chunk_id,
                        source_range=(chunk_start_char, chunk_end_char),
                    )
                )

                chunk_id += 1

                # Prepare for next chunk with overlap
                # Keep last ~200 chars of overlap for context
                overlap_text = (
                    chunk_text[-self.chunk_overlap :]
                    if len(chunk_text) > self.chunk_overlap
                    else chunk_text
                )

                current_chunk_sentences = [s for s in overlap_text.split(" ") if s]
                current_chunk_size = len(overlap_text) + 1
                chunk_start_char = chunk_end_char - len(overlap_text)

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_size += sentence_size

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_end_char = chunk_start_char + len(chunk_text)

            chunks.append(
                TextChunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    source_range=(chunk_start_char, chunk_end_char),
                )
            )

        return chunks

    def _augment_with_kb_insights(self, result: PreprocessedText, text: str) -> None:
        """
        Augment preprocessing result with KB domain/complexity analysis.

        Args:
            result: PreprocessedText object to augment
            text: Cleaned text for analysis
        """
        try:
            self._ensure_kb_initialized()
            if not self.enable_kb or not self._domain_classifier:
                return

            # Detect domain
            domain_result = self._domain_classifier.classify_domain(text)
            result.metadata["detected_domain"] = domain_result.domain
            result.metadata["domain_confidence"] = domain_result.confidence
            result.metadata["domain_indicators"] = domain_result.indicators

            # Detect complexity
            complexity_result = self._domain_classifier.analyze_complexity(text)
            result.metadata["detected_complexity"] = complexity_result.level
            result.metadata["complexity_score"] = complexity_result.score
            result.metadata["complexity_factors"] = complexity_result.factors

            # Recognize patterns
            if self._pattern_recognizer:
                pattern_results = self._pattern_recognizer.recognize_patterns(text)
                result.metadata["recognized_patterns"] = [p.pattern_id for p in pattern_results]
                result.metadata["pattern_hints"] = [
                    {
                        "pattern_id": p.pattern_id,
                        "pattern_category": p.pattern_category.value,
                        "confidence": p.confidence,
                        "locations": p.text_indices if hasattr(p, "text_indices") else [],
                    }
                    for p in pattern_results
                ]

                logger.debug(
                    f"Recognized {len(pattern_results)} patterns in text: "
                    f"{[p.pattern_id for p in pattern_results]}"
                )

            logger.debug(
                f"KB analysis complete: domain={result.metadata['detected_domain']}, "
                f"complexity={result.metadata['detected_complexity']}"
            )

        except Exception as e:
            logger.error(f"Error during KB augmentation: {e}", exc_info=True)
            # Don't fail preprocessing if KB analysis fails


__all__ = [
    "TextChunk",
    "PreprocessedText",
    "TextPreprocessor",
]
