"""
Tests for Stage 1: Text Pre-Processing

Tests the text preprocessing module including:
- Text cleaning and normalization
- Sentence segmentation
- Text chunking with overlap
- Input validation and safety checks
"""

import pytest
from bpmn_agent.stages.text_preprocessing import TextPreprocessor, PreprocessedText, TextChunk


class TestTextValidation:
    """Test input validation and safety checks."""
    
    def test_validate_input_too_short(self):
        """Should reject text shorter than minimum."""
        preprocessor = TextPreprocessor(min_length=10)
        with pytest.raises(ValueError, match="too short"):
            preprocessor._validate_input("short")
    
    def test_validate_input_too_long(self):
        """Should reject text longer than maximum."""
        preprocessor = TextPreprocessor(max_length=100)
        long_text = "a" * 101
        with pytest.raises(ValueError, match="too long"):
            preprocessor._validate_input(long_text)
    
    def test_validate_input_type_error(self):
        """Should reject non-string input."""
        preprocessor = TextPreprocessor()
        with pytest.raises(ValueError, match="must be string"):
            preprocessor._validate_input(123)
    
    def test_validate_input_suspicious_repeated_chars(self):
        """Should detect spam with excessive repeated characters."""
        preprocessor = TextPreprocessor()
        spam = "This is " + "a" * 50 + " spam message"
        with pytest.raises(ValueError, match="suspicious"):
            preprocessor._validate_input(spam)
    
    def test_validate_input_suspicious_code_injection(self):
        """Should detect code injection patterns."""
        preprocessor = TextPreprocessor()
        injection = "Normal text <script>alert('xss')</script> more text"
        with pytest.raises(ValueError, match="suspicious"):
            preprocessor._validate_input(injection)
    
    def test_validate_input_valid(self):
        """Should accept valid input."""
        preprocessor = TextPreprocessor()
        valid_text = "This is a perfectly normal process description."
        # Should not raise
        preprocessor._validate_input(valid_text)


class TestTextCleaning:
    """Test text cleaning and normalization."""
    
    def test_clean_normalize_unicode(self):
        """Should normalize Unicode properly."""
        preprocessor = TextPreprocessor()
        # Using combining characters
        text = "e\u0301"  # e + combining acute
        cleaned = preprocessor._clean_and_normalize(text)
        assert len(cleaned) > 0
        assert cleaned.isascii() or all(ord(c) < 128 for c in cleaned)
    
    def test_clean_normalize_whitespace(self):
        """Should normalize multiple spaces to single space."""
        preprocessor = TextPreprocessor()
        text = "Multiple    spaces    here"
        cleaned = preprocessor._clean_and_normalize(text)
        assert cleaned == "Multiple spaces here"
    
    def test_clean_normalize_newlines(self):
        """Should normalize line endings."""
        preprocessor = TextPreprocessor()
        text = "Line 1\r\nLine 2\rLine 3\nLine 4"
        cleaned = preprocessor._clean_and_normalize(text)
        assert "\r" not in cleaned
        assert "Line 1\nLine 2\nLine 3\nLine 4" in cleaned
    
    def test_clean_normalize_multiple_blank_lines(self):
        """Should collapse multiple blank lines."""
        preprocessor = TextPreprocessor()
        text = "Text\n\n\n\nMore text"
        cleaned = preprocessor._clean_and_normalize(text)
        assert "\n\n\n" not in cleaned
        assert "Text\n\nMore text" in cleaned


class TestSentenceSegmentation:
    """Test sentence segmentation."""
    
    def test_segment_single_sentence(self):
        """Should handle single sentence."""
        preprocessor = TextPreprocessor()
        text = "This is a single sentence."
        sentences = preprocessor._segment_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == "This is a single sentence."
    
    def test_segment_multiple_sentences(self):
        """Should split multiple sentences."""
        preprocessor = TextPreprocessor()
        text = "First sentence. Second sentence! Third sentence?"
        sentences = preprocessor._segment_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"
    
    def test_segment_with_abbreviations(self):
        """Should handle abbreviations without over-splitting."""
        preprocessor = TextPreprocessor()
        text = "Dr. Smith works at Inc. in the U.S.A. He processes orders."
        sentences = preprocessor._segment_sentences(text)
        # Should have at least 2 sentences, not split on abbreviations
        assert len(sentences) >= 2


class TestChunking:
    """Test text chunking with overlap."""
    
    def test_chunk_single_small_text(self):
        """Should create single chunk for small text."""
        preprocessor = TextPreprocessor(chunk_size=1000)
        text = "This is a short process description."
        sentences = preprocessor._segment_sentences(text)
        chunks = preprocessor._create_chunks(text, sentences)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == 0
    
    def test_chunk_large_text(self):
        """Should create multiple chunks for large text."""
        preprocessor = TextPreprocessor(chunk_size=100, chunk_overlap=20)
        # Create text larger than chunk size
        text = "This is sentence one. " * 20  # ~420 chars
        sentences = preprocessor._segment_sentences(text)
        chunks = preprocessor._create_chunks(text, sentences)
        assert len(chunks) > 1
    
    def test_chunk_preserves_range(self):
        """Should preserve source range for each chunk."""
        preprocessor = TextPreprocessor(chunk_size=500)
        text = "Sentence one. " * 10 + "Sentence two. " * 10
        sentences = preprocessor._segment_sentences(text)
        chunks = preprocessor._create_chunks(text, sentences)
        
        for chunk in chunks:
            assert chunk.source_range[0] >= 0
            assert chunk.source_range[1] <= len(text)
            assert chunk.source_range[0] < chunk.source_range[1]
    
    def test_chunk_metadata(self):
        """Should calculate chunk metadata correctly."""
        preprocessor = TextPreprocessor()
        text = "First sentence. Second sentence. Third sentence."
        sentences = preprocessor._segment_sentences(text)
        chunks = preprocessor._create_chunks(text, sentences)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.word_count > 0
            assert chunk.sentence_count > 0


class TestFullPreprocessing:
    """Test full preprocessing pipeline."""
    
    def test_preprocess_simple_text(self):
        """Should preprocess simple text successfully."""
        preprocessor = TextPreprocessor()
        text = "This is a process description. It has multiple sentences. And a clear structure."
        
        result = preprocessor.preprocess(text)
        
        assert isinstance(result, PreprocessedText)
        assert result.original_text == text
        assert len(result.cleaned_text) > 0
        assert len(result.sentences) == 3
        assert len(result.chunks) > 0
    
    def test_preprocess_sets_metadata(self):
        """Should set preprocessing metadata."""
        preprocessor = TextPreprocessor()
        text = "The customer submits an order. The system processes it. The order is shipped."
        
        result = preprocessor.preprocess(text)
        
        assert 'original_length' in result.metadata
        assert 'cleaned_length' in result.metadata
        assert 'sentence_count' in result.metadata
        assert 'chunk_count' in result.metadata
        assert 'total_words' in result.metadata
    
    def test_preprocess_complex_process_description(self):
        """Should handle real-world process descriptions."""
        text = """
        The customer submits a purchase order. The order is received by the sales department.
        Sales validates the order. If valid, it goes to warehouse. If invalid, customer is notified.
        
        Warehouse picks items. They are packed and labeled. Shipping receives packages.
        Packages are loaded onto truck. Truck departs. Delivery is tracked.
        
        Upon delivery, customer receives order. They can return items within 30 days.
        Returns are processed by the returns department.
        """
        
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess(text)
        
        assert result.original_text != result.cleaned_text  # Should be cleaned
        assert len(result.sentences) > 5
        assert len(result.chunks) > 0
        assert result.metadata['total_words'] > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
