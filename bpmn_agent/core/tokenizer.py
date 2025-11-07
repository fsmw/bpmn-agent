"""
Token Counting Utilities

Provides accurate token counting for different LLM models.
Uses model-specific tokenization when available, falls back to heuristics.
"""

import re
from enum import Enum
from typing import Optional

from loguru import logger


class ModelTokenizer(str, Enum):
    """Supported tokenization strategies by model."""

    OPENAI_GPT35 = "openai_gpt35"
    OPENAI_GPT4 = "openai_gpt4"
    CLAUDE = "claude"
    LLAMA = "llama"
    MISTRAL = "mistral"
    GENERIC = "generic"


class TokenCounter:
    """Accurate token counter for different model families."""

    # Token patterns for different languages
    WORD_PATTERN = re.compile(r"\b\w+\b")

    # Model families and their characteristics
    MODEL_STRATEGIES = {
        # OpenAI models
        "gpt-4": ModelTokenizer.OPENAI_GPT4,
        "gpt-3.5": ModelTokenizer.OPENAI_GPT35,
        "gpt-3": ModelTokenizer.OPENAI_GPT35,
        "text-davinci": ModelTokenizer.OPENAI_GPT35,
        "text-curie": ModelTokenizer.OPENAI_GPT35,
        # Anthropic models
        "claude": ModelTokenizer.CLAUDE,
        # Meta Llama models
        "llama": ModelTokenizer.LLAMA,
        "llama-2": ModelTokenizer.LLAMA,
        # Mistral models
        "mistral": ModelTokenizer.MISTRAL,
        "mixtral": ModelTokenizer.MISTRAL,
        # Ollama models (default to generic or specific if known)
        "ollama": ModelTokenizer.GENERIC,
    }

    @classmethod
    def get_strategy(cls, model_name: str) -> ModelTokenizer:
        """Determine tokenization strategy for a model."""
        model_lower = model_name.lower()

        for key, strategy in cls.MODEL_STRATEGIES.items():
            if key in model_lower:
                return strategy

        logger.debug(f"Using generic tokenizer for model: {model_name}")
        return ModelTokenizer.GENERIC

    @classmethod
    def count_tokens(cls, text: str, model_name: Optional[str] = None) -> int:
        """
        Count tokens with model-specific accuracy.

        Args:
            text: Text to tokenize
            model_name: Model name for strategy selection

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Try to use model-specific strategy if available
        if model_name:
            strategy = cls.get_strategy(model_name)
        else:
            strategy = ModelTokenizer.GENERIC

        if strategy == ModelTokenizer.OPENAI_GPT4:
            return cls._count_openai_tokens(text)
        elif strategy == ModelTokenizer.OPENAI_GPT35:
            return cls._count_openai_tokens(text)
        elif strategy == ModelTokenizer.CLAUDE:
            return cls._count_claude_tokens(text)
        elif strategy == ModelTokenizer.LLAMA:
            return cls._count_llama_tokens(text)
        elif strategy == ModelTokenizer.MISTRAL:
            return cls._count_mistral_tokens(text)
        else:
            return cls._count_generic_tokens(text)

    @staticmethod
    def _count_openai_tokens(text: str) -> int:
        """
        Count tokens using OpenAI approximation.

        OpenAI's tokenizers typically split on word boundaries and special characters.
        Average: ~1.3 tokens per word in English text.
        """
        # Split on word boundaries
        words = re.findall(r"\b\w+\b|\s+|[^\w\s]", text)

        token_count = 0
        for word in words:
            if re.match(r"\w+", word):
                # Actual words: ~1.3 tokens per word
                token_count += max(1, int(len(word) / 4 * 1.3))
            elif word.strip():
                # Punctuation and special chars: typically 1 token
                token_count += 1

        return max(1, token_count)

    @staticmethod
    def _count_claude_tokens(text: str) -> int:
        """
        Count tokens using Claude approximation.

        Claude uses a similar tokenization to OpenAI but with slight differences.
        Average: ~1.2 tokens per word.
        """
        words = re.findall(r"\b\w+\b|\s+|[^\w\s]", text)

        token_count = 0
        for word in words:
            if re.match(r"\w+", word):
                # ~1.2 tokens per word
                token_count += max(1, int(len(word) / 4.2 * 1.2))
            elif word.strip():
                token_count += 1

        return max(1, token_count)

    @staticmethod
    def _count_llama_tokens(text: str) -> int:
        """
        Count tokens using Llama approximation.

        Llama models use SentencePiece tokenization.
        Average: ~1.0 tokens per word (more aggressive tokenization).
        """
        words = re.findall(r"\b\w+\b|\s+|[^\w\s]", text)

        token_count = 0
        for word in words:
            if re.match(r"\w+", word):
                # ~1 token per word (more words become single tokens)
                token_count += max(1, len(word) // 4)
            elif word.strip():
                token_count += 1

        return max(1, token_count)

    @staticmethod
    def _count_mistral_tokens(text: str) -> int:
        """
        Count tokens using Mistral approximation.

        Mistral uses tokenization similar to Llama.
        Average: ~1.0 tokens per word.
        """
        words = re.findall(r"\b\w+\b|\s+|[^\w\s]", text)

        token_count = 0
        for word in words:
            if re.match(r"\w+", word):
                token_count += max(1, len(word) // 4)
            elif word.strip():
                token_count += 1

        return max(1, token_count)

    @staticmethod
    def _count_generic_tokens(text: str) -> int:
        """
        Generic token counting fallback.

        Uses simple approximation: ~4 characters per token on average.
        """
        if not text:
            return 0

        # Split on whitespace and count tokens
        words = text.split()

        # Rough estimate: 4 chars per token
        char_count = len(text)
        token_count = max(1, char_count // 4)

        # Ensure minimum token count based on word count
        return max(len(words), token_count)


# Convenience functions
def count_tokens(text: str, model_name: Optional[str] = None) -> int:
    """
    Count tokens in text with model-specific accuracy.

    Args:
        text: Text to tokenize
        model_name: Model name for strategy selection (optional)

    Returns:
        Estimated token count
    """
    return TokenCounter.count_tokens(text, model_name)


def estimate_cost(
    text: str,
    model_name: str = "gpt-3.5-turbo",
    input_cost_per_1k: float = 0.0005,
    output_tokens: Optional[int] = None,
    output_cost_per_1k: Optional[float] = None,
) -> dict:
    """
    Estimate API cost for text processing.

    Args:
        text: Input text
        model_name: Model being used
        input_cost_per_1k: Cost per 1K input tokens
        output_tokens: Estimated output tokens (if None, assumes similar to input)
        output_cost_per_1k: Cost per 1K output tokens (defaults to input_cost_per_1k)

    Returns:
        Dictionary with token counts and cost estimates
    """
    input_tokens = count_tokens(text, model_name)

    if output_tokens is None:
        output_tokens = input_tokens  # Rough estimate

    if output_cost_per_1k is None:
        output_cost_per_1k = input_cost_per_1k

    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
    }


__all__ = [
    "ModelTokenizer",
    "TokenCounter",
    "count_tokens",
    "estimate_cost",
]
