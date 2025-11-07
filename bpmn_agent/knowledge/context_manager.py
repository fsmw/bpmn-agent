"""
Context Selection and Token Optimization

Intelligently selects knowledge elements for LLM augmentation
while optimizing token usage.
"""

import logging
from typing import Dict, List, Optional

from bpmn_agent.models.knowledge_base import (
    BPMNPattern,
    ComplexityLevel,
    ContextPackage,
    DomainExample,
    DomainType,
    KnowledgeBase,
)

from .domain_classifier import ComplexityAnalyzer, DomainClassifier, PatternRecognizer
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Estimates token counts for different components.

    Provides rough estimates based on common LLM tokenization patterns.
    """

    # Rough tokens per component (averages)
    TOKENS_PER_WORD = 1.3  # Average English words to tokens ratio
    TOKENS_PER_PATTERN_BASE = 150  # Base tokens for pattern metadata
    TOKENS_PER_EXAMPLE_BASE = 200  # Base tokens for example
    TOKENS_PER_RULE = 50  # Validation rule tokens
    TOKENS_PER_TAG = 5  # Tag tokens
    TOKENS_PER_TERM = 3  # Terminology tokens

    @staticmethod
    def count_text_tokens(text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to count

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        words = len(text.split())
        return max(1, int(words * TokenCounter.TOKENS_PER_WORD))

    @staticmethod
    def count_pattern_tokens(pattern: BPMNPattern) -> int:
        """
        Estimate token count for a pattern.

        Args:
            pattern: Pattern to count

        Returns:
            Estimated token count
        """
        count = TokenCounter.TOKENS_PER_PATTERN_BASE
        count += TokenCounter.count_text_tokens(pattern.name)
        count += TokenCounter.count_text_tokens(pattern.description)

        for example in pattern.examples:
            count += TokenCounter.count_text_tokens(example)

        for _rule in pattern.validation_rules:
            count += TokenCounter.TOKENS_PER_RULE

        count += len(pattern.tags) * TokenCounter.TOKENS_PER_TAG

        return count

    @staticmethod
    def count_example_tokens(example: DomainExample) -> int:
        """
        Estimate token count for an example.

        Args:
            example: Example to count

        Returns:
            Estimated token count
        """
        count = TokenCounter.TOKENS_PER_EXAMPLE_BASE
        count += TokenCounter.count_text_tokens(example.text)

        # Add tokens for entities and relations
        for entity_list in example.entities_expected.values():
            for entity in entity_list:
                count += TokenCounter.count_text_tokens(entity)

        return count

    @staticmethod
    def count_terminology_tokens(terms: Dict[str, List[str]]) -> int:
        """
        Estimate token count for terminology.

        Args:
            terms: Domain terminology

        Returns:
            Estimated token count
        """
        count = 0
        for term_list in terms.values():
            count += len(term_list) * TokenCounter.TOKENS_PER_TERM
        return count


class ContextSelector:
    """
    Intelligently selects knowledge for LLM context augmentation.
    """

    def __init__(
        self, kb: Optional[KnowledgeBase] = None, vector_store: Optional[VectorStore] = None
    ):
        """
        Initialize context selector.

        Args:
            kb: Optional knowledge base
            vector_store: Optional vector store for semantic search
        """
        self.kb = kb
        self.vector_store = vector_store

        self.domain_classifier = DomainClassifier(vector_store)
        self.complexity_analyzer = ComplexityAnalyzer()
        self.pattern_recognizer = PatternRecognizer(vector_store, kb)

    def select_context(
        self,
        text: str,
        max_tokens: int = 4000,
        optimization_level: str = "balanced",
        domain: Optional[DomainType] = None,
        complexity: Optional[ComplexityLevel] = None,
        max_patterns: int = 3,
        max_examples: int = 3,
    ) -> ContextPackage:
        """
        Select context for a given input text.

        Args:
            text: Input process description
            max_tokens: Maximum tokens available for context
            optimization_level: Strategy - "minimal", "balanced", "comprehensive"
            domain: Optional pre-determined domain (if None, will be detected)
            complexity: Optional pre-determined complexity (if None, will be analyzed)
            max_patterns: Maximum number of patterns to select
            max_examples: Maximum number of examples to select

        Returns:
            ContextPackage with selected knowledge
        """
        package = ContextPackage(max_tokens=max_tokens, optimization_level=optimization_level)

        # Step 1: Detect domain or use provided domain
        if domain is None:
            domain_result = self.domain_classifier.classify_domain(text)
            domain = domain_result.domain
            domain_conf = domain_result.confidence
        else:
            domain_conf = 1.0  # Assume provided domain is correct
        package.detected_domain = domain

        # Step 2: Analyze complexity or use provided complexity
        if complexity is None:
            complexity_result = self.complexity_analyzer.analyze_complexity(text)
            complexity = complexity_result.level
            complexity_score = complexity_result.score
        else:
            complexity_score = 1.0  # Assume provided complexity is correct
        package.detected_complexity = complexity

        # Step 3: Recognize patterns
        patterns = self.pattern_recognizer.recognize_patterns(text, top_k=max_patterns)
        # Extract pattern IDs from result objects
        package.recognized_patterns = [p.pattern_id for p in patterns]

        # Step 4: Token budget allocation based on optimization level
        token_budgets = self._get_token_budgets(optimization_level, max_tokens)

        # Step 5: Select patterns
        patterns_to_select = self._select_patterns(
            text, domain, patterns, token_budgets["patterns"]
        )
        package.selected_patterns = patterns_to_select
        current_tokens = sum(TokenCounter.count_pattern_tokens(p) for p in patterns_to_select)

        # Step 6: Select examples
        examples_to_select = self._select_examples(
            text, domain, token_budgets["examples"], max_tokens - current_tokens
        )
        package.selected_examples = examples_to_select
        current_tokens += sum(TokenCounter.count_example_tokens(e) for e in examples_to_select)

        # Step 7: Extract domain terminology
        domain_terms = self._extract_domain_terminology(domain, token_budgets["terminology"])
        package.domain_terms = domain_terms
        current_tokens += TokenCounter.count_terminology_tokens(domain_terms)

        # Step 8: Select validation rules
        rules_to_select = self._select_validation_rules(patterns_to_select, token_budgets["rules"])
        package.validation_rules = rules_to_select
        current_tokens += len(rules_to_select) * TokenCounter.TOKENS_PER_RULE

        # Update package metadata
        package.token_count = current_tokens
        package.confidence = min(1.0, (domain_conf + complexity_score + len(patterns) * 0.2) / 3)
        package.reasoning = self._generate_reasoning(domain, complexity, patterns, domain_conf)

        return package

    @staticmethod
    def _get_token_budgets(optimization_level: str, max_tokens: int) -> Dict[str, int]:
        """
        Get token allocation by optimization level.

        Args:
            optimization_level: Strategy level
            max_tokens: Total tokens available

        Returns:
            Dictionary with token budgets for each component
        """
        budgets: Dict[str, Dict[str, float]] = {
            "minimal": {
                "patterns": 0.5,
                "examples": 0.3,
                "terminology": 0.1,
                "rules": 0.1,
            },
            "balanced": {
                "patterns": 0.3,
                "examples": 0.4,
                "terminology": 0.15,
                "rules": 0.15,
            },
            "comprehensive": {
                "patterns": 0.25,
                "examples": 0.35,
                "terminology": 0.25,
                "rules": 0.15,
            },
        }

        allocation = budgets.get(optimization_level, budgets["balanced"])

        return {component: int(max_tokens * ratio) for component, ratio in allocation.items()}

    def _select_patterns(
        self,
        text: str,
        domain: DomainType,
        recognized_patterns,  # Can be list of PatternRecognitionResult or tuples
        token_budget: int,
    ) -> List[BPMNPattern]:
        """
        Select most relevant patterns within token budget.

        Args:
            text: Input text
            domain: Detected domain
            recognized_patterns: List of PatternRecognitionResult or (pattern_id, category, confidence) tuples
            token_budget: Available tokens

        Returns:
            List of selected patterns
        """
        selected = []
        used_tokens = 0

        # Handle both result objects and tuples
        for pattern_item in recognized_patterns:
            if hasattr(pattern_item, "pattern_id"):
                # PatternRecognitionResult object
                pattern_id = pattern_item.pattern_id
                confidence = pattern_item.confidence
            else:
                # Tuple
                pattern_id, _, confidence = pattern_item

            if not self.kb:
                # If no KB, skip pattern selection
                continue

            pattern = self.kb.get_pattern(pattern_id)
            if not pattern:
                continue

            pattern_tokens = TokenCounter.count_pattern_tokens(pattern)

            if used_tokens + pattern_tokens <= token_budget:
                selected.append(pattern)
                used_tokens += pattern_tokens

        return selected

    def _select_examples(
        self, text: str, domain: DomainType, token_budget: int, remaining_tokens: int
    ) -> List[DomainExample]:
        """
        Select few-shot examples within token budget.

        Args:
            text: Input text
            domain: Detected domain
            token_budget: Preferred token allocation
            remaining_tokens: Total remaining tokens

        Returns:
            List of selected examples
        """
        # Adjust budget based on remaining tokens
        actual_budget = min(token_budget, remaining_tokens)

        # Get examples for domain
        if self.kb is None:
            return []
        domain_examples = self.kb.get_examples_by_domain(domain)

        # Sort by validation score and difficulty
        sorted_examples = sorted(
            domain_examples.values(),
            key=lambda e: (e.validation_score, e.difficulty != "hard"),
            reverse=True,
        )

        selected = []
        used_tokens = 0

        for example in sorted_examples[:10]:  # Limit search to top 10
            example_tokens = TokenCounter.count_example_tokens(example)

            if used_tokens + example_tokens <= actual_budget:
                selected.append(example)
                used_tokens += example_tokens

            if len(selected) >= 5:  # Maximum 5 examples
                break

        return selected

    @staticmethod
    def _extract_domain_terminology(domain: DomainType, token_budget: int) -> Dict[str, List[str]]:
        """
        Extract domain-specific terminology.

        Args:
            domain: Domain type
            token_budget: Available tokens

        Returns:
            Dictionary of terminology by category
        """
        # Predefined domain terminology
        terminology_db: Dict[DomainType, Dict[str, List[str]]] = {
            DomainType.HR: {
                "actors": ["Employee", "Manager", "HR", "Recruiter"],
                "activities": ["Recruit", "Onboard", "Train", "Review", "Terminate"],
                "data": ["Resume", "Offer Letter", "Contract", "Benefits"],
            },
            DomainType.FINANCE: {
                "actors": ["Finance Team", "Manager", "Approver", "Accountant"],
                "activities": ["Approve", "Invoice", "Pay", "Record", "Audit"],
                "data": ["Invoice", "PO", "Receipt", "Report"],
            },
            DomainType.IT: {
                "actors": ["IT Team", "Developer", "QA", "Operations"],
                "activities": ["Deploy", "Test", "Build", "Release", "Monitor"],
                "data": ["Code", "Ticket", "Log", "Artifact"],
            },
            DomainType.HEALTHCARE: {
                "actors": ["Patient", "Doctor", "Nurse", "Admin"],
                "activities": ["Admit", "Treat", "Diagnose", "Discharge", "Schedule"],
                "data": ["Record", "Prescription", "Test", "Chart"],
            },
            DomainType.MANUFACTURING: {
                "actors": ["Operator", "QC", "Supervisor", "Planner"],
                "activities": ["Produce", "Inspect", "Package", "Ship", "Maintain"],
                "data": ["Order", "Spec", "Report", "Schedule"],
            },
        }

        return terminology_db.get(domain, {})

    @staticmethod
    def _select_validation_rules(patterns: List[BPMNPattern], token_budget: int) -> List[str]:
        """
        Select validation rules from patterns.

        Args:
            patterns: Selected patterns
            token_budget: Available tokens

        Returns:
            List of validation rules
        """
        rules = []
        used_tokens = 0

        for pattern in patterns:
            for rule in pattern.validation_rules:
                rule_tokens = TokenCounter.TOKENS_PER_RULE

                if used_tokens + rule_tokens <= token_budget:
                    rules.append(rule)
                    used_tokens += rule_tokens

        return rules

    @staticmethod
    def _generate_reasoning(
        domain: DomainType, complexity: ComplexityLevel, patterns: List, domain_confidence: float
    ) -> str:
        """
        Generate explanation for context selection.

        Args:
            domain: Detected domain
            complexity: Detected complexity
            patterns: Recognized patterns
            domain_confidence: Domain detection confidence

        Returns:
            Reasoning explanation
        """
        parts = []

        if domain != DomainType.GENERIC:
            parts.append(f"Detected {domain.value} domain (confidence: {domain_confidence:.1%})")

        parts.append(f"Process complexity: {complexity.value}")

        if patterns:
            pattern_names = ", ".join([p.pattern_id for p in patterns[:3]])
            parts.append(f"Recognized patterns: {pattern_names}")

        return ". ".join(parts) + "."


class ContextOptimizer:
    """
    Optimizes context packages for token efficiency while maintaining quality.
    """

    @staticmethod
    def optimize_context(context: ContextPackage, max_tokens: int) -> ContextPackage:
        """
        Optimize a context package to fit token budget.

        Uses progressive disclosure: includes core elements first,
        then adds detail if space allows.

        Args:
            context: Context package to optimize
            max_tokens: Maximum tokens allowed

        Returns:
            Optimized context package
        """
        if context.token_count <= max_tokens:
            return context  # Already fits

        # Remove lowest priority elements progressively
        optimized = ContextPackage(
            detected_domain=context.detected_domain,
            detected_complexity=context.detected_complexity,
            recognized_patterns=context.recognized_patterns,
            max_tokens=max_tokens,
            optimization_level="optimized",
        )

        used_tokens = 0

        # Priority 1: Keep top patterns
        for pattern in context.selected_patterns:
            tokens = TokenCounter.count_pattern_tokens(pattern)
            if used_tokens + tokens <= max_tokens:
                optimized.selected_patterns.append(pattern)
                used_tokens += tokens

        # Priority 2: Keep best examples
        for example in context.selected_examples[:2]:  # Keep only top 2
            tokens = TokenCounter.count_example_tokens(example)
            if used_tokens + tokens <= max_tokens:
                optimized.selected_examples.append(example)
                used_tokens += tokens

        # Priority 3: Add terminology if space
        optimized.domain_terms = context.domain_terms
        term_tokens = TokenCounter.count_terminology_tokens(context.domain_terms)
        if used_tokens + term_tokens > max_tokens:
            optimized.domain_terms = {}
        else:
            used_tokens += term_tokens

        # Priority 4: Add rules if space
        for rule in context.validation_rules:
            rule_tokens = TokenCounter.TOKENS_PER_RULE
            if used_tokens + rule_tokens <= max_tokens:
                optimized.validation_rules.append(rule)
                used_tokens += rule_tokens

        optimized.token_count = used_tokens
        optimized.confidence = context.confidence * (used_tokens / context.token_count)
        optimized.reasoning = f"Optimized from {context.token_count} to {used_tokens} tokens"

        return optimized
