"""
Advanced Pattern Matching Engine

Provides sophisticated pattern matching capabilities including:
- Fuzzy keyword matching with edit distance
- Semantic similarity scoring using text embeddings
- Pattern recommendation based on context
- Confidence scoring and result ranking
- Domain-aware pattern selection
"""

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

from bpmn_agent.models.knowledge_base import (
    BPMNPattern,
    ComplexityLevel,
    DomainType,
    KnowledgeBase,
    PatternCategory,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a pattern match operation."""

    pattern: BPMNPattern
    match_score: float  # 0.0-1.0
    match_type: str  # "exact", "fuzzy", "semantic", "keyword", "category"
    matched_keywords: List[str] = field(default_factory=list)
    confidence: float = 1.0  # Confidence in the match
    reasoning: str = ""  # Explanation of why this matched
    relevance_factors: Dict[str, float] = field(
        default_factory=dict
    )  # Breakdown of scoring factors

    def __lt__(self, other):
        """Enable sorting by match score (descending)."""
        return self.match_score > other.match_score


@dataclass
class PatternRecommendation:
    """Pattern recommendation with reasoning."""

    patterns: List[Tuple[BPMNPattern, float]]  # (pattern, relevance_score)
    primary_domain: DomainType
    detected_complexity: ComplexityLevel
    detected_categories: List[PatternCategory]
    confidence: float
    reasoning: str
    alternative_patterns: List[Tuple[BPMNPattern, float]] = field(default_factory=list)


class AdvancedPatternMatcher:
    """
    Advanced pattern matching engine with fuzzy matching and semantic similarity.
    """

    def __init__(self, knowledge_base: KnowledgeBase, embedding_model: Optional[object] = None):
        """
        Initialize the advanced pattern matcher.

        Args:
            knowledge_base: The knowledge base containing patterns
            embedding_model: Optional pre-trained embedding model for semantic similarity
        """
        self.kb = knowledge_base
        self.embedding_model = embedding_model
        self.logger = logger

        # Build keyword indexes for faster matching
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build internal indexes for fast pattern lookup."""
        self.pattern_keywords: Dict[str, Set[str]] = {}
        self.pattern_tags: Dict[str, Set[str]] = {}
        self.pattern_category_map: Dict[PatternCategory, List[str]] = {}
        self.pattern_domain_map: Dict[DomainType, List[str]] = {}

        for pattern_id, pattern in self.kb.patterns.items():
            # Index keywords
            keywords = self._extract_keywords(pattern)
            self.pattern_keywords[pattern_id] = keywords

            # Index tags
            self.pattern_tags[pattern_id] = set(pattern.tags)

            # Index by category
            if pattern.category not in self.pattern_category_map:
                self.pattern_category_map[pattern.category] = []
            self.pattern_category_map[pattern.category].append(pattern_id)

            # Index by domain
            if pattern.domain not in self.pattern_domain_map:
                self.pattern_domain_map[pattern.domain] = []
            self.pattern_domain_map[pattern.domain].append(pattern_id)

    def _extract_keywords(self, pattern: BPMNPattern) -> Set[str]:
        """Extract searchable keywords from a pattern."""
        keywords = set()

        # From name and description
        for text in [pattern.name, pattern.description]:
            if text:
                words = text.lower().split()
                keywords.update(word.strip(".,!?;:") for word in words if len(word) > 2)

        # From examples
        for example in pattern.examples:
            if example:
                words = example.lower().split()
                keywords.update(word.strip(".,!?;:") for word in words if len(word) > 2)

        # From tags
        keywords.update(pattern.tags)

        # From category
        keywords.add(pattern.category.value)
        keywords.add(pattern.domain.value)
        keywords.add(pattern.complexity.value)

        return keywords

    def match_by_keywords(
        self,
        query: str,
        domain: Optional[DomainType] = None,
        fuzzy: bool = True,
        threshold: float = 0.6,
    ) -> List[MatchResult]:
        """
        Match patterns using keyword search with optional fuzzy matching.

        Args:
            query: Search query (space-separated keywords)
            domain: Optional domain filter
            fuzzy: Whether to use fuzzy matching (SequenceMatcher)
            threshold: Minimum match score (0.0-1.0)

        Returns:
            List of MatchResult objects sorted by relevance
        """
        query_keywords = set(query.lower().split())
        matches: List[MatchResult] = []

        # Get patterns to search
        if domain:
            pattern_ids = self.pattern_domain_map.get(domain, [])
        else:
            pattern_ids = list(self.kb.patterns.keys())

        for pattern_id in pattern_ids:
            pattern = self.kb.patterns[pattern_id]
            keywords = self.pattern_keywords[pattern_id]
            tags = self.pattern_tags[pattern_id]

            matched_keywords = []
            score = 0.0

            # Exact keyword matches
            exact_matches = query_keywords & keywords
            matched_keywords.extend(exact_matches)
            score += len(exact_matches) / len(query_keywords) if query_keywords else 0

            # Fuzzy keyword matches
            if fuzzy and not exact_matches:
                for q_keyword in query_keywords:
                    for keyword in keywords | tags:
                        if len(q_keyword) > 2 and len(keyword) > 2:
                            similarity = SequenceMatcher(None, q_keyword, keyword).ratio()
                            if similarity > 0.75:  # High similarity threshold
                                matched_keywords.append(keyword)
                                score += similarity * 0.5 / len(query_keywords)
                                break

            # Normalize score
            score = min(1.0, score)

            if score >= threshold:
                relevance_factors = {
                    "exact_matches": (
                        len(exact_matches) / len(query_keywords) if query_keywords else 0
                    ),
                    "keyword_coverage": score,
                    "pattern_confidence": pattern.confidence,
                    "usage_popularity": min(1.0, pattern.usage_count / 100),
                }

                match_result = MatchResult(
                    pattern=pattern,
                    match_score=score,
                    match_type="fuzzy" if fuzzy else "exact",
                    matched_keywords=matched_keywords,
                    confidence=pattern.confidence,
                    reasoning=f"Matched keywords: {', '.join(set(matched_keywords))}",
                    relevance_factors=relevance_factors,
                )
                matches.append(match_result)

        # Sort by match score
        matches.sort()
        return matches

    def match_by_category(
        self,
        category: PatternCategory,
        domain: Optional[DomainType] = None,
        min_confidence: float = 0.5,
    ) -> List[MatchResult]:
        """
        Match patterns by category.

        Args:
            category: Pattern category to match
            domain: Optional domain filter
            min_confidence: Minimum confidence threshold

        Returns:
            List of MatchResult objects sorted by relevance
        """
        pattern_ids = self.pattern_category_map.get(category, [])
        matches: List[MatchResult] = []

        for pattern_id in pattern_ids:
            pattern = self.kb.patterns[pattern_id]

            # Apply domain filter if specified
            if domain and pattern.domain != domain and pattern.domain != DomainType.GENERIC:
                continue

            if pattern.confidence >= min_confidence:
                match_result = MatchResult(
                    pattern=pattern,
                    match_score=pattern.confidence,
                    match_type="category",
                    confidence=pattern.confidence,
                    reasoning=f"Pattern category matches: {category.value}",
                    relevance_factors={
                        "category_match": 1.0,
                        "pattern_confidence": pattern.confidence,
                    },
                )
                matches.append(match_result)

        matches.sort()
        return matches

    def match_by_complexity(
        self,
        complexity: ComplexityLevel,
        domain: Optional[DomainType] = None,
    ) -> List[MatchResult]:
        """
        Match patterns by complexity level.

        Args:
            complexity: Complexity level to match
            domain: Optional domain filter

        Returns:
            List of MatchResult objects sorted by relevance
        """
        matches: List[MatchResult] = []

        for _pattern_id, pattern in self.kb.patterns.items():
            if pattern.complexity == complexity:
                if domain and pattern.domain != domain and pattern.domain != DomainType.GENERIC:
                    continue

                match_result = MatchResult(
                    pattern=pattern,
                    match_score=pattern.confidence,
                    match_type="complexity",
                    confidence=pattern.confidence,
                    reasoning=f"Pattern complexity matches: {complexity.value}",
                    relevance_factors={
                        "complexity_match": 1.0,
                        "pattern_confidence": pattern.confidence,
                    },
                )
                matches.append(match_result)

        matches.sort()
        return matches

    def semantic_similarity(
        self, query_text: str, domain: Optional[DomainType] = None, threshold: float = 0.5
    ) -> List[MatchResult]:
        """
        Match patterns using semantic similarity.

        Uses simple text similarity if embedding model not available.
        Can be enhanced with proper embedding model.

        Args:
            query_text: Natural language query
            domain: Optional domain filter
            threshold: Minimum similarity threshold

        Returns:
            List of MatchResult objects sorted by relevance
        """
        matches: List[MatchResult] = []

        # Get patterns to search
        if domain:
            pattern_ids = self.pattern_domain_map.get(domain, [])
        else:
            pattern_ids = list(self.kb.patterns.keys())

        for pattern_id in pattern_ids:
            pattern = self.kb.patterns[pattern_id]

            # Calculate similarity with pattern description and examples
            scores = []

            # Compare with pattern name
            name_sim = SequenceMatcher(None, query_text.lower(), pattern.name.lower()).ratio()
            scores.append(name_sim * 1.5)  # Weight name more heavily

            # Compare with pattern description
            desc_sim = self._text_similarity(query_text, pattern.description)
            scores.append(desc_sim)

            # Compare with examples
            for example in pattern.examples:
                example_sim = self._text_similarity(query_text, example)
                scores.append(example_sim * 0.8)  # Weight examples less

            # Average scores
            if scores:
                avg_score = sum(scores) / len(scores)
            else:
                avg_score = 0.0

            if avg_score >= threshold:
                match_result = MatchResult(
                    pattern=pattern,
                    match_score=avg_score,
                    match_type="semantic",
                    confidence=pattern.confidence,
                    reasoning="Semantic similarity to pattern description",
                    relevance_factors={
                        "semantic_similarity": avg_score,
                        "pattern_confidence": pattern.confidence,
                    },
                )
                matches.append(match_result)

        matches.sort()
        return matches

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using SequenceMatcher."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def composite_search(
        self,
        query: str,
        domain: Optional[DomainType] = None,
        complexity: Optional[ComplexityLevel] = None,
        category: Optional[PatternCategory] = None,
        max_results: int = 5,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[MatchResult]:
        """
        Perform composite search combining multiple matching strategies.

        Args:
            query: Search query text
            domain: Optional domain filter
            complexity: Optional complexity filter
            category: Optional category filter
            max_results: Maximum number of results to return
            weights: Weights for different match types
                    (default: keyword=0.4, semantic=0.4, category=0.1, complexity=0.1)

        Returns:
            Top N results sorted by composite score
        """
        if weights is None:
            weights = {
                "keyword": 0.4,
                "semantic": 0.4,
                "category": 0.1,
                "complexity": 0.1,
            }

        # Run multiple searches
        keyword_matches = self.match_by_keywords(query, domain, fuzzy=True, threshold=0.4)
        semantic_matches = self.semantic_similarity(query, domain, threshold=0.3)
        category_matches = self.match_by_category(category, domain) if category else []
        complexity_matches = self.match_by_complexity(complexity, domain) if complexity else []

        # Combine results with weighted scoring
        combined_scores: Dict[str, Tuple[MatchResult, float]] = {}

        for match in keyword_matches:
            pattern_id = match.pattern.id
            score = match.match_score * weights.get("keyword", 0.4)
            if pattern_id not in combined_scores:
                combined_scores[pattern_id] = (match, 0.0)
            combined_scores[pattern_id] = (match, combined_scores[pattern_id][1] + score)

        for match in semantic_matches:
            pattern_id = match.pattern.id
            score = match.match_score * weights.get("semantic", 0.4)
            if pattern_id not in combined_scores:
                combined_scores[pattern_id] = (match, 0.0)
            combined_scores[pattern_id] = (match, combined_scores[pattern_id][1] + score)

        for match in category_matches:
            pattern_id = match.pattern.id
            score = match.match_score * weights.get("category", 0.1)
            if pattern_id not in combined_scores:
                combined_scores[pattern_id] = (match, 0.0)
            combined_scores[pattern_id] = (match, combined_scores[pattern_id][1] + score)

        for match in complexity_matches:
            pattern_id = match.pattern.id
            score = match.match_score * weights.get("complexity", 0.1)
            if pattern_id not in combined_scores:
                combined_scores[pattern_id] = (match, 0.0)
            combined_scores[pattern_id] = (match, combined_scores[pattern_id][1] + score)

        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1][1], reverse=True)

        # Return top results
        results = []
        for _pattern_id, (match, score) in sorted_results[:max_results]:
            match.match_score = min(1.0, score)  # Update score to composite
            results.append(match)

        return results

    def get_pattern_recommendations(
        self,
        context_text: str,
        domain: Optional[DomainType] = None,
        max_patterns: int = 5,
        include_alternatives: bool = True,
    ) -> PatternRecommendation:
        """
        Get pattern recommendations based on context text.

        Args:
            context_text: Natural language description of the process
            domain: Optional domain hint
            max_patterns: Number of main recommendations
            include_alternatives: Whether to include alternative patterns

        Returns:
            PatternRecommendation object with reasoning
        """
        # Perform composite search
        matches = self.composite_search(context_text, domain=domain, max_results=max_patterns * 2)

        if not matches:
            return PatternRecommendation(
                patterns=[],
                primary_domain=domain or DomainType.GENERIC,
                detected_complexity=ComplexityLevel.MODERATE,
                detected_categories=[],
                confidence=0.0,
                reasoning="No patterns found matching the context",
            )

        # Extract top patterns and alternatives
        top_patterns = [(m.pattern, m.match_score) for m in matches[:max_patterns]]
        alternative_patterns = [(m.pattern, m.match_score) for m in matches[max_patterns:]]

        # Detect complexity and categories from results
        complexities = [m.pattern.complexity for m in matches[:max_patterns]]
        categories = [m.pattern.category for m in matches[:max_patterns]]

        # Determine primary complexity
        detected_complexity = (
            max(set(complexities), key=complexities.count)
            if complexities
            else ComplexityLevel.MODERATE
        )

        # Detect primary domain
        domains = [m.pattern.domain for m in matches[:max_patterns]]
        detected_domain = max(
            (d for d in set(domains) if d != DomainType.GENERIC),
            key=lambda x: sum(1 for d in domains if d == x),
            default=domain or DomainType.GENERIC,
        )

        # Build reasoning
        reasoning = f"Found {len(matches)} matching patterns. "
        reasoning += f"Primary domain: {detected_domain.value}. "
        reasoning += f"Detected complexity: {detected_complexity.value}. "
        reasoning += f"Pattern types: {', '.join({c.value for c in categories})}"

        # Average confidence
        avg_confidence = sum(m.match_score for m in matches[:max_patterns]) / len(
            matches[:max_patterns]
        )

        return PatternRecommendation(
            patterns=top_patterns,
            primary_domain=detected_domain,
            detected_complexity=detected_complexity,
            detected_categories=list(set(categories)),
            confidence=avg_confidence,
            reasoning=reasoning,
            alternative_patterns=alternative_patterns,
        )

    def find_similar_patterns(self, pattern_id: str, max_similar: int = 5) -> List[MatchResult]:
        """
        Find patterns similar to a given pattern.

        Args:
            pattern_id: ID of the reference pattern
            max_similar: Maximum number of similar patterns to return

        Returns:
            List of similar patterns sorted by relevance
        """
        pattern = self.kb.get_pattern(pattern_id)
        if not pattern:
            return []

        # Start with explicitly related patterns
        related_ids = set(pattern.related_patterns)
        matches: Dict[str, MatchResult] = {}

        # Add explicitly related patterns
        for rel_id in related_ids:
            rel_pattern = self.kb.get_pattern(rel_id)
            if rel_pattern:
                match = MatchResult(
                    pattern=rel_pattern,
                    match_score=0.95,  # High score for explicit relations
                    match_type="related",
                    confidence=1.0,
                    reasoning="Explicitly related pattern",
                )
                matches[rel_id] = match

        # Find patterns by same category
        category_matches = self.match_by_category(pattern.category, pattern.domain)
        for match in category_matches:
            if match.pattern.id != pattern_id:
                existing = matches.get(match.pattern.id)
                if not existing or match.match_score > existing.match_score:
                    matches[match.pattern.id] = match

        # Find patterns by keyword similarity
        keyword_matches = self.match_by_keywords(
            " ".join(pattern.tags), domain=pattern.domain, threshold=0.5
        )
        for match in keyword_matches:
            if match.pattern.id != pattern_id:
                existing = matches.get(match.pattern.id)
                if not existing or match.match_score > existing.match_score:
                    matches[match.pattern.id] = match

        # Sort and return top results
        sorted_matches = sorted(matches.values())
        return sorted_matches[:max_similar]

    def validate_pattern_match(
        self, pattern: BPMNPattern, text: str
    ) -> Tuple[bool, float, List[str]]:
        """
        Validate whether a pattern matches the given text.

        Args:
            pattern: Pattern to validate against
            text: Text to match

        Returns:
            Tuple of (is_valid, confidence, validation_details)
        """
        issues = []
        scores = []

        # Check keyword presence
        text_keywords = set(text.lower().split())
        pattern_keywords = self.pattern_keywords.get(pattern.id, set())
        keyword_overlap = len(text_keywords & pattern_keywords) / max(len(pattern_keywords), 1)
        scores.append(keyword_overlap)

        if keyword_overlap < 0.3:
            issues.append("Low keyword overlap with pattern")

        # Check text length appropriateness
        word_count = len(text.split())
        if pattern.complexity == ComplexityLevel.SIMPLE and word_count > 200:
            issues.append("Text too long for simple pattern")
            scores.append(0.5)
        elif pattern.complexity == ComplexityLevel.COMPLEX and word_count < 50:
            issues.append("Text too short for complex pattern")
            scores.append(0.5)
        else:
            scores.append(1.0)

        # Check example alignment
        example_similarities = []
        for example in pattern.examples:
            sim = self._text_similarity(text, example)
            example_similarities.append(sim)

        if example_similarities:
            avg_example_sim = sum(example_similarities) / len(example_similarities)
            scores.append(avg_example_sim)

        # Overall validation
        confidence = sum(scores) / len(scores) if scores else 0.0
        is_valid = confidence > 0.5 and len(issues) == 0

        return is_valid, confidence, issues
