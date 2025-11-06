"""
Integration of Advanced Pattern Matching with Process Graph Builder

Provides a bridge between the process extraction pipeline and advanced
pattern matching capabilities for enhanced process understanding.
"""

import logging
from typing import Dict, List, Optional, Tuple

from knowledge.advanced_pattern_matcher import (
    AdvancedPatternMatcher,
    MatchResult,
    PatternRecommendation,
)
from models.knowledge_base import (
    BPMNPattern,
    ComplexityLevel,
    DomainType,
    KnowledgeBase,
    PatternCategory,
)

logger = logging.getLogger(__name__)


class AdvancedPatternMatchingBridge:
    """
    Bridge class that integrates advanced pattern matching with the process graph builder.
    
    Provides methods to:
    - Enhance pattern recommendations for process extraction
    - Provide fuzzy pattern matching for entity resolution
    - Suggest related patterns based on context
    - Validate extracted patterns against knowledge base
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize the bridge with a knowledge base.
        
        Args:
            knowledge_base: The knowledge base containing patterns
        """
        self.kb = knowledge_base
        self.matcher = AdvancedPatternMatcher(knowledge_base)
        self.logger = logger
    
    def find_patterns_for_process(
        self,
        process_description: str,
        domain: Optional[DomainType] = None,
        max_patterns: int = 5
    ) -> PatternRecommendation:
        """
        Find relevant patterns for a given process description.
        
        This is the main entry point for enhanced pattern matching.
        
        Args:
            process_description: Natural language description of the process
            domain: Optional domain hint (e.g., healthcare, finance)
            max_patterns: Maximum number of patterns to recommend
            
        Returns:
            PatternRecommendation with ranked patterns and reasoning
        """
        return self.matcher.get_pattern_recommendations(
            context_text=process_description,
            domain=domain,
            max_patterns=max_patterns
        )
    
    def match_activity_to_patterns(
        self,
        activity_name: str,
        activity_description: str,
        domain: Optional[DomainType] = None
    ) -> List[MatchResult]:
        """
        Find patterns related to a specific activity.
        
        Useful for entity resolution and activity classification.
        
        Args:
            activity_name: Name of the activity
            activity_description: Description or context of the activity
            domain: Optional domain filter
            
        Returns:
            List of matching patterns sorted by relevance
        """
        query = f"{activity_name} {activity_description}".strip()
        return self.matcher.composite_search(
            query=query,
            domain=domain,
            max_results=5
        )
    
    def find_similar_patterns_for_pattern(
        self,
        pattern_id: str,
        max_similar: int = 3
    ) -> List[BPMNPattern]:
        """
        Find patterns similar to a given pattern.
        
        Useful for suggesting related patterns or alternatives.
        
        Args:
            pattern_id: ID of the reference pattern
            max_similar: Maximum number of similar patterns
            
        Returns:
            List of similar patterns
        """
        matches = self.matcher.find_similar_patterns(pattern_id, max_similar)
        return [m.pattern for m in matches]
    
    def validate_extracted_activities(
        self,
        activities: List[str],
        domain: Optional[DomainType] = None
    ) -> Dict[str, Tuple[bool, float, List[str]]]:
        """
        Validate extracted activities against patterns.
        
        Args:
            activities: List of activity names/descriptions
            domain: Optional domain filter
            
        Returns:
            Dictionary mapping activities to (is_valid, confidence, issues)
        """
        results = {}
        
        for activity in activities:
            # Find best matching patterns
            matches = self.matcher.match_by_keywords(activity, domain=domain)
            
            if matches:
                # Validate against the top match
                best_match = matches[0]
                is_valid, confidence, issues = self.matcher.validate_pattern_match(
                    best_match.pattern,
                    activity
                )
                results[activity] = (is_valid, confidence, issues)
            else:
                # No matching patterns
                results[activity] = (False, 0.0, ["No matching patterns found"])
        
        return results
    
    def enrich_pattern_context(
        self,
        pattern_id: str
    ) -> Dict:
        """
        Enrich a pattern with related patterns and context.
        
        Args:
            pattern_id: ID of the pattern to enrich
            
        Returns:
            Dictionary with pattern info and related patterns
        """
        pattern = self.kb.get_pattern(pattern_id)
        if not pattern:
            return {}
        
        # Find related patterns
        related_patterns = self.matcher.find_similar_patterns(pattern_id, max_similar=5)
        related_list = [
            {
                "id": m.pattern.id,
                "name": m.pattern.name,
                "score": m.match_score,
                "type": m.match_type
            }
            for m in related_patterns
        ]
        
        # Get explicit related patterns from KB
        kb_related = [
            {
                "id": rel_id,
                "name": self.kb.get_pattern(rel_id).name if self.kb.get_pattern(rel_id) else rel_id
            }
            for rel_id in pattern.related_patterns
        ]
        
        return {
            "pattern": {
                "id": pattern.id,
                "name": pattern.name,
                "description": pattern.description,
                "category": pattern.category.value,
                "domain": pattern.domain.value,
                "complexity": pattern.complexity.value,
                "confidence": pattern.confidence,
                "tags": list(pattern.tags),
                "usage_count": pattern.usage_count,
            },
            "similar_patterns": related_list,
            "related_patterns": kb_related,
            "examples": pattern.examples,
            "validation_rules": pattern.validation_rules,
            "anti_patterns": pattern.anti_patterns,
        }
    
    def suggest_patterns_by_domain(
        self,
        domain: DomainType,
        complexity: Optional[ComplexityLevel] = None,
        max_patterns: int = 10
    ) -> List[Dict]:
        """
        Get suggested patterns for a specific domain.
        
        Useful for domain-specific guidance.
        
        Args:
            domain: Domain to get patterns for
            complexity: Optional complexity filter
            max_patterns: Maximum patterns to return
            
        Returns:
            List of pattern info dictionaries
        """
        patterns = self.kb.get_patterns_by_domain(domain)
        
        # Filter by complexity if specified
        if complexity:
            patterns = {
                pid: p for pid, p in patterns.items()
                if p.complexity == complexity
            }
        
        # Sort by usage and confidence
        sorted_patterns = sorted(
            patterns.items(),
            key=lambda x: (x[1].usage_count, x[1].confidence),
            reverse=True
        )
        
        # Return top patterns
        result = []
        for pattern_id, pattern in sorted_patterns[:max_patterns]:
            result.append({
                "id": pattern.id,
                "name": pattern.name,
                "description": pattern.description,
                "domain": pattern.domain.value,
                "category": pattern.category.value,
                "complexity": pattern.complexity.value,
                "confidence": pattern.confidence,
                "usage_count": pattern.usage_count,
                "tags": list(pattern.tags),
            })
        
        return result
    
    def search_patterns(
        self,
        query: str,
        domain: Optional[DomainType] = None,
        category: Optional[PatternCategory] = None,
        complexity: Optional[ComplexityLevel] = None,
        max_results: int = 5
    ) -> List[MatchResult]:
        """
        Perform a comprehensive search across all patterns.
        
        Args:
            query: Search query
            domain: Optional domain filter
            category: Optional category filter
            complexity: Optional complexity filter
            max_results: Maximum results to return
            
        Returns:
            List of matching patterns
        """
        return self.matcher.composite_search(
            query=query,
            domain=domain,
            category=category,
            complexity=complexity,
            max_results=max_results
        )
    
    def get_pattern_statistics(self) -> Dict:
        """
        Get statistics about the pattern library.
        
        Returns:
            Dictionary with pattern statistics
        """
        stats = {
            "total_patterns": len(self.kb.patterns),
            "total_examples": len(self.kb.examples),
            "domains": {},
            "categories": {},
            "complexities": {},
        }
        
        for pattern in self.kb.patterns.values():
            # Count by domain
            domain = pattern.domain.value
            stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
            
            # Count by category
            category = pattern.category.value
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            
            # Count by complexity
            complexity = pattern.complexity.value
            stats["complexities"][complexity] = stats["complexities"].get(complexity, 0) + 1
        
        # Add metadata
        stats["metadata"] = {
            "version": self.kb.metadata.version,
            "created_at": self.kb.metadata.created_at.isoformat() if self.kb.metadata.created_at else None,
            "last_updated": self.kb.metadata.last_updated.isoformat() if self.kb.metadata.last_updated else None,
        }
        
        return stats
    
    def export_pattern_for_documentation(self, pattern_id: str) -> str:
        """
        Export a pattern in a human-readable format for documentation.
        
        Args:
            pattern_id: ID of the pattern to export
            
        Returns:
            Formatted pattern documentation
        """
        pattern = self.kb.get_pattern(pattern_id)
        if not pattern:
            return ""
        
        doc = f"""
# {pattern.name}

**ID:** `{pattern.id}`
**Domain:** {pattern.domain.value}
**Category:** {pattern.category.value}
**Complexity:** {pattern.complexity.value}
**Confidence:** {pattern.confidence:.1%}

## Description
{pattern.description}

## Tags
{', '.join(pattern.tags)}

## Examples
"""
        for i, example in enumerate(pattern.examples, 1):
            doc += f"\n{i}. {example}"
        
        if pattern.related_patterns:
            doc += "\n\n## Related Patterns\n"
            for rel_id in pattern.related_patterns:
                rel_pattern = self.kb.get_pattern(rel_id)
                if rel_pattern:
                    doc += f"- [{rel_pattern.name}]({rel_pattern.id})\n"
        
        if pattern.validation_rules:
            doc += "\n## Validation Rules\n"
            for rule in pattern.validation_rules:
                doc += f"- {rule}\n"
        
        if pattern.anti_patterns:
            doc += "\n## Anti-patterns to Avoid\n"
            for anti in pattern.anti_patterns:
                doc += f"- {anti}\n"
        
        return doc
