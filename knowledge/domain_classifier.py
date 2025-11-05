"""
Domain Classification and Pattern Recognition

Provides domain detection, complexity analysis, and pattern recognition
from natural language process descriptions.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from models.knowledge_base import (
    ComplexityLevel,
    DomainType,
    KnowledgeBase,
    PatternCategory,
)
from .vector_store import VectorStore


# Result models for classification and recognition
@dataclass
class DomainClassificationResult:
    """Result of domain classification."""
    domain: DomainType
    confidence: float
    indicators: List[str] = None  # Keywords that matched to determine domain
    
    def __post_init__(self):
        if self.indicators is None:
            self.indicators = []


@dataclass
class ComplexityAnalysisResult:
    """Result of complexity analysis."""
    level: ComplexityLevel
    score: float
    factors: Dict[str, float]


@dataclass
class PatternRecognitionResult:
    """Result of pattern recognition."""
    pattern_id: str
    pattern_category: PatternCategory
    confidence: float


class DomainClassifier:
    """
    Detects the business domain from text.
    
    Uses keyword matching and optionally semantic similarity.
    """
    
    # Domain-specific keywords
    DOMAIN_KEYWORDS: Dict[DomainType, Dict[str, float]] = {
        DomainType.HR: {
            "hiring": 1.0,
            "recruitment": 1.0,
            "employee": 0.9,
            "onboarding": 1.0,
            "training": 0.8,
            "performance review": 1.0,
            "leave": 0.9,
            "vacation": 0.8,
            "promotion": 0.9,
            "salary": 0.8,
            "payroll": 0.9,
            "hr": 0.7,
            "human resources": 1.0,
            "manager": 0.7,
            "department": 0.6,
            "termination": 0.9,
        },
        DomainType.FINANCE: {
            "invoice": 1.0,
            "payment": 0.9,
            "expense": 0.9,
            "budget": 0.9,
            "accounting": 1.0,
            "accounts payable": 1.0,
            "accounts receivable": 1.0,
            "purchase order": 1.0,
            "receipt": 0.8,
            "financial": 0.9,
            "reimbursement": 1.0,
            "approval": 0.7,
            "cost": 0.7,
            "transaction": 0.8,
            "audit": 0.9,
        },
        DomainType.IT: {
            "incident": 1.0,
            "ticket": 0.9,
            "deployment": 1.0,
            "release": 0.9,
            "change management": 1.0,
            "bug": 0.8,
            "issue": 0.7,
            "access request": 1.0,
            "provisioning": 0.9,
            "it service": 1.0,
            "system": 0.6,
            "application": 0.7,
            "production": 0.8,
            "downtime": 0.9,
            "troubleshooting": 0.9,
        },
        DomainType.HEALTHCARE: {
            "patient": 1.0,
            "appointment": 0.9,
            "medical": 0.9,
            "diagnosis": 1.0,
            "treatment": 0.9,
            "discharge": 1.0,
            "admission": 1.0,
            "prescription": 0.9,
            "surgery": 0.9,
            "hospital": 0.9,
            "doctor": 0.8,
            "clinical": 0.9,
            "health": 0.7,
            "nurse": 0.8,
            "emergency": 0.8,
        },
        DomainType.MANUFACTURING: {
            "production": 0.9,
            "quality control": 1.0,
            "assembly": 1.0,
            "equipment": 0.7,
            "maintenance": 0.8,
            "supply chain": 1.0,
            "inventory": 0.8,
            "warehouse": 0.8,
            "order fulfillment": 0.9,
            "material": 0.7,
            "defect": 0.9,
            "inspection": 0.9,
            "manufacturing": 1.0,
            "line": 0.6,
            "production planning": 1.0,
        },
    }
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the domain classifier.
        
        Args:
            vector_store: Optional vector store for semantic matching
        """
        self.vector_store = vector_store
        self.complexity_analyzer = ComplexityAnalyzer()
    
    def analyze_complexity(self, text: str) -> ComplexityAnalysisResult:
        """
        Analyze process complexity.
        
        Args:
            text: Process description
            
        Returns:
            ComplexityAnalysisResult with level, score, and factors
        """
        return self.complexity_analyzer.analyze_complexity(text)
    
    def classify_domain(self, text: str) -> DomainClassificationResult:
        """
        Classify the domain of a process description.
        
        Args:
            text: Process description text
            
        Returns:
            DomainClassificationResult with domain and confidence score
        """
        text_lower = text.lower()
        domain_scores: Dict[DomainType, float] = {domain: 0.0 for domain in DomainType}
        domain_indicators: Dict[DomainType, List[str]] = {domain: [] for domain in DomainType}
        
        # Keyword matching
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword, weight in keywords.items():
                if keyword in text_lower:
                    # Count occurrences
                    count = text_lower.count(keyword)
                    domain_scores[domain] += weight * count
                    if keyword not in domain_indicators[domain]:
                        domain_indicators[domain].append(keyword)
        
        # Normalize scores
        max_score = max(domain_scores.values()) if domain_scores else 0
        if max_score > 0:
            domain_scores = {
                domain: score / max_score for domain, score in domain_scores.items()
            }
        
        # If using vector store, get semantic matches
        if self.vector_store and max(domain_scores.values()) < 0.5:
            semantic_result = self._semantic_domain_match(text)
            if semantic_result.confidence > 0.3:
                return semantic_result
        
        # Find best domain
        best_domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[best_domain]
        
        # If confidence is very low, return generic
        if confidence < 0.1:
            confidence = 0.1
        
        return DomainClassificationResult(
            domain=best_domain,
            confidence=confidence,
            indicators=domain_indicators[best_domain]
        )
    
    def _semantic_domain_match(self, text: str) -> DomainClassificationResult:
        """
        Use vector store for semantic domain matching.
        
        Args:
            text: Process description
            
        Returns:
            DomainClassificationResult with domain and confidence
        """
        if not self.vector_store:
            return DomainClassificationResult(domain=DomainType.GENERIC, confidence=0.0)
        
        try:
            # Search for patterns to determine domain
            results, _ = self.vector_store.search_combined(
                text, top_k_patterns=3, top_k_examples=1
            )
            
            if results:
                # Get domain from top pattern
                domain = results[0].item.get("domain", "generic")
                confidence = results[0].similarity_score
                return DomainClassificationResult(
                    domain=DomainType(domain), confidence=float(confidence)
                )
        except Exception:
            pass
        
        return DomainClassificationResult(domain=DomainType.GENERIC, confidence=0.0)


class ComplexityAnalyzer:
    """
    Analyzes process complexity from text.
    
    Estimates complexity based on number of elements, decision points, actors, etc.
    """
    
    def analyze_complexity(self, text: str) -> ComplexityAnalysisResult:
        """
        Analyze process complexity.
        
        Args:
            text: Process description
            
        Returns:
            ComplexityAnalysisResult with level, score, and factors
        """
        # Count various process elements
        activity_count = self._count_activities(text)
        gateway_count = self._count_gateways(text)
        actor_count = self._count_actors(text)
        event_count = self._count_events(text)
        condition_count = self._count_conditions(text)
        loop_count = self._count_loops(text)
        
        # Total element count
        total_elements = (
            activity_count + gateway_count + event_count + actor_count
        )
        
        # Calculate complexity score
        # Base score from element count
        if total_elements <= 5:
            complexity_score = 0.2
        elif total_elements <= 15:
            complexity_score = 0.5
        else:
            complexity_score = min(1.0, 0.5 + (total_elements - 15) / 30.0)
        
        # Add factors for decision complexity
        decision_factor = min(0.3, gateway_count * 0.1)
        condition_factor = min(0.2, condition_count * 0.05)
        loop_factor = min(0.2, loop_count * 0.15)
        actor_factor = min(0.15, (actor_count - 1) * 0.05) if actor_count > 1 else 0
        
        complexity_score += decision_factor + condition_factor + loop_factor + actor_factor
        complexity_score = min(1.0, complexity_score)
        
        # Determine complexity level
        if complexity_score <= 0.33:
            level = ComplexityLevel.SIMPLE
        elif complexity_score <= 0.66:
            level = ComplexityLevel.MODERATE
        else:
            level = ComplexityLevel.COMPLEX
        
        # Build factors dictionary
        factors = {
            "activity_count": float(activity_count),
            "gateway_count": float(gateway_count),
            "actor_count": float(actor_count),
            "event_count": float(event_count),
            "condition_count": float(condition_count),
            "loop_count": float(loop_count),
            "total_elements": float(total_elements),
            "decision_factor": decision_factor,
            "condition_factor": condition_factor,
            "loop_factor": loop_factor,
            "actor_factor": actor_factor,
        }
        
        return ComplexityAnalysisResult(level=level, score=complexity_score, factors=factors)
    
    @staticmethod
    def _count_activities(text: str) -> int:
        """Count number of activities/tasks."""
        patterns = [
            r'\b(?:do|perform|execute|task|activity|activity|process|step|action)\b',
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',  # Capitalized phrases likely task names
        ]
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text))
        return max(1, count // 2)  # Avoid double counting
    
    @staticmethod
    def _count_gateways(text: str) -> int:
        """Count decision gateways and branches."""
        patterns = [
            r'\bif\b', r'\bthen\b', r'\belse\b', r'\band\b', r'\bor\b',
            r'\bgwateway\b', r'\bdecision\b', r'\bchoice\b', r'\bsplit\b',
            r'\bjoin\b', r'\bfork\b', r'\bmerge\b', r'\beither.*or\b'
        ]
        count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in patterns)
        return count
    
    @staticmethod
    def _count_actors(text: str) -> int:
        """Count number of actors/roles."""
        patterns = [
            r'\b(?:user|actor|role|person|manager|employee|customer|client)\b',
            r'\b(?:department|team|group|system|application)\b',
        ]
        count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in patterns)
        return max(1, count)
    
    @staticmethod
    def _count_events(text: str) -> int:
        """Count events in process."""
        patterns = [
            r'\b(?:start|end|event|trigger|occur|happens?)\b',
            r'\b(?:begin|finish|complete|submit|receive)\b',
        ]
        count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in patterns)
        return count
    
    @staticmethod
    def _count_conditions(text: str) -> int:
        """Count conditional statements."""
        patterns = [
            r'\bif\b', r'\bunless\b', r'\bwhen\b', r'\bcondition\b',
            r'\b(?:valid|invalid|approved|rejected|accept|reject)\b',
        ]
        count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in patterns)
        return count
    
    @staticmethod
    def _count_loops(text: str) -> int:
        """Count loop/iteration patterns."""
        patterns = [
            r'\b(?:loop|repeat|iterate|while|for|each)\b',
            r'\b(?:until|as long as)\b',
        ]
        count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in patterns)
        return count


class PatternRecognizer:
    """
    Recognizes BPMN patterns in text.
    
    Uses vector search and keyword matching to identify known patterns.
    """
    
    # Pattern keywords mapping
    PATTERN_KEYWORDS: Dict[PatternCategory, List[str]] = {
        PatternCategory.SEQUENTIAL: [
            "then", "after", "next", "follow", "sequence", "step by step"
        ],
        PatternCategory.PARALLEL: [
            "parallel", "concurrent", "simultaneous", "at the same time",
            "together", "alongside", "while"
        ],
        PatternCategory.EXCLUSIVE_CHOICE: [
            "either", "or", "if", "else", "choose", "exclusive", "mutually exclusive"
        ],
        PatternCategory.INCLUSIVE_CHOICE: [
            "and/or", "and or", "inclusive", "or both", "possibly both"
        ],
        PatternCategory.MULTI_INSTANCE: [
            "each", "every", "loop", "repeat", "iterate", "for each", "multiple"
        ],
        PatternCategory.EVENT_DRIVEN: [
            "when", "triggered by", "upon", "event", "occurs", "happens", "if"
        ],
        PatternCategory.EXCEPTION_HANDLING: [
            "error", "exception", "failure", "abort", "cancel", "reject", "invalid"
        ],
        PatternCategory.SYNCHRONIZATION: [
            "wait", "synchronize", "join", "merge", "all complete", "everyone done"
        ],
        PatternCategory.MESSAGE_PASSING: [
            "message", "communicate", "send", "receive", "notify", "inform"
        ],
        PatternCategory.DATA_FLOW: [
            "data", "input", "output", "use", "produce", "consume", "process"
        ],
    }
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        kb: Optional[KnowledgeBase] = None
    ):
        """
        Initialize pattern recognizer.
        
        Args:
            vector_store: Optional vector store for semantic search
            kb: Optional knowledge base for pattern metadata
        """
        self.vector_store = vector_store
        self.kb = kb
    
    def recognize_patterns(
        self, text: str, top_k: int = 5
    ) -> List[PatternRecognitionResult]:
        """
        Recognize patterns in text.
        
        Args:
            text: Process description
            top_k: Maximum number of patterns to return
            
        Returns:
            List of PatternRecognitionResult objects
        """
        recognized: Dict[PatternCategory, float] = {}
        
        # Keyword-based pattern recognition
        for category, keywords in self.PATTERN_KEYWORDS.items():
            score = self._keyword_match_score(text, keywords)
            if score > 0:
                recognized[category] = score
        
        # Vector-based pattern search if available
        if self.vector_store:
            pattern_results = self.vector_store.search_patterns(
                text, top_k=top_k, min_similarity=0.3
            )
            for result in pattern_results:
                # Could extract pattern category from result if available
                pass
        
        # Sort and format results
        results = []
        for category, score in sorted(
            recognized.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]:
            # Try to get pattern ID from KB if available
            pattern_id = f"pattern_{category.value}_001"
            if self.kb:
                matching_patterns = self.kb.get_patterns_by_category(category)
                if matching_patterns:
                    pattern_id = list(matching_patterns.keys())[0]
            
            results.append(PatternRecognitionResult(
                pattern_id=pattern_id,
                pattern_category=category,
                confidence=min(1.0, score)
            ))
        
        return results
    
    @staticmethod
    def _keyword_match_score(text: str, keywords: List[str]) -> float:
        """
        Calculate keyword match score.
        
        Args:
            text: Text to search
            keywords: Keywords to look for
            
        Returns:
            Match score (0-1)
        """
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        
        if not keywords:
            return 0.0
        
        # Score: 0-1, capped at 1.0
        score = matches / len(keywords)
        return min(1.0, score)
