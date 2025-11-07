"""
FastAPI REST endpoints for advanced pattern matching.

Provides a comprehensive REST API for:
- Pattern discovery and search
- Activity validation against patterns
- Domain-specific pattern suggestions
- Pattern enrichment and details
"""

from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field

from bpmn_agent.knowledge.pattern_matching_bridge import AdvancedPatternMatchingBridge
from bpmn_agent.models.knowledge_base import (
    ComplexityLevel,
    DomainType,
    KnowledgeBase,
    PatternCategory,
)

# ===========================
# Request/Response Models
# ===========================


class DomainTypeEnum(str, Enum):
    """Domain types for API."""

    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"
    LOGISTICS = "logistics"
    GENERIC = "generic"


class ComplexityLevelEnum(str, Enum):
    """Complexity levels for API."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class PatternCategoryEnum(str, Enum):
    """Pattern categories for API."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    EXCLUSIVE_CHOICE = "exclusive_choice"
    INCLUSIVE_CHOICE = "inclusive_choice"
    EVENT_BASED_CHOICE = "event_based_choice"
    COMPLEX_GATEWAY = "complex_gateway"
    SUBPROCESS = "subprocess"


class PatternMatchResult(BaseModel):
    """Result of pattern matching."""

    id: str
    name: str
    score: float = Field(..., ge=0.0, le=1.0, description="Match score between 0 and 1")
    category: str
    complexity: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)


class PatternRecommendation(BaseModel):
    """Pattern recommendation with alternatives."""

    best_pattern_id: Optional[str]
    best_pattern_name: Optional[str]
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: List[PatternMatchResult] = Field(default_factory=list)


class ActivityValidationResult(BaseModel):
    """Result of activity validation."""

    activity: str
    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    suggestions: List[str] = Field(default_factory=list)


class PatternSearchResult(BaseModel):
    """Result of pattern search."""

    patterns: List[PatternMatchResult] = Field(default_factory=list)
    total_count: int


class DomainPatternsResult(BaseModel):
    """Patterns for a specific domain."""

    domain: str
    patterns: List[PatternMatchResult] = Field(default_factory=list)
    total_count: int


class PatternDetailsResult(BaseModel):
    """Detailed information about a pattern."""

    id: str
    name: str
    description: str
    domain: str
    category: str
    complexity: str
    confidence: float
    tags: List[str]
    related_patterns: List[dict] = Field(default_factory=list)


# ===========================
# Initialize Router and Bridge
# ===========================

router = APIRouter(prefix="/api/v1/patterns", tags=["pattern-matching"])

# Global bridge instance (will be initialized on first use)
_bridge_instance: Optional[AdvancedPatternMatchingBridge] = None


def get_bridge() -> AdvancedPatternMatchingBridge:
    """Get or initialize the pattern matching bridge."""
    global _bridge_instance
    if _bridge_instance is None:
        try:
            kb = KnowledgeBase()
            _bridge_instance = AdvancedPatternMatchingBridge(kb)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize pattern matching bridge: {str(e)}"
            ) from e
    return _bridge_instance


# ===========================
# Endpoints
# ===========================


@router.post("/search", response_model=PatternSearchResult)
async def search_patterns(
    query: str = Query(..., min_length=1, description="Search query"),
    domain: Optional[DomainTypeEnum] = Query(None, description="Filter by domain"),
    category: Optional[PatternCategoryEnum] = Query(None, description="Filter by category"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum results to return"),
) -> PatternSearchResult:
    """
    Search for patterns using advanced matching.

    Supports:
    - Fuzzy keyword matching
    - Semantic similarity
    - Domain and category filtering
    """
    try:
        bridge = get_bridge()
        # Convert API enum values to model enum types using value-based mapping
        domain_type = None
        if domain:
            domain_mapping = {d.value: d for d in DomainType}
            domain_type = domain_mapping.get(domain.value.lower())
            if domain_type is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid domain: {domain.value}. Valid domains: {', '.join([d.value for d in DomainType])}",
                )

        category_type = None
        if category:
            category_mapping = {c.value: c for c in PatternCategory}
            category_type = category_mapping.get(category.value.lower())
            if category_type is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category: {category.value}. Valid categories: {', '.join([c.value for c in PatternCategory])}",
                )

        results = bridge.search_patterns(query, domain=domain_type, category=category_type)

        pattern_results = [
            PatternMatchResult(
                id=r.pattern.id,
                name=r.pattern.name,
                score=r.match_score,
                category=r.pattern.category.value,
                complexity=r.pattern.complexity.value,
                confidence=r.pattern.confidence,
                tags=list(r.pattern.tags),
            )
            for r in results[:max_results]
        ]

        return PatternSearchResult(patterns=pattern_results, total_count=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e


@router.post("/find-for-process", response_model=PatternRecommendation)
async def find_patterns_for_process(
    process_description: str = Query(..., min_length=1, description="Process description"),
    domain: Optional[DomainTypeEnum] = Query(None, description="Domain hint"),
) -> PatternRecommendation:
    """
    Find matching patterns for a process description.

    Returns the best matching pattern plus alternatives.
    """
    try:
        bridge = get_bridge()
        # Convert API enum value to model enum type using value-based mapping
        domain_type = None
        if domain:
            domain_mapping = {d.value: d for d in DomainType}
            domain_type = domain_mapping.get(domain.value.lower())
            if domain_type is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid domain: {domain.value}. Valid domains: {', '.join([d.value for d in DomainType])}",
                )

        recommendation = bridge.find_patterns_for_process(
            process_description, domain_hint=domain_type
        )

        alternatives = [
            PatternMatchResult(
                id=p.pattern.id,
                name=p.pattern.name,
                score=p.match_score,
                category=p.pattern.category.value,
                complexity=p.pattern.complexity.value,
                confidence=p.pattern.confidence,
                tags=list(p.pattern.tags),
            )
            for p in recommendation.patterns
        ]

        return PatternRecommendation(
            best_pattern_id=recommendation.best_pattern.id if recommendation.best_pattern else None,
            best_pattern_name=(
                recommendation.best_pattern.name if recommendation.best_pattern else None
            ),
            confidence=recommendation.confidence,
            alternatives=alternatives,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern finding failed: {str(e)}") from e


@router.post("/validate-activities", response_model=List[ActivityValidationResult])
async def validate_activities(
    activities: List[str] = Query(..., description="List of activity labels to validate"),
    domain: Optional[DomainTypeEnum] = Query(None, description="Domain for validation"),
) -> List[ActivityValidationResult]:
    """
    Validate extracted activities against known patterns.

    Checks if activities are recognized and provides suggestions.
    """
    try:
        if not activities:
            raise HTTPException(status_code=400, detail="Activities list cannot be empty")

        bridge = get_bridge()
        # Convert API enum value to model enum type using value-based mapping
        domain_type = None
        if domain:
            domain_mapping = {d.value: d for d in DomainType}
            domain_type = domain_mapping.get(domain.value.lower())
            if domain_type is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid domain: {domain.value}. Valid domains: {', '.join([d.value for d in DomainType])}",
                )

        results = bridge.validate_extracted_activities(activities, domain_hint=domain_type)

        validation_results = [
            ActivityValidationResult(
                activity=activity, is_valid=valid, confidence=score, suggestions=suggestions
            )
            for activity, (valid, score, suggestions) in results.items()
        ]

        return validation_results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Activity validation failed: {str(e)}")


@router.get("/by-domain/{domain}", response_model=DomainPatternsResult)
async def get_patterns_by_domain(
    domain: str = Path(..., description="Domain type"),
    max_patterns: int = Query(10, ge=1, le=100, description="Maximum patterns to return"),
    complexity: Optional[str] = Query(None, description="Filter by complexity"),
) -> DomainPatternsResult:
    """
    Get pattern suggestions for a specific domain.

    Useful for discovering available patterns in a domain.
    """
    try:
        bridge = get_bridge()
        # Validate domain - map lowercase string to DomainType enum
        domain_lower = domain.lower()
        domain_mapping = {d.value: d for d in DomainType}
        if domain_lower not in domain_mapping:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid domain: {domain}. Valid domains: {', '.join([d.value for d in DomainType])}",
            )
        domain_type = domain_mapping[domain_lower]

        complexity_level = None
        if complexity:
            # Map complexity string to ComplexityLevel
            complexity_mapping = {c.value: c for c in ComplexityLevel}
            complexity_lower = complexity.lower()
            if complexity_lower not in complexity_mapping:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid complexity: {complexity}. Valid levels: {', '.join([c.value for c in ComplexityLevel])}",
                )
            complexity_level = complexity_mapping[complexity_lower]

        patterns = bridge.suggest_patterns_by_domain(
            domain_type, complexity=complexity_level, max_patterns=max_patterns
        )

        pattern_results = [
            PatternMatchResult(
                id=p.get("id", ""),
                name=p.get("name", ""),
                score=1.0,
                category=p.get("category", ""),
                complexity=p.get("complexity", ""),
                confidence=p.get("confidence", 0.0),
                tags=p.get("tags", []),
            )
            for p in patterns
        ]

        return DomainPatternsResult(
            domain=domain, patterns=pattern_results, total_count=len(pattern_results)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Domain pattern retrieval failed: {str(e)}") from e


@router.get("/pattern/{pattern_id}", response_model=PatternDetailsResult)
async def get_pattern_details(
    pattern_id: str = Path(..., description="Pattern ID")
) -> PatternDetailsResult:
    """
    Get detailed information about a specific pattern.

    Includes related patterns and metadata.
    """
    try:
        bridge = get_bridge()

        # Enrich the pattern with context
        enriched = bridge.enrich_pattern_context(pattern_id)

        if not enriched:
            raise HTTPException(status_code=404, detail=f"Pattern not found: {pattern_id}")

        return PatternDetailsResult(
            id=enriched.get("id", pattern_id),
            name=enriched.get("name", ""),
            description=enriched.get("description", ""),
            domain=enriched.get("domain", ""),
            category=enriched.get("category", ""),
            complexity=enriched.get("complexity", ""),
            confidence=enriched.get("confidence", 0.0),
            tags=enriched.get("tags", []),
            related_patterns=enriched.get("related_patterns", []),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern retrieval failed: {str(e)}")


@router.get("/similar/{pattern_id}", response_model=PatternSearchResult)
async def find_similar_patterns(
    pattern_id: str = Path(..., description="Reference pattern ID"),
    max_patterns: int = Query(5, ge=1, le=50, description="Maximum similar patterns"),
) -> PatternSearchResult:
    """
    Find patterns similar to a given pattern.

    Based on category, complexity, domain, and content similarity.
    """
    try:
        bridge = get_bridge()

        similar = bridge.find_similar_patterns_for_pattern(pattern_id, max_similar=max_patterns)

        if not similar:
            # Pattern might not exist or have no similar patterns
            return PatternSearchResult(patterns=[], total_count=0)

        pattern_results = [
            PatternMatchResult(
                id=s.pattern.id,
                name=s.pattern.name,
                score=s.match_score,
                category=s.pattern.category.value,
                complexity=s.pattern.complexity.value,
                confidence=s.pattern.confidence,
                tags=list(s.pattern.tags),
            )
            for s in similar
        ]

        return PatternSearchResult(patterns=pattern_results, total_count=len(pattern_results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similar pattern search failed: {str(e)}")


@router.get("/statistics")
async def get_pattern_statistics():
    """
    Get statistics about the pattern library.

    Includes counts by domain, category, complexity, etc.
    """
    try:
        bridge = get_bridge()
        stats = bridge.get_pattern_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}") from e


# ===========================
# Health Checks
# ===========================


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        bridge = get_bridge()
        stats = bridge.get_pattern_statistics()
        return {"status": "healthy", "total_patterns": stats.get("total_patterns", 0)}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
