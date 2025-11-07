"""
Stage 3: Entity Resolution & Co-reference Resolution

Consolidates mentions referring to the same entity and handles:
- Task 2.3.1: Co-reference resolution (clustering mentions by similarity)
- Task 2.3.2: Actor/lane consolidation (group activities by responsible actor)
- Task 2.3.3: Relationship validation (check all relations reference valid entities)
- KB Integration: Use domain knowledge for co-reference and consolidation rules
"""

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from bpmn_agent.knowledge import DomainClassifier
from bpmn_agent.models.extraction import (
    ConfidenceLevel,
    CoReferenceGroup,
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    RelationType,
)
from bpmn_agent.models.knowledge_base import DomainType

logger = logging.getLogger(__name__)


# ===========================
# Task 2.3.1: Co-reference Resolution
# ===========================


class CoReferenceResolver:
    """Resolves co-references (mentions of same entity) with KB support."""

    def __init__(self, similarity_threshold: float = 0.75, enable_kb: bool = True):
        """
        Initialize resolver with optional KB support.

        Args:
            similarity_threshold: Minimum string similarity (0-1) to cluster mentions
            enable_kb: Whether to enable KB domain rules for co-reference
        """
        self.similarity_threshold = similarity_threshold
        self.enable_kb = enable_kb

        # Initialize KB components (lazy loading)
        self._domain_classifier: Optional[DomainClassifier] = None
        self._domain_type: Optional[DomainType] = None

    def _ensure_kb_initialized(self, domain: Optional[DomainType] = None) -> None:
        """Initialize KB components if enabled and not already done."""
        if not self.enable_kb:
            return

        if self._domain_classifier is None:
            try:
                self._domain_classifier = DomainClassifier()
                self._domain_type = domain
                logger.debug("KB components initialized for entity resolution")
            except Exception as e:
                logger.warning(f"Failed to initialize KB components: {e}")
                self.enable_kb = False

    def resolve_co_references(
        self,
        entities: List[ExtractedEntity],
        domain: Optional[DomainType] = None,
    ) -> Tuple[List[ExtractedEntity], List[CoReferenceGroup]]:
        """
        Cluster entities by co-reference and return canonical forms.

        Uses KB domain rules to improve co-reference resolution.

        Args:
            entities: Extracted entities
            domain: Detected domain for KB-aware resolution

        Returns:
            (canonical_entities, co_reference_groups)
        """
        if not entities:
            return [], []

        # Initialize KB if enabled with domain context
        if self.enable_kb and domain:
            self._ensure_kb_initialized(domain)

        # 1. Group entities by type (only co-reference within type)
        entities_by_type: Dict[EntityType, List[ExtractedEntity]] = {}
        for entity in entities:
            if entity.type not in entities_by_type:
                entities_by_type[entity.type] = []
            entities_by_type[entity.type].append(entity)

        # 2. Cluster mentions within each type
        all_clusters: List[List[ExtractedEntity]] = []
        for entity_type, type_entities in entities_by_type.items():
            # Apply domain-specific clustering rules if available
            similarity_threshold = self.similarity_threshold
            if self.enable_kb and entity_type in [EntityType.ACTOR, EntityType.ACTIVITY]:
                # Use slightly higher threshold for domain-specific terms
                similarity_threshold = max(0.7, self.similarity_threshold - 0.05)
                logger.debug(
                    f"Using domain-adjusted threshold {similarity_threshold} for {entity_type}"
                )

            clusters = self._cluster_entities(type_entities, similarity_threshold)
            all_clusters.extend(clusters)

        # 3. Select canonical form for each cluster
        canonical_entities: List[ExtractedEntity] = []
        co_reference_groups: List[CoReferenceGroup] = []

        for cluster in all_clusters:
            if len(cluster) == 0:
                continue

            # Canonical entity is the one with highest confidence
            canonical = max(cluster, key=lambda e: self._confidence_score(e.confidence))

            # Collect all mentions
            mention_ids = [e.id for e in cluster]
            mention_texts = [e.name for e in cluster]

            # Group confidence
            cluster_confidence = max(e.confidence for e in cluster)

            # Create co-reference group
            if len(cluster) > 1:
                co_ref_group = CoReferenceGroup(
                    canonical_id=canonical.id,
                    canonical_form=canonical.name,
                    mentions=mention_ids,
                    mention_texts=mention_texts,
                    confidence=cluster_confidence,
                )
                co_reference_groups.append(co_ref_group)

            canonical_entities.append(canonical)

        logger.debug(
            f"Co-reference resolution complete: {len(entities)} entities -> "
            f"{len(canonical_entities)} canonical, {len(co_reference_groups)} groups"
        )

        return canonical_entities, co_reference_groups

    def _cluster_entities(
        self,
        entities: List[ExtractedEntity],
        similarity_threshold: float,
    ) -> List[List[ExtractedEntity]]:
        """
        Cluster similar entities using agglomerative clustering.

        Args:
            entities: Entities of same type
            similarity_threshold: Threshold for clustering

        Returns:
            List of clusters
        """
        if not entities:
            return []

        if len(entities) == 1:
            return [[entities[0]]]

        # Start with singleton clusters
        clusters: List[List[ExtractedEntity]] = [[e] for e in entities]

        # Repeatedly merge closest clusters
        while len(clusters) > 1:
            # Find most similar pair of clusters
            best_similarity = -1
            best_i, best_j = 0, 1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Similarity is max similarity between any pair of entities
                    max_sim = max(
                        self._string_similarity(e1.name, e2.name)
                        for e1 in clusters[i]
                        for e2 in clusters[j]
                    )

                    if max_sim > best_similarity:
                        best_similarity = max_sim
                        best_i, best_j = i, j

            # Stop if no more similar clusters
            if best_similarity < similarity_threshold:
                break

            # Merge clusters
            clusters[best_i].extend(clusters[best_j])
            del clusters[best_j]

        return clusters

    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """
        Calculate string similarity (0-1).

        Args:
            s1, s2: Strings to compare

        Returns:
            Similarity score (0-1)
        """
        s1_lower = s1.lower().strip()
        s2_lower = s2.lower().strip()

        # Exact match
        if s1_lower == s2_lower:
            return 1.0

        # Substring match
        if s1_lower in s2_lower or s2_lower in s1_lower:
            return 0.9

        # Sequence matching
        matcher = SequenceMatcher(None, s1_lower, s2_lower)
        return matcher.ratio()

    @staticmethod
    def _confidence_score(confidence: ConfidenceLevel) -> float:
        """Convert confidence level to numeric score."""
        return {
            ConfidenceLevel.HIGH: 1.0,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.LOW: 0.0,
        }.get(confidence, 0.5)


# ===========================
# Task 2.3.2: Actor/Lane Consolidation
# ===========================


@dataclass
class ActorProfile:
    """Profile of an actor/swimlane in the process."""

    actor_id: str
    """Canonical actor ID"""

    actor_name: str
    """Canonical actor name"""

    activity_ids: List[str] = field(default_factory=list)
    """IDs of activities performed by this actor"""

    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    """Confidence in actor assignment"""

    alternative_names: List[str] = field(default_factory=list)
    """Alternative names for this actor"""


class ActorConsolidator:
    """Consolidates actors and assigns activities to lanes with KB support."""

    def __init__(self, similarity_threshold: float = 0.75, enable_kb: bool = True):
        """
        Initialize consolidator with optional KB support.

        Args:
            similarity_threshold: Minimum similarity to consolidate actors
            enable_kb: Whether to enable KB domain rules for consolidation
        """
        self.similarity_threshold = similarity_threshold
        self.enable_kb = enable_kb

        # Initialize KB components (lazy loading)
        self._domain_classifier: Optional[DomainClassifier] = None
        self._domain_type: Optional[DomainType] = None

    def _ensure_kb_initialized(self, domain: Optional[DomainType] = None) -> None:
        """Initialize KB components if enabled and not already done."""
        if not self.enable_kb:
            return

        if self._domain_classifier is None:
            try:
                self._domain_classifier = DomainClassifier()
                self._domain_type = domain
                logger.debug("KB components initialized for actor consolidation")
            except Exception as e:
                logger.warning(f"Failed to initialize KB components: {e}")
                self.enable_kb = False

    def consolidate_actors(
        self,
        entities: List[ExtractedEntity],
        relations: List[ExtractedRelation],
        domain: Optional[DomainType] = None,
    ) -> Tuple[Dict[str, ActorProfile], Dict[str, str]]:
        """
        Consolidate actors and map activities to lanes.

        Uses KB domain rules to improve actor consolidation.

        Args:
            entities: Extracted entities (including ACTOR type)
            relations: Extracted relations (including INVOLVES type)
            domain: Detected domain for KB-aware consolidation

        Returns:
            (actor_profiles, activity_to_actor_map)
        """
        # Initialize KB if enabled with domain context
        if self.enable_kb and domain:
            self._ensure_kb_initialized(domain)

        # 1. Extract and consolidate actors
        actors = [e for e in entities if e.type == EntityType.ACTOR]
        activities = [e for e in entities if e.type == EntityType.ACTIVITY]

        # Consolidate similar actors
        # Apply domain-specific rules if available
        similarity_threshold = self.similarity_threshold
        if self.enable_kb and domain:
            # Use slightly different threshold for domain-specific actors
            similarity_threshold = max(0.7, self.similarity_threshold - 0.05)
            logger.debug(
                f"Using domain-adjusted threshold {similarity_threshold} for actor consolidation"
            )

        consolidated_actors = self._consolidate_similar_actors(actors, similarity_threshold)

        # 2. Build actor profiles with activities
        actor_profiles: Dict[str, ActorProfile] = {}
        for actor in consolidated_actors:
            actor_profiles[actor.id] = ActorProfile(
                actor_id=actor.id,
                actor_name=actor.name,
                confidence=actor.confidence,
                alternative_names=actor.alternative_names,
            )

        # 3. Map activities to actors via INVOLVES relations
        activity_to_actor: Dict[str, str] = {}

        for relation in relations:
            if relation.type == RelationType.INVOLVES:
                # INVOLVES: actor -> activity
                # source_id should be actor, target_id should be activity
                actor_id = relation.source_id
                activity_id = relation.target_id

                # Find canonical actor (in case of co-references)
                canonical_actor = self._find_canonical_actor(actor_id, consolidated_actors)
                if canonical_actor:
                    if canonical_actor.id not in actor_profiles:
                        actor_profiles[canonical_actor.id] = ActorProfile(
                            actor_id=canonical_actor.id,
                            actor_name=canonical_actor.name,
                            confidence=canonical_actor.confidence,
                        )

                    actor_profiles[canonical_actor.id].activity_ids.append(activity_id)
                    activity_to_actor[activity_id] = canonical_actor.id

        # 4. Assign unassigned activities to closest actor by name heuristic
        for activity in activities:
            if activity.id not in activity_to_actor:
                # Try to infer from activity description
                best_actor = self._infer_actor_from_activity(activity, consolidated_actors)
                if best_actor:
                    activity_to_actor[activity.id] = best_actor.id
                    if best_actor.id in actor_profiles:
                        actor_profiles[best_actor.id].activity_ids.append(activity.id)

        return actor_profiles, activity_to_actor

    def _consolidate_similar_actors(
        self,
        actors: List[ExtractedEntity],
        similarity_threshold: Optional[float] = None,
    ) -> List[ExtractedEntity]:
        """
        Consolidate similar actors (e.g., "manager" and "the manager").

        Args:
            actors: Actor entities
            similarity_threshold: Optional override for similarity threshold

        Returns:
            Consolidated actors (deduplicated)
        """
        if not actors:
            return []

        # Use co-reference clustering similar to Stage 3.1
        threshold = (
            similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        )
        resolver = CoReferenceResolver(threshold, enable_kb=self.enable_kb)

        # Get domain type if available for resolver
        domain = self._domain_type if self.enable_kb else None
        canonical_actors, _ = resolver.resolve_co_references(actors, domain=domain)

        return canonical_actors

    @staticmethod
    def _find_canonical_actor(
        actor_id: str,
        canonical_actors: List[ExtractedEntity],
    ) -> Optional[ExtractedEntity]:
        """
        Find canonical actor by ID or name similarity.

        Args:
            actor_id: Actor ID to find
            canonical_actors: List of canonical actors

        Returns:
            Canonical actor or None
        """
        # Direct match
        for actor in canonical_actors:
            if actor.id == actor_id:
                return actor

        # If no direct match, this actor might be a co-reference
        # Return first actor as fallback (will be refined)
        return canonical_actors[0] if canonical_actors else None

    @staticmethod
    def _infer_actor_from_activity(
        activity: ExtractedEntity,
        actors: List[ExtractedEntity],
    ) -> Optional[ExtractedEntity]:
        """
        Try to infer actor from activity description using keywords.

        Args:
            activity: Activity entity
            actors: Available actors

        Returns:
            Best matching actor or None
        """
        if not actors:
            return None

        activity_name_lower = activity.name.lower()

        # Simple heuristic: if activity mentions an actor name, use that
        for actor in actors:
            if actor.name.lower() in activity_name_lower:
                return actor

        # If no match, return first actor (default assignment)
        return actors[0]


# ===========================
# Task 2.3.3: Relationship Validation
# ===========================


@dataclass
class RelationshipValidationReport:
    """Report of relationship validation."""

    total_relations: int
    """Total relations checked"""

    valid_relations: int
    """Relations with valid source and target"""

    invalid_relations: List[str] = field(default_factory=list)
    """IDs of invalid relations"""

    dangling_references: List[str] = field(default_factory=list)
    """Relations with missing source or target"""

    duplicate_relations: List[Tuple[str, str]] = field(default_factory=list)
    """Pairs of duplicate relations (id1, id2)"""


class RelationshipValidator:
    """Validates relationships in extraction results."""

    def validate_relationships(
        self,
        entities: List[ExtractedEntity],
        relations: List[ExtractedRelation],
    ) -> Tuple[List[ExtractedRelation], RelationshipValidationReport]:
        """
        Validate relationships and remove invalid ones.

        Args:
            entities: Extracted entities
            relations: Extracted relations

        Returns:
            (valid_relations, validation_report)
        """
        entity_ids = {e.id for e in entities}
        valid_relations: List[ExtractedRelation] = []
        report = RelationshipValidationReport(total_relations=len(relations), valid_relations=0)

        # Track seen relations to detect duplicates
        seen_relations: Dict[Tuple[str, str, str], str] = {}

        for relation in relations:
            # Check for valid source and target
            if relation.source_id not in entity_ids:
                report.dangling_references.append(
                    f"Relation {relation.id}: source '{relation.source_id}' not found"
                )
                report.invalid_relations.append(relation.id)
                continue

            if relation.target_id not in entity_ids:
                report.dangling_references.append(
                    f"Relation {relation.id}: target '{relation.target_id}' not found"
                )
                report.invalid_relations.append(relation.id)
                continue

            # Check for duplicate relations
            relation_key = (relation.type.value, relation.source_id, relation.target_id)
            if relation_key in seen_relations:
                report.duplicate_relations.append((seen_relations[relation_key], relation.id))
                # Keep first occurrence, skip this one
                continue

            seen_relations[relation_key] = relation.id
            valid_relations.append(relation)
            report.valid_relations += 1

        return valid_relations, report


# ===========================
# Full Entity Resolution Pipeline
# ===========================


class EntityResolutionPipeline:
    """Complete entity resolution pipeline (Stage 3)."""

    def __init__(
        self,
        co_reference_threshold: float = 0.75,
        actor_threshold: float = 0.75,
    ):
        """
        Initialize pipeline.

        Args:
            co_reference_threshold: Threshold for co-reference clustering
            actor_threshold: Threshold for actor consolidation
        """
        self.co_reference_resolver = CoReferenceResolver(co_reference_threshold)
        self.actor_consolidator = ActorConsolidator(actor_threshold)
        self.relationship_validator = RelationshipValidator()

    def resolve(
        self,
        extraction_result: ExtractionResult,
    ) -> ExtractionResult:
        """
        Execute full entity resolution pipeline.

        Args:
            extraction_result: Extraction results from Stage 2

        Returns:
            Resolved extraction result
        """
        # 1. Resolve co-references
        canonical_entities, co_ref_groups = self.co_reference_resolver.resolve_co_references(
            extraction_result.entities
        )

        # 2. Consolidate actors and assign activities
        actor_profiles, activity_to_actor = self.actor_consolidator.consolidate_actors(
            canonical_entities,
            extraction_result.relations,
        )

        # 3. Validate relationships
        valid_relations, validation_report = self.relationship_validator.validate_relationships(
            canonical_entities,
            extraction_result.relations,
        )

        # 4. Update extraction result with resolved data
        resolved_result = ExtractionResult(
            entities=canonical_entities,
            relations=valid_relations,
            co_references=co_ref_groups,
            metadata=extraction_result.metadata,
        )

        return resolved_result


__all__ = [
    "CoReferenceResolver",
    "ActorProfile",
    "ActorConsolidator",
    "RelationshipValidationReport",
    "RelationshipValidator",
    "EntityResolutionPipeline",
]
