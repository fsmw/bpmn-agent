"""
Extraction Stage Output Schemas

Defines the output data structures from the extraction pipeline stage,
which identifies and extracts process entities and relationships from
natural language input.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Confidence levels for extracted entities."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EntityType(str, Enum):
    """Types of extracted entities."""
    ACTIVITY = "activity"  # Task, Service, User action
    EVENT = "event"  # Start, end, intermediate
    GATEWAY = "gateway"  # Decision point
    ACTOR = "actor"  # Role, participant, swimlane
    DATA = "data"  # Data object, input/output
    CONSTRAINT = "constraint"  # Business rule, condition
    RESOURCE = "resource"  # System, application
    ORGANIZATION = "organization"  # Company, department


class RelationType(str, Enum):
    """Types of relationships between entities."""
    TRIGGERS = "triggers"  # Event triggers activity
    PRECEDES = "precedes"  # Sequence: A -> B
    FOLLOWS = "follows"  # Reverse sequence: A <- B
    ALTERNATIVE_TO = "alternative_to"  # XOR branch
    PARALLEL_TO = "parallel_to"  # AND branch
    SENDS_TO = "sends_to"  # Message flow
    RECEIVES_FROM = "receives_from"  # Receives message
    INVOLVES = "involves"  # Actor participates
    USES = "uses"  # Task uses resource/data
    PRODUCES = "produces"  # Task produces data
    CONSUMES = "consumes"  # Task consumes data
    CONDITIONAL = "conditional"  # Guard condition


class EntityAttribute(BaseModel):
    """Attributes of an extracted entity."""
    
    key: str = Field(..., description="Attribute key/name")
    value: Any = Field(..., description="Attribute value")
    value_type: str = Field(
        default="string",
        description="Value type: string, number, boolean, list, etc."
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Confidence in this attribute"
    )
    source_text: Optional[str] = Field(
        None,
        description="Source text snippet where this was extracted"
    )


class ExtractedEntity(BaseModel):
    """A single extracted entity from text."""
    
    id: str = Field(..., description="Unique entity identifier")
    type: EntityType = Field(..., description="Entity classification")
    name: str = Field(..., description="Entity name/label")
    description: Optional[str] = Field(None, description="Entity description")
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Confidence in entity extraction"
    )
    
    # Source information
    source_text: Optional[str] = Field(
        None,
        description="Original text snippet from input"
    )
    character_offsets: Optional[tuple[int, int]] = Field(
        None,
        description="Start and end character positions in input"
    )
    
    # Entity-specific attributes
    attributes: Dict[str, EntityAttribute] = Field(
        default_factory=dict,
        description="Entity-specific attributes"
    )
    
    # Metadata
    is_implicit: bool = Field(
        False,
        description="Whether entity was inferred vs. explicitly mentioned"
    )
    is_uncertain: bool = Field(
        False,
        description="Whether extraction is uncertain"
    )
    alternative_names: List[str] = Field(
        default_factory=list,
        description="Alternate names/aliases for this entity"
    )


class ExtractedRelation(BaseModel):
    """A relationship between two extracted entities."""
    
    id: str = Field(..., description="Unique relation identifier")
    type: RelationType = Field(..., description="Relationship type")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    label: Optional[str] = Field(None, description="Human-readable relationship label")
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Confidence in relationship extraction"
    )
    
    # Source information
    source_text: Optional[str] = Field(
        None,
        description="Original text snippet describing this relationship"
    )
    
    # Relationship attributes
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Relationship-specific attributes (e.g., condition expression)"
    )
    
    # Metadata
    is_implicit: bool = Field(
        False,
        description="Whether relationship was inferred vs. explicitly stated"
    )
    is_conditional: bool = Field(
        False,
        description="Whether this relationship is conditional"
    )
    condition_expression: Optional[str] = Field(
        None,
        description="Guard condition if conditional"
    )


class CoReferenceGroup(BaseModel):
    """
    Group of entity mentions that refer to the same real-world entity
    (co-reference resolution).
    """
    
    canonical_id: str = Field(
        ...,
        description="ID of canonical entity representation"
    )
    canonical_form: str = Field(
        ...,
        description="Preferred/canonical form of the entity name"
    )
    mentions: List[str] = Field(
        default_factory=list,
        description="IDs of extracted entities that refer to this canonical form"
    )
    mention_texts: List[str] = Field(
        default_factory=list,
        description="Original text forms of mentions"
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Confidence in co-reference resolution"
    )


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""
    
    input_text: str = Field(..., description="Original input text")
    input_length: int = Field(..., description="Input text length in characters")
    extraction_timestamp: str = Field(..., description="ISO 8601 timestamp of extraction")
    extraction_duration_ms: float = Field(..., description="Time taken for extraction (ms)")
    llm_model: str = Field(..., description="LLM model used")
    llm_temperature: float = Field(..., description="LLM temperature setting")
    stage: str = Field(default="extraction", description="Pipeline stage identifier")
    
    # Quality metrics
    total_entities_extracted: int = Field(default=0, description="Total entities found")
    high_confidence_entities: int = Field(default=0, description="High-confidence entities")
    medium_confidence_entities: int = Field(default=0, description="Medium-confidence entities")
    low_confidence_entities: int = Field(default=0, description="Low-confidence entities")
    
    total_relations_extracted: int = Field(default=0, description="Total relations found")
    high_confidence_relations: int = Field(default=0, description="High-confidence relations")
    
    co_reference_groups: int = Field(default=0, description="Co-reference groups formed")
    
    # Warnings and notes
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings during extraction (ambiguities, truncations, etc.)"
    )
    notes: Optional[str] = Field(None, description="General extraction notes")


class ExtractionResult(BaseModel):
    """Complete extraction stage output."""
    
    # Core extracted elements
    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="All extracted entities"
    )
    relations: List[ExtractedRelation] = Field(
        default_factory=list,
        description="All extracted relationships"
    )
    co_references: List[CoReferenceGroup] = Field(
        default_factory=list,
        description="Co-reference groups (entity disambiguation)"
    )
    
    # Metadata
    metadata: ExtractionMetadata = Field(..., description="Extraction metadata")
    
    # Statistics
    @property
    def entity_count(self) -> int:
        """Total extracted entities."""
        return len(self.entities)
    
    @property
    def relation_count(self) -> int:
        """Total extracted relations."""
        return len(self.relations)
    
    @property
    def average_entity_confidence(self) -> float:
        """Average confidence of entities."""
        if not self.entities:
            return 0.0
        
        confidence_map = {
            ConfidenceLevel.HIGH: 1.0,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.LOW: 0.0,
        }
        
        total = sum(confidence_map.get(e.confidence, 0.5) for e in self.entities)
        return total / len(self.entities)
    
    @property
    def entity_types_distribution(self) -> Dict[EntityType, int]:
        """Distribution of entity types."""
        distribution: Dict[EntityType, int] = {}
        for entity in self.entities:
            distribution[entity.type] = distribution.get(entity.type, 0) + 1
        return distribution
    
    @property
    def relation_types_distribution(self) -> Dict[RelationType, int]:
        """Distribution of relation types."""
        distribution: Dict[RelationType, int] = {}
        for relation in self.relations:
            distribution[relation.type] = distribution.get(relation.type, 0) + 1
        return distribution
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[ExtractedEntity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities if e.type == entity_type]
    
    def get_relations_for_entity(self, entity_id: str) -> tuple[List[ExtractedRelation], List[ExtractedRelation]]:
        """Get outgoing and incoming relations for an entity."""
        outgoing = [r for r in self.relations if r.source_id == entity_id]
        incoming = [r for r in self.relations if r.target_id == entity_id]
        return outgoing, incoming
    
    def get_entity_by_id(self, entity_id: str) -> Optional[ExtractedEntity]:
        """Get entity by ID."""
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None
    
    def resolve_co_references(self, entity_id: str) -> str:
        """
        Resolve co-reference group for an entity ID.
        
        Returns the canonical form (preferred name) of the entity.
        If the entity is not in any co-reference group, returns the entity ID itself.
        
        Args:
            entity_id: ID of the entity to resolve
            
        Returns:
            Canonical form (ID) of the entity
        """
        for group in self.co_references:
            if entity_id in group.mentions or entity_id == group.canonical_id:
                return group.canonical_id
        return entity_id


class ExtractionError(BaseModel):
    """Error during extraction stage."""
    
    error_type: str = Field(..., description="Error classification")
    message: str = Field(..., description="Error message")
    severity: str = Field(
        default="error",
        description="Severity level: error, warning, info"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Contextual information about the error"
    )
    recoverable: bool = Field(
        True,
        description="Whether pipeline can continue after this error"
    )


class ExtractionResultWithErrors(ExtractionResult):
    """Extraction result with error tracking."""
    
    errors: List[ExtractionError] = Field(
        default_factory=list,
        description="Errors encountered during extraction"
    )
    
    @property
    def has_errors(self) -> bool:
        """Whether extraction encountered errors."""
        return len(self.errors) > 0
    
    @property
    def has_fatal_errors(self) -> bool:
        """Whether there are non-recoverable errors."""
        return any(not e.recoverable for e in self.errors)


__all__ = [
    "ConfidenceLevel",
    "EntityType",
    "RelationType",
    "EntityAttribute",
    "ExtractedEntity",
    "ExtractedRelation",
    "CoReferenceGroup",
    "ExtractionMetadata",
    "ExtractionResult",
    "ExtractionError",
    "ExtractionResultWithErrors",
]
