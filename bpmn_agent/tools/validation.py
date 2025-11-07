"""
Validation Tools for BPMN Agent

Provides comprehensive validation capabilities:
- XML XSD schema validation
- Graph semantic validation
- Extraction quality assessment
- Compliance checking
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from bpmn_agent.models.extraction import (
    ExtractionResultWithErrors,
)
from bpmn_agent.models.graph import ProcessGraph

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(str, Enum):
    """Categories of validation issues."""

    XML_STRUCTURE = "xml_structure"
    XML_COMPLIANCE = "xml_compliance"
    GRAPH_CONNECTIVITY = "graph_connectivity"
    GRAPH_SEMANTICS = "graph_semantics"
    EXTRACTION_QUALITY = "extraction_quality"
    BUSINESS_LOGIC = "business_logic"
    BPMN_SPEC = "bpmn_spec"


@dataclass
class ValidationIssue:
    """A validation issue found during validation."""

    level: ValidationLevel
    category: ValidationCategory
    message: str
    element_id: Optional[str] = None
    element_type: Optional[str] = None
    suggestion: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    location: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result."""

    is_valid: bool
    total_issues: int
    issues_by_level: Dict[ValidationLevel, int] = field(default_factory=dict)
    issues_by_category: Dict[ValidationCategory, int] = field(default_factory=dict)
    issues: List[ValidationIssue] = field(default_factory=list)
    overall_score: float = 0.0  # 0-100 quality score
    metrics: Dict[str, Any] = field(default_factory=dict)


class XMLValidator:
    """XML validation against BPMN XSD schema."""

    BPMN_XSD_PATH = Path(__file__).parent.parent / "knowledge" / "schemas" / "BPMN20.xsd"

    def __init__(self):
        """Initialize XML validator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.xsd_schema = None
        self._load_xsd_schema()

    def _load_xsd_schema(self) -> None:
        """Load BPMN XSD schema for validation."""
        try:
            if self.BPMN_XSD_PATH.exists():
                import lxml.etree as etree

                with open(self.BPMN_XSD_PATH, "rb") as schema_file:
                    schema_doc = etree.parse(schema_file)
                    self.xsd_schema = etree.XMLSchema(schema_doc)
                self.logger.info("BPMN XSD schema loaded successfully")
            else:
                self.logger.warning(f"BPMN XSD schema not found at {self.BPMN_XSD_PATH}")
        except Exception as e:
            self.logger.error(f"Failed to load BPMN XSD schema: {e}")

    def validate_xml_against_xsd(self, xml_content: str) -> ValidationResult:
        """Validate XML content against BPMN XSD schema.

        Args:
            xml_content: XML string to validate

        Returns:
            ValidationResult with XSD validation issues
        """
        result = ValidationResult(is_valid=True, total_issues=0)

        if not self.xsd_schema:
            # Fallback to basic structure validation
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.XML_COMPLIANCE,
                    message="XSD schema not available, performing basic validation only",
                    suggestion="Install BPMN XSD schema for comprehensive validation",
                )
            )
            result.total_issues = 1
            result.issues_by_level[ValidationLevel.WARNING] = 1
            return self._validate_xml_structure(xml_content, result)

        try:
            import lxml.etree as etree

            xml_doc = etree.fromstring(xml_content.encode("utf-8"))

            # Validate against XSD
            if self.xsd_schema.validate(xml_doc):
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.INFO,
                        category=ValidationCategory.XML_COMPLIANCE,
                        message="XML is valid against BPMN 2.0 XSD schema",
                    )
                )
                result.issues_by_level[ValidationLevel.INFO] = 1
            else:
                result.is_valid = False
                error_log = self.xsd_schema.error_log
                for error in error_log:
                    level = (
                        ValidationLevel.ERROR
                        if error.level_name == "ERROR"
                        else ValidationLevel.WARNING
                    )
                    result.issues.append(
                        ValidationIssue(
                            level=level,
                            category=ValidationCategory.XML_COMPLIANCE,
                            message=str(error.message),
                            line_number=error.line,
                            column_number=error.column,
                            suggestion="Fix XML structure to comply with BPMN 2.0 specification",
                        )
                    )
                    result.issues_by_level[level] = result.issues_by_level.get(level, 0) + 1

        except Exception as e:
            result.is_valid = False
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.XML_STRUCTURE,
                    message=f"XML parsing error: {str(e)}",
                    suggestion="Ensure well-formed XML structure",
                )
            )
            result.issues_by_level[ValidationLevel.ERROR] = (
                result.issues_by_level.get(ValidationLevel.ERROR, 0) + 1
            )

        result.total_issues = sum(result.issues_by_level.values())
        result.overall_score = self._calculate_validation_score(result)

        return result

    def _validate_xml_structure(
        self, xml_content: str, result: ValidationResult
    ) -> ValidationResult:
        """Fallback basic XML structure validation."""
        try:
            # Try parsing with ElementTree
            root = ET.fromstring(xml_content)

            # Basic BPMN structure checks
            if root.tag != "definitions":
                result.is_valid = False
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        category=ValidationCategory.XML_STRUCTURE,
                        message="Root element must be 'definitions'",
                        suggestion="Change root element to 'definitions' namespace",
                    )
                )
                result.issues_by_level[ValidationLevel.ERROR] = (
                    result.issues_by_level.get(ValidationLevel.ERROR, 0) + 1
                )

            # Check for process element
            process_elements = root.findall(".//process") or root.findall(".//{*}process")
            if not process_elements:
                result.is_valid = False
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        category=ValidationCategory.XML_STRUCTURE,
                        message="No 'process' element found",
                        suggestion="Add at least one 'process' element to contain BPMN elements",
                    )
                )
                result.issues_by_level[ValidationLevel.ERROR] = (
                    result.issues_by_level.get(ValidationLevel.ERROR, 0) + 1
                )

            # Check for start events
            start_events = root.findall(".//startEvent") or root.findall(".//{*}startEvent")
            if not start_events:
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category=ValidationCategory.BPMN_SPEC,
                        message="No start events found",
                        suggestion="Add at least one start event to make process executable",
                    )
                )
                result.issues_by_level[ValidationLevel.WARNING] = (
                    result.issues_by_level.get(ValidationLevel.WARNING, 0) + 1
                )

            # Check for end events
            end_events = root.findall(".//endEvent") or root.findall(".//{*}endEvent")
            if not end_events:
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category=ValidationCategory.BPMN_SPEC,
                        message="No end events found",
                        suggestion="Add at least one end event for process termination",
                    )
                )
                result.issues_by_level[ValidationLevel.WARNING] = (
                    result.issues_by_level.get(ValidationLevel.WARNING, 0) + 1
                )

        except ET.ParseError as e:
            result.is_valid = False
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.XML_STRUCTURE,
                    message=f"XML well-formedness error: {str(e)}",
                    line_number=e.lineno if hasattr(e, "lineno") else None,
                    suggestion="Fix XML syntax to ensure well-formedness",
                )
            )
            result.issues_by_level[ValidationLevel.ERROR] = (
                result.issues_by_level.get(ValidationLevel.ERROR, 0) + 1
            )

        result.total_issues = sum(result.issues_by_level.values())
        result.overall_score = self._calculate_validation_score(result)

        return result

    def _calculate_validation_score(self, result: ValidationResult) -> float:
        """Calculate validation quality score (0-100)."""
        if result.is_valid and not result.issues:
            return 100.0

        score = 100.0
        score -= result.issues_by_level.get(ValidationLevel.CRITICAL, 0) * 25
        score -= result.issues_by_level.get(ValidationLevel.ERROR, 0) * 15
        score -= result.issues_by_level.get(ValidationLevel.WARNING, 0) * 5
        score -= result.issues_by_level.get(ValidationLevel.INFO, 0) * 1

        return max(0.0, score)


class GraphValidator:
    """Graph semantic validation."""

    def __init__(self):
        """Initialize graph validator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def validate_graph_semantics(
        self, graph: ProcessGraph, extraction_result: Optional[ExtractionResultWithErrors] = None
    ) -> ValidationResult:
        """Validate graph semantic correctness.

        Args:
            graph: Process graph to validate
            extraction_result: Original extraction for context

        Returns:
            ValidationResult with semantic issues
        """
        result = ValidationResult(is_valid=True, total_issues=0)

        # Validate connectivity
        self._validate_connectivity(graph, result)

        # Validate gateways
        self._validate_gateways(graph, result)

        # Validate events
        self._validate_events(graph, result)

        # Validate swimlanes
        self._validate_swimlanes(graph, result)

        # Validate business logic
        if extraction_result:
            self._validate_business_logic(graph, extraction_result, result)

        result.total_issues = sum(result.issues_by_level.values())
        result.overall_score = self._calculate_graph_score(result)
        result.is_valid = (
            result.issues_by_level.get(ValidationLevel.ERROR, 0) == 0
            and result.issues_by_level.get(ValidationLevel.CRITICAL, 0) == 0
        )

        return result

    def _validate_connectivity(self, graph: ProcessGraph, result: ValidationResult) -> None:
        """Validate graph connectivity."""
        # Check for disconnected components
        if graph.nodes and not graph.edges:
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.GRAPH_CONNECTIVITY,
                    message="Graph has nodes but no edges",
                    suggestion="Add sequence flows to connect nodes",
                )
            )
            result.issues_by_level[ValidationLevel.WARNING] = (
                result.issues_by_level.get(ValidationLevel.WARNING, 0) + 1
            )
            return

        # Check for start events
        start_nodes = [n for n in graph.nodes if n.type.lower() == "start"]
        if not start_nodes:
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.GRAPH_CONNECTIVITY,
                    message="No start events found in graph",
                    suggestion="Add at least one start event",
                )
            )
            result.issues_by_level[ValidationLevel.WARNING] = (
                result.issues_by_level.get(ValidationLevel.WARNING, 0) + 1
            )

        # Check connectivity from start to other nodes
        all_node_ids = {n.id for n in graph.nodes}
        reachable_from_start = set()

        for start_node in start_nodes:
            reachable = self._get_reachable_nodes(graph, start_node.id)
            reachable_from_start.update(reachable)

        unreachable = all_node_ids - reachable_from_start
        if unreachable:
            for node_id in unreachable:
                node = next((n for n in graph.nodes if n.id == node_id), None)
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category=ValidationCategory.GRAPH_CONNECTIVITY,
                        message=f"Node {node_id} (type: {node.type if node else 'unknown'}) is not reachable from any start event",
                        element_id=node_id,
                        element_type=node.type if node else "unknown",
                        suggestion="Add path from start event to this node",
                    )
                )
            result.issues_by_level[ValidationLevel.WARNING] = result.issues_by_level.get(
                ValidationLevel.WARNING, 0
            ) + len(unreachable)

    def _validate_gateways(self, graph: ProcessGraph, result: ValidationResult) -> None:
        """Validate gateway usage."""
        gateways = [n for n in graph.nodes if "gateway" in n.type.lower()]

        for gateway in gateways:
            outgoing_count = sum(1 for e in graph.edges if e.source_id == gateway.id)
            incoming_count = sum(1 for e in graph.edges if e.target_id == gateway.id)

            # Check gateway connections
            if outgoing_count == 0 and incoming_count == 0:
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        category=ValidationCategory.GRAPH_SEMANTICS,
                        message=f"Gateway {gateway.id} has no connections",
                        element_id=gateway.id,
                        element_type=gateway.type,
                        suggestion="Connect gateway to process flow or remove unused gateway",
                    )
                )
                result.issues_by_level[ValidationLevel.ERROR] = (
                    result.issues_by_level.get(ValidationLevel.ERROR, 0) + 1
                )

            # Check split gateway (multiple outgoing)
            if outgoing_count > 1:
                # Parallel gateway should have matching join
                if gateway.type.lower() in ["parallel_gateway", "and_gateway"]:
                    # Look for matching parallel join
                    matching_join = self._find_matching_gateway_join(
                        graph, gateway.id, gateway.type
                    )
                    if not matching_join:
                        result.issues.append(
                            ValidationIssue(
                                level=ValidationLevel.WARNING,
                                category=ValidationCategory.GRAPH_SEMANTICS,
                                message=f"Parallel gateway {gateway.id} splits but no matching join found",
                                element_id=gateway.id,
                                element_type=gateway.type,
                                suggestion="Add parallel gateway to join parallel flows",
                            )
                        )
                        result.issues_by_level[ValidationLevel.WARNING] = (
                            result.issues_by_level.get(ValidationLevel.WARNING, 0) + 1
                        )

            # Check join gateway (multiple incoming)
            if incoming_count > 1:
                if gateway.type.lower() in ["exclusive_gateway", "xor_gateway"]:
                    result.issues.append(
                        ValidationIssue(
                            level=ValidationLevel.WARNING,
                            category=ValidationCategory.GRAPH_SEMANTICS,
                            message=f"XOR gateway {gateway.id} has multiple incoming edges (unusual pattern)",
                            element_id=gateway.id,
                            element_type=gateway.type,
                            suggestion="Consider using parallel gateway if flows are concurrent",
                        )
                    )
                    result.issues_by_level[ValidationLevel.WARNING] = (
                        result.issues_by_level.get(ValidationLevel.WARNING, 0) + 1
                    )

    def _validate_events(self, graph: ProcessGraph, result: ValidationResult) -> None:
        """Validate event usage."""
        events = [n for n in graph.nodes if "event" in n.type.lower()]

        for event in events:
            outgoing_count = sum(1 for e in graph.edges if e.source_id == event.id)
            incoming_count = sum(1 for e in graph.edges if e.target_id == event.id)

            # Start event validation
            if event.type.lower() == "start":
                if incoming_count > 0:
                    result.issues.append(
                        ValidationIssue(
                            level=ValidationLevel.ERROR,
                            category=ValidationCategory.BPMN_SPEC,
                            message=f"Start event {event.id} should have no incoming edges",
                            element_id=event.id,
                            element_type=event.type,
                            suggestion="Remove incoming edges from start event",
                        )
                    )
                    result.issues_by_level[ValidationLevel.ERROR] = (
                        result.issues_by_level.get(ValidationLevel.ERROR, 0) + 1
                    )

                if outgoing_count == 0:
                    result.issues.append(
                        ValidationIssue(
                            level=ValidationLevel.ERROR,
                            category=ValidationCategory.GRAPH_CONNECTIVITY,
                            message=f"Start event {event.id} has no outgoing edges",
                            element_id=event.id,
                            element_type=event.type,
                            suggestion="Add outgoing edges from start event",
                        )
                    )
                    result.issues_by_level[ValidationLevel.ERROR] = (
                        result.issues_by_level.get(ValidationLevel.ERROR, 0) + 1
                    )

            # End event validation
            elif event.type.lower() == "end":
                if outgoing_count > 0:
                    result.issues.append(
                        ValidationIssue(
                            level=ValidationLevel.ERROR,
                            category=ValidationCategory.BPMN_SPEC,
                            message=f"End event {event.id} should have no outgoing edges",
                            element_id=event.id,
                            element_type=event.type,
                            suggestion="Remove outgoing edges from end event",
                        )
                    )
                    result.issues_by_level[ValidationLevel.ERROR] = (
                        result.issues_by_level.get(ValidationLevel.ERROR, 0) + 1
                    )

                if incoming_count == 0:
                    result.issues.append(
                        ValidationIssue(
                            level=ValidationLevel.WARNING,
                            category=ValidationCategory.GRAPH_CONNECTIVITY,
                            message=f"End event {event.id} has no incoming edges (orphaned)",
                            element_id=event.id,
                            element_type=event.type,
                            suggestion="Add incoming edges to connect to process flow",
                        )
                    )
                    result.issues_by_level[ValidationLevel.WARNING] = (
                        result.issues_by_level.get(ValidationLevel.WARNING, 0) + 1
                    )

    def _validate_swimlanes(self, graph: ProcessGraph, result: ValidationResult) -> None:
        """Validate swimlane assignments."""
        # Check if nodes have lane assignments
        nodes_with_lanes = [n for n in graph.nodes if hasattr(n, "lane_id") and n.lane_id]
        nodes_without_lanes = [n for n in graph.nodes if not (hasattr(n, "lane_id") and n.lane_id)]

        if nodes_with_lanes and nodes_without_lanes:
            # Mixed assignment - some nodes have lanes, others don't
            for node in nodes_without_lanes[:5]:  # Limit to first 5
                result.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.INFO,
                        category=ValidationCategory.BPMN_SPEC,
                        message=f"Node {node.id} is not assigned to any lane",
                        element_id=node.id,
                        element_type=node.type,
                        suggestion="Consider assigning node to appropriate lane for clarity",
                    )
                )
            result.issues_by_level[ValidationLevel.INFO] = result.issues_by_level.get(
                ValidationLevel.INFO, 0
            ) + min(5, len(nodes_without_lanes))

    def _validate_business_logic(
        self,
        graph: ProcessGraph,
        extraction_result: ExtractionResultWithErrors,
        result: ValidationResult,
    ) -> None:
        """Validate business logic consistency."""
        # Check for missing transitions
        entities = extraction_result.entities
        relations = extraction_result.relations

        # Build entity lookup by ID and by name+type
        entity_lookup_by_id = {e.id: e for e in entities}
        entity_lookup = {f"{e.name}_{e.type}": e for e in entities}

        missing_entities = []
        for relation in relations:
            source_entity = entity_lookup_by_id.get(relation.source_id)
            target_entity = entity_lookup_by_id.get(relation.target_id)
            
            if source_entity:
                source_key = f"{source_entity.name}_{source_entity.type}"
                if source_key not in entity_lookup:
                    missing_entities.append(source_entity.name)
            else:
                missing_entities.append(f"entity_{relation.source_id}")
                
            if target_entity:
                target_key = f"{target_entity.name}_{target_entity.type}"
                if target_key not in entity_lookup:
                    missing_entities.append(target_entity.name)
            else:
                missing_entities.append(f"entity_{relation.target_id}")

        if missing_entities:
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.BUSINESS_LOGIC,
                    message=f"Relations reference {len(set(missing_entities))} entities not found in extraction: {', '.join(set(missing_entities))}",
                    suggestion="Review extraction to ensure all referenced entities are captured",
                )
            )
            result.issues_by_level[ValidationLevel.WARNING] = (
                result.issues_by_level.get(ValidationLevel.WARNING, 0) + 1
            )

    def _get_reachable_nodes(self, graph: ProcessGraph, start_id: str) -> Set[str]:
        """Get all nodes reachable from start node."""
        reachable = set()
        to_visit = [start_id]

        adjacency: Dict[str, List[str]] = self._build_adjacency(graph)

        while to_visit:
            current = to_visit.pop()
            if current not in reachable:
                reachable.add(current)
                to_visit.extend(adjacency.get(current, []))

        return reachable

    def _build_adjacency(self, graph: ProcessGraph) -> Dict[str, List[str]]:
        """Build adjacency list from graph edges."""
        adjacency: Dict[str, List[str]] = {}
        for edge in graph.edges:
            if edge.source_id not in adjacency:
                adjacency[edge.source_id] = []
            adjacency[edge.source_id].append(edge.target_id)
        return adjacency

    def _find_matching_gateway_join(
        self, graph: ProcessGraph, split_gateway_id: str, gateway_type: str
    ) -> Optional[str]:
        """Find matching gateway join for a split."""
        # This is a simplified implementation
        # In practice, would need more sophisticated matching logic
        gateways = [n for n in graph.nodes if gateway_type.lower() in n.type.lower()]
        for gateway in gateways:
            if gateway.id != split_gateway_id:
                incoming_count = sum(1 for e in graph.edges if e.target_id == gateway.id)
                if incoming_count > 1:
                    return gateway.id
        return None

    def _calculate_graph_score(self, result: ValidationResult) -> float:
        """Calculate graph validation quality score (0-100)."""
        score = 100.0
        score -= result.issues_by_level.get(ValidationLevel.CRITICAL, 0) * 20
        score -= result.issues_by_level.get(ValidationLevel.ERROR, 0) * 10
        score -= result.issues_by_level.get(ValidationLevel.WARNING, 0) * 3
        score -= result.issues_by_level.get(ValidationLevel.INFO, 0) * 1

        return max(0.0, score)


class ExtractionValidator:
    """Extraction quality validation."""

    def __init__(self):
        """Initialize extraction validator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def validate_extraction(
        self, extraction_result: ExtractionResultWithErrors
    ) -> ValidationResult:
        """Validate extraction quality and completeness.

        Args:
            extraction_result: Extraction result to validate

        Returns:
            ValidationResult with extraction quality issues
        """
        result = ValidationResult(is_valid=True, total_issues=0)

        # Validate entities
        self._validate_entities(extraction_result, result)

        # Validate relations
        self._validate_relations(extraction_result, result)

        # Validate metadata
        self._validate_metadata(extraction_result, result)

        result.total_issues = sum(result.issues_by_level.values())
        result.overall_score = self._calculate_extraction_score(result, extraction_result)

        return result

    def _validate_entities(
        self, extraction_result: ExtractionResultWithErrors, result: ValidationResult
    ) -> None:
        """Validate extracted entities."""
        entities = extraction_result.entities

        if not entities:
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.EXTRACTION_QUALITY,
                    message="No entities extracted from text",
                    suggestion="Review input text and extraction prompts",
                )
            )
            result.issues_by_level[ValidationLevel.WARNING] = (
                result.issues_by_level.get(ValidationLevel.WARNING, 0) + 1
            )
            return

        # Check confidence levels
        low_confidence_entities = [e for e in entities if e.confidence == "low"]
        if len(low_confidence_entities) > len(entities) * 0.5:
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.EXTRACTION_QUALITY,
                    message=f"High proportion of low confidence entities: {len(low_confidence_entities)}/{len(entities)}",
                    suggestion="Review extraction quality and consider refining prompts",
                )
            )
            result.issues_by_level[ValidationLevel.WARNING] = (
                result.issues_by_level.get(ValidationLevel.WARNING, 0) + 1
            )

        # Check for duplicates
        entity_names = [f"{e.name}_{e.type}" for e in entities]
        duplicate_count = len(entity_names) - len(set(entity_names))
        if duplicate_count > 0:
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.INFO,
                    category=ValidationCategory.EXTRACTION_QUALITY,
                    message=f"Found {duplicate_count} duplicate entities",
                    suggestion="Consider entity resolution to handle duplicates",
                )
            )
            result.issues_by_level[ValidationLevel.INFO] = (
                result.issues_by_level.get(ValidationLevel.INFO, 0) + 1
            )

    def _validate_relations(
        self, extraction_result: ExtractionResultWithErrors, result: ValidationResult
    ) -> None:
        """Validate extracted relations."""
        relations = extraction_result.relations
        entities = extraction_result.entities

        if not relations and len(entities) > 1:
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.EXTRACTION_QUALITY,
                    message=f"Found {len(entities)} entities but no relations between them",
                    suggestion="Check if relationships between entities are being captured",
                )
            )
            result.issues_by_level[ValidationLevel.WARNING] = (
                result.issues_by_level.get(ValidationLevel.WARNING, 0) + 1
            )

        # Check relation validity against entities
        entity_lookup_by_id = {e.id: e for e in entities}
        entity_names = {f"{e.name}_{e.type}": e for e in entities}
        missing_references = []

        for relation in relations:
            source_entity = entity_lookup_by_id.get(relation.source_id)
            target_entity = entity_lookup_by_id.get(relation.target_id)
            
            if source_entity:
                source_key = f"{source_entity.name}_{source_entity.type}"
                if source_key not in entity_names:
                    missing_references.append(source_entity.name)
            else:
                missing_references.append(f"entity_{relation.source_id}")
                
            if target_entity:
                target_key = f"{target_entity.name}_{target_entity.type}"
                if target_key not in entity_names:
                    missing_references.append(target_entity.name)
            else:
                missing_references.append(f"entity_{relation.target_id}")

        if missing_references:
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.EXTRACTION_QUALITY,
                    message=f"Relations reference missing entities: {', '.join(set(missing_references))}",
                    suggestion="Ensure all referenced entities are extracted",
                )
            )
            result.issues_by_level[ValidationLevel.ERROR] = (
                result.issues_by_level.get(ValidationLevel.ERROR, 0) + 1
            )

    def _validate_metadata(
        self, extraction_result: ExtractionResultWithErrors, result: ValidationResult
    ) -> None:
        """Validate extraction metadata."""
        metadata = extraction_result.metadata

        if not metadata:
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.INFO,
                    category=ValidationCategory.EXTRACTION_QUALITY,
                    message="Missing extraction metadata",
                    suggestion="Add metadata for better traceability",
                )
            )
            result.issues_by_level[ValidationLevel.INFO] = (
                result.issues_by_level.get(ValidationLevel.INFO, 0) + 1
            )
            return

        # Check extraction duration
        if (
            metadata.extraction_duration_ms and metadata.extraction_duration_ms > 30000
        ):  # 30 seconds
            result.issues.append(
                ValidationIssue(
                    level=ValidationLevel.INFO,
                    category=ValidationCategory.EXTRACTION_QUALITY,
                    message=f"Long extraction duration: {metadata.extraction_duration_ms}ms",
                    suggestion="Consider optimization for extraction performance",
                )
            )
            result.issues_by_level[ValidationLevel.INFO] = (
                result.issues_by_level.get(ValidationLevel.INFO, 0) + 1
            )

    def _calculate_extraction_score(
        self, result: ValidationResult, extraction_result: ExtractionResultWithErrors
    ) -> float:
        """Calculate extraction quality score."""
        score = 100.0

        # Base deduction for issues
        score -= result.issues_by_level.get(ValidationLevel.CRITICAL, 0) * 25
        score -= result.issues_by_level.get(ValidationLevel.ERROR, 0) * 15
        score -= result.issues_by_level.get(ValidationLevel.WARNING, 0) * 5
        score -= result.issues_by_level.get(ValidationLevel.INFO, 0) * 1

        # Quality bonuses
        entities = extraction_result.entities
        if entities:
            # High confidence bonus
            high_confidence = [e for e in entities if e.confidence == "high"]
            confidence_ratio = len(high_confidence) / len(entities)
            score += confidence_ratio * 10

        relations = extraction_result.relations
        if entities and relations:
            # Good entity-relation ratio
            ratio = len(relations) / len(entities)
            if 0.5 <= ratio <= 2.0:  # Good range
                score += 5

        return max(0.0, min(100.0, score))


# Convenience functions for validation
def validate_xml_against_xsd(xml_content: str) -> ValidationResult:
    """Convenience function to validate XML against BPMN XSD."""
    validator = XMLValidator()
    return validator.validate_xml_against_xsd(xml_content)


def validate_graph_semantics(
    graph: ProcessGraph, extraction_result: Optional[ExtractionResultWithErrors] = None
) -> ValidationResult:
    """Convenience function to validate graph semantics."""
    validator = GraphValidator()
    return validator.validate_graph_semantics(graph, extraction_result)


def validate_extraction(extraction_result: ExtractionResultWithErrors) -> ValidationResult:
    """Convenience function to validate extraction quality."""
    validator = ExtractionValidator()
    return validator.validate_extraction(extraction_result)
