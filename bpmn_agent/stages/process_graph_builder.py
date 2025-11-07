"""
Stage 4: Semantic Graph Construction

Converts extracted entities and relationships to a ProcessGraph intermediate representation.

Handles:
- Task 2.4.1: Convert extraction results to graph nodes and edges
- Task 2.4.2: Build lane structure from consolidated actors
- Task 2.4.3: Validate graph structure
- Task 2.4.4: Infer implicit flows for join points
- KB Integration: Use KB patterns for implicit flow inference and domain-specific layout
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from bpmn_agent.knowledge import PatternRecognizer
from bpmn_agent.knowledge.pattern_matching_bridge import AdvancedPatternMatchingBridge
from bpmn_agent.models.extraction import (
    ConfidenceLevel,
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    RelationType,
)
from bpmn_agent.models.graph import (
    EdgeType,
    GraphEdge,
    GraphMetrics,
    GraphNode,
    GraphValidationReport,
    NodeType,
    ProcessGraph,
)
from bpmn_agent.models.knowledge_base import DomainType
from bpmn_agent.stages.entity_resolution import ActorProfile

logger = logging.getLogger(__name__)


# ===========================
# KB Integration Helpers
# ===========================


class KBGraphEnricher:
    """Enriches graph construction with KB patterns and domain insights."""

    def __init__(self, enable_kb: bool = True):
        """
        Initialize KB enricher.

        Args:
            enable_kb: Whether to enable KB integration
        """
        self.enable_kb = enable_kb
        self._pattern_recognizer: Optional[PatternRecognizer] = None
        self._advanced_pattern_bridge: Optional[AdvancedPatternMatchingBridge] = None

    def _get_pattern_recognizer(self) -> Optional[PatternRecognizer]:
        """Lazy load pattern recognizer."""
        if not self.enable_kb:
            return None
        if self._pattern_recognizer is None:
            try:
                self._pattern_recognizer = PatternRecognizer()
                logger.info("Pattern recognizer initialized for KB-aware graph enrichment")
            except Exception as e:
                logger.warning(f"Failed to initialize pattern recognizer: {e}")
                return None
        return self._pattern_recognizer

    def _get_advanced_pattern_bridge(self) -> Optional["AdvancedPatternMatchingBridge"]:
        """Lazy load advanced pattern matching bridge."""
        if not self.enable_kb:
            return None
        if self._advanced_pattern_bridge is None:
            try:
                from knowledge.pattern_matching_bridge import AdvancedPatternMatchingBridge
                from models.knowledge_base import KnowledgeBase

                kb = KnowledgeBase()
                self._advanced_pattern_bridge = AdvancedPatternMatchingBridge(kb)
                logger.info("Advanced pattern matching bridge initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize advanced pattern bridge: {e}")
                return None
        return self._advanced_pattern_bridge

    def get_relevant_patterns(
        self, domain: Optional[DomainType], max_patterns: int = 3
    ) -> List[Dict]:
        """
        Get relevant patterns for the domain.

        Args:
            domain: Domain type to get patterns for
            max_patterns: Maximum number of patterns to return

        Returns:
            List of relevant patterns
        """
        if not self.enable_kb or not domain:
            return []

        recognizer = self._get_pattern_recognizer()
        if not recognizer:
            return []

        try:
            patterns = recognizer.recognize_patterns_for_domain(domain)
            # Sort by confidence and return top patterns
            sorted_patterns = sorted(patterns, key=lambda p: p.get("confidence", 0), reverse=True)
            return sorted_patterns[:max_patterns]
        except Exception as e:
            logger.warning(f"Error getting patterns for domain {domain}: {e}")
            return []

    def get_pattern_for_flow_type(self, pattern_name: str) -> Optional[Dict]:
        """
        Get a specific pattern by name.

        Args:
            pattern_name: Name of pattern to retrieve

        Returns:
            Pattern definition or None
        """
        if not self.enable_kb:
            return None

        recognizer = self._get_pattern_recognizer()
        if not recognizer:
            return None

        try:
            # Get pattern from recognizer
            patterns = recognizer.get_all_patterns()
            for pattern in patterns:
                if pattern.get("name") == pattern_name:
                    return pattern
        except Exception as e:
            logger.warning(f"Error getting pattern {pattern_name}: {e}")

        return None

    def suggest_implicit_flows_for_pattern(
        self, pattern_name: str, nodes: List[GraphNode], edges: List[GraphEdge]
    ) -> List[Tuple[str, str]]:
        """
        Suggest implicit flows based on recognized patterns.

        Args:
            pattern_name: Name of the pattern
            nodes: List of graph nodes
            edges: List of graph edges

        Returns:
            List of (source_id, target_id) tuples representing suggested flows
        """
        pattern = self.get_pattern_for_flow_type(pattern_name)
        if not pattern:
            return []

        suggestions: List[Tuple[str, str]] = []
        pattern_structure = pattern.get("structure", {})

        # Get pattern elements
        sequence = pattern_structure.get("sequence", [])
        parallel = pattern_structure.get("parallel", [])
        choice = pattern_structure.get("choice", {})

        # Find nodes matching pattern element names
        node_map = {node.label.lower(): node.id for node in nodes}
        edge_ids = {(e.source_id, e.target_id) for e in edges}

        # Suggest sequential flows
        for i in range(len(sequence) - 1):
            current_label = sequence[i].lower()
            next_label = sequence[i + 1].lower()

            current_id = node_map.get(current_label)
            next_id = node_map.get(next_label)

            if current_id and next_id and (current_id, next_id) not in edge_ids:
                suggestions.append((current_id, next_id))

        return suggestions

    def find_patterns_for_process(
        self, process_description: str, domain: Optional[DomainType] = None
    ) -> Optional[Dict]:
        """
        Find matching patterns for a process description using advanced matching.

        Args:
            process_description: Description of the process
            domain: Optional domain hint for better matching

        Returns:
            Pattern recommendation or None
        """
        bridge = self._get_advanced_pattern_bridge()
        if not bridge:
            return None

        try:
            recommendation = bridge.find_patterns_for_process(process_description, domain=domain)
            return (
                {
                    "pattern_id": (
                        recommendation.best_pattern.id if recommendation.best_pattern else None
                    ),
                    "pattern_name": (
                        recommendation.best_pattern.name if recommendation.best_pattern else None
                    ),
                    "confidence": recommendation.confidence,
                    "patterns": [
                        {
                            "id": p.pattern.id,
                            "name": p.pattern.name,
                            "category": p.pattern.category.value,
                            "score": p.match_score,
                        }
                        for p in recommendation.patterns
                    ],
                }
                if recommendation
                else None
            )
        except Exception as e:
            logger.warning(f"Error finding patterns for process: {e}")
            return None

    def match_activities_to_patterns(
        self, activity_labels: List[str], domain: Optional[DomainType] = None
    ) -> Dict[str, Dict]:
        """
        Match extracted activities to known patterns for validation.

        Args:
            activity_labels: List of activity labels from extraction
            domain: Optional domain for domain-specific matching

        Returns:
            Dictionary mapping activity label to match info (valid, score, suggestions)
        """
        bridge = self._get_advanced_pattern_bridge()
        if not bridge:
            return {}

        try:
            results = bridge.validate_extracted_activities(activity_labels, domain=domain)
            return {
                activity: {
                    "is_valid": valid,
                    "confidence": score,
                    "suggestions": suggestions,
                }
                for activity, (valid, score, suggestions) in results.items()
            }
        except Exception as e:
            logger.warning(f"Error validating activities: {e}")
            return {}

    def suggest_patterns_by_domain(self, domain: DomainType, max_patterns: int = 5) -> List[Dict]:
        """
        Get domain-specific pattern suggestions for graph construction.

        Args:
            domain: Domain type
            max_patterns: Maximum patterns to return

        Returns:
            List of pattern suggestions with metadata
        """
        bridge = self._get_advanced_pattern_bridge()
        if not bridge:
            return []

        try:
            suggestions = bridge.suggest_patterns_by_domain(domain, max_patterns=max_patterns)
            return [
                {
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "category": p.get("category"),
                    "complexity": p.get("complexity"),
                    "confidence": p.get("confidence"),
                    "tags": p.get("tags", []),
                }
                for p in suggestions
            ]
        except Exception as e:
            logger.warning(f"Error getting domain patterns: {e}")
            return []

    def search_patterns(
        self, query: str, domain: Optional[DomainType] = None, category: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for patterns using advanced pattern matching.

        Args:
            query: Search query
            domain: Optional domain filter
            category: Optional category filter

        Returns:
            List of matching patterns
        """
        bridge = self._get_advanced_pattern_bridge()
        if not bridge:
            return []

        try:
            results = bridge.search_patterns(query, domain=domain, category=category)
            return [
                {
                    "id": r.pattern.id,
                    "name": r.pattern.name,
                    "score": r.match_score,
                    "match_type": r.match_type,
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Error searching patterns: {e}")
            return []


# ===========================
# Task 2.4.1: Build Graph Nodes and Edges
# ===========================


class ProcessGraphBuilder:
    """Converts extraction results to ProcessGraph with optional KB integration."""

    def __init__(self, enable_kb: bool = True):
        """
        Initialize builder.

        Args:
            enable_kb: Whether to enable KB integration
        """
        self._node_counter = 0
        self._edge_counter = 0
        self.kb_enricher = KBGraphEnricher(enable_kb=enable_kb)
        self.enable_kb = enable_kb

    def build_from_extraction(
        self,
        extraction_result: ExtractionResult,
        actor_profiles: Dict[str, ActorProfile],
        domain: Optional[DomainType] = None,
    ) -> ProcessGraph:
        """
        Convert extraction results to ProcessGraph.

        Args:
            extraction_result: Resolved extraction results from Stage 3
            actor_profiles: Actor profiles with activity assignments
            domain: Domain type for KB-aware layout decisions

        Returns:
            ProcessGraph with nodes and edges
        """
        # Initialize graph
        graph_id = f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []

        # 1. Build nodes from entities
        entity_to_node_map: Dict[str, str] = {}
        for entity in extraction_result.entities:
            node = self._entity_to_node(entity)
            nodes.append(node)
            entity_to_node_map[entity.id] = node.id

        # 2. Build edges from relations
        for relation in extraction_result.relations:
            if (
                relation.source_id in entity_to_node_map
                and relation.target_id in entity_to_node_map
            ):
                edge = self._relation_to_edge(
                    relation,
                    entity_to_node_map[relation.source_id],
                    entity_to_node_map[relation.target_id],
                )
                edges.append(edge)

        # 3. Create graph
        process_name = "Extracted Process"
        process_description = ""

        graph = ProcessGraph(
            id=graph_id,
            name=process_name,
            description=process_description,
            nodes=nodes,
            edges=edges,
            created_timestamp=datetime.now().isoformat(),
            metadata={
                "num_entities": len(extraction_result.entities),
                "num_relations": len(extraction_result.relations),
                "num_actors": len(actor_profiles),
                "domain": domain.value if domain else None,
                "kb_enabled": self.enable_kb,
            },
        )

        # Detect and upgrade gateway types from DECISION to PARALLEL_FORK/PARALLEL_JOIN
        graph = self._upgrade_gateway_types(graph)

        # Inject synthetic start/end nodes if needed
        graph = self._inject_synthetic_start_end_nodes(graph)

        return graph

    def _upgrade_gateway_types(self, graph: ProcessGraph) -> ProcessGraph:
        """
        Detect and upgrade gateway node types from DECISION to PARALLEL_FORK/PARALLEL_JOIN.

        A gateway is considered:
        - PARALLEL_FORK if it has 1 incoming edge and 2+ outgoing edges
        - PARALLEL_JOIN if it has 2+ incoming edges and 1 outgoing edge
        - DECISION otherwise (exclusive choice)

        Args:
            graph: ProcessGraph to enhance

        Returns:
            Updated ProcessGraph with upgraded gateway types
        """
        updated_nodes = []

        for node in graph.nodes:
            if node.type != NodeType.DECISION:
                updated_nodes.append(node)
                continue

            # Get control flow edges only
            incoming = [
                e for e in graph.edges if e.target_id == node.id and e.type == EdgeType.CONTROL_FLOW
            ]
            outgoing = [
                e for e in graph.edges if e.source_id == node.id and e.type == EdgeType.CONTROL_FLOW
            ]

            # Determine gateway type based on incoming/outgoing edges
            if len(incoming) == 1 and len(outgoing) >= 2:
                # Fork: single path splits into multiple
                node.type = NodeType.PARALLEL_FORK
                node.bpmn_type = "ParallelGateway"
            elif len(incoming) >= 2 and len(outgoing) == 1:
                # Join: multiple paths converge
                node.type = NodeType.PARALLEL_JOIN
                node.bpmn_type = "ParallelGateway"

            updated_nodes.append(node)

        # Create new graph with updated nodes
        return ProcessGraph(
            id=graph.id,
            name=graph.name,
            description=graph.description,
            nodes=updated_nodes,
            edges=graph.edges,
            created_timestamp=graph.created_timestamp,
            metadata=graph.metadata,
        )

    def _inject_synthetic_start_end_nodes(self, graph: ProcessGraph) -> ProcessGraph:
        """
        Inject synthetic START and END nodes if they don't exist.

        This ensures the graph has explicit start and end events for proper BPMN generation.

        Args:
            graph: ProcessGraph to enhance

        Returns:
            Updated ProcessGraph with synthetic start/end nodes
        """
        # Check if explicit START/END nodes exist
        has_start = any(n.type == NodeType.START for n in graph.nodes)
        has_end = any(n.type == NodeType.END for n in graph.nodes)

        if has_start and has_end:
            # Already has proper start/end nodes
            return graph

        # Find implicit start and end nodes
        implicit_start_nodes = [n for n in graph.get_start_nodes() if n.type != NodeType.START]
        implicit_end_nodes = [n for n in graph.get_end_nodes() if n.type != NodeType.END]

        nodes_to_add = []
        edges_to_add = []

        # Create synthetic START node if needed
        if implicit_start_nodes:
            start_node = GraphNode(
                id=f"synthetic_start_{uuid.uuid4().hex[:8]}",
                type=NodeType.START,
                label="Start",
                bpmn_type="StartEvent",
                confidence=1.0,
                metadata={"synthetic": True},
                x=None,
                y=None,
                is_abstract=False,
            )
            nodes_to_add.append(start_node)

            # Connect synthetic start to all implicit start nodes
            for implicit_start in implicit_start_nodes:
                edge = GraphEdge(
                    id=f"edge_{uuid.uuid4().hex[:8]}",
                    source_id=start_node.id,
                    target_id=implicit_start.id,
                    type=EdgeType.CONTROL_FLOW,
                    label="",
                    confidence=1.0,
                    metadata={"synthetic": True},
                    condition=None,
                    is_default=False,
                )
                edges_to_add.append(edge)

        # Create synthetic END node if needed
        if implicit_end_nodes:
            end_node = GraphNode(
                id=f"synthetic_end_{uuid.uuid4().hex[:8]}",
                type=NodeType.END,
                label="End",
                bpmn_type="EndEvent",
                confidence=1.0,
                metadata={"synthetic": True},
            )
            nodes_to_add.append(end_node)

            # Connect all implicit end nodes to synthetic end
            for implicit_end in implicit_end_nodes:
                edge = GraphEdge(
                    id=f"edge_{uuid.uuid4().hex[:8]}",
                    source_id=implicit_end.id,
                    target_id=end_node.id,
                    type=EdgeType.CONTROL_FLOW,
                    label="",
                    confidence=1.0,
                    metadata={"synthetic": True},
                    condition=None,
                    is_default=False,
                )
                edges_to_add.append(edge)

        # Add synthetic nodes and edges to graph
        if nodes_to_add or edges_to_add:
            graph.nodes.extend(nodes_to_add)
            graph.edges.extend(edges_to_add)
            graph._build_indexes()
            logger.debug(
                f"Injected {len(nodes_to_add)} synthetic nodes and {len(edges_to_add)} synthetic edges"
            )

        return graph

    def _entity_to_node(self, entity: ExtractedEntity) -> GraphNode:
        """
        Convert extracted entity to graph node.

        Args:
            entity: Extracted entity

        Returns:
            GraphNode
        """
        # Map entity type to node type
        node_type = self._map_entity_to_node_type(entity.type)
        bpmn_type = ProcessGraphBuilder._map_entity_to_bpmn_type_with_attributes(entity)

        # Convert confidence level to numeric score
        confidence_score = {
            ConfidenceLevel.HIGH: 1.0,
            ConfidenceLevel.MEDIUM: 0.7,
            ConfidenceLevel.LOW: 0.4,
        }.get(entity.confidence, 0.7)

        node = GraphNode(
            id=entity.id,
            type=node_type,
            label=entity.name,
            bpmn_type=bpmn_type,
            confidence=confidence_score,
            properties={
                "entity_type": entity.type.value,
                "description": entity.description,
                "alternative_names": entity.alternative_names,
            },
            metadata={
                "original_id": entity.id,
                "source_text": entity.source_text,
            },
            x=None,
            y=None,
            is_abstract=False,
        )

        return node

    def _relation_to_edge(
        self,
        relation: ExtractedRelation,
        source_node_id: str,
        target_node_id: str,
    ) -> GraphEdge:
        """
        Convert extracted relation to graph edge.

        Args:
            relation: Extracted relation
            source_node_id: Target source node ID
            target_node_id: Target target node ID

        Returns:
            GraphEdge
        """
        # Map relation type to edge type
        edge_type = self._map_relation_to_edge_type(relation.type)

        # Convert confidence
        confidence_score = {
            ConfidenceLevel.HIGH: 1.0,
            ConfidenceLevel.MEDIUM: 0.7,
            ConfidenceLevel.LOW: 0.4,
        }.get(relation.confidence, 0.7)

        edge = GraphEdge(
            id=f"edge_{uuid.uuid4().hex[:8]}",
            source_id=source_node_id,
            target_id=target_node_id,
            type=edge_type,
            label=relation.label or relation.type.value,
            confidence=confidence_score,
            metadata={
                "original_relation_type": relation.type.value,
                "source_text": relation.source_text,
            },
            condition=relation.condition_expression,
            is_default=False,
        )

        return edge

    @staticmethod
    def _map_entity_to_node_type(entity_type: EntityType) -> NodeType:
        """Map entity type to node type."""
        mapping = {
            EntityType.ACTIVITY: NodeType.TASK,
            EntityType.EVENT: NodeType.EVENT,
            EntityType.GATEWAY: NodeType.DECISION,
            EntityType.ACTOR: NodeType.ACTOR,
            EntityType.DATA: NodeType.DATA,
        }
        return mapping.get(entity_type, NodeType.TASK)

    @staticmethod
    def _map_entity_to_bpmn_type(entity_type: EntityType) -> str:
        """Map entity type to BPMN element type."""
        mapping = {
            EntityType.ACTIVITY: "Task",
            EntityType.EVENT: "IntermediateCatchEvent",
            EntityType.GATEWAY: "ExclusiveGateway",
            EntityType.ACTOR: "Lane",
            EntityType.DATA: "DataObject",
        }
        return mapping.get(entity_type, "Task")

    @staticmethod
    def _map_entity_to_bpmn_type_with_attributes(entity) -> str:
        """
        Map entity to BPMN element type, using entity attributes for refinement.

        For events, checks for event_type and trigger attributes to determine
        the specific event kind (e.g., BoundaryEvent, TimerEvent, etc.)
        """
        from bpmn_agent.models.extraction import EntityType

        # Base type mapping
        base_mapping = {
            EntityType.ACTIVITY: "Task",
            EntityType.EVENT: "IntermediateCatchEvent",
            EntityType.GATEWAY: "ExclusiveGateway",
            EntityType.ACTOR: "Lane",
            EntityType.DATA: "DataObject",
        }

        base_type = base_mapping.get(entity.type, "Task")

        # For events, check attributes for more specific type
        if entity.type == EntityType.EVENT and entity.attributes:
            # Check for event_type attribute
            event_type_attr = entity.attributes.get("event_type")
            if event_type_attr:
                event_type_value = (
                    event_type_attr.value if hasattr(event_type_attr, "value") else event_type_attr
                )
                if "boundary" in str(event_type_value).lower():
                    return "BoundaryEvent"

            # Check for trigger attribute to determine timer/signal/message events
            trigger_attr = entity.attributes.get("trigger")
            if trigger_attr:
                trigger_value = (
                    trigger_attr.value if hasattr(trigger_attr, "value") else trigger_attr
                )
                trigger_str = str(trigger_value).lower()

                if "timer" in trigger_str:
                    return "TimerIntermediateCatchEvent"
                elif "signal" in trigger_str:
                    return "SignalIntermediateCatchEvent"
                elif "message" in trigger_str:
                    return "MessageIntermediateCatchEvent"
                elif "error" in trigger_str:
                    return "ErrorIntermediateCatchEvent"

        # For activities, check for subprocess attribute
        if entity.type == EntityType.ACTIVITY and entity.attributes:
            activity_type_attr = entity.attributes.get("activity_type")
            if activity_type_attr:
                activity_type_value = (
                    activity_type_attr.value
                    if hasattr(activity_type_attr, "value")
                    else activity_type_attr
                )
                if "subprocess" in str(activity_type_value).lower():
                    return "SubProcess"

        return base_type

    @staticmethod
    def _map_relation_to_edge_type(relation_type: RelationType) -> EdgeType:
        """Map relation type to edge type."""
        mapping = {
            RelationType.PRECEDES: EdgeType.CONTROL_FLOW,
            RelationType.FOLLOWS: EdgeType.CONTROL_FLOW,
            RelationType.TRIGGERS: EdgeType.CONTROL_FLOW,
            RelationType.INVOLVES: EdgeType.SWIMLANE,
            RelationType.USES: EdgeType.DATA_FLOW,
            RelationType.PRODUCES: EdgeType.DATA_FLOW,
            RelationType.CONSUMES: EdgeType.DATA_FLOW,
            RelationType.SENDS_TO: EdgeType.MESSAGE_FLOW,
            RelationType.RECEIVES_FROM: EdgeType.MESSAGE_FLOW,
            RelationType.ALTERNATIVE_TO: EdgeType.CONTROL_FLOW,
            RelationType.PARALLEL_TO: EdgeType.CONTROL_FLOW,
            RelationType.CONDITIONAL: EdgeType.CONTROL_FLOW,
        }
        return mapping.get(relation_type, EdgeType.ASSOCIATION)


# ===========================
# Task 2.4.2: Build Lane Structure
# ===========================


@dataclass
class LaneStructure:
    """Represents a swimlane/participant in the process."""

    lane_id: str
    """Unique lane identifier"""

    lane_name: str
    """Lane name (actor name)"""

    node_ids: List[str] = field(default_factory=list)
    """IDs of nodes in this lane"""

    actor_id: str = ""
    """Reference to original actor entity"""

    confidence: float = 1.0
    """Confidence in lane structure"""

    metadata: Dict = field(default_factory=dict)
    """Additional metadata"""


class LaneStructureBuilder:
    """Builds swimlane/lane structure from actors with optional domain-specific layout."""

    def __init__(self, enable_kb: bool = True):
        """
        Initialize lane structure builder.

        Args:
            enable_kb: Whether to enable KB domain-aware decisions
        """
        self.enable_kb = enable_kb
        self.kb_enricher = KBGraphEnricher(enable_kb=enable_kb)

    def build_lanes_from_actors(
        self,
        graph: ProcessGraph,
        actor_profiles: Dict[str, ActorProfile],
        domain: Optional[DomainType] = None,
    ) -> Tuple[ProcessGraph, Dict[str, LaneStructure]]:
        """
        Build lane structure from actor profiles with domain-aware layout.

        Args:
            graph: ProcessGraph with nodes
            actor_profiles: Actor profiles with activity assignments
            domain: Domain type for KB-aware layout decisions

        Returns:
            (updated_graph_with_lane_edges, lane_structures)
        """
        lane_structures: Dict[str, LaneStructure] = {}
        new_edges: List[GraphEdge] = []

        # 1. Create lane structures for each actor
        for actor_id, profile in actor_profiles.items():
            lane_id = f"lane_{uuid.uuid4().hex[:8]}"

            lane = LaneStructure(
                lane_id=lane_id,
                lane_name=profile.actor_name,
                node_ids=profile.activity_ids,
                actor_id=actor_id,
                confidence=self._confidence_to_float(profile.confidence),
            )
            lane_structures[actor_id] = lane

        # 2. Apply domain-specific layout decisions if KB enabled
        if self.enable_kb and domain:
            lane_structures = self._apply_domain_layout_decisions(graph, lane_structures, domain)

        # 3. Create swimlane edges from activities to actor lanes
        for actor_id, lane in lane_structures.items():
            for activity_id in lane.node_ids:
                # Find actor node
                actor_node = graph.get_node(actor_id)
                activity_node = graph.get_node(activity_id)

                if actor_node and activity_node:
                    # Create edge: actor -> activity (swimlane assignment)
                    edge = GraphEdge(
                        id=f"swimlane_{uuid.uuid4().hex[:8]}",
                        source_id=actor_id,
                        target_id=activity_id,
                        type=EdgeType.SWIMLANE,
                        label=f"Assigned to {actor_node.label}",
                        confidence=lane.confidence,
                        metadata={"lane_id": lane.lane_id},
                        condition=None,
                        is_default=False,
                    )
                    new_edges.append(edge)

        # 4. Update graph with swimlane edges
        graph.edges.extend(new_edges)
        graph._build_indexes()

        return graph, lane_structures

    def _apply_domain_layout_decisions(
        self, graph: ProcessGraph, lane_structures: Dict[str, LaneStructure], domain: DomainType
    ) -> Dict[str, LaneStructure]:
        """
        Apply domain-specific layout decisions to lane structures.

        Args:
            graph: ProcessGraph
            lane_structures: Current lane structures
            domain: Domain type

        Returns:
            Updated lane structures
        """
        try:
            # Get domain patterns to understand typical actor roles
            patterns = self.kb_enricher.get_relevant_patterns(domain, max_patterns=3)

            for pattern in patterns:
                # Check if pattern suggests actor ordering or grouping
                if "actors" in pattern:
                    suggested_order = pattern.get("actors", [])
                    lane_structures = self._reorder_lanes_by_pattern(
                        lane_structures, suggested_order
                    )

                # Check if pattern suggests lane isolation or grouping
                if "swimlane_hints" in pattern:
                    hints = pattern.get("swimlane_hints", {})
                    lane_structures = self._apply_swimlane_hints(lane_structures, hints)

            logger.info(f"Applied domain-specific layout decisions for domain {domain}")

        except Exception as e:
            logger.warning(f"Error applying domain layout decisions: {e}")

        return lane_structures

    @staticmethod
    def _reorder_lanes_by_pattern(
        lane_structures: Dict[str, LaneStructure], suggested_order: List[str]
    ) -> Dict[str, LaneStructure]:
        """
        Reorder lanes based on pattern suggestions.

        Args:
            lane_structures: Current lane structures
            suggested_order: Suggested order of actor names

        Returns:
            Reordered lane structures
        """
        # Create new ordered dict
        ordered_lanes = {}

        # First, add lanes that match the suggested order
        for actor_name in suggested_order:
            for actor_id, lane in lane_structures.items():
                if lane.lane_name.lower() == actor_name.lower() and actor_id not in ordered_lanes:
                    ordered_lanes[actor_id] = lane

        # Then add any remaining lanes
        for actor_id, lane in lane_structures.items():
            if actor_id not in ordered_lanes:
                ordered_lanes[actor_id] = lane

        return ordered_lanes

    @staticmethod
    def _apply_swimlane_hints(
        lane_structures: Dict[str, LaneStructure], hints: Dict
    ) -> Dict[str, LaneStructure]:
        """
        Apply swimlane hints from domain patterns.

        Args:
            lane_structures: Current lane structures
            hints: Hints about lane organization (grouped, isolated, etc.)

        Returns:
            Updated lane structures with hints in metadata
        """
        for actor_id, lane in lane_structures.items():
            if "grouped_with" in hints:
                lane.metadata["grouped_with"] = hints["grouped_with"]

            if "isolation_level" in hints:
                lane.metadata["isolation_level"] = hints["isolation_level"]

        return lane_structures

    @staticmethod
    def _confidence_to_float(confidence: ConfidenceLevel) -> float:
        """Convert confidence level to float."""
        mapping = {
            ConfidenceLevel.HIGH: 1.0,
            ConfidenceLevel.MEDIUM: 0.7,
            ConfidenceLevel.LOW: 0.4,
        }
        return mapping.get(confidence, 0.7)


# ===========================
# Task 2.4.3: Graph Validation
# ===========================


@dataclass
class GraphValidationIssue:
    """Represents a validation issue found in graph."""

    category: str
    """Issue category: error, warning, suggestion"""

    message: str
    """Issue description"""

    affected_nodes: List[str] = field(default_factory=list)
    """IDs of affected nodes"""

    severity: str = "warning"
    """Issue severity: error, warning, info"""


class GraphValidator:
    """Validates ProcessGraph structure and correctness."""

    def validate_graph(
        self, graph: ProcessGraph
    ) -> Tuple[bool, List[GraphValidationIssue], GraphMetrics]:
        """
        Validate graph structure.

        Args:
            graph: ProcessGraph to validate

        Returns:
            (is_valid, issues, metrics)
        """
        issues: List[GraphValidationIssue] = []

        # 1. Check for isolated nodes
        isolated_nodes = self._find_isolated_nodes(graph)
        if isolated_nodes:
            issue = GraphValidationIssue(
                category="structure",
                message=f"Found {len(isolated_nodes)} isolated nodes with no connections",
                affected_nodes=isolated_nodes,
                severity="warning",
            )
            issues.append(issue)

        # 2. Check for start/end events
        start_nodes = graph.get_start_nodes()
        end_nodes = graph.get_end_nodes()

        if not start_nodes:
            issue = GraphValidationIssue(
                category="flow",
                message="No start node found in process",
                severity="error",
            )
            issues.append(issue)

        if not end_nodes:
            issue = GraphValidationIssue(
                category="flow",
                message="No end node found in process",
                severity="error",
            )
            issues.append(issue)

        # 3. Check for cycles
        has_cycles, cycle_nodes = self._detect_cycles(graph)
        if has_cycles:
            issue = GraphValidationIssue(
                category="flow",
                message=f"Graph contains cycles: {len(cycle_nodes)} nodes involved",
                affected_nodes=cycle_nodes,
                severity="warning",
            )
            issues.append(issue)

        # 4. Check for unreachable nodes
        unreachable = self._find_unreachable_nodes(graph)
        if unreachable:
            issue = GraphValidationIssue(
                category="connectivity",
                message=f"Found {len(unreachable)} unreachable nodes",
                affected_nodes=unreachable,
                severity="warning",
            )
            issues.append(issue)

        # 5. Check gateway validation
        gateway_issues = self._validate_gateways(graph)
        issues.extend(gateway_issues)

        # 6. Compute metrics
        metrics = self._compute_metrics(graph, issues)

        # Graph is valid if no errors (warnings are OK)
        is_valid = not any(issue.severity == "error" for issue in issues)

        return is_valid, issues, metrics

    @staticmethod
    def _find_isolated_nodes(graph: ProcessGraph) -> List[str]:
        """Find nodes with no incoming or outgoing edges."""
        isolated = []
        for node in graph.nodes:
            incoming = graph.get_incoming_edges(node.id)
            outgoing = graph.get_outgoing_edges(node.id)

            # Exclude ACTOR nodes (they don't have control flow)
            if node.type != NodeType.ACTOR:
                if not incoming and not outgoing:
                    isolated.append(node.id)

        return isolated

    @staticmethod
    def _detect_cycles(graph: ProcessGraph) -> Tuple[bool, List[str]]:
        """Detect if graph contains cycles using DFS."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        cycle_nodes: List[str] = []

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for edge in graph.get_outgoing_edges(node_id):
                target_id = edge.target_id

                if target_id not in visited:
                    if dfs(target_id):
                        cycle_nodes.append(node_id)
                        return True
                elif target_id in rec_stack:
                    cycle_nodes.extend([node_id, target_id])
                    return True

            rec_stack.remove(node_id)
            return False

        for node in graph.nodes:
            if node.id not in visited:
                if dfs(node.id):
                    return True, list(set(cycle_nodes))

        return False, []

    @staticmethod
    def _find_unreachable_nodes(graph: ProcessGraph) -> List[str]:
        """Find nodes unreachable from start nodes via control flow edges."""
        if not graph.get_start_nodes():
            return [n.id for n in graph.nodes]

        reachable: Set[str] = set()

        def bfs(start_id: str) -> None:
            queue = [start_id]
            visited = {start_id}

            while queue:
                current_id = queue.pop(0)
                reachable.add(current_id)

                # Only traverse control flow edges for reachability
                control_flow_edges = [
                    e
                    for e in graph.get_outgoing_edges(current_id)
                    if e.type == EdgeType.CONTROL_FLOW
                ]
                for edge in control_flow_edges:
                    if edge.target_id not in visited:
                        visited.add(edge.target_id)
                        queue.append(edge.target_id)

        # Find reachable nodes from all start nodes
        for start_node in graph.get_start_nodes():
            bfs(start_node.id)

        # Unreachable are all nodes not in reachable set
        unreachable = [
            n.id for n in graph.nodes if n.id not in reachable and n.type != NodeType.ACTOR
        ]

        return unreachable

    @staticmethod
    def _validate_gateways(graph: ProcessGraph) -> List[GraphValidationIssue]:
        """Validate gateway nodes."""
        issues: List[GraphValidationIssue] = []
        gateways = graph.get_decision_nodes()

        for gateway in gateways:
            incoming = graph.get_incoming_edges(gateway.id)
            outgoing = graph.get_outgoing_edges(gateway.id)

            # Gateways should have at least one incoming and two outgoing (or join with multiple incoming)
            if len(incoming) == 0:
                issue = GraphValidationIssue(
                    category="gateway",
                    message=f"Gateway '{gateway.label}' has no incoming edges",
                    affected_nodes=[gateway.id],
                    severity="warning",
                )
                issues.append(issue)

            if len(outgoing) < 2 and len(incoming) < 2:
                issue = GraphValidationIssue(
                    category="gateway",
                    message=f"Gateway '{gateway.label}' is not used for branching or joining",
                    affected_nodes=[gateway.id],
                    severity="info",
                )
                issues.append(issue)

        return issues

    @staticmethod
    def _compute_metrics(graph: ProcessGraph, issues: List[GraphValidationIssue]) -> GraphMetrics:
        """Compute graph metrics."""
        # Basic counts
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)

        # Density (actual edges / possible edges)
        max_edges = node_count * (node_count - 1)
        density = edge_count / max_edges if max_edges > 0 else 0.0

        # Cyclomatic complexity: E - N + 2P (E=edges, N=nodes, P=connected components)
        complexity = max(0.0, edge_count - node_count + 2)

        # Average degree
        total_degree = sum(
            len(graph.get_incoming_edges(n.id)) + len(graph.get_outgoing_edges(n.id))
            for n in graph.nodes
        )
        avg_degree = total_degree / node_count if node_count > 0 else 0.0

        # Path metrics
        paths = graph.get_parallel_paths()
        longest_path = max(len(p) for p in paths) if paths else 1

        # Decision points
        decision_count = len(graph.get_decision_nodes())

        # Data dependencies
        data_edges = [e for e in graph.edges if e.type == EdgeType.DATA_FLOW]

        # Completeness (0-1 based on presence of start/end)
        has_start = len(graph.get_start_nodes()) > 0
        has_end = len(graph.get_end_nodes()) > 0
        completeness = 1.0 if (has_start and has_end) else 0.5

        # Ambiguity based on unresolved issues
        error_issues = [i for i in issues if i.severity == "error"]
        ambiguity = len(error_issues) / max(node_count, 1) * 0.5

        metrics = GraphMetrics(
            node_count=node_count,
            edge_count=edge_count,
            density=density,
            cyclomatic_complexity=complexity,
            average_node_degree=avg_degree,
            longest_path_length=longest_path,
            num_parallel_paths=len(paths),
            num_decision_points=decision_count,
            num_data_dependencies=len(data_edges),
            structural_completeness=completeness,
            ambiguity_score=ambiguity,
        )

        return metrics


# ===========================
# Task 2.4.4: Implicit Flow Inference
# ===========================


@dataclass
class ImplicitFlow:
    """Represents an inferred implicit flow."""

    source_id: str
    """Source node ID"""

    target_id: str
    """Target node ID"""

    reason: str
    """Reason for inference"""

    confidence: float = 0.5
    """Confidence in inferred flow (0-1)"""


class ImplicitFlowInferrer:
    """Infers implicit flows and suggests join points with optional KB pattern awareness."""

    def __init__(self, enable_kb: bool = True):
        """
        Initialize flow inferrer.

        Args:
            enable_kb: Whether to enable KB pattern-based inference
        """
        self.enable_kb = enable_kb
        self.kb_enricher = KBGraphEnricher(enable_kb=enable_kb)

    def infer_implicit_flows(
        self, graph: ProcessGraph, domain: Optional[DomainType] = None
    ) -> List[ImplicitFlow]:
        """
        Infer implicit flows based on graph structure and KB patterns.

        Args:
            graph: ProcessGraph to analyze
            domain: Domain type for pattern-based inference

        Returns:
            List of inferred implicit flows
        """
        inferred_flows: List[ImplicitFlow] = []

        # 1. Try KB pattern-based inference if enabled
        if self.enable_kb and domain:
            pattern_flows = self._infer_flows_from_kb_patterns(graph, domain)
            inferred_flows.extend(pattern_flows)

        # 2. Detect fork/join patterns needing implicit flows
        fork_join_flows = self._infer_fork_join_flows(graph)
        inferred_flows.extend(fork_join_flows)

        # 3. Detect sequential flow gaps
        sequential_flows = self._infer_sequential_flows(graph)
        inferred_flows.extend(sequential_flows)

        # 4. Detect data flow dependencies as implicit control flow
        data_flow_implications = self._infer_data_flow_dependencies(graph)
        inferred_flows.extend(data_flow_implications)

        # Remove duplicates
        unique_flows: Dict[Tuple[str, str], GraphEdge] = {}
        for flow in inferred_flows:
            key = (flow.source_id, flow.target_id)
            if key not in unique_flows or flow.confidence > unique_flows[key].confidence:
                unique_flows[key] = flow

        return list(unique_flows.values())

    def _infer_flows_from_kb_patterns(
        self, graph: ProcessGraph, domain: DomainType
    ) -> List[ImplicitFlow]:
        """
        Infer flows based on recognized KB patterns.

        Args:
            graph: ProcessGraph to analyze
            domain: Domain type

        Returns:
            List of inferred implicit flows
        """
        flows: List[ImplicitFlow] = []

        try:
            # Get relevant patterns for domain
            patterns = self.kb_enricher.get_relevant_patterns(domain, max_patterns=5)

            for pattern in patterns:
                pattern_name = pattern.get("name", "")
                pattern_structure = pattern.get("structure", {})

                # Check for sequential patterns
                if "sequence" in pattern_structure:
                    suggested_flows = self._infer_sequence_from_pattern(
                        graph, pattern_name, pattern_structure.get("sequence", [])
                    )
                    flows.extend(suggested_flows)

                # Check for parallel patterns
                if "parallel" in pattern_structure:
                    suggested_flows = self._infer_parallel_from_pattern(
                        graph, pattern_name, pattern_structure.get("parallel", [])
                    )
                    flows.extend(suggested_flows)

                # Check for choice patterns
                if "choice" in pattern_structure:
                    suggested_flows = self._infer_choice_from_pattern(
                        graph, pattern_name, pattern_structure.get("choice", {})
                    )
                    flows.extend(suggested_flows)

            logger.info(
                f"KB patterns generated {len(flows)} potential implicit flows for domain {domain}"
            )

        except Exception as e:
            logger.warning(f"Error inferring flows from KB patterns: {e}")

        return flows

    def _infer_sequence_from_pattern(
        self, graph: ProcessGraph, pattern_name: str, sequence: List[str]
    ) -> List[ImplicitFlow]:
        """
        Infer sequential flows from pattern.

        Args:
            graph: ProcessGraph
            pattern_name: Pattern name
            sequence: List of activity names in sequence

        Returns:
            List of inferred flows
        """
        flows: List[ImplicitFlow] = []

        # Find nodes matching sequence
        node_label_map = {node.label.lower(): node for node in graph.nodes}
        existing_edges = {(e.source_id, e.target_id) for e in graph.edges}

        for i in range(len(sequence) - 1):
            current_label = sequence[i].lower()
            next_label = sequence[i + 1].lower()

            current_node = node_label_map.get(current_label)
            next_node = node_label_map.get(next_label)

            if current_node and next_node:
                edge_key = (current_node.id, next_node.id)
                if edge_key not in existing_edges:
                    flow = ImplicitFlow(
                        source_id=current_node.id,
                        target_id=next_node.id,
                        reason=f"Pattern '{pattern_name}' suggests sequential flow",
                        confidence=0.75,
                    )
                    flows.append(flow)

        return flows

    def _infer_parallel_from_pattern(
        self, graph: ProcessGraph, pattern_name: str, parallel_activities: List[List[str]]
    ) -> List[ImplicitFlow]:
        """
        Infer parallel flow patterns.

        Args:
            graph: ProcessGraph
            pattern_name: Pattern name
            parallel_activities: List of parallel activity sequences

        Returns:
            List of inferred flows
        """
        flows: List[ImplicitFlow] = []

        # Find fork and join points for parallel pattern
        node_label_map = {node.label.lower(): node for node in graph.nodes}
        existing_edges = {(e.source_id, e.target_id) for e in graph.edges}

        if parallel_activities and len(parallel_activities) > 1:
            # Look for common predecessor (fork point)
            first_sequence = parallel_activities[0]
            if first_sequence:
                first_activity_label = first_sequence[0].lower()
                fork_node = node_label_map.get(first_activity_label)

                if fork_node:
                    # For each parallel branch, create join suggestions
                    for branch_seq in parallel_activities:
                        if branch_seq and len(branch_seq) > 1:
                            last_activity_label = branch_seq[-1].lower()
                            join_node = node_label_map.get(last_activity_label)

                            if join_node:
                                edge_key = (join_node.id, fork_node.id)
                                if edge_key not in existing_edges:
                                    # Don't suggest self-join
                                    if join_node.id != fork_node.id:
                                        flow = ImplicitFlow(
                                            source_id=join_node.id,
                                            target_id=fork_node.id,
                                            reason=f"Pattern '{pattern_name}' suggests join after parallel",
                                            confidence=0.65,
                                        )
                                        flows.append(flow)

        return flows

    def _infer_choice_from_pattern(
        self, graph: ProcessGraph, pattern_name: str, choice_spec: Dict
    ) -> List[ImplicitFlow]:
        """
        Infer choice (decision) flow patterns.

        Args:
            graph: ProcessGraph
            pattern_name: Pattern name
            choice_spec: Choice pattern specification

        Returns:
            List of inferred flows
        """
        flows: List[ImplicitFlow] = []

        # This would involve inferring conditional branches from pattern
        # For now, we'll keep it simple
        logger.debug(f"Choice pattern inference for '{pattern_name}' is a placeholder")

        return flows

    @staticmethod
    def _infer_fork_join_flows(graph: ProcessGraph) -> List[ImplicitFlow]:
        """Infer flows for fork/join parallelism."""
        flows: List[ImplicitFlow] = []

        # Find fork nodes (multiple outgoing edges)
        for node in graph.nodes:
            outgoing = graph.get_outgoing_edges(node.id)
            if len(outgoing) > 1:
                # This is a fork - infer join point
                # Find nodes that should be joined after parallel paths

                # Get all nodes reachable from each outgoing branch
                reachable_sets = []
                for edge in outgoing:
                    reachable = ImplicitFlowInferrer._find_reachable_from_node(
                        graph, edge.target_id
                    )
                    reachable_sets.append(reachable)

                # Find common reachable nodes (candidates for join)
                if reachable_sets:
                    common_reachable = (
                        set.intersection(*reachable_sets) if len(reachable_sets) > 1 else set()
                    )

                    # The first common reachable node is the join point
                    if common_reachable:
                        join_node_id = min(common_reachable)  # Arbitrary choice of first

                        # Create implicit flows from each branch end to join
                        for reachable_set in reachable_sets:
                            for node_id in reachable_set:
                                outgoing_count = len(graph.get_outgoing_edges(node_id))
                                if outgoing_count == 0:  # Leaf node in branch
                                    flow = ImplicitFlow(
                                        source_id=node_id,
                                        target_id=join_node_id,
                                        reason="Inferred join from parallel fork",
                                        confidence=0.6,
                                    )
                                    flows.append(flow)

        return flows

    @staticmethod
    def _find_reachable_from_node(graph: ProcessGraph, node_id: str) -> Set[str]:
        """Find all nodes reachable from a given node."""
        reachable = set()
        queue = [node_id]
        visited = {node_id}

        while queue:
            current = queue.pop(0)
            reachable.add(current)

            for edge in graph.get_outgoing_edges(current):
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append(edge.target_id)

        return reachable

    @staticmethod
    def _infer_sequential_flows(graph: ProcessGraph) -> List[ImplicitFlow]:
        """Infer flows for sequential activities."""
        flows: List[ImplicitFlow] = []

        # Look for activities without sequence but that should be connected
        tasks = [n for n in graph.nodes if n.type == NodeType.TASK]

        for task in tasks:
            outgoing = graph.get_outgoing_edges(task.id)

            # If task has no outgoing edges but there are more tasks, infer flow
            if len(outgoing) == 0:
                # Find the most likely next task
                remaining_tasks = [t for t in tasks if t.id != task.id]
                if remaining_tasks:
                    # Simple heuristic: connect to next unconnected task
                    next_task = remaining_tasks[0]
                    if not graph.get_incoming_edges(next_task.id):
                        flow = ImplicitFlow(
                            source_id=task.id,
                            target_id=next_task.id,
                            reason="Inferred sequential flow",
                            confidence=0.4,
                        )
                        flows.append(flow)

        return flows

    @staticmethod
    def _infer_data_flow_dependencies(graph: ProcessGraph) -> List[ImplicitFlow]:
        """Infer control flow from data flow dependencies."""
        flows: List[ImplicitFlow] = []

        # If task A outputs data that task B uses, infer control flow
        data_edges = [e for e in graph.edges if e.type == EdgeType.DATA_FLOW]

        for data_edge in data_edges:
            source = graph.get_node(data_edge.source_id)
            target = graph.get_node(data_edge.target_id)

            if source and target:
                # If source is task and target is task, infer control flow
                if source.type == NodeType.TASK and target.type == NodeType.TASK:
                    # Check if control flow already exists
                    existing_flow = any(
                        e.source_id == source.id
                        and e.target_id == target.id
                        and e.type == EdgeType.CONTROL_FLOW
                        for e in graph.edges
                    )

                    if not existing_flow:
                        flow = ImplicitFlow(
                            source_id=source.id,
                            target_id=target.id,
                            reason="Inferred from data dependency",
                            confidence=0.5,
                        )
                        flows.append(flow)

        return flows


# ===========================
# Full Stage 4 Pipeline
# ===========================


class SemanticGraphConstructionPipeline:
    """Complete semantic graph construction pipeline (Stage 4) with KB integration."""

    def __init__(self, enable_kb: bool = True):
        """
        Initialize pipeline.

        Args:
            enable_kb: Whether to enable KB integration
        """
        self.graph_builder = ProcessGraphBuilder(enable_kb=enable_kb)
        self.lane_builder = LaneStructureBuilder(enable_kb=enable_kb)
        self.validator = GraphValidator()
        self.flow_inferrer = ImplicitFlowInferrer(enable_kb=enable_kb)
        self.enable_kb = enable_kb

    def construct_graph(
        self,
        extraction_result: ExtractionResult,
        actor_profiles: Dict[str, ActorProfile],
        domain: Optional[DomainType] = None,
    ) -> Tuple[ProcessGraph, GraphValidationReport, List[ImplicitFlow]]:
        """
        Execute full semantic graph construction pipeline.

        Args:
            extraction_result: Resolved extraction results from Stage 3
            actor_profiles: Actor profiles with activity assignments
            domain: Domain type for KB-aware decisions

        Returns:
            (process_graph, validation_report, inferred_flows)
        """
        # 1. Build initial graph from extraction (with domain-aware layout if KB enabled)
        graph = self.graph_builder.build_from_extraction(
            extraction_result, actor_profiles, domain=domain
        )

        # 2. Add lane structure (with domain-aware decisions if KB enabled)
        graph, lane_structures = self.lane_builder.build_lanes_from_actors(
            graph, actor_profiles, domain=domain
        )

        # 3. Validate graph
        is_valid, issues, metrics = self.validator.validate_graph(graph)

        # 4. Infer implicit flows (with KB pattern awareness if enabled)
        inferred_flows = self.flow_inferrer.infer_implicit_flows(graph, domain=domain)

        # 5. Create validation report
        validation_report = GraphValidationReport(
            graph_id=graph.id,
            is_valid=is_valid,
            metrics=metrics,
            issues=[issue.message for issue in issues if issue.severity == "error"],
            warnings=[issue.message for issue in issues if issue.severity == "warning"],
            suggestions=[
                f"Inferred implicit flow: {flow.reason} ({flow.confidence:.0%} confidence)"
                for flow in inferred_flows
            ],
        )

        return graph, validation_report, inferred_flows


__all__ = [
    "KBGraphEnricher",
    "ProcessGraphBuilder",
    "LaneStructure",
    "LaneStructureBuilder",
    "GraphValidationIssue",
    "GraphValidator",
    "ImplicitFlow",
    "ImplicitFlowInferrer",
    "SemanticGraphConstructionPipeline",
]
