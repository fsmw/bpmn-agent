"""
Process Graph Intermediate Representation

Defines the graph-based intermediate representation used internally
to structure extracted entities and relationships before BPMN XML generation.

Supports graph algorithms for validation, optimization, and refinement.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field


class NodeType(str, Enum):
    """Types of nodes in the process graph."""

    TASK = "task"
    DECISION = "decision"
    START = "start"
    END = "end"
    EVENT = "event"
    SUBPROCESS = "subprocess"
    PARALLEL_FORK = "parallel_fork"
    PARALLEL_JOIN = "parallel_join"
    DATA = "data"
    ACTOR = "actor"


class EdgeType(str, Enum):
    """Types of edges in the process graph."""

    CONTROL_FLOW = "control_flow"  # Sequence flow
    MESSAGE_FLOW = "message_flow"  # Asynchronous communication
    DATA_FLOW = "data_flow"  # Data dependency
    ASSOCIATION = "association"  # Non-control connection
    SWIMLANE = "swimlane"  # Assignment to actor/lane


class GraphNode(BaseModel):
    """Node in the process graph."""

    id: str = Field(..., description="Unique node identifier")
    type: NodeType = Field(..., description="Node type")
    label: str = Field(..., description="Node label/name")

    # BPMN attributes
    bpmn_type: str = Field(..., description="Corresponding BPMN element type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node-specific properties")

    # Position for visualization
    x: Optional[float] = Field(None, description="X coordinate for layout")
    y: Optional[float] = Field(None, description="Y coordinate for layout")

    # Metadata
    is_abstract: bool = Field(False, description="Whether this is an abstract/synthetic node")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GraphEdge(BaseModel):
    """Edge in the process graph."""

    id: str = Field(..., description="Unique edge identifier")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    type: EdgeType = Field(..., description="Edge type")

    # Edge attributes
    label: Optional[str] = Field(None, description="Edge label")
    condition: Optional[str] = Field(None, description="Guard condition (for decisions)")
    is_default: bool = Field(False, description="Whether this is default path")

    # Weight/probability
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Edge weight/probability")

    # Metadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ProcessGraph(BaseModel):
    """Complete process graph structure with optimized indexing."""

    id: str = Field(..., description="Graph identifier")
    name: str = Field(..., description="Process name")
    description: Optional[str] = Field(None, description="Process description")

    # Graph structure
    nodes: List[GraphNode] = Field(default_factory=list, description="All nodes")
    edges: List[GraphEdge] = Field(default_factory=list, description="All edges")

    # Graph properties
    is_acyclic: bool = Field(default=True, description="Whether graph is acyclic (DAG)")
    is_connected: bool = Field(default=True, description="Whether graph is connected")
    has_implicit_parallelism: bool = Field(
        default=False, description="Whether graph uses implicit parallelism"
    )

    # Metrics
    complexity: float = Field(default=0.0, description="Cyclomatic complexity score")

    # Metadata
    version: str = Field(default="1.0", description="Graph version")
    created_timestamp: str = Field(..., description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Internal indexes for O(1) lookups (not serialized)
    _node_index: Dict[str, GraphNode] = {}
    _edge_index: Dict[str, GraphEdge] = {}
    _outgoing_edges: Dict[str, List[GraphEdge]] = {}
    _incoming_edges: Dict[str, List[GraphEdge]] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initialize graph and build indexes."""
        super().__init__(**data)
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build internal indexes for O(1) lookups."""
        # Clear existing indexes
        self._node_index = {}
        self._edge_index = {}
        self._outgoing_edges = {}
        self._incoming_edges = {}

        # Build node index
        for node in self.nodes:
            self._node_index[node.id] = node

        # Build edge index and adjacency lists
        for edge in self.edges:
            self._edge_index[edge.id] = edge

            # Add to outgoing edges
            if edge.source_id not in self._outgoing_edges:
                self._outgoing_edges[edge.source_id] = []
            self._outgoing_edges[edge.source_id].append(edge)

            # Add to incoming edges
            if edge.target_id not in self._incoming_edges:
                self._incoming_edges[edge.target_id] = []
            self._incoming_edges[edge.target_id].append(edge)

    # Helper methods
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID - O(1) lookup."""
        return self._node_index.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        """Get edge by ID - O(1) lookup."""
        return self._edge_index.get(edge_id)

    def get_incoming_edges(self, node_id: str) -> List[GraphEdge]:
        """Get all edges targeting this node - O(1) lookup."""
        return self._incoming_edges.get(node_id, [])

    def get_outgoing_edges(self, node_id: str) -> List[GraphEdge]:
        """Get all edges from this node - O(1) lookup."""
        return self._outgoing_edges.get(node_id, [])

    def get_start_nodes(self) -> List[GraphNode]:
        """Get start nodes (no incoming control flow edges)."""
        start_ids = set()
        for node in self.nodes:
            if node.type == NodeType.START:
                start_ids.add(node.id)
            elif node.type in (NodeType.TASK, NodeType.EVENT, NodeType.DECISION):
                # Check for incoming control flow edges only (not swimlane/data/association)
                control_flow_incoming = [
                    e for e in self.get_incoming_edges(node.id) if e.type == EdgeType.CONTROL_FLOW
                ]
                if not control_flow_incoming:
                    start_ids.add(node.id)
        return [n for n in self.nodes if n.id in start_ids]

    def get_end_nodes(self) -> List[GraphNode]:
        """Get end nodes (no outgoing control flow edges)."""
        end_ids = set()
        for node in self.nodes:
            if node.type == NodeType.END:
                end_ids.add(node.id)
            elif node.type in (NodeType.TASK, NodeType.EVENT, NodeType.DECISION):
                # Check for outgoing control flow edges only (not swimlane/data/association)
                control_flow_outgoing = [
                    e for e in self.get_outgoing_edges(node.id) if e.type == EdgeType.CONTROL_FLOW
                ]
                if not control_flow_outgoing:
                    end_ids.add(node.id)
        return [n for n in self.nodes if n.id in end_ids]

    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes if n.type == node_type]

    def get_decision_nodes(self) -> List[GraphNode]:
        """Get all decision/gateway nodes."""
        return self.get_nodes_by_type(NodeType.DECISION)

    def get_parallel_paths(self) -> List[List[GraphNode]]:
        """
        Identify all parallel execution paths in the graph.

        Returns list of path sequences, where each path is a sequence of nodes
        representing one possible execution path through the graph.
        Handles parallelism by exploring all branches from fork nodes.
        """
        paths: List[List[GraphNode]] = []

        for start_node in self.get_start_nodes():
            # Find all complete paths from this start node
            all_paths = self._find_all_paths(start_node.id, set())
            paths.extend(all_paths)

        return paths

    def _find_all_paths(self, node_id: str, visited: Set[str]) -> List[List[GraphNode]]:
        """
        Find all execution paths from a given node.

        Recursively explores all outgoing edges, building complete execution paths.
        Handles parallel branches by exploring all options.
        """
        if node_id in visited:
            # Cycle detected - return empty to prevent infinite loops
            return []

        node = self.get_node(node_id)
        if not node:
            return []

        # Mark as visited for this branch
        new_visited = visited | {node_id}

        # Get all outgoing edges
        outgoing = self.get_outgoing_edges(node_id)

        if not outgoing:
            # End of path - return the node as a complete path
            return [[node]]

        # For each outgoing edge, find all continuations
        all_paths: List[List[GraphNode]] = []

        for edge in outgoing:
            # Recursively find all paths from the target node
            target_paths = self._find_all_paths(edge.target_id, new_visited)

            # Prepend current node to each continuation path
            for path in target_paths:
                all_paths.append([node] + path)

        return all_paths

    def get_data_dependencies(self, node_id: str) -> Tuple[List[str], List[str]]:
        """
        Get data dependencies for a node.

        Returns (inputs, outputs) - IDs of data nodes this node depends on.
        """
        incoming_data = []
        outgoing_data = []

        for edge in self.edges:
            if edge.type == EdgeType.DATA_FLOW:
                if edge.target_id == node_id:
                    incoming_data.append(edge.source_id)
                elif edge.source_id == node_id:
                    outgoing_data.append(edge.target_id)

        return incoming_data, outgoing_data

    def get_actor_assignments(self, node_id: str) -> List[str]:
        """Get actor/swimlane assignments for a node."""
        assignments = []
        for edge in self.edges:
            if edge.type == EdgeType.SWIMLANE and edge.target_id == node_id:
                assignments.append(edge.source_id)
        return assignments

    def validate_structure(self) -> Tuple[bool, List[str]]:
        """
        Validate graph structure for common issues.

        Returns (is_valid, error_messages).
        """
        errors: List[str] = []

        # Check for orphaned nodes
        node_ids = {n.id for n in self.nodes}
        edge_nodes = set()
        for edge in self.edges:
            edge_nodes.add(edge.source_id)
            edge_nodes.add(edge.target_id)

        orphaned = node_ids - edge_nodes
        if orphaned:
            errors.append(f"Orphaned nodes (no edges): {orphaned}")

        # Check for dangling edges
        for edge in self.edges:
            if edge.source_id not in node_ids:
                errors.append(f"Edge {edge.id} references non-existent source: {edge.source_id}")
            if edge.target_id not in node_ids:
                errors.append(f"Edge {edge.id} references non-existent target: {edge.target_id}")

        # Check for multiple start nodes (only if not explicit)
        starts = self.get_start_nodes()
        if len(starts) > 1:
            # This is only an error if they're not connected by initial fork
            has_fork = any(n.type == NodeType.PARALLEL_FORK for n in starts)
            if not has_fork:
                errors.append("Multiple disconnected start points found")

        # Check for unreachable end nodes
        for end_node in self.get_end_nodes():
            if not self._is_reachable_from_start(end_node.id):
                errors.append(f"End node '{end_node.label}' is unreachable from start")

        return len(errors) == 0, errors

    def _is_reachable_from_start(self, target_id: str, visited: Optional[Set[str]] = None) -> bool:
        """Check if target node is reachable from any start node."""
        if visited is None:
            visited = set()

        for start_node in self.get_start_nodes():
            if start_node.id == target_id:
                return True
            if self._can_reach(start_node.id, target_id, visited.copy()):
                return True

        return False

    def _can_reach(self, from_id: str, to_id: str, visited: Set[str]) -> bool:
        """Check if we can reach 'to_id' from 'from_id' via control flow edges."""
        if from_id == to_id:
            return True
        if from_id in visited:
            return False

        visited.add(from_id)

        # Only traverse control flow edges for reachability analysis
        control_flow_edges = [
            e for e in self.get_outgoing_edges(from_id) if e.type == EdgeType.CONTROL_FLOW
        ]
        for edge in control_flow_edges:
            if self._can_reach(edge.target_id, to_id, visited):
                return True

        return False


class GraphMetrics(BaseModel):
    """Quality metrics for a process graph."""

    node_count: int = Field(..., description="Total number of nodes")
    edge_count: int = Field(..., description="Total number of edges")
    density: float = Field(..., description="Graph density (0-1)")
    cyclomatic_complexity: float = Field(..., description="Cyclomatic complexity")
    average_node_degree: float = Field(..., description="Average degree per node")
    longest_path_length: int = Field(..., description="Longest execution path length")
    num_parallel_paths: int = Field(..., description="Number of parallel execution paths")
    num_decision_points: int = Field(..., description="Number of decision gateways")
    num_data_dependencies: int = Field(..., description="Number of data flow edges")

    # Quality scores
    structural_completeness: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Score indicating graph is complete (0-1)"
    )
    ambiguity_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Score indicating ambiguity in graph (0-1)"
    )


class GraphValidationReport(BaseModel):
    """Validation report for a process graph."""

    graph_id: str = Field(..., description="Graph ID")
    is_valid: bool = Field(..., description="Whether graph is valid")
    metrics: GraphMetrics = Field(..., description="Graph metrics")

    issues: List[str] = Field(default_factory=list, description="Structural issues found")
    warnings: List[str] = Field(default_factory=list, description="Warnings and recommendations")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")


__all__ = [
    "NodeType",
    "EdgeType",
    "GraphNode",
    "GraphEdge",
    "ProcessGraph",
    "GraphMetrics",
    "GraphValidationReport",
]
