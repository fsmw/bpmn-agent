"""
Graph Analysis Tools for BPMN Agent

Provides sophisticated analysis capabilities for process graphs:
- Pattern detection and anomaly identification
- Structural analysis and validation
- Optimization suggestions
- Quality metrics calculation
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from bpmn_agent.models.extraction import ExtractionResultWithErrors
from bpmn_agent.models.graph import GraphNode, ProcessGraph

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Types of graph anomalies detected."""

    ISOLATED_NODE = "isolated_node"
    ORPHANED_NODE = "orphaned_node"
    DEAD_END = "dead_end"
    MULTI_INCOMING_GATEWAY = "multi_incoming_gateway"
    MULTI_OUTGOING_EVENT = "multi_outgoing_event"
    CYCLE_DETECTED = "cycle_detected"
    UNCLOSED_GATEWAY = "unclosed_gateway"
    INVALID_FLOW = "invalid_flow"
    LANE_ASSIGNMENT_ISSUE = "lane_assignment_issue"


class StructureType(str, Enum):
    """Type of graph structures detected."""

    LINEAR = "linear"
    PARALLEL = "parallel"
    CHOICE = "choice"
    LOOP = "loop"
    HIERARCHICAL = "hierarchical"
    COMPLEX = "complex"


@dataclass
class GraphAnomaly:
    """Detected graph anomaly."""

    anomaly_type: AnomalyType
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    description: str = ""
    severity: str = "medium"  # low, medium, high, critical
    suggestion: str = ""
    location: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphStructure:
    """Detected graph structure."""

    structure_type: StructureType
    nodes: Set[str] = field(default_factory=set)
    edges: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphAnalysisResult:
    """Complete graph analysis result."""

    graph_id: str
    total_nodes: int
    total_edges: int
    anomalies: List[GraphAnomaly] = field(default_factory=list)
    structures: List[GraphStructure] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    complexity_level: str = "simple"


class GraphAnalyzer:
    """Advanced graph analysis capabilities."""

    def __init__(self):
        """Initialize graph analyzer."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def analyze_graph_structure(
        self, graph: ProcessGraph, extraction_result: Optional[ExtractionResultWithErrors] = None
    ) -> GraphAnalysisResult:
        """Perform comprehensive graph structure analysis.

        Args:
            graph: Process graph to analyze
            extraction_result: Original extraction result for context

        Returns:
            Complete analysis result
        """
        self.logger.info(f"Analyzing graph structure: {graph.id}")

        result = GraphAnalysisResult(
            graph_id=graph.id, total_nodes=len(graph.nodes), total_edges=len(graph.edges)
        )

        # Basic metrics
        result.metrics = self._calculate_basic_metrics(graph)

        # Detect anomalies
        result.anomalies = self._detect_anomalies(graph)

        # Identify structures
        result.structures = self._identify_structures(graph)

        # Calculate quality score
        result.quality_score = self._calculate_quality_score(result)

        # Generate suggestions
        result.suggestions = self._generate_suggestions(result)

        # Determine complexity
        result.complexity_level = self._determine_complexity(result)

        self.logger.info(
            f"Analysis complete: {len(result.anomalies)} anomalies, quality score: {result.quality_score:.2f}"
        )

        return result

    def find_isolated_nodes(self, graph: ProcessGraph) -> List[str]:
        """Find nodes with no incoming or outgoing edges.

        Args:
            graph: Process graph to analyze

        Returns:
            List of isolated node IDs
        """
        isolated = []

        # Find nodes mentioned in edges
        connected_nodes = set()
        for edge in graph.edges:
            connected_nodes.add(edge.source_id)
            connected_nodes.add(edge.target_id)

        # Find nodes that are not connected (but allow start/end events)
        for node in graph.nodes:
            if node.id not in connected_nodes:
                # Allow isolated start/end events
                if node.type.lower() not in ["start", "end"]:
                    isolated.append(node.id)

        self.logger.info(f"Found {len(isolated)} isolated nodes")
        return isolated

    def find_orphaned_nodes(self, graph: ProcessGraph) -> List[str]:
        """Find nodes that can't be reached from a start event.

        Args:
            graph: Process graph to analyze

        Returns:
            List of orphaned node IDs
        """
        # Find all start events
        start_nodes = [n for n in graph.nodes if n.type.lower() == "start"]

        if not start_nodes:
            # If no start events, all non-start nodes are orphaned
            return [n.id for n in graph.nodes if n.type.lower() != "start"]

        def get_reachable_nodes(start_id: str) -> Set[str]:
            """Get all nodes reachable from a start node."""
            reachable = set()
            to_visit = deque([start_id])

            adj = self._build_adjacency_list(graph)

            while to_visit:
                current = to_visit.popleft()
                if current not in reachable:
                    reachable.add(current)
                    to_visit.extend(adj.get(current, []))

            return reachable

        # Get all reachable nodes from any start event
        reachable = set()
        for start in start_nodes:
            reachable.update(get_reachable_nodes(start.id))

        # Orphaned nodes are those not reachable
        orphaned = [n.id for n in graph.nodes if n.id not in reachable]

        self.logger.info(f"Found {len(orphaned)} orphaned nodes")
        return orphaned

    def detect_cycles(self, graph: ProcessGraph) -> List[List[str]]:
        """Detect cycles in the graph using DFS.

        Args:
            graph: Process graph to analyze

        Returns:
            List of detected cycles (each as list of node IDs)
        """
        adj = self._build_adjacency_list(graph)
        all_nodes = {n.id for n in graph.nodes}

        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node_id: str) -> bool:
            """DFS with cycle detection."""
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for neighbor in adj.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle.copy())
                    return True

            rec_stack.remove(node_id)
            path.pop()
            return False

        for node in all_nodes:
            if node not in visited:
                dfs(node)

        self.logger.info(f"Detected {len(cycles)} cycles in graph")
        return cycles

    def suggest_implicit_joins(self, graph: ProcessGraph) -> List[Dict[str, Any]]:
        """Suggest implicit gateway joins to close open branches.

        Args:
            graph: Process graph to analyze

        Returns:
            List of suggested joins with metadata
        """
        suggestions = []
        adj = self._build_adjacency_list(graph)

        # Find gateways with multiple outgoing edges
        Gateways = [n for n in graph.nodes if "gateway" in n.type.lower()]

        for gateway in Gateways:
            outgoing = [e for e in graph.edges if e.source_id == gateway.id]
            if len(outgoing) > 1:
                # Analyze downstream flows
                destinations = set()
                for edge in outgoing:
                    downstream = self._trace_downstream(graph, edge.target_id)
                    destinations.update(downstream)

                # Check if all paths eventually converge
                if len(destinations) > 1:
                    # Look for common downstream nodes
                    converge_points = self._find_convergence_points(
                        graph, [e.target_id for e in outgoing]
                    )

                    if converge_points:
                        suggestion = {
                            "type": "implicit_join",
                            "gateway_id": gateway.id,
                            "gateway_type": gateway.type,
                            "convergence_points": list(converge_points),
                            "branches": [e.target_id for e in outgoing],
                            "reason": "Multiple branches from gateway could converge",
                            "confidence": 0.8,
                        }
                        suggestions.append(suggestion)

        # Suggest joins for dead-end parallel flows
        for edge in graph.edges:
            target_node = self._get_node(graph, edge.target_id)
            if target_node and not self._has_outgoing_edges(graph, edge.target_id):
                # Check if this node is part of a parallel flow that should merge
                incoming = self._count_incoming_edges(graph, edge.target_id)
                if incoming > 1:
                    # This node has multiple incoming but no outgoing - potential merge point
                    upstream_sources = [
                        e.source_id for e in graph.edges if e.target_id == edge.target_id
                    ]
                    if len(upstream_sources) > 1:
                        suggestion = {
                            "type": "merge_point",
                            "node_id": edge.target_id,
                            "sources": upstream_sources,
                            "reason": "Multiple incoming flows but no outgoing flow",
                            "suggestion": "Add parallel gateway or merge node",
                            "confidence": 0.7,
                        }
                        suggestions.append(suggestion)

        self.logger.info(f"Generated {len(suggestions)} implicit join suggestions")
        return suggestions

    def _build_adjacency_list(self, graph: ProcessGraph) -> Dict[str, List[str]]:
        """Build adjacency list for graph traversal."""
        adj = defaultdict(list)
        for edge in graph.edges:
            adj[edge.source_id].append(edge.target_id)
        return dict(adj)

    def _trace_downstream(self, graph: ProcessGraph, node_id: str) -> Set[str]:
        """Trace all downstream nodes from a given node."""
        reachable = set()
        to_visit = deque([node_id])
        adj = self._build_adjacency_list(graph)

        while to_visit:
            current = to_visit.popleft()
            if current not in reachable:
                reachable.add(current)
                to_visit.extend(adj.get(current, []))

        return reachable

    def _find_convergence_points(self, graph: ProcessGraph, start_nodes: List[str]) -> Set[str]:
        """Find nodes that can be reached from multiple start nodes."""
        downstream_sets = []
        for start in start_nodes:
            downstream = self._trace_downstream(graph, start)
            downstream_sets.append(downstream)

        # Find intersection of downstream sets (convergence points)
        convergence = downstream_sets[0].copy()
        for downstream in downstream_sets[1:]:
            convergence.intersection_update(downstream)

        return convergence

    def _get_node(self, graph: ProcessGraph, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        for node in graph.nodes:
            if node.id == node_id:
                return node
        return None

    def _has_outgoing_edges(self, graph: ProcessGraph, node_id: str) -> bool:
        """Check if node has outgoing edges."""
        return any(edge.source_id == node_id for edge in graph.edges)

    def _count_incoming_edges(self, graph: ProcessGraph, node_id: str) -> int:
        """Count incoming edges for a node."""
        return sum(1 for edge in graph.edges if edge.target_id == node_id)

    def _calculate_basic_metrics(self, graph: ProcessGraph) -> Dict[str, Any]:
        """Calculate basic graph metrics."""
        node_types: dict[str, int] = defaultdict(int)
        for node in graph.nodes:
            node_types[node.type] += 1

        edge_types: dict[str, int] = defaultdict(int)
        for edge in graph.edges:
            edge_types[edge.type] += 1

        return {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "node_degree_avg": (
                sum(
                    sum(1 for e in graph.edges if e.source_id == n.id or e.target_id == n.id)
                    for n in graph.nodes
                )
                / len(graph.nodes)
                if graph.nodes
                else 0
            ),
            "is_connected": len(graph.edges) > 0
            and any(
                self._get_node(graph, n.id) is not None
                and self._get_node(graph, n.id).type.lower() == "start"  # type: ignore[union-attr]
                for n in graph.nodes
            ),
        }

    def _detect_anomalies(self, graph: ProcessGraph) -> List[GraphAnomaly]:
        """Detect various graph anomalies."""
        anomalies = []

        # Isolated nodes
        isolated = self.find_isolated_nodes(graph)
        for node_id in isolated:
            anomalies.append(
                GraphAnomaly(
                    anomaly_type=AnomalyType.ISOLATED_NODE,
                    node_id=node_id,
                    description=f"Node {node_id} is not connected to any edges",
                    severity="medium",
                    suggestion="Connect node to process flow or remove if unused",
                )
            )

        # Orphaned nodes
        orphaned = self.find_orphaned_nodes(graph)
        for node_id in orphaned[:10]:  # Limit to first 10
            anomalies.append(
                GraphAnomaly(
                    anomaly_type=AnomalyType.ORPHANED_NODE,
                    node_id=node_id,
                    description=f"Node {node_id} cannot be reached from any start event",
                    severity="high",
                    suggestion="Add connecting edge from existing process or make accessible from start",
                )
            )

        # Dead ends
        dead_ends = self._find_dead_ends(graph)
        for node_id in dead_ends:
            node = self._get_node(graph, node_id)
            if node and node.type.lower() not in ["end"]:
                severity = "critical" if node.type.lower() == "task" else "medium"
                anomalies.append(
                    GraphAnomaly(
                        anomaly_type=AnomalyType.DEAD_END,
                        node_id=node_id,
                        description=f"Node {node_id} has no outgoing edges but is not an end event",
                        severity=severity,
                        suggestion="Add outgoing edges or convert to end event",
                    )
                )

        # Cycles
        cycles = self.detect_cycles(graph)
        for cycle in cycles:
            anomalies.append(
                GraphAnomaly(
                    anomaly_type=AnomalyType.CYCLE_DETECTED,
                    description=f"Cycle detected: {' -> '.join(cycle)}",
                    severity="medium",
                    suggestion="Review cycle logic, consider using timer events or error handling",
                    location={"cycle": cycle},
                )
            )

        return anomalies

    def _find_dead_ends(self, graph: ProcessGraph) -> List[str]:
        """Find nodes with no outgoing edges (except end events)."""
        dead_ends = []
        nodes_with_outgoing = {e.source_id for e in graph.edges}

        for node in graph.nodes:
            if node.id not in nodes_with_outgoing:
                if node.type.lower() not in ["end"]:
                    dead_ends.append(node.id)

        return dead_ends

    def _identify_structures(self, graph: ProcessGraph) -> List[GraphStructure]:
        """Identify common graph structures."""
        structures = []

        # Linear structure check
        if self._is_linear(graph):
            structures.append(
                GraphStructure(
                    structure_type=StructureType.LINEAR,
                    confidence=0.9,
                    description="Simple linear process flow",
                )
            )

        # Parallel structure check
        parallel_nodes = self._find_parallel_structures(graph)
        if parallel_nodes:
            structures.append(
                GraphStructure(
                    structure_type=StructureType.PARALLEL,
                    nodes=set(parallel_nodes),
                    confidence=0.8,
                    description="Parallel processing detected",
                )
            )

        # Choice structure check
        choice_nodes = self._find_choice_structures(graph)
        if choice_nodes:
            structures.append(
                GraphStructure(
                    structure_type=StructureType.CHOICE,
                    nodes=set(choice_nodes),
                    confidence=0.8,
                    description="Decision/choice flow detected",
                )
            )

        # Loop structure check
        if cycles := self.detect_cycles(graph):
            structures.append(
                GraphStructure(
                    structure_type=StructureType.LOOP,
                    nodes=set(node for cycle in cycles for node in cycle),
                    confidence=0.7,
                    description="Loop structure detected",
                )
            )

        return structures

    def _is_linear(self, graph: ProcessGraph) -> bool:
        """Check if graph has simple linear structure."""
        if len(graph.nodes) <= 2:
            return True

        # Count incoming/outgoing edges per node
        incoming_count: Dict[str, int] = defaultdict(int)
        outgoing_count: Dict[str, int] = defaultdict(int)

        for edge in graph.edges:
            outgoing_count[edge.source_id] += 1
            incoming_count[edge.target_id] += 1

        # Linear flow: max 1 incoming and max 1 outgoing per node
        for node in graph.nodes:
            if incoming_count[node.id] > 1 or outgoing_count[node.id] > 1:
                # Allow special cases for gateways
                if "gateway" not in node.type.lower():
                    return False

        return True

    def _find_parallel_structures(self, graph: ProcessGraph) -> List[str]:
        """Find parallel gateway structures."""
        parallel_nodes = []

        # Find AND gateways with multiple outgoing edges
        and_gateways = [
            n
            for n in graph.nodes
            if n.type.lower() == "parallel_gateway" and self._count_outgoing_edges(graph, n.id) > 1
        ]

        for gateway in and_gateways:
            parallel_nodes.append(gateway.id)

        return parallel_nodes

    def _find_choice_structures(self, graph: ProcessGraph) -> List[str]:
        """Find exclusive choice (XOR gateway) structures."""
        choice_nodes = []

        # Find XOR gateways with multiple outgoing edges
        xor_gateways = [
            n
            for n in graph.nodes
            if n.type.lower() in ("exclusive_gateway", "xor_gateway")
            and self._count_outgoing_edges(graph, n.id) > 1
        ]

        for gateway in xor_gateways:
            choice_nodes.append(gateway.id)

        return choice_nodes

    def _count_outgoing_edges(self, graph: ProcessGraph, node_id: str) -> int:
        """Count outgoing edges for a node."""
        return sum(1 for edge in graph.edges if edge.source_id == node_id)

    def _calculate_quality_score(self, result: GraphAnalysisResult) -> float:
        """Calculate overall graph quality score (0-100)."""
        score = 100.0

        # Penalize anomalies
        for anomaly in result.anomalies:
            if anomaly.severity == "critical":
                score -= 20
            elif anomaly.severity == "high":
                score -= 10
            elif anomaly.severity == "medium":
                score -= 5
            elif anomaly.severity == "low":
                score -= 2

        # Bonus for structure quality
        if any(s.structure_type == StructureType.LINEAR for s in result.structures):
            score += 5

        # Ensure score doesn't go negative
        return max(0.0, score)

    def _generate_suggestions(self, result: GraphAnalysisResult) -> List[str]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []

        # Suggestions based on anomalies
        if result.anomalies:
            anomaly_types = {a.anomaly_type for a in result.anomalies}

            if AnomalyType.ISOLATED_NODE in anomaly_types:
                suggestions.append("Review and connect isolated nodes to main process flow")

            if AnomalyType.ORPHANED_NODE in anomaly_types:
                suggestions.append("Ensure all nodes are accessible from start events")

            if AnomalyType.DEAD_END in anomaly_types:
                suggestions.append("Add proper termination paths for all process branches")

            if AnomalyType.CYCLE_DETECTED in anomaly_types:
                suggestions.append(
                    "Review cycles for potential infinite loops or missing exit conditions"
                )

        # Suggestions for improvement
        if result.quality_score < 70:
            suggestions.append(
                "Consider reviewing overall process structure for quality improvements"
            )

        # Add implicit join suggestions
        implicit_joins = [
            s for s in result.structures if s.structure_type == StructureType.PARALLEL
        ]
        if implicit_joins:
            suggestions.append(
                "Consider adding explicit parallel gateways to improve process clarity"
            )

        return suggestions

    def _determine_complexity(self, result: GraphAnalysisResult) -> str:
        """Determine graph complexity level."""
        complexity_score = 0

        # Node count complexity
        if result.total_nodes > 20:
            complexity_score += 2
        elif result.total_nodes > 10:
            complexity_score += 1

        # Edge count complexity
        if result.total_edges > result.total_nodes * 2:
            complexity_score += 2
        elif result.total_edges > result.total_nodes:
            complexity_score += 1

        # Anomaly complexity
        critical_anomalies = sum(1 for a in result.anomalies if a.severity == "critical")
        if critical_anomalies > 3:
            complexity_score += 2
        elif critical_anomalies > 0:
            complexity_score += 1

        # Structure complexity
        structure_types = {s.structure_type for s in result.structures}
        if len(structure_types) > 2:
            complexity_score += 1

        if complexity_score >= 4:
            return "complex"
        elif complexity_score >= 2:
            return "moderate"
        else:
            return "simple"


# Convenience functions for quick analysis
def analyze_graph_structure(graph: ProcessGraph) -> GraphAnalysisResult:
    """Convenience function to analyze graph structure."""
    analyzer = GraphAnalyzer()
    return analyzer.analyze_graph_structure(graph)


def find_isolated_nodes(graph: ProcessGraph) -> List[str]:
    """Convenience function to find isolated nodes."""
    analyzer = GraphAnalyzer()
    return analyzer.find_isolated_nodes(graph)


def find_orphaned_nodes(graph: ProcessGraph) -> List[str]:
    """Convenience function to find orphaned nodes."""
    analyzer = GraphAnalyzer()
    return analyzer.find_orphaned_nodes(graph)


def detect_cycles(graph: ProcessGraph) -> List[List[str]]:
    """Convenience function to detect cycles."""
    analyzer = GraphAnalyzer()
    return analyzer.detect_cycles(graph)


def suggest_implicit_joins(graph: ProcessGraph) -> List[Dict[str, Any]]:
    """Convenience function to suggest implicit joins."""
    analyzer = GraphAnalyzer()
    return analyzer.suggest_implicit_joins(graph)
