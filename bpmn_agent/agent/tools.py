"""
Agent Tools and Utilities

Analysis, validation, and refinement tools for the BPMN Agent.
Provides utilities for graph analysis, XML validation, and process refinement.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from bpmn_agent.models.extraction import ExtractionResultWithErrors
from bpmn_agent.models.graph import ProcessGraph

logger = logging.getLogger(__name__)


@dataclass
class GraphAnalysisResult:
    """Result from graph analysis."""

    total_nodes: int
    total_edges: int
    isolated_nodes: List[str]
    cyclic_paths: List[List[str]]
    disconnected_components: int
    complex_gateways: List[str]
    implicit_joins_needed: List[Tuple[str, str]]
    issues: List[str]
    warnings: List[str]


class GraphAnalyzer:
    """Analyzes process graphs for structural issues and patterns."""

    @staticmethod
    def analyze_structure(graph: ProcessGraph) -> GraphAnalysisResult:
        """Perform comprehensive graph structural analysis.

        Args:
            graph: ProcessGraph to analyze

        Returns:
            GraphAnalysisResult with findings
        """
        issues = []
        warnings = []

        # Analyze nodes
        isolated_nodes = GraphAnalyzer._find_isolated_nodes(graph)
        if isolated_nodes:
            issues.append(f"Found {len(isolated_nodes)} isolated nodes: {isolated_nodes}")

        # Analyze cycles
        cyclic_paths = GraphAnalyzer._detect_cycles(graph)
        if cyclic_paths:
            warnings.append(f"Found {len(cyclic_paths)} cyclic paths (may be intentional)")

        # Analyze connectivity
        disconnected = GraphAnalyzer._count_disconnected_components(graph)
        if disconnected > 1:
            issues.append(f"Graph has {disconnected} disconnected components")

        # Analyze gateways
        complex_gateways = GraphAnalyzer._find_complex_gateways(graph)

        # Detect implicit joins
        implicit_joins = GraphAnalyzer._detect_implicit_joins(graph)
        if implicit_joins:
            warnings.append(f"Found {len(implicit_joins)} potential implicit joins")

        return GraphAnalysisResult(
            total_nodes=len(graph.nodes),
            total_edges=len(graph.edges),
            isolated_nodes=isolated_nodes,
            cyclic_paths=cyclic_paths,
            disconnected_components=disconnected,
            complex_gateways=complex_gateways,
            implicit_joins_needed=implicit_joins,
            issues=issues,
            warnings=warnings,
        )

    @staticmethod
    def _find_isolated_nodes(graph: ProcessGraph) -> List[str]:
        """Find nodes with no incoming or outgoing edges.

        Args:
            graph: ProcessGraph to analyze

        Returns:
            List of isolated node IDs
        """
        isolated = []

        for node in graph.nodes:
            node_id = node.id
            incoming = [e for e in graph.edges if e.target_id == node_id]
            outgoing = [e for e in graph.edges if e.source_id == node_id]

            # Start events can have no incoming, end events can have no outgoing
            node_type = node.type if hasattr(node, "type") else None

            if not incoming and not outgoing and node_type not in ["start_event", "end_event"]:
                isolated.append(node_id)
            elif not incoming and node_type != "start_event":
                isolated.append(f"{node_id} (no incoming)")
            elif not outgoing and node_type != "end_event":
                isolated.append(f"{node_id} (no outgoing)")

        return isolated

    @staticmethod
    def _detect_cycles(graph: ProcessGraph) -> List[List[str]]:
        """Detect cyclic paths in the graph.

        Args:
            graph: ProcessGraph to analyze

        Returns:
            List of cyclic paths (each path is a list of node IDs)
        """
        cycles = []
        visited_global = set()

        def dfs_cycle(node_id: str, path: List[str], visited: Set[str]) -> None:
            """DFS to find cycles."""
            if node_id in visited:
                # Found a cycle
                if node_id in path:
                    cycle_start_idx = path.index(node_id)
                    cycle = path[cycle_start_idx:] + [node_id]
                    cycles.append(cycle)
                return

            if node_id in visited_global:
                return

            visited.add(node_id)
            path.append(node_id)

            # Find outgoing edges
            for edge in graph.edges:
                if edge.source_id == node_id:
                    dfs_cycle(edge.target_id, path.copy(), visited.copy())

        # Start DFS from each node
        for node in graph.nodes:
            node_id = node.id
            if node_id not in visited_global:
                dfs_cycle(node_id, [], set())
                visited_global.add(node_id)

        return cycles

    @staticmethod
    def _count_disconnected_components(graph: ProcessGraph) -> int:
        """Count disconnected components in the graph.

        Args:
            graph: ProcessGraph to analyze

        Returns:
            Number of disconnected components
        """
        if not graph.nodes:
            return 0

        visited = set()
        components = 0

        def bfs(start_node: str) -> None:
            """BFS to find connected component."""
            queue = [start_node]
            visited.add(start_node)

            while queue:
                node_id = queue.pop(0)

                # Find adjacent nodes
                for edge in graph.edges:
                    if edge.source_id == node_id and edge.target_id not in visited:
                        visited.add(edge.target_id)
                        queue.append(edge.target_id)
                    elif edge.target_id == node_id and edge.source_id not in visited:
                        visited.add(edge.source_id)
                        queue.append(edge.source_id)

        # Count components
        for node in graph.nodes:
            node_id = node.id
            if node_id not in visited:
                bfs(node_id)
                components += 1

        return components

    @staticmethod
    def _find_complex_gateways(graph: ProcessGraph) -> List[str]:
        """Find gateways with complex branching patterns.

        Args:
            graph: ProcessGraph to analyze

        Returns:
            List of complex gateway IDs
        """
        complex_gateways = []

        for node in graph.nodes:
            node_id = node.id
            node_type = node.type if hasattr(node, "type") else None

            if node_type and "gateway" in str(node_type).lower():
                # Count outgoing branches
                outgoing = sum(1 for e in graph.edges if e.source_id == node_id)
                incoming = sum(1 for e in graph.edges if e.target_id == node_id)

                # Complex if many branches
                if outgoing > 2 or (outgoing > 1 and incoming > 1):
                    complex_gateways.append(node_id)

        return complex_gateways

    @staticmethod
    def _detect_implicit_joins(graph: ProcessGraph) -> List[Tuple[str, str]]:
        """Detect implicit joins (gates with missing explicit join).

        Args:
            graph: ProcessGraph to analyze

        Returns:
            List of (source_gateway, implicit_join) tuples
        """
        implicit_joins = []

        for node in graph.nodes:
            node_id = node.id
            node_type = node.type if hasattr(node, "type") else None

            if node_type and "gateway" in str(node_type).lower():
                # Check if this is a split gateway (multiple outgoing)
                outgoing = [e for e in graph.edges if e.source_id == node_id]

                if len(outgoing) > 1:
                    # Potential split, look for corresponding join
                    split_target_nodes = {e.target_id for e in outgoing}

                    # Find nodes that might need joining
                    for target_node_id in split_target_nodes:
                        # Check if all paths from target eventually converge
                        # Simplified check: look for common descendant
                        descendants = GraphAnalyzer._get_descendants(graph, target_node_id)

                        if descendants:
                            # Find common node among all split paths
                            join_node = GraphAnalyzer._find_common_ancestor(
                                graph, split_target_nodes
                            )
                            if join_node and join_node != node_id:
                                implicit_joins.append((node_id, join_node))
                                break

        return implicit_joins

    @staticmethod
    def _get_descendants(graph: ProcessGraph, node_id: str) -> Set[str]:
        """Get all descendants of a node.

        Args:
            graph: ProcessGraph
            node_id: Starting node ID

        Returns:
            Set of descendant node IDs
        """
        descendants = set()
        queue = [node_id]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)

            for edge in graph.edges:
                if edge.source_id == current and edge.target_id not in visited:
                    descendants.add(edge.target_id)
                    queue.append(edge.target_id)

        return descendants

    @staticmethod
    def _find_common_ancestor(graph: ProcessGraph, node_ids: Set[str]) -> Optional[str]:
        """Find common ancestor of a set of nodes.

        Args:
            graph: ProcessGraph
            node_ids: Set of node IDs

        Returns:
            Common ancestor node ID or None
        """
        if not node_ids:
            return None

        # Find ancestors for each node
        ancestors_sets = []
        for node_id in node_ids:
            ancestors = {node_id}
            queue = [node_id]

            while queue:
                current = queue.pop(0)
                for edge in graph.edges:
                    if edge.target_id == current:
                        ancestors.add(edge.source_id)
                        queue.append(edge.source_id)

            ancestors_sets.append(ancestors)

        # Find intersection
        if ancestors_sets:
            common = ancestors_sets[0]
            for ancestor_set in ancestors_sets[1:]:
                common = common.intersection(ancestor_set)

            # Filter out the original nodes
            common = common - node_ids
            return list(common)[0] if common else None

        return None


class XMLValidator:
    """Validates BPMN XML output."""

    @staticmethod
    def validate_xml_structure(xml_str: str) -> Tuple[bool, List[str]]:
        """Validate basic XML structure.

        Args:
            xml_str: XML string to validate

        Returns:
            (is_valid, errors) tuple
        """
        errors = []

        try:
            from lxml import etree

            parser = etree.XMLParser(remove_blank_text=True)
            etree.fromstring(xml_str.encode(), parser=parser)

        except etree.XMLSyntaxError as e:
            errors.append(f"XML Syntax Error: {str(e)}")
        except Exception as e:
            errors.append(f"XML Parsing Error: {str(e)}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_bpmn_elements(xml_str: str) -> Tuple[bool, List[str]]:
        """Validate that required BPMN elements are present.

        Args:
            xml_str: XML string to validate

        Returns:
            (is_valid, warnings) tuple
        """
        warnings = []

        try:
            from lxml import etree

            root = etree.fromstring(xml_str.encode())

            # Check for required elements
            namespaces = {
                "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
            }

            processes = root.findall(".//bpmn:process", namespaces)
            if not processes:
                warnings.append("No process element found")

            for process in processes:
                start_events = process.findall("bpmn:startEvent", namespaces)
                end_events = process.findall("bpmn:endEvent", namespaces)

                if not start_events:
                    warnings.append("Process has no start event")
                if not end_events:
                    warnings.append("Process has no end event")

        except Exception as e:
            warnings.append(f"Validation error: {str(e)}")

        return len(warnings) == 0, warnings


class ProcessRefinementTools:
    """Tools for refining and improving process definitions."""

    @staticmethod
    def suggest_clarification_questions(
        extraction_result: ExtractionResultWithErrors,
        graph: ProcessGraph,
    ) -> List[str]:
        """Generate questions to clarify ambiguous extractions.

        Args:
            extraction_result: Extraction results
            graph: Process graph

        Returns:
            List of suggested clarification questions
        """
        questions = []

        # Check for low-confidence extractions
        low_confidence = [
            e
            for e in extraction_result.entities
            if hasattr(e, "confidence") and e.confidence in ["low", "medium"]
        ]
        if low_confidence:
            entity_names = [e.name for e in low_confidence[:3]]
            questions.append(
                f"The following entities had low confidence: {', '.join(entity_names)}. "
                "Could you confirm these are correct?"
            )

        # Check for missing elements
        if len(graph.nodes) < 3:
            questions.append(
                "The process appears to be very simple. Are there any additional steps or decision points?"
            )

        # Check for gateways without conditions
        gateways = [
            n
            for n in graph.nodes
            if hasattr(n, "type") and "gateway" in str(n.type).lower()
        ]
        if gateways:
            questions.append(
                "Could you describe the conditions or criteria for each decision point?"
            )

        # Check for actors/lanes
        lanes = [n for n in graph.nodes if hasattr(n, "type") and n.type == "lane"]
        if not lanes:
            questions.append("Which roles or departments are involved in this process?")

        return questions

    @staticmethod
    def suggest_improvements(graph: GraphAnalysisResult) -> List[str]:
        """Suggest improvements to the process graph.

        Args:
            graph: Graph analysis result

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Suggest fixes for issues
        if graph.isolated_nodes:
            suggestions.append(
                f"Consider connecting the isolated nodes: {', '.join(graph.isolated_nodes)}"
            )

        if graph.cyclic_paths:
            suggestions.append("The process contains loops. Verify these are intentional.")

        if graph.disconnected_components > 1:
            suggestions.append(
                f"The process has {graph.disconnected_components} disconnected parts. "
                "Consider if they should be connected."
            )

        if graph.implicit_joins_needed:
            join_count = len(graph.implicit_joins_needed)
            suggestions.append(
                f"Found {join_count} implicit joins that may need explicit gateways."
            )

        # Structural suggestions
        if graph.total_edges == 0:
            suggestions.append("No flows detected. Verify that relationships were captured.")

        complexity = graph.total_nodes
        if complexity > 20:
            suggestions.append(
                f"This process is complex ({complexity} nodes). "
                "Consider breaking it into sub-processes."
            )

        return suggestions
