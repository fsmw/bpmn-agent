"""
BPMN 2.0 XML Generation Stage (Stage 5)

Converts ProcessGraph intermediate representation to valid BPMN 2.0 XML.
Includes support for visual layout (BPMN Diagram Interchange).

Supports:
- XML generation from graph with pattern compliance
- ID and reference management with pattern tracking
- BPMN diagram interchange (DI) generation
- XSD schema compliance validation
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from lxml import etree
from pydantic import BaseModel, Field

from bpmn_agent.core import Timer
from bpmn_agent.knowledge.loader import PatternLibraryLoader
from bpmn_agent.models.bpmn_elements import (
    BPMNElementType,
    EndEvent,
    ExclusiveGateway,
    IntermediateEvent,
    Lane,
    LaneSet,
    ManualTask,
    ParallelGateway,
    Process,
    ReceiveTask,
    ScriptTask,
    SendTask,
    SequenceFlow,
    ServiceTask,
    StartEvent,
    Task,
    UserTask,
    Waypoint,
)
from bpmn_agent.models.graph import EdgeType, GraphEdge, GraphNode, NodeType, ProcessGraph

logger = logging.getLogger(__name__)

# BPMN 2.0 Namespaces
BPMN_NAMESPACE = "http://www.omg.org/spec/BPMN/20100524/MODEL"
BPMNDI_NAMESPACE = "http://www.omg.org/spec/BPMN/20100524/DI"
DC_NAMESPACE = "http://www.omg.org/spec/DD/20100524/DC"
DI_NAMESPACE = "http://www.omg.org/spec/DD/20100524/DI"

# Element sizes for layout
DEFAULT_TASK_WIDTH = 100
DEFAULT_TASK_HEIGHT = 80
DEFAULT_EVENT_WIDTH = 36
DEFAULT_EVENT_HEIGHT = 36
DEFAULT_GATEWAY_WIDTH = 50
DEFAULT_GATEWAY_HEIGHT = 50
LANE_HEIGHT = 200
LANE_WIDTH = 800

# Spacing
HORIZONTAL_SPACING = 150
VERTICAL_SPACING = 120


class PatternReference(BaseModel):
    """Reference to knowledge base pattern that influenced element generation."""

    pattern_id: str = Field(..., description="Pattern ID from knowledge base")
    pattern_name: str = Field(..., description="Pattern name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    matched_rules: List[str] = Field(default_factory=list, description="Rules that matched")


class IDMapping(BaseModel):
    """Maps graph node IDs to BPMN element IDs."""

    graph_id: str = Field(..., description="Graph node/edge ID")
    bpmn_id: str = Field(..., description="BPMN element ID")
    element_type: str = Field(..., description="BPMN element type")
    pattern_reference: Optional[PatternReference] = Field(
        None, description="Pattern that influenced this element"
    )


class LayoutInfo(BaseModel):
    """Layout information for visual diagram elements."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    width: float = Field(..., description="Element width")
    height: float = Field(..., description="Element height")
    waypoints: Optional[List[Waypoint]] = Field(None, description="Path waypoints for edges")


class BPMNXMLGenerator:
    """Generates valid BPMN 2.0 XML from ProcessGraph representation.

    Supports:
    - All major BPMN element types
    - Proper ID and reference management
    - BPMN Diagram Interchange (visual layout)
    - XSD schema compliance
    - Knowledge base pattern tracking
    """

    def __init__(self, enable_kb: bool = True):
        """Initialize XML generator.

        Args:
            enable_kb: Whether to track knowledge base patterns
        """
        self.enable_kb = enable_kb
        self.kb_loader: Optional[PatternLibraryLoader] = None
        if enable_kb:
            self._ensure_kb_initialized()

        self.id_mappings: List[IDMapping] = []
        self.layout_info: Dict[str, LayoutInfo] = {}
        self.process: Optional[Process] = None
        self.lanes_by_id: Dict[str, Lane] = {}
        self.elements_by_id: Dict[str, Any] = {}
        self.graph_id_to_bpmn_id: Dict[str, str] = (
            {}
        )  # Reverse mapping: graph node ID -> BPMN element ID

    def _ensure_kb_initialized(self) -> None:
        """Ensure knowledge base is loaded."""
        if self.kb_loader is None:
            self.kb_loader = PatternLibraryLoader()

    def generate_xml(self, graph: ProcessGraph, process_name: Optional[str] = None) -> str:
        """Generate BPMN 2.0 XML from process graph.

        Args:
            graph: ProcessGraph to convert
            process_name: Optional override for process name

        Returns:
            Valid BPMN 2.0 XML string

        Raises:
            ValueError: If graph structure is invalid
        """
        with Timer("xml_generation"):
            try:
                # Validate graph
                is_valid, errors = graph.validate_structure()
                if not is_valid:
                    logger.warning(f"Graph validation warnings: {errors}")

                # Reset state
                self.id_mappings = []
                self.layout_info = {}
                self.elements_by_id = {}
                self.lanes_by_id = {}
                self.graph_id_to_bpmn_id = {}

                # Create process
                self.process = self._create_process(graph, process_name or graph.name)

                # Convert graph to BPMN elements
                self._convert_nodes(graph)
                self._convert_edges(graph)
                self._organize_lanes(graph)

                # Generate XML
                root = self._build_xml_root()

                # Validate against XSD (if available)
                self._validate_xml(root)

                return etree.tostring(
                    root, pretty_print=True, xml_declaration=True, encoding="utf-8"
                ).decode("utf-8")
            except Exception as e:
                logger.error(f"Exception during XML generation: {e}", exc_info=True)
                raise

    def _create_process(self, graph: ProcessGraph, name: str) -> Process:
        """Create BPMN Process element."""
        process_id = self._generate_id("Process", name)
        process = Process(
            id=process_id,
            name=name or graph.name,
            is_executable=True,
            documentation=None,
            process_type="private",
            lane_set=None,
        )
        self.elements_by_id[process_id] = process
        return process

    def _convert_nodes(self, graph: ProcessGraph) -> None:
        """Convert graph nodes to BPMN flow nodes."""
        # Layout nodes in grid
        x_pos = 50
        y_pos = 50
        col_count = 0
        max_cols = 5

        for node in graph.nodes:
            # Calculate position if not set
            if node.x is None or node.y is None:
                node.x = x_pos
                node.y = y_pos
                col_count += 1
                if col_count >= max_cols:
                    col_count = 0
                    y_pos += VERTICAL_SPACING + DEFAULT_TASK_HEIGHT
                    x_pos = 50
                else:
                    x_pos += HORIZONTAL_SPACING + DEFAULT_TASK_WIDTH

            # Skip ACTOR and DATA nodes (they're represented as swim lanes/data stores, not flow objects)
            if node.type == NodeType.ACTOR or node.type == NodeType.DATA:
                continue

            # Convert based on type
            if node.type == NodeType.START:
                element = self._convert_start_event(node)
            elif node.type == NodeType.END:
                element = self._convert_end_event(node)
            elif node.type == NodeType.DECISION:
                element = self._convert_gateway(node)
            elif node.type == NodeType.PARALLEL_FORK:
                element = self._convert_parallel_fork(node)
            elif node.type == NodeType.PARALLEL_JOIN:
                element = self._convert_parallel_join(node)
            elif node.type == NodeType.EVENT:
                element = self._convert_intermediate_event(node)
            elif node.type == NodeType.TASK:
                element = self._convert_task(node)
            else:
                logger.warning(f"Unsupported node type: {node.type}")
                continue

            if element:
                self.elements_by_id[element.id] = element
                self.graph_id_to_bpmn_id[node.id] = element.id  # Store reverse mapping
                self.process.flow_nodes.append(element)

                # Record layout
                self.layout_info[element.id] = LayoutInfo(
                    x=node.x,
                    y=node.y,
                    width=self._get_element_width(node),
                    height=self._get_element_height(node),
                    waypoints=None,
                )

                # Record ID mapping
                self._record_id_mapping(node.id, element.id, node.bpmn_type)

    def _convert_start_event(self, node: GraphNode) -> Optional[StartEvent]:
        """Convert graph node to StartEvent."""
        event_id = self._generate_id("StartEvent", node.label)
        event = StartEvent(
            id=event_id,
            name=node.label,
            documentation=node.properties.get("description"),
            is_interrupting=True,
        )
        return event

    def _convert_end_event(self, node: GraphNode) -> Optional[EndEvent]:
        """Convert graph node to EndEvent."""
        event_id = self._generate_id("EndEvent", node.label)
        event = EndEvent(
            id=event_id,
            name=node.label,
            documentation=node.properties.get("description"),
        )
        return event

    def _convert_intermediate_event(self, node: GraphNode) -> Optional[IntermediateEvent]:
        """Convert graph node to IntermediateEvent."""
        event_id = self._generate_id("IntermediateEvent", node.label)
        event = IntermediateEvent(
            id=event_id,
            name=node.label,
            documentation=node.properties.get("description"),
            is_interrupting=False,
        )
        return event

    def _convert_gateway(self, node: GraphNode) -> Optional[ExclusiveGateway]:
        """Convert graph decision node to ExclusiveGateway."""
        gateway_id = self._generate_id("ExclusiveGateway", node.label)
        gateway = ExclusiveGateway(
            id=gateway_id,
            name=node.label,
            documentation=node.properties.get("description"),
            default_flow=None,  # Will be set during edge conversion
        )
        return gateway

    def _convert_parallel_fork(self, node: GraphNode) -> Optional[ParallelGateway]:
        """Convert parallel fork node to ParallelGateway."""
        gateway_id = self._generate_id("ParallelGateway", node.label)
        gateway = ParallelGateway(
            id=gateway_id,
            name=node.label,
            documentation=node.properties.get("description"),
        )
        return gateway

    def _convert_parallel_join(self, node: GraphNode) -> Optional[ParallelGateway]:
        """Convert parallel join node to ParallelGateway."""
        gateway_id = self._generate_id("ParallelGateway", node.label)
        gateway = ParallelGateway(
            id=gateway_id,
            name=node.label,
            documentation=node.properties.get("description"),
        )
        return gateway

    def _convert_task(self, node: GraphNode) -> Optional[Task]:
        """Convert graph task node to appropriate Task subtype."""
        task_id = self._generate_id("Task", node.label)
        task_type = node.properties.get("task_type", "task").lower()

        if task_type == "usertask":
            task = UserTask(
                id=task_id,
                name=node.label,
                documentation=node.properties.get("description"),
                is_for_compensation=False,
                loop_characteristics=None,
                implementation="##unspecified",
                rendering=None,
            )
        elif task_type == "servicetask":
            task = ServiceTask(
                id=task_id,
                name=node.label,
                documentation=node.properties.get("description"),
                is_for_compensation=False,
                loop_characteristics=None,
                implementation="##webService",
                operation_ref=None,
            )
        elif task_type == "scripttask":
            task = ScriptTask(
                id=task_id,
                name=node.label,
                documentation=node.properties.get("description"),
                script_language=node.properties.get("script_language"),
                script=node.properties.get("script"),
            )
        elif task_type == "sendtask":
            task = SendTask(
                id=task_id,
                name=node.label,
                documentation=node.properties.get("description"),
                is_for_compensation=False,
                loop_characteristics=None,
                implementation=None,
                message_ref=None,
            )
        elif task_type == "receivetask":
            task = ReceiveTask(
                id=task_id,
                name=node.label,
                documentation=node.properties.get("description"),
                is_for_compensation=False,
                loop_characteristics=None,
                implementation=None,
                message_ref=None,
                instantiate=False,
            )
        elif task_type == "manualtask":
            task = ManualTask(
                id=task_id,
                name=node.label,
                documentation=node.properties.get("description"),
                is_for_compensation=False,
                loop_characteristics=None,
                implementation=None,
            )
        else:
            task = Task(
                id=task_id,
                name=node.label,
                documentation=node.properties.get("description"),
                is_for_compensation=False,
                loop_characteristics=None,
                implementation=None,
            )

        return task

    def _convert_edges(self, graph: ProcessGraph) -> None:
        """Convert graph edges to BPMN sequence flows."""
        for edge in graph.edges:
            if edge.type == EdgeType.CONTROL_FLOW:
                self._create_sequence_flow(edge, graph)

    def _create_sequence_flow(self, edge: GraphEdge, graph: ProcessGraph) -> None:
        """Create BPMN SequenceFlow from graph edge."""
        # Skip flows involving ACTOR or DATA nodes (they're not in the BPMN flow)
        source_node = next((n for n in graph.nodes if n.id == edge.source_id), None)
        target_node = next((n for n in graph.nodes if n.id == edge.target_id), None)

        if source_node and source_node.type in (NodeType.ACTOR, NodeType.DATA):
            return
        if target_node and target_node.type in (NodeType.ACTOR, NodeType.DATA):
            return

        # Look up BPMN IDs using the reverse mapping
        source_bpmn_id = self.graph_id_to_bpmn_id.get(edge.source_id)
        target_bpmn_id = self.graph_id_to_bpmn_id.get(edge.target_id)

        source_elem = self.elements_by_id.get(source_bpmn_id) if source_bpmn_id else None
        target_elem = self.elements_by_id.get(target_bpmn_id) if target_bpmn_id else None

        if not source_elem or not target_elem:
            logger.warning(
                f"Cannot create flow: missing elements {edge.source_id} or {edge.target_id}"
            )
            return

        flow_id = self._generate_id(
            "SequenceFlow", edge.label or f"{edge.source_id}_to_{edge.target_id}"
        )
        flow = SequenceFlow(
            id=flow_id,
            name=edge.label,
            source_ref=source_elem.id,
            target_ref=target_elem.id,
            is_default=edge.is_default,
            documentation=None,
            condition_expression=edge.properties.get("condition"),
        )

        # Add condition if present
        if edge.condition:
            flow.condition_expression = edge.condition

        # Add to process
        self.process.sequence_flows.append(flow)

        # Update node references
        if hasattr(source_elem, "outgoing"):
            source_elem.outgoing.append(flow_id)
        if hasattr(target_elem, "incoming"):
            target_elem.incoming.append(flow_id)

        # Record layout (waypoints for edge)
        self.layout_info[flow_id] = self._calculate_edge_layout(source_elem, target_elem)

    def _organize_lanes(self, graph: ProcessGraph) -> None:
        """Organize tasks into lanes based on actor assignments."""
        # Get actor assignments
        actor_tasks: Dict[str, List[str]] = {}

        for node in graph.nodes:
            if node.type == NodeType.ACTOR:
                actor_id = node.id
                actor_tasks[actor_id] = []

        # Group tasks by actor
        for edge in graph.edges:
            if edge.type == EdgeType.SWIMLANE:
                target_node = next((n for n in graph.nodes if n.id == edge.target_id), None)
                if target_node and target_node.type == NodeType.ACTOR:
                    actor_id = target_node.id
                    source_task_id = edge.source_id
                    if actor_id not in actor_tasks:
                        actor_tasks[actor_id] = []
                    actor_tasks[actor_id].append(source_task_id)

        # Create lane set
        if actor_tasks:
            lane_set = LaneSet(
                id=self._generate_id("LaneSet", "LaneSet"),
                name=None,
                documentation=None,
                parent_lane=None,
            )

            for actor_id, task_ids in actor_tasks.items():
                actor_node = next((n for n in graph.nodes if n.id == actor_id), None)
                if actor_node:
                    lane = Lane(
                        id=self._generate_id("Lane", actor_node.label),
                        name=actor_node.label,
                        documentation=None,
                        parent_lane=None,
                        partition_element_ref=None,
                    )

                    # Add task references to lane
                    for task_id in task_ids:
                        task_elem = self.elements_by_id.get(task_id)
                        if task_elem:
                            lane.flow_node_ref.append(task_elem.id)

                    lane_set.lanes.append(lane)
                    self.lanes_by_id[lane.id] = lane

            if lane_set.lanes:
                self.process.lane_set = lane_set

    def _build_xml_root(self) -> etree._Element:
        """Build XML root element with all BPMN content."""
        nsmap = {
            None: BPMN_NAMESPACE,
            "bpmndi": BPMNDI_NAMESPACE,
            "dc": DC_NAMESPACE,
            "di": DI_NAMESPACE,
        }

        root = etree.Element(
            "definitions",
            nsmap=nsmap,
            id=self._generate_id("Definitions", "ProcessDiagram"),
            targetNamespace=BPMN_NAMESPACE,
        )

        # Add process element
        process_elem = self._build_process_element()
        root.append(process_elem)

        # Add BPMN Diagram Interchange
        diagram_elem = self._build_diagram_element()
        root.append(diagram_elem)

        return root

    def _build_process_element(self) -> etree._Element:
        """Build process XML element."""
        process_elem = etree.Element("process")
        process_elem.set("id", self.process.id)
        if self.process.name:
            process_elem.set("name", self.process.name)
        process_elem.set("isExecutable", "true")

        # Add lane set if present
        if self.process.lane_set:
            lane_set_elem = self._build_lane_set_element(self.process.lane_set)
            process_elem.append(lane_set_elem)

        # Add flow nodes
        for node in self.process.flow_nodes:
            node_elem = self._build_flow_node_element(node)
            process_elem.append(node_elem)

        # Add sequence flows
        for flow in self.process.sequence_flows:
            flow_elem = self._build_sequence_flow_element(flow)
            process_elem.append(flow_elem)

        return process_elem

    def _build_flow_node_element(self, node: Any) -> etree._Element:
        """Build XML element for a flow node."""
        element_type = node.element_type

        if element_type == BPMNElementType.START_EVENT:
            tag = "startEvent"
        elif element_type == BPMNElementType.END_EVENT:
            tag = "endEvent"
        elif element_type == BPMNElementType.INTERMEDIATE_EVENT:
            tag = "intermediateEvent"
        elif element_type == BPMNElementType.EXCLUSIVE_GATEWAY:
            tag = "exclusiveGateway"
        elif element_type == BPMNElementType.PARALLEL_GATEWAY:
            tag = "parallelGateway"
        elif element_type == BPMNElementType.INCLUSIVE_GATEWAY:
            tag = "inclusiveGateway"
        elif element_type == BPMNElementType.USER_TASK:
            tag = "userTask"
        elif element_type == BPMNElementType.SERVICE_TASK:
            tag = "serviceTask"
        elif element_type == BPMNElementType.SCRIPT_TASK:
            tag = "scriptTask"
        elif element_type == BPMNElementType.SEND_TASK:
            tag = "sendTask"
        elif element_type == BPMNElementType.RECEIVE_TASK:
            tag = "receiveTask"
        elif element_type == BPMNElementType.MANUAL_TASK:
            tag = "manualTask"
        else:
            tag = "task"

        elem = etree.Element(tag)
        elem.set("id", node.id)
        if node.name:
            elem.set("name", node.name)

        # Add incoming/outgoing references
        if hasattr(node, "incoming"):
            for incoming_id in node.incoming:
                in_elem = etree.SubElement(elem, "incoming")
                in_elem.text = incoming_id

        if hasattr(node, "outgoing"):
            for outgoing_id in node.outgoing:
                out_elem = etree.SubElement(elem, "outgoing")
                out_elem.text = outgoing_id

        # Add documentation if present
        if node.documentation:
            doc_elem = etree.SubElement(elem, "documentation")
            doc_elem.text = node.documentation

        return elem

    def _build_sequence_flow_element(self, flow: SequenceFlow) -> etree._Element:
        """Build XML element for a sequence flow."""
        elem = etree.Element("sequenceFlow")
        elem.set("id", flow.id)
        elem.set("sourceRef", flow.source_ref)
        elem.set("targetRef", flow.target_ref)

        if flow.name:
            elem.set("name", flow.name)

        if flow.is_default:
            elem.set("default", "true")

        if flow.condition_expression:
            cond_elem = etree.SubElement(elem, "conditionExpression")
            cond_elem.set("{http://www.w3.org/2001/XMLSchema-instance}type", "tFormalExpression")
            cond_elem.text = flow.condition_expression

        return elem

    def _build_lane_set_element(self, lane_set: LaneSet) -> etree._Element:
        """Build XML element for lane set."""
        elem = etree.Element("laneSet")
        elem.set("id", lane_set.id)

        for lane in lane_set.lanes:
            lane_elem = etree.SubElement(elem, "lane")
            lane_elem.set("id", lane.id)
            if lane.name:
                lane_elem.set("name", lane.name)

            # Add flow node references
            for ref_id in lane.flow_node_ref:
                ref_elem = etree.SubElement(lane_elem, "flowNodeRef")
                ref_elem.text = ref_id

        return elem

    def _build_diagram_element(self) -> etree._Element:
        """Build BPMN Diagram Interchange element."""
        diagram = etree.Element("{%s}BPMNDiagram" % BPMNDI_NAMESPACE)
        diagram.set("id", self._generate_id("BPMNDiagram", "Diagram"))

        plane = etree.SubElement(diagram, "{%s}BPMNPlane" % BPMNDI_NAMESPACE)
        plane.set("id", self._generate_id("BPMNPlane", "Plane"))
        plane.set("bpmnElement", self.process.id)

        # Add shapes for nodes
        for elem_id, layout in self.layout_info.items():
            if elem_id in self.elements_by_id or any(
                f.id == elem_id for f in self.process.sequence_flows
            ):
                if layout.waypoints:
                    # This is an edge
                    edge_elem = self._build_edge_diagram(elem_id, layout)
                    plane.append(edge_elem)
                else:
                    # This is a node
                    shape_elem = self._build_shape_diagram(elem_id, layout)
                    plane.append(shape_elem)

        return diagram

    def _build_shape_diagram(self, element_id: str, layout: LayoutInfo) -> etree._Element:
        """Build BPMN shape diagram element."""
        shape = etree.Element("{%s}BPMNShape" % BPMNDI_NAMESPACE)
        shape.set("id", self._generate_id("Shape", element_id))
        shape.set("bpmnElement", element_id)

        bounds = etree.SubElement(shape, "{%s}Bounds" % DC_NAMESPACE)
        bounds.set("x", str(layout.x))
        bounds.set("y", str(layout.y))
        bounds.set("width", str(layout.width))
        bounds.set("height", str(layout.height))

        return shape

    def _build_edge_diagram(self, element_id: str, layout: LayoutInfo) -> etree._Element:
        """Build BPMN edge diagram element."""
        edge = etree.Element("{%s}BPMNEdge" % BPMNDI_NAMESPACE)
        edge.set("id", self._generate_id("Edge", element_id))
        edge.set("bpmnElement", element_id)

        if layout.waypoints:
            for waypoint in layout.waypoints:
                wp = etree.SubElement(edge, "{%s}waypoint" % DI_NAMESPACE)
                wp.set("x", str(waypoint.x))
                wp.set("y", str(waypoint.y))

        return edge

    def _calculate_edge_layout(self, source_elem: Any, target_elem: Any) -> LayoutInfo:
        """Calculate layout for edge between two elements."""
        source_layout = self.layout_info.get(source_elem.id)
        target_layout = self.layout_info.get(target_elem.id)

        if not source_layout or not target_layout:
            return LayoutInfo(x=0, y=0, width=1, height=1, waypoints=[])

        # Create waypoints from source to target
        waypoints = [
            Waypoint(
                x=source_layout.x + source_layout.width / 2,
                y=source_layout.y + source_layout.height / 2,
            ),
            Waypoint(
                x=target_layout.x + target_layout.width / 2,
                y=target_layout.y + target_layout.height / 2,
            ),
        ]

        return LayoutInfo(x=0, y=0, width=1, height=1, waypoints=waypoints)

    def _validate_xml(self, root: etree._Element) -> bool:
        """Validate XML against BPMN 2.0 XSD schema (if available).

        Currently logs validation status. Full XSD validation
        will be implemented in Phase 4.
        """
        try:
            # Check basic structure
            process_elem = root.find(".//{%s}process" % BPMN_NAMESPACE)
            if process_elem is None:
                logger.warning("XML missing process element")
                return False

            logger.info("XML basic structure validation passed")
            return True
        except Exception as e:
            logger.warning(f"XML validation error: {e}")
            return False

    def _generate_id(self, element_type: str, label: str = "") -> str:
        """Generate unique ID for BPMN element."""
        # Create consistent ID from label if available
        if label:
            sanitized = "".join(c if c.isalnum() else "_" for c in label)
            return f"{element_type}_{sanitized[:20]}_{uuid4().hex[:8]}"
        return f"{element_type}_{uuid4().hex[:12]}"

    def _get_element_width(self, node: GraphNode) -> float:
        """Get width for element based on type."""
        if node.type in [NodeType.START, NodeType.END, NodeType.EVENT]:
            return DEFAULT_EVENT_WIDTH
        elif node.type in [NodeType.DECISION, NodeType.PARALLEL_FORK, NodeType.PARALLEL_JOIN]:
            return DEFAULT_GATEWAY_WIDTH
        else:
            return DEFAULT_TASK_WIDTH

    def _get_element_height(self, node: GraphNode) -> float:
        """Get height for element based on type."""
        if node.type in [NodeType.START, NodeType.END, NodeType.EVENT]:
            return DEFAULT_EVENT_HEIGHT
        elif node.type in [NodeType.DECISION, NodeType.PARALLEL_FORK, NodeType.PARALLEL_JOIN]:
            return DEFAULT_GATEWAY_HEIGHT
        else:
            return DEFAULT_TASK_HEIGHT

    def _record_id_mapping(self, graph_id: str, bpmn_id: str, element_type: str) -> None:
        """Record mapping from graph ID to BPMN ID."""
        mapping = IDMapping(
            graph_id=graph_id,
            bpmn_id=bpmn_id,
            element_type=element_type,
            pattern_reference=None,
        )
        self.id_mappings.append(mapping)

    def get_id_mappings(self) -> List[IDMapping]:
        """Get all ID mappings created during generation."""
        return self.id_mappings

    def get_layout_info(self) -> Dict[str, LayoutInfo]:
        """Get layout information for all elements."""
        return self.layout_info
