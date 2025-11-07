"""
BPMN 2.0 Domain Model

Pydantic-based models representing BPMN 2.0 concepts aligned with specification:
https://www.omg.org/spec/BPMN/2.0.2/

These models serve as the target representation that the agent generates.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class BPMNElementType(str, Enum):
    """BPMN element types."""

    PROCESS = "process"
    TASK = "task"
    SERVICE_TASK = "serviceTask"
    USER_TASK = "userTask"
    MANUAL_TASK = "manualTask"
    SCRIPT_TASK = "scriptTask"
    SEND_TASK = "sendTask"
    RECEIVE_TASK = "receiveTask"
    START_EVENT = "startEvent"
    END_EVENT = "endEvent"
    INTERMEDIATE_EVENT = "intermediateEvent"
    BOUNDARY_EVENT = "boundaryEvent"
    GATEWAY = "gateway"
    EXCLUSIVE_GATEWAY = "exclusiveGateway"
    INCLUSIVE_GATEWAY = "inclusiveGateway"
    PARALLEL_GATEWAY = "parallelGateway"
    EVENT_BASED_GATEWAY = "eventBasedGateway"
    SUBPROCESS = "subprocess"
    CALL_ACTIVITY = "callActivity"
    LANE = "lane"
    POOL = "pool"
    COLLABORATION = "collaboration"
    SEQUENCE_FLOW = "sequenceFlow"
    MESSAGE_FLOW = "messageFlow"
    ASSOCIATION = "association"
    DATA_OBJECT = "dataObject"
    DATA_STORE = "dataStore"
    TEXT_ANNOTATION = "textAnnotation"
    GROUP = "group"


class EventType(str, Enum):
    """BPMN event types."""

    START = "start"
    END = "end"
    INTERMEDIATE = "intermediate"


class EventTrigger(str, Enum):
    """BPMN event triggers."""

    NONE = "none"
    MESSAGE = "message"
    TIMER = "timer"
    ERROR = "error"
    SIGNAL = "signal"
    ESCALATION = "escalation"
    CONDITIONAL = "conditional"
    LINK = "link"
    MULTIPLE = "multiple"


class GatewayType(str, Enum):
    """BPMN gateway types."""

    EXCLUSIVE = "exclusive"
    INCLUSIVE = "inclusive"
    PARALLEL = "parallel"
    EVENT_BASED = "eventBased"


class FlowNodeType(str, Enum):
    """BPMN flow node types."""

    TASK = "task"
    EVENT = "event"
    GATEWAY = "gateway"


class BaseElement(BaseModel):
    """Base class for all BPMN elements."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique element ID")
    name: Optional[str] = Field(None, description="Element name/label")
    documentation: Optional[str] = Field(None, description="Element documentation")

    model_config = ConfigDict(use_enum_values=False)


class Bounds(BaseModel):
    """Graphical bounds for diagram elements."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    width: float = Field(..., description="Width")
    height: float = Field(..., description="Height")


class DiagramElement(BaseModel):
    """Graphical representation information."""

    element_id: str = Field(..., description="Reference to model element ID")
    bounds: Bounds = Field(..., description="Bounding rectangle")
    label_bounds: Optional[Bounds] = Field(None, description="Label bounds (if text present)")


class Waypoint(BaseModel):
    """A point along a connection path."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class Message(BaseElement):
    """BPMN Message (used in MessageFlow)."""

    item_ref: Optional[str] = Field(None, description="Reference to ItemDefinition")


class ItemDefinition(BaseElement):
    """BPMN ItemDefinition (data structure definition)."""

    structure_ref: Optional[str] = Field(None, description="Structure reference (e.g., XSD type)")
    is_collection: bool = Field(False, description="Whether this is a collection")


class Operation(BaseElement):
    """BPMN Operation (interface definition)."""

    input_message: Optional[str] = Field(None, description="Input message ID")
    output_message: Optional[str] = Field(None, description="Output message ID")
    error_ref: List[str] = Field(default_factory=list, description="Error references")


class Interface(BaseElement):
    """BPMN Interface."""

    implementation_ref: Optional[str] = Field(None, description="Implementation reference")
    operations: List[str] = Field(default_factory=list, description="Operation IDs")


class Performance(BaseModel):
    """Performance/timing information."""

    duration_ms: Optional[int] = Field(None, description="Expected duration in milliseconds")
    priority: Optional[int] = Field(None, description="Priority level")


class Task(BaseElement):
    """Generic BPMN Task."""

    element_type: str = Field(default=BPMNElementType.TASK, description="Element type")
    incoming: List[str] = Field(default_factory=list, description="Incoming sequence flow IDs")
    outgoing: List[str] = Field(default_factory=list, description="Outgoing sequence flow IDs")
    is_for_compensation: bool = Field(False, description="Compensation handler flag")
    loop_characteristics: Optional[str] = Field(
        None, description="Loop type: standard/multiInstance"
    )

    # Task-specific attributes
    implementation: Optional[str] = Field(None, description="Implementation reference")


class ServiceTask(Task):
    """Service Task (automated service invocation)."""

    element_type: str = Field(default=BPMNElementType.SERVICE_TASK)
    operation_ref: Optional[str] = Field(None, description="Operation reference")
    implementation: str = Field("##webService", description="Implementation type")


class UserTask(Task):
    """User Task (manual work performed by a resource)."""

    element_type: str = Field(default=BPMNElementType.USER_TASK)
    implementation: str = Field("##unspecified", description="Implementation type")
    rendering: Optional[str] = Field(None, description="UI rendering hint")


class ManualTask(Task):
    """Manual Task (non-system work)."""

    element_type: str = Field(default=BPMNElementType.MANUAL_TASK)


class ScriptTask(Task):
    """Script Task (automated task running a script)."""

    element_type: str = Field(default=BPMNElementType.SCRIPT_TASK)
    script_language: Optional[str] = Field(None, description="Script language (e.g., 'JavaScript')")
    script: Optional[str] = Field(None, description="Script content")


class SendTask(Task):
    """Send Task (sending a message)."""

    element_type: str = Field(default=BPMNElementType.SEND_TASK)
    message_ref: Optional[str] = Field(None, description="Message reference")


class ReceiveTask(Task):
    """Receive Task (waiting for a message)."""

    element_type: str = Field(default=BPMNElementType.RECEIVE_TASK)
    message_ref: Optional[str] = Field(None, description="Message reference")
    instantiate: bool = Field(False, description="Whether this creates a new process instance")


class SubProcess(Task):
    """SubProcess (nested process container)."""

    element_type: str = Field(default=BPMNElementType.SUBPROCESS)

    # Nested process content
    flow_nodes: List[
        Union[
            Task,
            "ServiceTask",
            "UserTask",
            "ManualTask",
            "ScriptTask",
            "Event",
            "StartEvent",
            "EndEvent",
            "IntermediateEvent",
            "BoundaryEvent",
            "Gateway",
            "ExclusiveGateway",
            "InclusiveGateway",
            "ParallelGateway",
        ]
    ] = Field(default_factory=list, description="Nested task and event elements")
    sequence_flows: List["SequenceFlow"] = Field(
        default_factory=list, description="Nested control flows"
    )

    # Subprocess-specific attributes
    is_triggering: bool = Field(False, description="Whether subprocess is event-triggered")
    trigger_event_ref: Optional[str] = Field(
        None, description="Triggering event ID for event subprocesses"
    )

    # Data and structure
    data_objects: List["DataObject"] = Field(
        default_factory=list, description="Data objects in subprocess"
    )
    lane_set: Optional["LaneSet"] = Field(None, description="Lane set for subprocess")
    associations: List["Association"] = Field(
        default_factory=list, description="Associations in subprocess"
    )

    # Execution context
    is_independent: bool = Field(True, description="Whether subprocess is independent")
    wait_for_completion: bool = Field(
        True, description="Whether parent waits for subprocess completion"
    )


class Event(BaseElement):
    """Base class for BPMN Events."""

    element_type: str = Field(default=BPMNElementType.START_EVENT, description="Event type")
    event_type: EventType = Field(..., description="Event classification")
    trigger: EventTrigger = Field(default=EventTrigger.NONE, description="Event trigger type")
    incoming: List[str] = Field(default_factory=list, description="Incoming sequence flow IDs")
    outgoing: List[str] = Field(default_factory=list, description="Outgoing sequence flow IDs")


class StartEvent(Event):
    """Start Event (process initiation point)."""

    element_type: str = Field(default=BPMNElementType.START_EVENT)
    event_type: EventType = Field(default=EventType.START)
    is_interrupting: bool = Field(True, description="Whether it interrupts other flows")


class EndEvent(Event):
    """End Event (process termination point)."""

    element_type: str = Field(default=BPMNElementType.END_EVENT)
    event_type: EventType = Field(default=EventType.END)


class IntermediateEvent(Event):
    """Intermediate Event (triggered during process execution)."""

    element_type: str = Field(default=BPMNElementType.INTERMEDIATE_EVENT)
    event_type: EventType = Field(default=EventType.INTERMEDIATE)
    is_interrupting: bool = Field(False, description="Whether it interrupts current flow")


class BoundaryEvent(Event):
    """Boundary Event (attached to a task, triggered during task execution)."""

    element_type: str = Field(default=BPMNElementType.BOUNDARY_EVENT)
    event_type: EventType = Field(default=EventType.INTERMEDIATE)
    attached_to_ref: Optional[str] = Field(None, description="Task ID this event is attached to")
    is_interrupting: bool = Field(True, description="Whether it interrupts the attached task")
    cancel_activity: bool = Field(True, description="Whether completion cancels the task")


class Gateway(BaseElement):
    """Base class for BPMN Gateways."""

    element_type: str = Field(default=BPMNElementType.GATEWAY, description="Gateway type")
    gateway_type: GatewayType = Field(..., description="Gateway classification")
    incoming: List[str] = Field(default_factory=list, description="Incoming sequence flow IDs")
    outgoing: List[str] = Field(default_factory=list, description="Outgoing sequence flow IDs")


class ExclusiveGateway(Gateway):
    """Exclusive Gateway (XOR - single path selection)."""

    element_type: str = Field(default=BPMNElementType.EXCLUSIVE_GATEWAY)
    gateway_type: GatewayType = Field(default=GatewayType.EXCLUSIVE)
    default_flow: Optional[str] = Field(None, description="Default outgoing flow ID")


class InclusiveGateway(Gateway):
    """Inclusive Gateway (multiple path selection)."""

    element_type: str = Field(default=BPMNElementType.INCLUSIVE_GATEWAY)
    gateway_type: GatewayType = Field(default=GatewayType.INCLUSIVE)
    default_flow: Optional[str] = Field(None, description="Default outgoing flow ID")


class ParallelGateway(Gateway):
    """Parallel Gateway (concurrent path splits/joins)."""

    element_type: str = Field(default=BPMNElementType.PARALLEL_GATEWAY)
    gateway_type: GatewayType = Field(default=GatewayType.PARALLEL)


class EventBasedGateway(Gateway):
    """Event-Based Gateway (triggers based on event occurrence)."""

    element_type: str = Field(default=BPMNElementType.EVENT_BASED_GATEWAY)
    gateway_type: GatewayType = Field(default=GatewayType.EVENT_BASED)
    instantiate: bool = Field(False, description="Whether it instantiates process instances")


class SequenceFlow(BaseElement):
    """Sequence Flow (control flow between elements)."""

    element_type: str = Field(default=BPMNElementType.SEQUENCE_FLOW)
    source_ref: str = Field(..., description="Source element ID")
    target_ref: str = Field(..., description="Target element ID")
    condition_expression: Optional[str] = Field(None, description="Guard condition")
    is_default: bool = Field(False, description="Whether this is default flow")
    waypoints: List[Waypoint] = Field(default_factory=list, description="Flow path points")


class MessageFlow(BaseElement):
    """Message Flow (communication between processes)."""

    element_type: str = Field(default=BPMNElementType.MESSAGE_FLOW)
    source_ref: str = Field(..., description="Source element ID (participant/flow node)")
    target_ref: str = Field(..., description="Target element ID (participant/flow node)")
    message_ref: Optional[str] = Field(None, description="Message reference")
    waypoints: List[Waypoint] = Field(default_factory=list, description="Flow path points")


class Association(BaseElement):
    """Association (non-control flow connection)."""

    element_type: str = Field(default=BPMNElementType.ASSOCIATION)
    source_ref: str = Field(..., description="Source element ID")
    target_ref: str = Field(..., description="Target element ID")
    direction: Optional[str] = Field(None, description="Direction: 'none', 'one', 'both'")


class DataObject(BaseElement):
    """Data Object (data representation)."""

    element_type: str = Field(default=BPMNElementType.DATA_OBJECT)
    item_subject_ref: Optional[str] = Field(None, description="ItemDefinition reference")
    is_collection: bool = Field(False, description="Whether this is a collection")


class DataStore(BaseElement):
    """Data Store (persistent data repository)."""

    element_type: str = Field(default=BPMNElementType.DATA_STORE)
    item_subject_ref: Optional[str] = Field(None, description="ItemDefinition reference")
    capacity: Optional[int] = Field(None, description="Storage capacity")


class TextAnnotation(BaseElement):
    """Text Annotation (documentation in diagram)."""

    element_type: str = Field(default=BPMNElementType.TEXT_ANNOTATION)
    text: str = Field(default="", description="Annotation text")


class Group(BaseElement):
    """Group (logical grouping of elements)."""

    element_type: str = Field(default=BPMNElementType.GROUP)
    category_ref: Optional[str] = Field(None, description="Category reference")
    elements: List[str] = Field(default_factory=list, description="Grouped element IDs")


class Lane(BaseElement):
    """Lane (swimlane for organizing flow nodes)."""

    element_type: str = Field(default=BPMNElementType.LANE)
    parent_lane: Optional[str] = Field(None, description="Parent lane ID (for nested lanes)")
    flow_node_ref: List[str] = Field(default_factory=list, description="Flow node IDs in lane")
    partition_element_ref: Optional[str] = Field(None, description="Participant/Resource reference")


class LaneSet(BaseElement):
    """Lane Set (collection of lanes)."""

    lanes: List[Lane] = Field(default_factory=list, description="Lanes in this set")
    parent_lane: Optional[str] = Field(None, description="Parent lane ID")


class Process(BaseElement):
    """BPMN Process (main workflow container)."""

    element_type: str = Field(default=BPMNElementType.PROCESS)
    is_executable: bool = Field(True, description="Whether process is executable")
    process_type: str = Field("private", description="Process type: private/public/none")

    # Flow elements
    flow_nodes: List[
        Union[
            Task,
            ServiceTask,
            UserTask,
            ManualTask,
            ScriptTask,
            SubProcess,
            Event,
            StartEvent,
            EndEvent,
            IntermediateEvent,
            BoundaryEvent,
            Gateway,
            ExclusiveGateway,
            InclusiveGateway,
            ParallelGateway,
        ]
    ] = Field(default_factory=list, description="Task, SubProcess, and Event elements")
    sequence_flows: List[SequenceFlow] = Field(default_factory=list, description="Control flows")

    # Data and structure
    data_objects: List[DataObject] = Field(default_factory=list, description="Data objects")
    data_stores: List[DataStore] = Field(default_factory=list, description="Data stores")
    lane_set: Optional[LaneSet] = Field(None, description="Lane set for process")

    # Associations
    associations: List[Association] = Field(
        default_factory=list, description="Non-control associations"
    )
    text_annotations: List[TextAnnotation] = Field(default_factory=list, description="Annotations")


class Participant(BaseElement):
    """Participant (process participant in collaboration)."""

    element_type: str = Field(default=BPMNElementType.POOL)
    process_ref: Optional[str] = Field(None, description="Process ID for this participant")
    interface_ref: List[str] = Field(default_factory=list, description="Interface references")


class Collaboration(BaseElement):
    """Collaboration (multiple interacting processes)."""

    element_type: str = Field(default=BPMNElementType.COLLABORATION)
    participants: List[Participant] = Field(default_factory=list, description="Participants")
    message_flows: List[MessageFlow] = Field(default_factory=list, description="Message flows")
    conversation_links: List[Association] = Field(
        default_factory=list, description="Conversation links"
    )


class Definitions(BaseElement):
    """BPMN Definitions (root container)."""

    element_type: str = Field(default="definitions", description="Root element type")
    target_namespace: str = Field(..., description="Target namespace URI")

    # Contained elements
    messages: List[Message] = Field(default_factory=list, description="Message definitions")
    item_definitions: List[ItemDefinition] = Field(
        default_factory=list, description="Item definitions"
    )
    interfaces: List[Interface] = Field(default_factory=list, description="Interfaces")

    # Main content
    processes: List[Process] = Field(default_factory=list, description="Process definitions")
    collaborations: List[Collaboration] = Field(default_factory=list, description="Collaborations")

    # Diagram information (visual representation)
    diagrams: List[Dict[str, Any]] = Field(default_factory=list, description="Diagram metadata")


class BPMNDiagram:
    """Complete BPMN 2.0 diagram (root representation)."""

    def __init__(self, definitions: Definitions):
        """Initialize BPMN diagram."""
        self.definitions = definitions

    @property
    def process(self) -> Optional[Process]:
        """Get primary process (first process if available)."""
        return self.definitions.processes[0] if self.definitions.processes else None

    @property
    def collaboration(self) -> Optional[Collaboration]:
        """Get collaboration (if present)."""
        return self.definitions.collaborations[0] if self.definitions.collaborations else None

    def get_flow_nodes(self) -> List[BaseElement]:
        """Get all flow nodes from all processes."""
        nodes = []
        for process in self.definitions.processes:
            nodes.extend(process.flow_nodes)
        return nodes

    def get_all_flows(self) -> List[Union[SequenceFlow, MessageFlow]]:
        """Get all flows (sequence and message)."""
        flows = []
        for process in self.definitions.processes:
            flows.extend(process.sequence_flows)
        for collaboration in self.definitions.collaborations:
            flows.extend(collaboration.message_flows)
        return flows

    def validate_cross_references(self) -> tuple[bool, List[str]]:
        """
        Validate all cross-references in the BPMN diagram.

        Checks that:
        - All flow references point to existing elements
        - All message references exist
        - All item definition references exist
        - All interface references exist
        - Lane references are valid
        - Data object/store references are valid

        Returns:
            (is_valid, error_messages)
        """
        errors: List[str] = []

        # Build sets of all valid element IDs
        all_element_ids: Set[str] = set()
        all_element_ids.update(msg.id for msg in self.definitions.messages)
        all_element_ids.update(item.id for item in self.definitions.item_definitions)
        all_element_ids.update(iface.id for iface in self.definitions.interfaces)

        # Collect all flow node IDs and sequence flow IDs
        flow_node_ids: Set[str] = set()
        sequence_flow_ids: Set[str] = set()

        for process in self.definitions.processes:
            # Add process ID
            all_element_ids.add(process.id)

            # Collect flow nodes
            for node in process.flow_nodes:
                flow_node_ids.add(node.id)
                all_element_ids.add(node.id)

            # Collect sequence flows
            for flow in process.sequence_flows:
                sequence_flow_ids.add(flow.id)
                all_element_ids.add(flow.id)

            # Collect data objects and stores
            for data_obj in process.data_objects:
                all_element_ids.add(data_obj.id)
            for data_store in process.data_stores:
                all_element_ids.add(data_store.id)

            # Collect lanes
            if process.lane_set:
                for lane in process.lane_set.lanes:
                    all_element_ids.add(lane.id)

            # Collect associations and annotations
            for assoc in process.associations:
                all_element_ids.add(assoc.id)
            for annot in process.text_annotations:
                all_element_ids.add(annot.id)

        # Add collaboration elements
        for collab in self.definitions.collaborations:
            all_element_ids.add(collab.id)
            for participant in collab.participants:
                all_element_ids.add(participant.id)
            for msg_flow in collab.message_flows:
                all_element_ids.add(msg_flow.id)
            for conv_link in collab.conversation_links:
                all_element_ids.add(conv_link.id)

        # Validate sequence flow references
        for process in self.definitions.processes:
            for flow in process.sequence_flows:
                # Check source and target exist
                if flow.source_ref not in flow_node_ids:
                    errors.append(
                        f"SequenceFlow '{flow.id}' references non-existent source: {flow.source_ref}"
                    )
                if flow.target_ref not in flow_node_ids:
                    errors.append(
                        f"SequenceFlow '{flow.id}' references non-existent target: {flow.target_ref}"
                    )

            # Validate flow node incoming/outgoing references
            for node in process.flow_nodes:
                for incoming_id in node.incoming:
                    if incoming_id not in sequence_flow_ids:
                        errors.append(
                            f"Node '{node.id}' references non-existent incoming flow: {incoming_id}"
                        )
                for outgoing_id in node.outgoing:
                    if outgoing_id not in sequence_flow_ids:
                        errors.append(
                            f"Node '{node.id}' references non-existent outgoing flow: {outgoing_id}"
                        )

            # Validate data object references
            for data_obj in process.data_objects:
                if data_obj.item_subject_ref and data_obj.item_subject_ref not in all_element_ids:
                    errors.append(
                        f"DataObject '{data_obj.id}' references non-existent ItemDefinition: {data_obj.item_subject_ref}"
                    )

            # Validate data store references
            for data_store in process.data_stores:
                if (
                    data_store.item_subject_ref
                    and data_store.item_subject_ref not in all_element_ids
                ):
                    errors.append(
                        f"DataStore '{data_store.id}' references non-existent ItemDefinition: {data_store.item_subject_ref}"
                    )

            # Validate lane references
            if process.lane_set:
                lane_ids = {lane.id for lane in process.lane_set.lanes}
                for lane in process.lane_set.lanes:
                    for node_id in lane.flow_node_ref:
                        if node_id not in flow_node_ids:
                            errors.append(
                                f"Lane '{lane.id}' references non-existent flow node: {node_id}"
                            )
                    if lane.parent_lane and lane.parent_lane not in lane_ids:
                        errors.append(
                            f"Lane '{lane.id}' references non-existent parent lane: {lane.parent_lane}"
                        )

            # Validate association references
            for assoc in process.associations:
                if assoc.source_ref not in all_element_ids:
                    errors.append(
                        f"Association '{assoc.id}' references non-existent source: {assoc.source_ref}"
                    )
                if assoc.target_ref not in all_element_ids:
                    errors.append(
                        f"Association '{assoc.id}' references non-existent target: {assoc.target_ref}"
                    )

            # Validate task-specific references
            for node in process.flow_nodes:
                if isinstance(node, ServiceTask) and node.operation_ref:
                    if node.operation_ref not in all_element_ids:
                        errors.append(
                            f"ServiceTask '{node.id}' references non-existent Operation: {node.operation_ref}"
                        )
                elif isinstance(node, SendTask) and node.message_ref:
                    if node.message_ref not in all_element_ids:
                        errors.append(
                            f"SendTask '{node.id}' references non-existent Message: {node.message_ref}"
                        )
                elif isinstance(node, ReceiveTask) and node.message_ref:
                    if node.message_ref not in all_element_ids:
                        errors.append(
                            f"ReceiveTask '{node.id}' references non-existent Message: {node.message_ref}"
                        )

            # Validate gateway default flows
            for node in process.flow_nodes:
                if isinstance(node, (ExclusiveGateway, InclusiveGateway)) and node.default_flow:
                    if node.default_flow not in sequence_flow_ids:
                        errors.append(
                            f"Gateway '{node.id}' references non-existent default flow: {node.default_flow}"
                        )

        # Validate message flow references
        for collab in self.definitions.collaborations:
            for msg_flow in collab.message_flows:
                if msg_flow.source_ref not in all_element_ids:
                    errors.append(
                        f"MessageFlow '{msg_flow.id}' references non-existent source: {msg_flow.source_ref}"
                    )
                if msg_flow.target_ref not in all_element_ids:
                    errors.append(
                        f"MessageFlow '{msg_flow.id}' references non-existent target: {msg_flow.target_ref}"
                    )
                if msg_flow.message_ref and msg_flow.message_ref not in all_element_ids:
                    errors.append(
                        f"MessageFlow '{msg_flow.id}' references non-existent Message: {msg_flow.message_ref}"
                    )

            # Validate participant references
            for participant in collab.participants:
                if participant.process_ref and participant.process_ref not in all_element_ids:
                    errors.append(
                        f"Participant '{participant.id}' references non-existent Process: {participant.process_ref}"
                    )
                for iface_id in participant.interface_ref:
                    if iface_id not in all_element_ids:
                        errors.append(
                            f"Participant '{participant.id}' references non-existent Interface: {iface_id}"
                        )

        return len(errors) == 0, errors


# Rebuild Pydantic models to resolve forward references and Union types
Process.model_rebuild()
Definitions.model_rebuild()


__all__ = [
    "BPMNElementType",
    "EventType",
    "EventTrigger",
    "GatewayType",
    "FlowNodeType",
    "BaseElement",
    "Bounds",
    "DiagramElement",
    "Waypoint",
    "Message",
    "ItemDefinition",
    "Operation",
    "Interface",
    "Performance",
    "Task",
    "ServiceTask",
    "UserTask",
    "ManualTask",
    "ScriptTask",
    "SendTask",
    "ReceiveTask",
    "SubProcess",
    "Event",
    "StartEvent",
    "EndEvent",
    "IntermediateEvent",
    "BoundaryEvent",
    "Gateway",
    "ExclusiveGateway",
    "InclusiveGateway",
    "ParallelGateway",
    "EventBasedGateway",
    "SequenceFlow",
    "MessageFlow",
    "Association",
    "DataObject",
    "DataStore",
    "TextAnnotation",
    "Group",
    "Lane",
    "LaneSet",
    "Process",
    "Participant",
    "Collaboration",
    "Definitions",
    "BPMNDiagram",
]
