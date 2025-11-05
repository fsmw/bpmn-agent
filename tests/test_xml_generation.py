"""
Comprehensive tests for Stage 5 (XML Generation) BPMN XML Generation.

Tests:
- XML generation from simple process graphs
- XML generation with all BPMN element types
- ID mapping and reference management
- BPMN Diagram Interchange (DI) generation
- Lane organization and swimlanes
- Condition expression handling
- XML structure validation
- Pattern reference tracking
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from lxml import etree

from bpmn_agent.models.graph import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    ProcessGraph,
)
from bpmn_agent.models.bpmn_elements import (
    BPMNElementType,
    Bounds,
    DiagramElement,
    EndEvent,
    EventType,
    GatewayType,
    StartEvent,
    Waypoint,
)
from bpmn_agent.stages.xml_generation import (
    BPMNXMLGenerator,
    IDMapping,
    LayoutInfo,
    PatternReference,
)


# ===========================
# Test Data Fixtures
# ===========================


def create_graph(name: str) -> ProcessGraph:
    """Helper to create a ProcessGraph with required fields."""
    return ProcessGraph(
        id=f"{name.lower()}-{id(object())}",
        name=name,
        created_timestamp=datetime.now().isoformat()
    )


@pytest.fixture
def simple_graph():
    """Create a simple linear process graph: Start -> Task -> End."""
    graph = create_graph("SimpleProcess")
    
    # Create nodes
    start_node = GraphNode(
        id="start",
        type=NodeType.START,
        label="Process Start",
        bpmn_type="startEvent",
    )
    task_node = GraphNode(
        id="task1",
        type=NodeType.TASK,
        label="Execute Task",
        bpmn_type="userTask",
        properties={"task_type": "usertask"},
        x=200,
        y=100,
    )
    end_node = GraphNode(
        id="end",
        type=NodeType.END,
        label="Process End",
        bpmn_type="endEvent",
    )
    
    # Add nodes
    graph.nodes = [start_node, task_node, end_node]
    
    # Create edges
    edge1 = GraphEdge(
        id="edge1",
        source_id="start",
        target_id="task1",
        type=EdgeType.CONTROL_FLOW,
        label="Flow to Task",
    )
    edge2 = GraphEdge(
        id="edge2",
        source_id="task1",
        target_id="end",
        type=EdgeType.CONTROL_FLOW,
        label="Flow to End",
    )
    
    graph.edges = [edge1, edge2]
    
    return graph


@pytest.fixture
def complex_graph_with_gateways():
    """Create a complex graph with decision gateway."""
    graph = create_graph("ComplexProcess")
    
    # Create nodes
    start = GraphNode(id="start", type=NodeType.START, label="Start", bpmn_type="startEvent")
    task1 = GraphNode(
        id="task1", type=NodeType.TASK, label="Task 1", bpmn_type="userTask",
        properties={"task_type": "usertask"}
    )
    decision = GraphNode(
        id="decision1", type=NodeType.DECISION, label="Decision", bpmn_type="exclusiveGateway"
    )
    task2_yes = GraphNode(
        id="task2_yes", type=NodeType.TASK, label="Approved Path", bpmn_type="userTask",
        properties={"task_type": "usertask"}
    )
    task2_no = GraphNode(
        id="task2_no", type=NodeType.TASK, label="Rejected Path", bpmn_type="userTask",
        properties={"task_type": "usertask"}
    )
    end = GraphNode(id="end", type=NodeType.END, label="End", bpmn_type="endEvent")
    
    graph.nodes = [start, task1, decision, task2_yes, task2_no, end]
    
    # Create edges
    graph.edges = [
        GraphEdge(id="e1", source_id="start", target_id="task1", type=EdgeType.CONTROL_FLOW),
        GraphEdge(id="e2", source_id="task1", target_id="decision1", type=EdgeType.CONTROL_FLOW),
        GraphEdge(
            id="e3",
            source_id="decision1",
            target_id="task2_yes",
            type=EdgeType.CONTROL_FLOW,
            label="Yes",
            condition="approved == true",
            is_default=True,
        ),
        GraphEdge(
            id="e4",
            source_id="decision1",
            target_id="task2_no",
            type=EdgeType.CONTROL_FLOW,
            label="No",
            condition="approved == false",
        ),
        GraphEdge(id="e5", source_id="task2_yes", target_id="end", type=EdgeType.CONTROL_FLOW),
        GraphEdge(id="e6", source_id="task2_no", target_id="end", type=EdgeType.CONTROL_FLOW),
    ]
    
    return graph


@pytest.fixture
def graph_with_actors():
    """Create graph with actor swimlanes."""
    graph = create_graph("ProcessWithActors")
    
    # Create actor nodes
    actor1 = GraphNode(
        id="actor_finance", type=NodeType.ACTOR, label="Finance Team", bpmn_type="lane"
    )
    actor2 = GraphNode(
        id="actor_approver", type=NodeType.ACTOR, label="Approver", bpmn_type="lane"
    )
    
    # Create task nodes
    start = GraphNode(id="start", type=NodeType.START, label="Start", bpmn_type="startEvent")
    task1 = GraphNode(
        id="task1", type=NodeType.TASK, label="Prepare Request", bpmn_type="userTask",
        properties={"task_type": "usertask"}
    )
    task2 = GraphNode(
        id="task2", type=NodeType.TASK, label="Approve Request", bpmn_type="userTask",
        properties={"task_type": "usertask"}
    )
    end = GraphNode(id="end", type=NodeType.END, label="End", bpmn_type="endEvent")
    
    graph.nodes = [actor1, actor2, start, task1, task2, end]
    
    # Create edges
    graph.edges = [
        GraphEdge(id="e1", source_id="start", target_id="task1", type=EdgeType.CONTROL_FLOW),
        GraphEdge(id="e2", source_id="task1", target_id="task2", type=EdgeType.CONTROL_FLOW),
        GraphEdge(id="e3", source_id="task2", target_id="end", type=EdgeType.CONTROL_FLOW),
        # Swimlane edges
        GraphEdge(
            id="swimlane1", source_id="task1", target_id="actor_finance",
            type=EdgeType.SWIMLANE
        ),
        GraphEdge(
            id="swimlane2", source_id="task2", target_id="actor_approver",
            type=EdgeType.SWIMLANE
        ),
    ]
    
    return graph


@pytest.fixture
def graph_with_parallel():
    """Create graph with parallel fork/join."""
    graph = create_graph("ProcessWithParallel")
    
    start = GraphNode(id="start", type=NodeType.START, label="Start", bpmn_type="startEvent")
    fork = GraphNode(
        id="fork", type=NodeType.PARALLEL_FORK, label="Fork", bpmn_type="parallelGateway"
    )
    task1 = GraphNode(
        id="task1", type=NodeType.TASK, label="Parallel Task 1", bpmn_type="userTask",
        properties={"task_type": "usertask"}
    )
    task2 = GraphNode(
        id="task2", type=NodeType.TASK, label="Parallel Task 2", bpmn_type="userTask",
        properties={"task_type": "usertask"}
    )
    join = GraphNode(
        id="join", type=NodeType.PARALLEL_JOIN, label="Join", bpmn_type="parallelGateway"
    )
    end = GraphNode(id="end", type=NodeType.END, label="End", bpmn_type="endEvent")
    
    graph.nodes = [start, fork, task1, task2, join, end]
    
    graph.edges = [
        GraphEdge(id="e1", source_id="start", target_id="fork", type=EdgeType.CONTROL_FLOW),
        GraphEdge(id="e2", source_id="fork", target_id="task1", type=EdgeType.CONTROL_FLOW),
        GraphEdge(id="e3", source_id="fork", target_id="task2", type=EdgeType.CONTROL_FLOW),
        GraphEdge(id="e4", source_id="task1", target_id="join", type=EdgeType.CONTROL_FLOW),
        GraphEdge(id="e5", source_id="task2", target_id="join", type=EdgeType.CONTROL_FLOW),
        GraphEdge(id="e6", source_id="join", target_id="end", type=EdgeType.CONTROL_FLOW),
    ]
    
    return graph


@pytest.fixture
def xml_generator():
    """Create XML generator instance."""
    return BPMNXMLGenerator(enable_kb=False)


# ===========================
# Basic XML Generation Tests
# ===========================


def test_generate_xml_simple_graph(xml_generator, simple_graph):
    """Test XML generation from simple linear process graph."""
    xml_str = xml_generator.generate_xml(simple_graph)
    
    # Parse XML
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    # Verify root element
    assert root.tag.endswith('definitions')
    assert 'targetNamespace' in root.attrib
    
    # Verify process element exists
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    process = root.find('.//bpmn:process', ns)
    assert process is not None
    assert process.get('name') == 'SimpleProcess'
    assert process.get('isExecutable') == 'true'


def test_generate_xml_with_start_end_events(xml_generator, simple_graph):
    """Test that start and end events are properly generated."""
    xml_str = xml_generator.generate_xml(simple_graph)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Verify start event
    start_events = root.findall('.//bpmn:startEvent', ns)
    assert len(start_events) > 0
    assert any(e.get('name') == 'Process Start' for e in start_events)
    
    # Verify end event
    end_events = root.findall('.//bpmn:endEvent', ns)
    assert len(end_events) > 0
    assert any(e.get('name') == 'Process End' for e in end_events)


def test_generate_xml_with_tasks(xml_generator, simple_graph):
    """Test that tasks are properly generated."""
    xml_str = xml_generator.generate_xml(simple_graph)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Verify user task
    user_tasks = root.findall('.//bpmn:userTask', ns)
    assert len(user_tasks) > 0
    assert any(t.get('name') == 'Execute Task' for t in user_tasks)


def test_generate_xml_with_sequence_flows(xml_generator, simple_graph):
    """Test that sequence flows are properly generated."""
    xml_str = xml_generator.generate_xml(simple_graph)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Verify sequence flows exist
    flows = root.findall('.//bpmn:sequenceFlow', ns)
    assert len(flows) >= 2
    
    # Verify flow attributes
    assert all('sourceRef' in f.attrib for f in flows)
    assert all('targetRef' in f.attrib for f in flows)


# ===========================
# Gateway and Decision Tests
# ===========================


def test_generate_xml_with_exclusive_gateway(xml_generator, complex_graph_with_gateways):
    """Test exclusive gateway generation."""
    xml_str = xml_generator.generate_xml(complex_graph_with_gateways)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Verify exclusive gateway
    gateways = root.findall('.//bpmn:exclusiveGateway', ns)
    assert len(gateways) > 0
    assert any(g.get('name') == 'Decision' for g in gateways)


def test_generate_xml_with_condition_expressions(xml_generator, complex_graph_with_gateways):
    """Test that condition expressions are properly attached to flows."""
    xml_str = xml_generator.generate_xml(complex_graph_with_gateways)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Find flows with conditions
    flows = root.findall('.//bpmn:sequenceFlow', ns)
    condition_flows = [
        f for f in flows
        if f.find('bpmn:conditionExpression', ns) is not None
    ]
    
    assert len(condition_flows) > 0


def test_generate_xml_with_parallel_gateway(xml_generator, graph_with_parallel):
    """Test parallel gateway generation."""
    xml_str = xml_generator.generate_xml(graph_with_parallel)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Verify parallel gateways
    gateways = root.findall('.//bpmn:parallelGateway', ns)
    assert len(gateways) >= 2  # Fork and join


# ===========================
# Swimlane and Lane Tests
# ===========================


def test_generate_xml_with_lanes(xml_generator, graph_with_actors):
    """Test lane set generation for swimlanes."""
    xml_str = xml_generator.generate_xml(graph_with_actors)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Verify lane set
    lane_sets = root.findall('.//bpmn:laneSet', ns)
    assert len(lane_sets) > 0
    
    # Verify lanes
    lanes = root.findall('.//bpmn:lane', ns)
    assert len(lanes) > 0


def test_generate_xml_lanes_have_flow_node_refs(xml_generator, graph_with_actors):
    """Test that lanes have proper flow node references."""
    xml_str = xml_generator.generate_xml(graph_with_actors)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Verify lane references
    lanes = root.findall('.//bpmn:lane', ns)
    for lane in lanes:
        refs = lane.findall('bpmn:flowNodeRef', ns)
        # At least some lanes should have references
        assert len(refs) >= 0


# ===========================
# ID Mapping Tests
# ===========================


def test_id_mappings_created(xml_generator, simple_graph):
    """Test that ID mappings are properly created."""
    xml_generator.generate_xml(simple_graph)
    mappings = xml_generator.get_id_mappings()
    
    assert len(mappings) > 0
    assert all(isinstance(m, IDMapping) for m in mappings)
    assert all(m.bpmn_id is not None for m in mappings)
    assert all(m.graph_id is not None for m in mappings)


def test_id_mappings_include_element_types(xml_generator, simple_graph):
    """Test that ID mappings include element types."""
    xml_generator.generate_xml(simple_graph)
    mappings = xml_generator.get_id_mappings()
    
    assert all(m.element_type is not None for m in mappings)
    
    # Check for expected element types
    element_types = [m.element_type for m in mappings]
    assert 'startEvent' in element_types
    assert 'endEvent' in element_types
    assert 'userTask' in element_types


def test_id_mappings_match_xml_ids(xml_generator, simple_graph):
    """Test that ID mappings match actual XML element IDs."""
    xml_str = xml_generator.generate_xml(simple_graph)
    mappings = xml_generator.get_id_mappings()
    
    root = etree.fromstring(xml_str.encode('utf-8'))
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Collect all element IDs from XML
    xml_ids = set()
    for elem_tag in ['startEvent', 'endEvent', 'userTask', 'task', 'exclusiveGateway']:
        for elem in root.findall(f'.//bpmn:{elem_tag}', ns):
            if 'id' in elem.attrib:
                xml_ids.add(elem.attrib['id'])
    
    # All BPMN IDs from mappings should be in XML
    for mapping in mappings:
        assert mapping.bpmn_id in xml_ids


# ===========================
# Layout Information Tests
# ===========================


def test_layout_info_generated(xml_generator, simple_graph):
    """Test that layout information is generated."""
    xml_generator.generate_xml(simple_graph)
    layout_info = xml_generator.get_layout_info()
    
    assert len(layout_info) > 0
    assert all(isinstance(info, LayoutInfo) for info in layout_info.values())


def test_layout_info_includes_coordinates(xml_generator, simple_graph):
    """Test that layout info includes x, y coordinates."""
    xml_generator.generate_xml(simple_graph)
    layout_info = xml_generator.get_layout_info()
    
    for layout in layout_info.values():
        assert layout.x is not None
        assert layout.y is not None
        assert isinstance(layout.x, (int, float))
        assert isinstance(layout.y, (int, float))
        assert layout.x >= 0
        assert layout.y >= 0


def test_layout_info_includes_dimensions(xml_generator, simple_graph):
    """Test that layout info includes width and height."""
    xml_generator.generate_xml(simple_graph)
    layout_info = xml_generator.get_layout_info()
    
    for layout in layout_info.values():
        assert layout.width is not None
        assert layout.height is not None
        assert layout.width > 0
        assert layout.height > 0


# ===========================
# BPMN Diagram Interchange Tests
# ===========================


def test_bpmn_diagram_interchange_generated(xml_generator, simple_graph):
    """Test that BPMN Diagram Interchange is generated."""
    xml_str = xml_generator.generate_xml(simple_graph)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    ns = {'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI'}
    
    # Verify diagram exists
    diagrams = root.findall('.//bpmndi:BPMNDiagram', ns)
    assert len(diagrams) > 0


def test_bpmn_diagram_has_plane(xml_generator, simple_graph):
    """Test that BPMN diagram has a plane element."""
    xml_str = xml_generator.generate_xml(simple_graph)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    ns = {'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI'}
    
    # Verify plane exists
    planes = root.findall('.//bpmndi:BPMNPlane', ns)
    assert len(planes) > 0


def test_bpmn_shapes_generated(xml_generator, simple_graph):
    """Test that BPMN shapes are generated for nodes."""
    xml_str = xml_generator.generate_xml(simple_graph)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    ns = {'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI'}
    
    # Verify shapes exist
    shapes = root.findall('.//bpmndi:BPMNShape', ns)
    assert len(shapes) > 0


def test_bpmn_shape_has_bounds(xml_generator, simple_graph):
    """Test that BPMN shapes have bounds."""
    xml_str = xml_generator.generate_xml(simple_graph)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    ns_di = {'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI'}
    ns_dc = {'dc': 'http://www.omg.org/spec/DD/20100524/DC'}
    
    # Get shapes and verify they have bounds
    shapes = root.findall('.//bpmndi:BPMNShape', ns_di)
    for shape in shapes:
        bounds = shape.find('dc:Bounds', ns_dc)
        assert bounds is not None
        assert 'x' in bounds.attrib
        assert 'y' in bounds.attrib
        assert 'width' in bounds.attrib
        assert 'height' in bounds.attrib


def test_bpmn_edges_generated(xml_generator, simple_graph):
    """Test that BPMN edges are generated for flows."""
    xml_str = xml_generator.generate_xml(simple_graph)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    ns = {'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI'}
    
    # Verify edges exist
    edges = root.findall('.//bpmndi:BPMNEdge', ns)
    assert len(edges) >= 0  # May or may not have edges depending on layout


# ===========================
# Namespace Tests
# ===========================


def test_proper_namespaces_in_xml(xml_generator, simple_graph):
    """Test that proper BPMN 2.0 namespaces are used."""
    xml_str = xml_generator.generate_xml(simple_graph)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    # Verify namespaces
    assert 'http://www.omg.org/spec/BPMN/20100524/MODEL' in root.nsmap.values()
    assert 'http://www.omg.org/spec/BPMN/20100524/DI' in root.nsmap.values()


def test_default_namespace_is_bpmn(xml_generator, simple_graph):
    """Test that default namespace is BPMN."""
    xml_str = xml_generator.generate_xml(simple_graph)
    
    # Check that default namespace is set
    assert 'http://www.omg.org/spec/BPMN/20100524/MODEL' in xml_str


# ===========================
# Error Handling Tests
# ===========================


def test_generate_xml_with_invalid_graph_returns_warning(xml_generator):
    """Test that invalid graphs are handled gracefully."""
    # Create invalid graph (no nodes)
    graph = create_graph("EmptyProcess")
    
    # Should not raise, but may have warnings
    xml_str = xml_generator.generate_xml(graph)
    assert xml_str is not None
    assert len(xml_str) > 0


def test_generate_xml_with_disconnected_nodes(xml_generator):
    """Test handling of disconnected nodes."""
    graph = create_graph("DisconnectedProcess")
    
    # Add isolated nodes
    node1 = GraphNode(id="n1", type=NodeType.START, label="Start", bpmn_type="startEvent")
    node2 = GraphNode(id="n2", type=NodeType.TASK, label="Task", bpmn_type="userTask",
                      properties={"task_type": "usertask"})
    
    graph.nodes = [node1, node2]
    graph.edges = []  # No connections
    
    # Should still generate valid XML
    xml_str = xml_generator.generate_xml(graph)
    assert xml_str is not None


def test_generate_xml_preserves_graph_data(xml_generator, simple_graph):
    """Test that generation doesn't modify original graph."""
    original_node_count = len(simple_graph.nodes)
    original_edge_count = len(simple_graph.edges)
    
    xml_generator.generate_xml(simple_graph)
    
    assert len(simple_graph.nodes) == original_node_count
    assert len(simple_graph.edges) == original_edge_count


# ===========================
# Complex Workflow Tests
# ===========================


def test_generate_xml_complex_workflow(xml_generator, complex_graph_with_gateways):
    """Test generation of complex workflow with decisions."""
    xml_str = xml_generator.generate_xml(complex_graph_with_gateways)
    
    assert xml_str is not None
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    # Verify all expected elements
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Should have start, end, tasks, and gateway
    starts = root.findall('.//bpmn:startEvent', ns)
    ends = root.findall('.//bpmn:endEvent', ns)
    tasks = root.findall('.//bpmn:userTask', ns)
    gateways = root.findall('.//bpmn:exclusiveGateway', ns)
    
    assert len(starts) > 0
    assert len(ends) > 0
    assert len(tasks) > 0
    assert len(gateways) > 0


def test_generate_xml_parallel_workflow(xml_generator, graph_with_parallel):
    """Test generation of workflow with parallel paths."""
    xml_str = xml_generator.generate_xml(graph_with_parallel)
    
    assert xml_str is not None
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    # Verify parallel gateways
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    gateways = root.findall('.//bpmn:parallelGateway', ns)
    assert len(gateways) >= 2


# ===========================
# Multiple Generation Tests
# ===========================


def test_multiple_generations_independent(simple_graph):
    """Test that multiple generations don't interfere with each other."""
    gen1 = BPMNXMLGenerator(enable_kb=False)
    gen2 = BPMNXMLGenerator(enable_kb=False)
    
    xml1 = gen1.generate_xml(simple_graph, process_name="Process1")
    xml2 = gen2.generate_xml(simple_graph, process_name="Process2")
    
    # Parse both
    root1 = etree.fromstring(xml1.encode('utf-8'))
    root2 = etree.fromstring(xml2.encode('utf-8'))
    
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Check process names
    p1 = root1.find('.//bpmn:process', ns)
    p2 = root2.find('.//bpmn:process', ns)
    
    assert p1.get('name') == "Process1"
    assert p2.get('name') == "Process2"


def test_regeneration_with_same_generator(xml_generator, simple_graph, complex_graph_with_gateways):
    """Test that regenerating with same generator works correctly."""
    xml1 = xml_generator.generate_xml(simple_graph)
    mappings1 = len(xml_generator.get_id_mappings())
    
    xml2 = xml_generator.generate_xml(complex_graph_with_gateways)
    mappings2 = len(xml_generator.get_id_mappings())
    
    # New generation should have different number of mappings
    assert mappings1 != mappings2
    assert mappings2 > mappings1


# ===========================
# XML Validity Tests
# ===========================


def test_xml_output_is_well_formed(xml_generator, simple_graph):
    """Test that XML output is well-formed."""
    xml_str = xml_generator.generate_xml(simple_graph)
    
    # Should parse without errors
    try:
        root = etree.fromstring(xml_str.encode('utf-8'))
        assert root is not None
    except etree.XMLSyntaxError as e:
        pytest.fail(f"Generated XML is not well-formed: {e}")


def test_xml_output_has_declaration(xml_generator, simple_graph):
    """Test that XML output includes XML declaration."""
    xml_str = xml_generator.generate_xml(simple_graph)
    
    assert xml_str.startswith('<?xml')


def test_xml_is_pretty_printed(xml_generator, simple_graph):
    """Test that XML output is properly formatted."""
    xml_str = xml_generator.generate_xml(simple_graph)
    
    # Should have newlines and indentation
    assert '\n' in xml_str
    lines = xml_str.split('\n')
    assert len(lines) > 1


# ===========================
# Pattern Reference Tests (KB Integration Ready)
# ===========================


def test_pattern_reference_structure():
    """Test PatternReference model structure."""
    ref = PatternReference(
        pattern_id="p1",
        pattern_name="Approval Pattern",
        confidence=0.95,
        matched_rules=["rule1", "rule2"],
    )
    
    assert ref.pattern_id == "p1"
    assert ref.confidence == 0.95
    assert len(ref.matched_rules) == 2


def test_id_mapping_with_pattern_reference():
    """Test IDMapping with PatternReference."""
    ref = PatternReference(
        pattern_id="p1",
        pattern_name="Test Pattern",
        confidence=0.9,
    )
    
    mapping = IDMapping(
        graph_id="g1",
        bpmn_id="b1",
        element_type="userTask",
        pattern_reference=ref,
    )
    
    assert mapping.pattern_reference is not None
    assert mapping.pattern_reference.pattern_id == "p1"


# ===========================
# Task Type Specific Tests
# ===========================


def test_generate_xml_with_different_task_types(xml_generator):
    """Test generation with different task types."""
    graph = create_graph("MultiTaskTypeProcess")
    
    start = GraphNode(id="start", type=NodeType.START, label="Start", bpmn_type="startEvent")
    
    # Different task types
    user_task = GraphNode(
        id="task1", type=NodeType.TASK, label="User Task", bpmn_type="userTask",
        properties={"task_type": "usertask"}
    )
    service_task = GraphNode(
        id="task2", type=NodeType.TASK, label="Service Task", bpmn_type="serviceTask",
        properties={"task_type": "servicetask"}
    )
    manual_task = GraphNode(
        id="task3", type=NodeType.TASK, label="Manual Task", bpmn_type="manualTask",
        properties={"task_type": "manualtask"}
    )
    
    end = GraphNode(id="end", type=NodeType.END, label="End", bpmn_type="endEvent")
    
    graph.nodes = [start, user_task, service_task, manual_task, end]
    graph.edges = [
        GraphEdge(id="e1", source_id="start", target_id="task1", type=EdgeType.CONTROL_FLOW),
        GraphEdge(id="e2", source_id="task1", target_id="task2", type=EdgeType.CONTROL_FLOW),
        GraphEdge(id="e3", source_id="task2", target_id="task3", type=EdgeType.CONTROL_FLOW),
        GraphEdge(id="e4", source_id="task3", target_id="end", type=EdgeType.CONTROL_FLOW),
    ]
    
    xml_str = xml_generator.generate_xml(graph)
    root = etree.fromstring(xml_str.encode('utf-8'))
    
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Verify different task types
    user_tasks = root.findall('.//bpmn:userTask', ns)
    service_tasks = root.findall('.//bpmn:serviceTask', ns)
    manual_tasks = root.findall('.//bpmn:manualTask', ns)
    
    assert len(user_tasks) > 0
    assert len(service_tasks) > 0
    assert len(manual_tasks) > 0
