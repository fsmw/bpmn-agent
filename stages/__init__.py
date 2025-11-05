"""
BPMN Extraction Pipeline Stages

Implements the 5-stage extraction pipeline:
1. Text Pre-Processing: Clean and prepare input text
2. Entity & Relation Extraction: Extract entities and relationships from text using LLM
3. Entity Resolution: Consolidate co-references and resolve entities
4. Semantic Graph Construction: Build intermediate process graph
5. BPMN XML Generation: Convert graph to valid BPMN 2.0 XML
"""

from bpmn_agent.stages.text_preprocessing import (
    TextChunk,
    PreprocessedText,
    TextPreprocessor,
)

from bpmn_agent.stages.extraction_prompts import (
    ExtractionPrompt,
    create_extraction_prompt,
    render_full_prompt,
    BPMNTypeMappings,
)

from bpmn_agent.stages.entity_extraction import (
    JSONParser,
    EntityExtractor,
)

from bpmn_agent.stages.entity_resolution import (
    CoReferenceResolver,
    ActorProfile,
    ActorConsolidator,
    RelationshipValidationReport,
    RelationshipValidator,
    EntityResolutionPipeline,
)

from bpmn_agent.stages.process_graph_builder import (
    ProcessGraphBuilder,
    LaneStructure,
    LaneStructureBuilder,
    GraphValidationIssue,
    GraphValidator,
    ImplicitFlow,
    ImplicitFlowInferrer,
    SemanticGraphConstructionPipeline,
)

__all__ = [
    # Stage 1: Text Pre-Processing
    "TextChunk",
    "PreprocessedText",
    "TextPreprocessor",
    
    # Stage 2: Entity & Relation Extraction
    "ExtractionPrompt",
    "create_extraction_prompt",
    "render_full_prompt",
    "BPMNTypeMappings",
    "JSONParser",
    "EntityExtractor",
    
    # Stage 3: Entity Resolution
    "CoReferenceResolver",
    "ActorProfile",
    "ActorConsolidator",
    "RelationshipValidationReport",
    "RelationshipValidator",
    "EntityResolutionPipeline",
    
    # Stage 4: Semantic Graph Construction
    "ProcessGraphBuilder",
    "LaneStructure",
    "LaneStructureBuilder",
    "GraphValidationIssue",
    "GraphValidator",
    "ImplicitFlow",
    "ImplicitFlowInferrer",
    "SemanticGraphConstructionPipeline",
]
