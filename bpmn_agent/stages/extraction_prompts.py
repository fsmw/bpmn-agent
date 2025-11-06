"""
Stage 2: Entity & Relation Extraction

Extracts BPMN-relevant entities and relationships from preprocessed text using LLM.

This stage:
1. Designs role-playing prompts for LLM as "BPMN 2.0 Process Analysis Expert"
2. Sends text to LLM with JSON schema for expected output
3. Parses JSON response with error recovery
4. Maps extracted entities to canonical BPMN types
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from bpmn_agent.models.extraction import (
    ConfidenceLevel,
    EntityAttribute,
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    RelationType,
)


# ===========================
# Task 2.2.1: Prompting Strategy
# ===========================

@dataclass
class ExtractionPrompt:
    """Complete LLM prompt for entity and relation extraction."""
    
    system_message: str
    """System role definition for LLM"""
    
    user_message_template: str
    """Template for user message with {text} placeholder"""
    
    few_shot_examples: List[Dict[str, Any]]
    """Few-shot examples of text → JSON extraction"""
    
    output_schema: Dict[str, Any]
    """JSON schema definition for expected output"""
    
    constraints: List[str]
    """Explicit constraints and rules for extraction"""
    
    def render_user_message(self, text: str) -> str:
        """
        Render user message with text to extract from.
        
        Args:
            text: Process description text
            
        Returns:
            Complete user message
        """
        return self.user_message_template.format(text=text)


def create_extraction_prompt() -> ExtractionPrompt:
    """
    Create the complete extraction prompt for LLM.
    
    Returns:
        ExtractionPrompt with system message, user template, examples, schema, and constraints
    """
    
    # 1. System message - define LLM role
    system_message = """You are a BPMN 2.0 Process Analysis Expert with deep knowledge of business process modeling.

Your task is to analyze process descriptions and extract:
1. **Entities**: Activities (tasks, subprocesses), Events (start, end, intermediate, boundary), Gateways (decisions), Actors (participants), and Data Objects (data inputs/outputs)
2. **Relationships**: Connections between entities showing process flow, decision paths, actor involvement, and data flow

You must:
- Extract ONLY entities explicitly mentioned or strongly implied in the text
- Assign appropriate confidence levels (high/medium/low) based on certainty
- Flag uncertain or implicit entities for review
- Preserve original entity names where possible
- Return structured JSON with no additional commentary

Key BPMN concepts:
- **Task/Activity**: Units of work performed by actors
- **SubProcess**: Nested process containing multiple activities (keywords: sub-process, subprocess, nested process, decomposed process, detailed steps)
- **Event**: Start/end points or intermediate occurrences (messages, timers, conditions)
  - **Start Events**: Process initiation (keywords: start, begin, initiate, trigger)
  - **End Events**: Process completion (keywords: end, complete, finish, terminate)
  - **Intermediate Events**: Occur during process execution
    - **Timer Events**: Time-based triggers (keywords: wait, delay, timeout, after X minutes/hours, schedule, periodic)
    - **Signal Events**: External notifications (keywords: signal, alert, notification, broadcast, message received)
    - **Message Events**: Message reception/sending (keywords: message, receive, send, incoming message, outgoing message)
    - **Error Events**: Exception handling (keywords: error, exception, failure, fault, catch)
  - **Boundary Events**: Attached to tasks, triggered during task execution (keywords: occurs during, interrupts, catches, while task is running, exception handler)
- **Gateway**: Decision points (XOR exclusive, AND parallel) or join points (keywords: decide, choice, split, join, merge, branch)
- **Lane/Actor**: Participant or role performing activities (keywords: responsible, department, team, person, system)
- **Data Object**: Represents data used or produced by activities (keywords: data, document, record, information, input, output, order, invoice, report, form, file, database, table)
  - Can be inputs consumed by tasks: "task receives order data"
  - Can be outputs produced by tasks: "task generates report"
  - Can be used between tasks: "order data flows from order entry to fulfillment"
  - Examples: orders, invoices, reports, customer records, payment details, confirmations, documents
- **Sequence Flow**: Ordering between activities
- **Decision Path**: Conditional branches (if/then/else, based on condition, depending on)
- **Data Flow**: Movement of data between activities or data objects (relations: uses, produces, consumes)

Data Object Detection Guide:
- Look for nouns that represent business objects or information: order, invoice, document, record, data, form, request, confirmation
- Activities typically use data: "process the order" → activity "process order" + data object "order"
- Activities typically produce data: "generate a report" → activity "generate report" + data object "report"
- Data can flow between activities: if text mentions "the order data" or "order information", extract as data object
- Common data objects: customer data, order details, payment info, confirmation message, invoice, receipt
- If uncertain whether something is data, err on the side of inclusion

Event Trigger Detection Guide:
- Timer events typically follow patterns like "wait N time units", "timeout", "schedule", "repeat"
- Boundary events are described as happening "while a task is running" or "if a task fails"
- Signal/Message events mention "notification", "alert", "message arrives"
- Error events involve "catches exception" or "handles error"

Subprocess Detection:
- If text mentions "process steps", "detailed workflow", "contains", "includes", "consists of": mark as subprocess
- If text describes phases or stages that are broken down: these are subprocesses
- If multiple related activities are grouped: consider as subprocess candidate
"""
    
    # 2. Few-shot examples
    few_shot_examples = [
        {
            "input": "The customer submits an order. Sales verifies it. If valid, warehouse picks items. If not valid, customer is notified.",
            "output": {
                "entities": [
                    {"id": "cust_1", "type": "actor", "name": "customer", "confidence": "high"},
                    {"id": "sales_1", "type": "actor", "name": "sales", "confidence": "high"},
                    {"id": "wh_1", "type": "actor", "name": "warehouse", "confidence": "high"},
                    {"id": "act_1", "type": "activity", "name": "submit order", "confidence": "high"},
                    {"id": "act_2", "type": "activity", "name": "verify order", "confidence": "high"},
                    {"id": "act_3", "type": "activity", "name": "pick items", "confidence": "high"},
                    {"id": "act_4", "type": "activity", "name": "notify customer", "confidence": "high"},
                    {"id": "evt_1", "type": "event", "name": "order received", "confidence": "medium"},
                    {"id": "gw_1", "type": "gateway", "name": "order valid", "confidence": "high"}
                ],
                "relations": [
                    {"id": "rel_1", "type": "precedes", "source_id": "act_1", "target_id": "act_2", "confidence": "high"},
                    {"id": "rel_2", "type": "conditional", "source_id": "gw_1", "target_id": "act_3", "condition": "valid", "confidence": "high"},
                    {"id": "rel_3", "type": "conditional", "source_id": "gw_1", "target_id": "act_4", "condition": "invalid", "confidence": "high"},
                    {"id": "rel_4", "type": "involves", "source_id": "cust_1", "target_id": "act_1", "confidence": "high"},
                    {"id": "rel_5", "type": "involves", "source_id": "sales_1", "target_id": "act_2", "confidence": "high"},
                ]
            }
        },
        {
            "input": "Manager reviews requests daily. If approved, accountant processes payment. Finance confirms transaction.",
            "output": {
                "entities": [
                    {"id": "actor_1", "type": "actor", "name": "manager", "confidence": "high"},
                    {"id": "actor_2", "type": "actor", "name": "accountant", "confidence": "high"},
                    {"id": "actor_3", "type": "actor", "name": "finance", "confidence": "high"},
                    {"id": "act_1", "type": "activity", "name": "review requests", "confidence": "high"},
                    {"id": "act_2", "type": "activity", "name": "process payment", "confidence": "high"},
                    {"id": "act_3", "type": "activity", "name": "confirm transaction", "confidence": "high"},
                    {"id": "gw_1", "type": "gateway", "name": "approved", "confidence": "high"}
                ],
                "relations": [
                    {"id": "r1", "type": "precedes", "source_id": "act_1", "target_id": "gw_1", "confidence": "high"},
                    {"id": "r2", "type": "conditional", "source_id": "gw_1", "target_id": "act_2", "condition": "approved", "confidence": "high"},
                    {"id": "r3", "type": "precedes", "source_id": "act_2", "target_id": "act_3", "confidence": "high"},
                    {"id": "r4", "type": "involves", "source_id": "actor_1", "target_id": "act_1", "confidence": "high"},
                    {"id": "r5", "type": "involves", "source_id": "actor_2", "target_id": "act_2", "confidence": "high"},
                ]
            }
         },
        {
            "input": "The order fulfillment process consists of order validation (check inventory, verify customer data, check payment) and fulfillment (pick items, pack order, arrange shipping). Both phases are executed in sequence.",
            "output": {
                "entities": [
                    {"id": "act_1", "type": "activity", "name": "order fulfillment", "confidence": "high", "attributes": {"activity_type": "subprocess"}},
                    {"id": "act_2", "type": "activity", "name": "order validation", "confidence": "high", "attributes": {"activity_type": "subprocess"}},
                    {"id": "act_3", "type": "activity", "name": "check inventory", "confidence": "high"},
                    {"id": "act_4", "type": "activity", "name": "verify customer data", "confidence": "high"},
                    {"id": "act_5", "type": "activity", "name": "check payment", "confidence": "high"},
                    {"id": "act_6", "type": "activity", "name": "fulfillment", "confidence": "high", "attributes": {"activity_type": "subprocess"}},
                    {"id": "act_7", "type": "activity", "name": "pick items", "confidence": "high"},
                    {"id": "act_8", "type": "activity", "name": "pack order", "confidence": "high"},
                    {"id": "act_9", "type": "activity", "name": "arrange shipping", "confidence": "high"}
                ],
                "relations": [
                    {"id": "r1", "type": "precedes", "source_id": "act_2", "target_id": "act_6", "confidence": "high"},
                    {"id": "r2", "type": "precedes", "source_id": "act_3", "target_id": "act_4", "confidence": "medium"},
                    {"id": "r3", "type": "precedes", "source_id": "act_7", "target_id": "act_8", "confidence": "medium"}
                ]
            }
        },
        {
             "input": "A payment processing task is performed by the accountant. If the payment fails, an error notification is sent immediately. Additionally, if the task takes longer than 30 minutes, a timeout alert is triggered. Once payment is confirmed, the system sends a confirmation message.",
             "output": {
                 "entities": [
                     {"id": "actor_1", "type": "actor", "name": "accountant", "confidence": "high"},
                     {"id": "act_1", "type": "activity", "name": "payment processing", "confidence": "high"},
                     {"id": "evt_1", "type": "event", "name": "payment fails", "confidence": "high", "attributes": {"event_type": "boundary_event", "trigger": "error"}},
                     {"id": "evt_2", "type": "event", "name": "timeout alert", "confidence": "high", "attributes": {"event_type": "boundary_event", "trigger": "timer"}},
                     {"id": "evt_3", "type": "event", "name": "confirmation message", "confidence": "high", "attributes": {"event_type": "intermediate_event", "trigger": "message"}},
                     {"id": "act_2", "type": "activity", "name": "send error notification", "confidence": "high"},
                     {"id": "act_3", "type": "activity", "name": "send timeout notification", "confidence": "medium"}
                 ],
                 "relations": [
                     {"id": "r1", "type": "involves", "source_id": "actor_1", "target_id": "act_1", "confidence": "high"},
                     {"id": "r2", "type": "precedes", "source_id": "evt_1", "target_id": "act_2", "confidence": "high"},
                     {"id": "r3", "type": "precedes", "source_id": "evt_2", "target_id": "act_3", "confidence": "high"}
                 ]
             }
         },
        {
            "input": "The customer submits their invoice through the online portal. The finance department reviews the invoice details. If the invoice is approved, the accountant processes the payment and generates a receipt. The customer receives the receipt via email.",
            "output": {
                "entities": [
                    {"id": "actor_1", "type": "actor", "name": "customer", "confidence": "high"},
                    {"id": "actor_2", "type": "actor", "name": "finance department", "confidence": "high"},
                    {"id": "actor_3", "type": "actor", "name": "accountant", "confidence": "high"},
                    {"id": "act_1", "type": "activity", "name": "submit invoice", "confidence": "high"},
                    {"id": "act_2", "type": "activity", "name": "review invoice", "confidence": "high"},
                    {"id": "act_3", "type": "activity", "name": "process payment", "confidence": "high"},
                    {"id": "act_4", "type": "activity", "name": "generate receipt", "confidence": "high"},
                    {"id": "gw_1", "type": "gateway", "name": "invoice approved", "confidence": "high"},
                    {"id": "data_1", "type": "data", "name": "invoice", "confidence": "high"},
                    {"id": "data_2", "type": "data", "name": "receipt", "confidence": "high"}
                ],
                "relations": [
                    {"id": "r1", "type": "involves", "source_id": "actor_1", "target_id": "act_1", "confidence": "high"},
                    {"id": "r2", "type": "produces", "source_id": "act_1", "target_id": "data_1", "confidence": "high"},
                    {"id": "r3", "type": "involves", "source_id": "actor_2", "target_id": "act_2", "confidence": "high"},
                    {"id": "r4", "type": "uses", "source_id": "act_2", "target_id": "data_1", "confidence": "high"},
                    {"id": "r5", "type": "conditional", "source_id": "gw_1", "target_id": "act_3", "condition": "approved", "confidence": "high"},
                    {"id": "r6", "type": "involves", "source_id": "actor_3", "target_id": "act_3", "confidence": "high"},
                    {"id": "r7", "type": "uses", "source_id": "act_3", "target_id": "data_1", "confidence": "high"},
                    {"id": "r8", "type": "precedes", "source_id": "act_3", "target_id": "act_4", "confidence": "high"},
                    {"id": "r9", "type": "produces", "source_id": "act_4", "target_id": "data_2", "confidence": "high"},
                    {"id": "r10", "type": "involves", "source_id": "actor_1", "target_id": "act_4", "confidence": "medium"}
                ]
            }
        }
      ]
    
    # 3. Output schema - JSON schema for structured extraction
    output_schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "description": "Extracted process entities",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Unique entity ID"},
                        "type": {
                            "type": "string",
                            "enum": ["activity", "event", "gateway", "actor", "data"],
                            "description": "Entity type"
                        },
                        "name": {"type": "string", "description": "Entity name"},
                        "description": {
                            "type": "string",
                            "description": "Optional entity description"
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Extraction confidence"
                        },
                        "attributes": {
                            "type": "object",
                            "description": "Entity-specific attributes"
                        }
                    },
                    "required": ["id", "type", "name", "confidence"]
                }
            },
            "relations": {
                "type": "array",
                "description": "Extracted relationships",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Unique relation ID"},
                        "type": {
                            "type": "string",
                            "enum": [
                                "precedes", "conditional", "parallel",
                                "involves", "uses", "produces", "consumes",
                                "sends_to", "receives_from"
                            ],
                            "description": "Relationship type"
                        },
                        "source_id": {"type": "string", "description": "Source entity ID"},
                        "target_id": {"type": "string", "description": "Target entity ID"},
                        "label": {"type": "string", "description": "Optional relationship label"},
                        "condition": {"type": "string", "description": "Guard condition if applicable"},
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Extraction confidence"
                        }
                    },
                    "required": ["id", "type", "source_id", "target_id", "confidence"]
                }
            },
            "warnings": {
                "type": "array",
                "description": "Extraction warnings (ambiguities, truncations, etc.)",
                "items": {"type": "string"}
            }
        },
        "required": ["entities", "relations"]
    }
    
    # 4. User message template
    user_message_template = """Analyze this process description and extract entities and relationships in JSON format.

Process Description:
{text}

Return ONLY valid JSON matching the schema. No additional text or explanation.
"""
    
    # 5. Explicit constraints
    constraints = [
        "Extract ONLY entities mentioned in or strongly implied by the text",
        "Every activity must have at least one actor involved (use 'involves' relation)",
        "Decision gateways must have at least 2 outgoing conditional paths",
        "Use past/present tense verbs as activity names, lowercase",
        "Actor names should be roles (manager, customer, system) not specific people",
        "Never hallucinate entities or relationships not in the text",
        "Assign confidence scores: high for explicit mentions, medium for inferences, low for ambiguous",
        "If you detect any truncation or incompleteness, add warning",
        "Return IDs in format: entity_type_number (e.g., act_1, gw_2, actor_3)",
    ]
    
    return ExtractionPrompt(
        system_message=system_message,
        user_message_template=user_message_template,
        few_shot_examples=few_shot_examples,
        output_schema=output_schema,
        constraints=constraints,
    )


def render_full_prompt(prompt: ExtractionPrompt, text: str) -> str:
    """
    Render complete prompt including few-shot examples.
    
    Args:
        prompt: ExtractionPrompt template
        text: Text to extract from
        
    Returns:
        Complete formatted prompt with examples
    """
    prompt_parts = [
        prompt.system_message,
        "\n" + "=" * 60,
        "CONSTRAINTS:",
        "\n".join(f"- {c}" for c in prompt.constraints),
        "\n" + "=" * 60,
        "JSON SCHEMA:",
        json.dumps(prompt.output_schema, indent=2),
        "\n" + "=" * 60,
        "FEW-SHOT EXAMPLES:",
    ]
    
    for i, example in enumerate(prompt.few_shot_examples, 1):
        prompt_parts.append(f"\nExample {i}:")
        prompt_parts.append(f"Input: {example['input']}")
        prompt_parts.append(f"Output: {json.dumps(example['output'], indent=2)}")
    
    prompt_parts.append("\n" + "=" * 60)
    prompt_parts.append("YOUR TASK:")
    prompt_parts.append(prompt.render_user_message(text))
    
    return "\n".join(prompt_parts)


# ===========================
# Task 2.2.4: Taxonomy Mapper (included here for cohesion)
# ===========================

class BPMNTypeMappings:
    """Maps extracted entity types to canonical BPMN types."""
    
    # Activity mappings
    ACTIVITY_MAPPINGS = {
        "activity": "Task",
        "task": "Task",
        "action": "Task",
        "step": "Task",
        "work": "Task",
        "process": "Task",
        "execute": "Task",
        "perform": "Task",
        "handle": "Task",
        "subprocess": "SubProcess",
        "service": "ServiceTask",
        "user": "UserTask",
        "manual": "ManualTask",
        "script": "ScriptTask",
    }
    
    # Event mappings
    EVENT_MAPPINGS = {
        "event": "IntermediateEvent",
        "start": "StartEvent",
        "end": "EndEvent",
        "begin": "StartEvent",
        "finish": "EndEvent",
        "occur": "IntermediateEvent",
        "triggered": "IntermediateEvent",
        "message": "IntermediateMessageEvent",
        "timer": "IntermediateTimerEvent",
        "signal": "IntermediateSignalEvent",
    }
    
    # Gateway mappings
    GATEWAY_MAPPINGS = {
        "gateway": "ExclusiveGateway",
        "decision": "ExclusiveGateway",
        "choice": "ExclusiveGateway",
        "xor": "ExclusiveGateway",
        "exclusive": "ExclusiveGateway",
        "if": "ExclusiveGateway",
        "and": "ParallelGateway",
        "parallel": "ParallelGateway",
        "fork": "ParallelGateway",
        "join": "ParallelGateway",
        "synchronization": "ParallelGateway",
        "or": "InclusiveGateway",
        "inclusive": "InclusiveGateway",
    }
    
    @classmethod
    def map_entity_type(cls, extracted_type: str, entity_name: str = "") -> str:
        """
        Map extracted entity type to BPMN type.
        
        Args:
            extracted_type: Type from LLM extraction (activity, event, gateway, etc.)
            entity_name: Entity name for additional context
            
        Returns:
            Canonical BPMN element type
        """
        extracted_lower = extracted_type.lower().strip()
        name_lower = entity_name.lower()
        
        # Try direct mapping
        if extracted_lower in cls.ACTIVITY_MAPPINGS:
            return cls.ACTIVITY_MAPPINGS[extracted_lower]
        if extracted_lower in cls.EVENT_MAPPINGS:
            return cls.EVENT_MAPPINGS[extracted_lower]
        if extracted_lower in cls.GATEWAY_MAPPINGS:
            return cls.GATEWAY_MAPPINGS[extracted_lower]
        
        # Try name-based inference for activities
        activity_keywords = ["process", "execute", "handle", "perform", "send", "receive", "check", "review", "approve"]
        if any(kw in name_lower for kw in activity_keywords):
            return "Task"
        
        # Try name-based inference for gateways
        gateway_keywords = ["if", "decision", "choice", "approve", "verify", "validate"]
        if any(kw in name_lower for kw in gateway_keywords):
            return "ExclusiveGateway"
        
        # Default based on entity type
        if extracted_lower in ["activity", "action", "task"]:
            return "Task"
        elif extracted_lower in ["event", "occurrence"]:
            return "IntermediateEvent"
        elif extracted_lower == "gateway":
            return "ExclusiveGateway"
        
        # Ultimate fallback
        return "Task"


__all__ = [
    "ExtractionPrompt",
    "create_extraction_prompt",
    "render_full_prompt",
    "BPMNTypeMappings",
]
