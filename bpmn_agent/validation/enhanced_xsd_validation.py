"""
Enhanced BPMN 2.0 XSD Validation Framework with Knowledge Base Integration

Provides comprehensive XML validation using BPMN 2.0 XSD schema with:
- Pattern-based error analysis
- Knowledge base remediation suggestions
- Semantic validation integration
- Extended error reporting and debugging
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
import re

from bpmn_agent.models.graph import GraphNode, GraphEdge, ProcessGraph
from bpmn_agent.models.extraction import ExtractionResultWithErrors
from bpmn_agent.knowledge.domain_classifier import DomainClassifier, DomainType
from bpmn_agent.tools.graph_analysis import GraphAnalyzer, GraphAnomaly, AnomalyType

logger = logging.getLogger(__name__)


class XSDValidationErrorLevel(str, Enum):
    """XSD validation error severity levels."""
    
    FATAL = "fatal"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationErrorCategory(str, Enum):
    """Categories of XSD validation errors."""
    
    SCHEMA = "schema"
    STRUCTURE = "structure"
    ELEMENT_MISSING = "element_missing"
    ATTRIBUTE_MISSING = "attribute_missing"
    ATTRIBUTE_INVALID = "attribute_invalid"
    NAMESPACE = "namespace"
    CONNECTIVITY = "connectivity"
    PATTERN_VIOLATION = "pattern_violation"
    SEMANTIC = "semantic"
    COMPLIANCE = "compliance"


@dataclass
class XSDValidationError:
    """Detailed XSD validation error."""
    
    level: XSDValidationErrorLevel
    category: ValidationErrorCategory
    message: str
    element_id: Optional[str] = None
    element_name: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    xpath: Optional[str] = None
    namespace: Optional[str] = None
    suggestion: Optional[str] = None
    pattern_reference: Optional[str] = None  # Knowledge base pattern reference
    code: Optional[str] = None  # Error code for cross-referencing
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class XSDValidationResult:
    """Complete XSD validation result with enhanced reporting."""
    
    is_valid: bool
    total_errors: int
    total_warnings: int
    errors_by_level: Dict[XSDValidationErrorLevel, int] = field(default_factory=dict)
    errors_by_category: Dict[ValidationErrorCategory, int] = field(default_factory=dict)
    errors: List[XSDValidationError] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    remediation_plan: List[str] = field(default_factory=list)
    pattern_matches: List[str] = field(default_factory=list)  # Patterns that were applied


class EnhancedXSDValidator:
    """Enhanced BPMN 2.0 XSD validator with knowledge base integration."""
    
    # Standard BPMN 2.0 XSD URL
    BPMN_XSD_URL = "https://www.omg.org/spec/BPMN/20100501/BPMN20.xsd"
    
    # Local XSD fallback path
    LOCAL_XSD_PATH = Path(__file__).parent.parent / "knowledge" / "schemas" / "BPMN20.xsd"
    
    # Common BPMN XML namespaces
    NAMESPACES = {
        "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
        "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }
    
    def __init__(self, enable_kb_patterns: bool = True):
        """Initialize enhanced XSD validator.
        
        Args:
            enable_kb_patterns: Enable knowledge base pattern integration
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.enable_kb_patterns = enable_kb_patterns
        self.xsd_content = None
        self.xsd_schema = None
        self.domain_classifier = DomainClassifier()
        self._load_xsd_schema()
    
    def _load_xsd_schema(self) -> None:
        """Load BPMN 2.0 XSD schema from multiple sources."""
        try:
            # Try to validate using lxml (more capable)
            import lxml.etree as etree
            
            # First try to load from local file if exists
            if self.LOCAL_XSD_PATH.exists():
                try:
                    with open(self.LOCAL_XSD_PATH, 'rb') as f:
                        # Use XMLParser with recover=True to handle missing external schemas gracefully
                        parser = etree.XMLParser(recover=True, no_network=False)
                        schema_doc = etree.parse(f, parser=parser)
                        # Try to create schema, but don't fail if external references are missing
                        try:
                            self.xsd_schema = etree.XMLSchema(schema_doc)
                            self.logger.info(f"BPMN 2.0 XSD schema loaded from local file: {self.LOCAL_XSD_PATH}")
                            return
                        except etree.XMLSchemaParseError as schema_error:
                            # If schema parsing fails due to missing external files, log warning but continue
                            self.logger.warning(f"XSD schema loaded but external references may be missing: {schema_error}")
                            # Still try to use it for basic validation
                            self.xsd_schema = None
                            with open(self.LOCAL_XSD_PATH, 'r', encoding='utf-8') as f2:
                                self.xsd_content = f2.read()
                            return
                except Exception as e:
                    self.logger.warning(f"Failed to load local XSD, trying download: {e}")
            
            # Try to download from URL
            self.xsd_content = self._download_xsd_schema()
            
            # Parse XSD content - handle encoding declaration properly
            if isinstance(self.xsd_content, str):
                # Remove XML declaration if present to avoid encoding issues
                content = self.xsd_content
                if content.strip().startswith('<?xml'):
                    # Find end of XML declaration
                    end_decl = content.find('?>')
                    if end_decl != -1:
                        content = content[end_decl + 2:].strip()
                # Parse as bytes to avoid encoding declaration issues
                schema_doc = etree.parse(StringIO(content))
            else:
                schema_doc = etree.parse(StringIO(self.xsd_content.decode('utf-8')))
            
            # Create XSD schema
            self.xsd_schema = etree.XMLSchema(schema_doc)
            self.logger.info("BPMN 2.0 XSD schema loaded successfully using lxml")
            
        except ImportError:
            # Fallback to basic XML validation using xml.etree
            self.logger.warning("lxml not available, using basic validation")
            if self.LOCAL_XSD_PATH.exists():
                with open(self.LOCAL_XSD_PATH, 'r', encoding='utf-8') as f:
                    self.xsd_content = f.read()
            else:
                self.xsd_content = self._download_xsd_schema()
        except Exception as e:
            self.logger.error(f"Failed to load BPMN XSD schema: {e}")
            # Try local file as last resort
            if self.LOCAL_XSD_PATH.exists():
                try:
                    with open(self.LOCAL_XSD_PATH, 'r', encoding='utf-8') as f:
                        self.xsd_content = f.read()
                    self.logger.info("Using local XSD file as fallback")
                except Exception as e2:
                    self.logger.warning(f"Failed to load local XSD: {e2}")
                    self.xsd_content = self._create_minimal_xsd_schema()
            else:
                self.xsd_content = self._create_minimal_xsd_schema()
    
    def _download_xsd_schema(self) -> str:
        """Download BPMN 2.0 XSD schema from OMG source."""
        try:
            import requests
            response = requests.get(self.BPMN_XSD_URL, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.logger.warning(f"Failed to download XSD schema: {e}")
            # Return minimal schema for basic validation
            return self._create_minimal_xsd_schema()
    
    def _create_minimal_xsd_schema(self) -> str:
        """Create minimal BPMN XSD schema for basic validation."""
        # Use the provided XSD structure but remove external dependencies
        return """<?xml version="1.0" encoding="UTF-8"?>
<xsd:schema xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI" elementFormDefault="qualified" attributeFormDefault="unqualified" targetNamespace="http://www.omg.org/spec/BPMN/20100524/MODEL">
<!-- Note: External imports (BPMNDI.xsd, Semantic.xsd) removed for standalone use -->
<xsd:element name="definitions" type="tDefinitions"/>
<xsd:complexType name="tDefinitions">
<xsd:sequence>
<xsd:element ref="import" minOccurs="0" maxOccurs="unbounded"/>
<xsd:element ref="extension" minOccurs="0" maxOccurs="unbounded"/>
<xsd:element ref="rootElement" minOccurs="0" maxOccurs="unbounded"/>
<xsd:element ref="bpmndi:BPMNDiagram" minOccurs="0" maxOccurs="unbounded"/>
<xsd:element ref="relationship" minOccurs="0" maxOccurs="unbounded"/>
</xsd:sequence>
<xsd:attribute name="id" type="xsd:ID" use="optional"/>
<xsd:attribute name="name" type="xsd:string"/>
<xsd:attribute name="targetNamespace" type="xsd:anyURI" use="required"/>
<xsd:attribute name="expressionLanguage" type="xsd:anyURI" use="optional" default="http://www.w3.org/1999/XPath"/>
<xsd:attribute name="typeLanguage" type="xsd:anyURI" use="optional" default="http://www.w3.org/2001/XMLSchema"/>
<xsd:attribute name="exporter" type="xsd:string"/>
<xsd:attribute name="exporterVersion" type="xsd:string"/>
<xsd:anyAttribute namespace="##other" processContents="lax"/>
</xsd:complexType>
<xsd:element name="import" type="tImport"/>
<xsd:complexType name="tImport">
<xsd:attribute name="namespace" type="xsd:anyURI" use="required"/>
<xsd:attribute name="location" type="xsd:string" use="required"/>
<xsd:attribute name="importType" type="xsd:anyURI" use="required"/>
</xsd:complexType>
<xsd:element name="extension"/>
<xsd:element name="rootElement"/>
<xsd:element name="relationship"/>
</xsd:schema>"""
    def validate_xml_against_xsd(
        self,
        xml_content: str,
        graph: Optional[ProcessGraph] = None,
        extraction_result: Optional[ExtractionResultWithErrors] = None,
        domain: Optional[str] = None,
        patterns_applied: Optional[List[str]] = None
    ) -> XSDValidationResult:
        """Enhanced XSD validation with knowledge base integration.
        
        Args:
            xml_content: XML content to validate
            graph: Optional process graph for context
            extraction_result: Optional extraction result for semantic validation
            domain: Optional process domain
            patterns_applied: Optional list of applied patterns
            
        Returns:
            Enhanced validation result
        """
        self.logger.info("Starting enhanced XSD validation")
        
        result = XSDValidationResult(
            is_valid=True,
            total_errors=0,
            total_warnings=0
        )
        
        try:
            # Parse XML content
            xml_doc = ET.fromstring(xml_content)
            
            # Stage 1: Basic structural validation
            self._validate_xml_structure(xml_doc, result)
            
            # Stage 2: XSD schema validation (if available)
            if self.xsd_schema and hasattr(self.xsd_schema, 'validate'):
                is_valid = self.xsd_schema.validate(xml_doc)
                if not is_valid:
                    result.is_valid = False
                    self._parse_xsd_errors(xml_doc, result)
            
            # Stage 3: Connect and semantic validation (knowledge base enhanced)
            if self.enable_kb_patterns and (graph or extraction_result):
                self._validate_semantic_compliance(xml_content, result, graph, extraction_result, domain, patterns_applied)
            
            # Stage 4: Calculate quality scores and metrics
            self._calculate_quality_metrics(result, xml_content, graph, extraction_result)
            
            # Stage 5: Generate remediation plan
            self._generate_remediation_plan(result)
            
            # Stage 6: Pattern compliance checking
            if self.enable_kb_patterns and patterns_applied:
                self._validate_pattern_compliance(result, patterns_applied)
            
        except ET.ParseError as e:
            # Handle XML parsing errors
            result.is_valid = False
            result.errors.append(XSDValidationError(
                level=XSDValidationErrorLevel.FATAL,
                category=ValidationErrorCategory.STRUCTURE,
                message=f"XML parsing error: {str(e)}",
                line_number=getattr(e, 'lineno', None),
                column_number=getattr(e, 'position', None),
                suggestion="Ensure XML is well-formed with proper nesting and quotes"
            ))
            result.total_errors = 1
            result.errors_by_level[XSDValidationErrorLevel.FATAL] = 1
            result.errors_by_category[ValidationErrorCategory.STRUCTURE] = 1
        
        # Calculate final metrics
        result.total_errors = sum(
            result.errors_by_level[level] for level in result.errors_by_level
            if level in [XSDValidationErrorLevel.FATAL, XSDValidationErrorLevel.ERROR]
        )
        result.total_warnings = sum(
            result.errors_by_level[level] for level in result.errors_by_level
            if level in [XSDValidationErrorLevel.WARNING]
        )
        
        self.logger.info(
            f"Enhanced XSD validation completed: "
            f"valid={result.is_valid}, errors={result.total_errors}, warnings={result.total_warnings}"
        )
        
        return result
    
    def _parse_xsd_errors(self, xml_doc: ET.Element, result: XSDValidationResult) -> None:
        """Parse XSD validation errors from lxml error log."""
        if not self.xsd_schema or not hasattr(self.xsd_schema, 'error_log'):
            return
        
        error_log = self.xsd_schema.error_log
        
        for error in error_log:
            level = self._determine_error_level(error.type_name)
            category = self._categorize_error(error)
            
            # Extract element information
            xpath = error.path
            element_name = None
            element_id = None
            
            if xpath:
                try:
                    # Extract element name from XPath
                    element_name = xpath.split('/')[-1].split('[')[0] if '[' in xpath else xpath.split('/')[-1]
                    # Try to extract ID
                    element = xml_doc.xpath(xpath)
                    if element:
                        element_id = element[0].get('id', None)
                except Exception:
                    pass  # XPath parsing failed
            
            result.errors.append(XSDValidationError(
                level=level,
                category=category,
                message=str(error.message),
                element_id=element_id,
                element_name=element_name,
                line_number=error.line,
                column_number=error.column,
                xpath=xpath,
                suggestion=self._generate_suggestion_for_error(error),
                code=error.type_name
            ))
            
            result.errors_by_level[level] = result.errors_by_level.get(level, 0) + 1
            result.errors_by_category[category] = result.errors_by_category.get(category, 0) + 1
    
    def _determine_error_level(self, error_type: str) -> XSDValidationErrorLevel:
        """Determine error severity level from XSD error type."""
        fatal_types = ['SchemaError', 'XMLSchemaValidationError', 'ValidationError']
        error_types = ['XMLError']
        warning_types = ['XMLSchemaWarning']
        
        if error_type in fatal_types:
            return XSDValidationErrorLevel.FATAL
        elif error_type in error_types:
            return XSDValidationErrorLevel.ERROR
        elif error_type in warning_types:
            return XSDValidationErrorLevel.WARNING
        else:
            return XSDValidationErrorLevel.INFO
    
    def _categorize_error(self, error) -> ValidationErrorCategory:
        """Categorize XSD validation error for better organization."""
        message_lower = error.message.lower()
        
        if 'required' in message_lower and 'element' in message_lower:
            return ValidationErrorCategory.ELEMENT_MISSING
        elif 'required' in message_lower and 'attribute' in message_lower:
            return ValidationErrorCategory.ATTRIBUTE_MISSING
        elif 'attribute' in message_lower and ('invalid' in message_lower or 'illegal' in message_lower):
            return ValidationErrorCategory.ATTRIBUTE_INVALID
        elif 'namespace' in message_lower:
            return ValidationErrorCategory.NAMESPACE
        elif 'schema' in message_lower:
            return ValidationErrorCategory.SCHEMA
        elif 'structure' in message_lower:
            return ValidationErrorCategory.STRUCTURE
        else:
            return ValidationErrorCategory.COMPLIANCE
    
    def _generate_suggestion_for_error(self, error) -> str:
        """Generate remediation suggestion for XSD validation error."""
        message = error.message.lower()
        
        # Sugerencias basadas en tipo de error
        if "element" in message and "not allowed" in message:
            return "Remove or replace the invalid element. Check BPMN 2.0 specification for allowed elements."
        elif "required" in message and "attribute" in message:
            return "Add the required attribute. Check element definition in BPMN 2.0 XSD schema."
        elif "required" in message and "element" in message:
            return "Add the required element. Check BPMN 2.0 specification for required child elements."
        elif "namespace" in message:
            return "Fix namespace declaration. Ensure proper BPMN namespace is used: http://www.omg.org/spec/BPMN/20100524/MODEL"
        elif "invalid" in message or "illegal" in message:
            return "Check attribute value against BPMN 2.0 specification. Ensure value matches expected type."
        elif "structure" in message or "hierarchy" in message:
            return "Check element hierarchy. Ensure elements are nested according to BPMN 2.0 structure."
        else:
            return "Review BPMN 2.0 specification for this element type and ensure compliance."
    
    def _validate_xml_structure(self, xml_doc: ET.Element, result: XSDValidationResult) -> None:
        """Validate basic XML structure and BPMN requirements."""
        
        # 1. Validar namespace BPMN
        root_tag = xml_doc.tag
        if not any(ns in root_tag for ns in ['definitions', '{http://www.omg.org/spec/BPMN']):
            # Try to find definitions element
            definitions = xml_doc.find('.//{http://www.omg.org/spec/BPMN/20100524/MODEL}definitions')
            if definitions is None:
                # Check if root is definitions without namespace
                if 'definitions' not in root_tag.lower():
                    result.errors.append(XSDValidationError(
                        level=XSDValidationErrorLevel.ERROR,
                        category=ValidationErrorCategory.NAMESPACE,
                        message="Root element must be 'definitions' with BPMN namespace",
                        suggestion="Ensure XML starts with <definitions xmlns='http://www.omg.org/spec/BPMN/20100524/MODEL'>"
                    ))
                    result.is_valid = False
        
        # 2. Validar que existe al menos un proceso
        # Try with namespace first
        processes = xml_doc.findall('.//{http://www.omg.org/spec/BPMN/20100524/MODEL}process')
        # Also try without namespace (for compatibility)
        if not processes:
            processes = xml_doc.findall('.//process')
        
        if not processes:
            result.errors.append(XSDValidationError(
                level=XSDValidationErrorLevel.ERROR,
                category=ValidationErrorCategory.ELEMENT_MISSING,
                message="No process element found",
                suggestion="Add at least one <process> element inside <definitions>"
            ))
            result.is_valid = False
        
        # 3. Validar estructura básica de cada proceso
        for process in processes:
            process_id = process.get('id')
            if not process_id:
                result.errors.append(XSDValidationError(
                    level=XSDValidationErrorLevel.ERROR,
                    category=ValidationErrorCategory.ATTRIBUTE_MISSING,
                    message="Process missing required 'id' attribute",
                    element_id=process_id,
                    suggestion="Add 'id' attribute to process element"
                ))
                result.is_valid = False
            
            # Validate that it has startEvent (warning, not error)
            start_events = process.findall('.//{http://www.omg.org/spec/BPMN/20100524/MODEL}startEvent')
            if not start_events:
                start_events = process.findall('.//startEvent')
            
            if not start_events:
                result.errors.append(XSDValidationError(
                    level=XSDValidationErrorLevel.WARNING,
                    category=ValidationErrorCategory.STRUCTURE,
                    message=f"Process '{process_id}' has no start event",
                    element_id=process_id,
                    suggestion="Add at least one startEvent to the process"
                ))
        
        # Update counters
        if result.errors:
            for error in result.errors:
                result.errors_by_level[error.level] = result.errors_by_level.get(error.level, 0) + 1
                result.errors_by_category[error.category] = result.errors_by_category.get(error.category, 0) + 1
    
    def _validate_semantic_compliance(
        self,
        xml_content: str,
        result: XSDValidationResult,
        graph: Optional[ProcessGraph],
        extraction_result: Optional[ExtractionResultWithErrors],
        domain: Optional[str],
        patterns_applied: Optional[List[str]]
    ) -> None:
        """Validate semantic compliance with knowledge base enhancement."""
        
        # 1. Basic semantic validation using GraphAnalyzer if available
        if graph:
            try:
                from bpmn_agent.tools.graph_analysis import GraphAnalyzer
                analyzer = GraphAnalyzer()
                analysis_result = analyzer.analyze_graph_structure(graph, extraction_result)
                
                # Convert anomalies to validation errors
                for anomaly in analysis_result.anomalies:
                    if anomaly.severity in ["high", "critical"]:
                        result.errors.append(XSDValidationError(
                            level=XSDValidationErrorLevel.ERROR,
                            category=ValidationErrorCategory.SEMANTIC,
                            message=f"Graph anomaly: {anomaly.description}",
                            element_id=anomaly.node_id,
                            suggestion=anomaly.suggestion
                        ))
                        result.is_valid = False
                    elif anomaly.severity == "medium":
                        result.errors.append(XSDValidationError(
                            level=XSDValidationErrorLevel.WARNING,
                            category=ValidationErrorCategory.SEMANTIC,
                            message=f"Graph anomaly: {anomaly.description}",
                            element_id=anomaly.node_id,
                            suggestion=anomaly.suggestion
                        ))
            except Exception as e:
                self.logger.warning(f"Graph analysis failed during semantic validation: {e}")
        
        # 2. Validación de dominio específica
        if domain:
            self._validate_domain_semantics(xml_content, result, domain)
        
        # 3. Validation of applied patterns
        if patterns_applied:
            self._validate_pattern_application(xml_content, result, patterns_applied)
    
    def _validate_domain_semantics(self, xml_content: str, result: XSDValidationResult, domain: str) -> None:
        """Validate domain-specific semantic rules."""
        domain_lower = domain.lower()
        
        # HR domain specific checks
        if domain_lower in ["hr", "humanresources", "recruitment"]:
            self._validate_hr_semantics(xml_content, result)
        # Finance domain specific checks
        elif domain_lower in ["finance", "financial", "accounting"]:
            self._validate_finance_semantics(xml_content, result)
        # IT domain specific checks
        elif domain_lower in ["it", "technology", "system"]:
            self._validate_it_semantics(xml_content, result)
        # Healthcare domain specific checks
        elif domain_lower in ["healthcare", "medical", "hospital"]:
            self._validate_healthcare_semantics(xml_content, result)
    
    def _validate_hr_semantics(self, xml_content: str, result: XSDValidationResult) -> None:
        """Validate HR domain-specific BPMN semantics."""
        import xml.etree.ElementTree as ET
        try:
            xml_doc = ET.fromstring(xml_content)
        except:
            return
        
        # Check for HR-specific elements that should be included
        hr_elements = ["userTask", "managerTask", "approval", "review", "assessment"]
        found_elements = set()
        
        # Common HR process patterns
        hr_patterns = [
            {
                "name": "approval_sequence",
                "description": "HR processes should have proper approval sequences",
                "elements": ["approval", "review", "assessment"],
                "validation": "tasks should exist and be properly connected"
            },
            {
                "name": "documentation_requirements",
                "description": "HR processes should have proper documentation",
                "elements": ["documentation"],
                "validation": "process should include task documentation"
            }
        ]
        
        # Search for elements
        for element_name in hr_elements:
            found_elements.update([elem.tag for elem in xml_doc.iter() if element_name.lower() in elem.tag.lower()])
        
        missing_elements = [elem for elem in hr_elements if elem not in found_elements and elem not in ["task"]]
        if missing_elements:
            result.errors.append(XSDValidationError(
                level=XSDValidationErrorLevel.INFO,
                category=ValidationErrorCategory.DOMAIN_SPECIFIC,
                message=f"Consider adding HR-specific elements: {', '.join(missing_elements)}",
                suggestion=f"Add HR elements like {', '.join(missing_elements[:3])} to improve process description",
                pattern_reference="hr_domain_patterns"
            ))
            result.errors_by_level[XSDValidationErrorLevel.INFO] += 1
            result.errors_by_category[ValidationErrorCategory.DOMAIN_SPECIFIC] += 1
    
    def _validate_finance_semantics(self, xml_content: str, result: XSDValidationResult) -> None:
        """Validate finance domain-specific BPMN semantics."""
        import xml.etree.ElementTree as ET
        try:
            xml_doc = ET.fromstring(xml_content)
        except:
            return
        
        # Check for financial control elements
        finance_elements = ["approval", "invoice", "payment", "audit", "compliance"]
        found_elements = set()
        
        for element_name in finance_elements:
            found_elements.update([elem.tag for elem in xml_doc.iter() if element_name.lower() in elem.tag.lower()])
        
        if "approval" not in found_elements:
            result.errors.append(XSDValidationError(
                level=XSDValidationErrorLevel.WARNING,
                category=ValidationErrorCategory.DOMAIN_SPECIFIC,
                message="Financial processes should include approval mechanisms",
                suggestion="Add approval steps for financial transactions",
                pattern_reference="finance_domain_patterns"
            ))
            result.errors_by_level[XSDValidationErrorLevel.WARNING] += 1
            result.errors_by_category[ValidationErrorCategory.DOMAIN_SPECIFIC] += 1
    
    def _validate_it_semantics(self, xml_content: str, result: XSDValidationResult) -> None:
        """Validate IT domain-specific BPMN semantics."""
        import xml.etree.ElementTree as ET
        try:
            xml_doc = ET.fromstring(xml_content)
        except:
            return
        
        # Check for IT domain elements
        it_elements = ["system", "deployment", "monitoring", "error_handling", "backup"]
        found_elements = set()
        
        for element_name in it_elements:
            found_elements.update([elem.tag for elem in xml_doc.iter() if element_name.lower() in elem.tag.lower()])
        
        if "error_handling" not in found_elements:
            result.errors.append(XSDValidationError(
                level=XSDValidationErrorLevel.WARNING,
                category=ValidationErrorCategory.DOMAIN_SPECIFIC,
                message="IT processes should include error handling mechanisms",
                suggestion="Add boundary error handling events and compensation flows",
                pattern_reference="it_domain_patterns"
            ))
            result.errors_by_level[XSDValidationErrorLevel.WARNING] += 1
            result.errors_by_category[ValidationErrorCategory.DOMAIN_SPECIFIC] += 1
    
    def _validate_healthcare_semantics(self, xml_content: str, result: XSDValidationResult) -> None:
        """Validate healthcare domain-specific BPMN semantics."""
        import xml.etree.ElementTree as ET
        try:
            xml_doc = ET.fromstring(xml_content)
        except:
            return
        
        # Check for healthcare domain elements
        healthcare_elements = ["patient", "treatment", "assessment", "discharge", "appointment"]
        found_elements = set()
        
        for element_name in healthcare_elements:
            found_elements.update([elem.tag for elem in xml_doc.iter() if element_name.lower() in elem.tag.lower()])
        
        if "patient" not in found_elements:
            result.errors.append(XSDValidationError(
                level=XSDValidationErrorLevel.WARNING,
                category=ValidationErrorCategory.DOMAIN_SPECIFIC,
                message="Healthcare processes should focus on patient-related activities",
                suggestion="Ensure process includes patient-centric elements",
                pattern_reference="healthcare_domain_patterns"
            ))
            result.errors_by_level[XSDValidationErrorLevel.WARNING] += 1
            result.errors_by_category[ValidationErrorCategory.DOMAIN_SPECIFIC] += 1
    
    def _validate_pattern_application(self, xml_content: str, result: XSDValidationResult, patterns_applied: List[str]) -> None:
        """Validate that applied patterns are properly reflected in the XML."""
        result.pattern_matches = patterns_applied  # Store applied patterns
        
        for pattern in patterns_applied:
            # Check if pattern-specific elements are present
            if "approval_sequence" in pattern.lower():
                if "approval" not in xml_content.lower():
                    result.errors.append(XSDValidationError(
                        level=XSDValidationErrorLevel.INFO,
                        category=ValidationErrorCategory.PATTERN_VIOLATION,
                        message=f"Applied pattern '{pattern}' should be reflected in XML content",
                        suggestion="Ensure approval sequence is properly implemented in the generated BPMN",
                        pattern_reference=pattern
                    ))
                    result.errors_by_level[XSDValidationErrorLevel.INFO] += 1
                    result.errors_by_category[ValidationErrorCategory.PATTERN_VIOLATION] += 1
    
    def _convert_validation_level(self, level: ValidationLevel) -> XSDValidationErrorLevel:
        """Convert ValidationLevel enum to XSDValidationErrorLevel."""
        mapping = {
            ValidationLevel.CRITICAL: XSDValidationErrorLevel.FATAL,
            ValidationLevel.ERROR: XSDValidationErrorLevel.ERROR,
            ValidationLevel.WARNING: XSDValidationErrorLevel.WARNING,
            ValidationLevel.INFO: XSDValidationErrorLevel.INFO
        }
        return mapping.get(level, XSDValidationErrorLevel.INFO)
    
    def _calculate_quality_metrics(
        self,
        result: XSDValidationResult,
        xml_content: str,
        graph: Optional[ProcessGraph],
        extraction_result: Optional[ExtractionResultWithErrors]
    ) -> None:
        """Calculate comprehensive quality metrics."""
        metrics = {}
        
        # Structural metrics
        metrics["xml_size_bytes"] = len(xml_content.encode('utf-8'))
        metrics["xml_lines"] = xml_content.count('\n') + 1
        
        # Count BPMN elements
        element_counts = {}
        import xml.etree.ElementTree as ET
        try:
            xml_doc = ET.fromstring(xml_content)
            for elem in xml_doc.iter():
                tag_name = elem.tag.split('}')[-1]  # Remove namespace
                element_counts[tag_name] = element_counts.get(tag_name, 0) + 1
        except:
            pass
        
        metrics["element_counts"] = element_counts
        
        # Compliance metrics
        metrics["xsd_compliance_rate"] = max(0.0, 100.0 - (result.total_errors * 10))
        if result.total_errors + result.total_warnings > 0:
            metrics["xsd_compliance_rate"] = 100.0 - ((result.total_errors * 10) / (result.total_errors + result.total_warnings))
        
        # Semantic metrics (if available)
        semantic_errors = result.errors_by_category.get(ValidationErrorCategory.SEMANTIC, 0)
        metrics["semantic_validity_rate"] = max(0.0, 100.0 - (semantic_errors * 5))
        
        if result.total_errors + result.total_warnings + semantic_errors > 0:
            metrics["semantic_validity_rate"] = 100.0 - (
                (result.total_errors * 10 + semantic_errors * 5) / 
                (result.total_errors + result.total_warnings + semantic_errors)
            )
        
        # Domain-specific metrics
        domain_errors = result.errors_by_category.get(ValidationErrorCategory.DOMAIN_SPECIFIC, 0)
        metrics["domain_compliance_rate"] = max(0.0, 100.0 - (domain_errors * 2))
        
        if result.total_errors + result.total_warnings + semantic_errors + domain_errors > 0:
            metrics["domain_compliance_rate"] = 100.0 - (
                (result.total_errors * 10 + semantic_errors * 5 + domain_errors * 2) / 
                (result.total_errors + result.total_warnings + semantic_errors + domain_errors)
            )
        
        result.metrics = metrics
        result.quality_score = (
            metrics.get("xsd_compliance_rate", 0) * 0.4 +
            metrics.get("semantic_validity_rate", 0) * 0.3 +
            metrics.get("domain_compliance_rate", 0) * 0.2 +
            metrics.get("structure_quality", 0) * 0.1
        )
    
    def _generate_remediation_plan(self, result: XSDValidationResult) -> None:
        """Generate prioritized remediation plan for validation issues."""
        if not result.errors:
            result.remediation_plan.append("No validation issues found - process is compliant")
            return
        
        # Group errors by priority
        critical_errors = [e for e in result.errors if e.level == XSDValidationErrorLevel.FATAL]
        error_errors = [e for e in result.errors if e.level == XSDValidationErrorLevel.ERROR]
        warning_errors = [e for e in result.errors if e.level == XSDValidationErrorLevel.WARNING]
        info_errors = [e for e in result.errors if e.level == XSDValidationErrorLevel.INFO]
        
        plan = []
        
        # Generate plan based on priority
        if critical_errors:
            plan.append("🔥 CRITICAL FIXES REQUIRED:")
            for error in critical_errors[:3]:  # Limit to top 3
                plan.append(f"  - {error.message}")
                if error.suggestion:
                    plan.append(f"     → {error.suggestion}")
                plan.append("")
        
        if error_errors:
            plan.append("❌ ERRORS TO FIX:")
            for error in error_errors[:5]:  # Limit to top 5
                plan.append(f"  - {error.message}")
                if error.suggestion:
                    plan.append(f"     → {error.suggestion}")
                plan.append("")
        
        if warning_errors:
            plan.append("⚠️ WARNINGS TO ADDRESS:")
            for error in warning_errors[:3]:  # Limit to top 3
                plan.append(f"  - {error.message}")
                plan.append(f"     → {error.suggestion}")
                plan.append("")
        
        if info_errors:
            plan.append("ℹ️ IMPROVEMENTS:")
            for error in info_errors[:2]:  # Limit to top 2
                plan.append(f"  - {error.message}")
                if error.suggestion:
                    plan.append(f"     → {error.suggestion}")
                plan.append("")
        
        result.remediation_plan = plan
    
    def _validate_pattern_compliance(self, result: XSDValidationResult, patterns_applied: List[str]) -> None:
        """Validate pattern compliance and update metrics."""
        if not patterns_applied:
            return
        
        # Count pattern-related issues
        pattern_issues = [
            e for e in result.errors 
            if e.category == ValidationErrorCategory.PATTERN_VIOLIATION
        ]
        
        if pattern_issues:
            result.errors_by_category[ValidationErrorCategory.PATTERN_VIOLATION] = len(pattern_issues)
            
            # Adjust metrics for pattern compliance
            pattern_compliance_rate = max(0.0, 100.0 - (len(pattern_issues) * 5))
            result.metrics["pattern_compliance_rate"] = pattern_compliance_rate
            
            # Recalculate overall quality score with pattern compliance
            result.quality_score = (
                result.quality_score * 0.8 + pattern_compliance_rate * 0.2
            )
            
            # Add pattern validation summary
            result.remediation_plan.append("ℹ️ PATTERN COMPLIANCE:")
            for issue in pattern_issues:
                result.remediation_plan.append(f"  - {issue.message}")
                if issue.suggestion:
                    result.remediation_plan.append(f"     → {issue.suggestion}")
            result.remediation_plan.append("")


# Enhanced validation function for direct use
def validate_xml_with_enhanced_xsd(
    xml_content: str,
    graph: Optional[ProcessGraph] = None,
    extraction_result: Optional[ExtractionResultWithErrors] = None,
    domain: Optional[str] = None,
    patterns_applied: Optional[List[str]] = None
) -> XSDValidationResult:
    """Convenience function for enhanced XSD validation.
    
    Args:
        xml_content: XML content to validate
        graph: Optional process graph for context
        extraction_result: Optional extraction result for semantic validation
        domain: Optional process domain
        patterns_applied: Optional list of applied patterns
        
    Returns:
        Enhanced validation result
    """
    validator = EnhancedXSDValidator()
    return validator.validate_xml_against_xsd(
        xml_content, graph, extraction_result, domain, patterns_applied
    )


# Backward compatibility with existing validate_xml_against_xsd
def validate_xml_against_xsd(xml_content: str) -> "XSDValidationResult":
    """Backward compatibility wrapper for enhanced validation."""
    return validate_xml_with_enhanced_xsd(xml_content)