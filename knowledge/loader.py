"""
Pattern Library Loader

Loads BPMN pattern libraries from JSON files and populates the knowledge base.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from models.knowledge_base import (
    BPMNPattern,
    ComplexityLevel,
    ContextPackage,
    DomainExample,
    DomainType,
    GraphStructure,
    KnowledgeBase,
)

logger = logging.getLogger(__name__)


class PatternLibraryLoader:
    """
    Loads BPMN pattern and example libraries from JSON files.
    """
    
    def __init__(self, patterns_dir: Optional[Path] = None):
        """
        Initialize the pattern loader.
        
        Args:
            patterns_dir: Directory containing pattern JSON files.
                         If None, uses default location relative to this module.
        """
        if patterns_dir is None:
            # Default to patterns directory in same location as this module
            patterns_dir = Path(__file__).parent / "patterns"
        
        self.patterns_dir = Path(patterns_dir)
        self.logger = logger
    
    def load_all_patterns(self) -> KnowledgeBase:
        """
        Load all patterns from the patterns directory.
        
        Returns:
            Populated KnowledgeBase object
        """
        kb = KnowledgeBase()
        
        # Load patterns from each domain file
        domain_files = [
            (DomainType.HR, "hr_patterns.json"),
            (DomainType.FINANCE, "finance_patterns.json"),
            (DomainType.IT, "it_patterns.json"),
            (DomainType.HEALTHCARE, "healthcare_patterns.json"),
            (DomainType.MANUFACTURING, "manufacturing_patterns.json"),
            (DomainType.GENERIC, "generic_patterns.json"),
        ]
        
        for domain, filename in domain_files:
            patterns = self.load_domain_patterns(domain, filename)
            for pattern_id, pattern in patterns.items():
                kb.add_pattern(pattern)
            self.logger.info(
                f"Loaded {len(patterns)} patterns from {filename} ({domain.value})"
            )
        
        # Load examples if available
        examples_file = self.patterns_dir / "examples.json"
        if examples_file.exists():
            try:
                with open(examples_file, 'r') as f:
                    examples_data = json.load(f)
                
                for example_id, example_dict in examples_data.items():
                    example = self._dict_to_example(example_id, example_dict)
                    kb.add_example(example)
                
                self.logger.info(f"Loaded {len(examples_data)} examples")
            except Exception as e:
                self.logger.warning(f"Failed to load examples: {e}")
        
        return kb
    
    def load_domain_patterns(
        self,
        domain: DomainType,
        filename: Optional[str] = None
    ) -> Dict[str, BPMNPattern]:
        """
        Load patterns for a specific domain.
        
        Args:
            domain: Domain type
            filename: Optional specific filename. If None, constructs from domain name.
            
        Returns:
            Dictionary of patterns indexed by ID
        """
        if filename is None:
            filename = f"{domain.value}_patterns.json"
        
        filepath = self.patterns_dir / filename
        patterns: Dict[str, BPMNPattern] = {}
        
        if not filepath.exists():
            self.logger.warning(f"Pattern file not found: {filepath}")
            return patterns
        
        try:
            with open(filepath, 'r') as f:
                patterns_data = json.load(f)
            
            for pattern_id, pattern_dict in patterns_data.items():
                try:
                    pattern = self._dict_to_pattern(pattern_id, pattern_dict)
                    patterns[pattern_id] = pattern
                except Exception as e:
                    self.logger.error(f"Failed to parse pattern {pattern_id}: {e}")
            
            self.logger.info(f"Loaded {len(patterns)} patterns from {filename}")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from {filepath}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load patterns from {filepath}: {e}")
        
        return patterns
    
    @staticmethod
    def _dict_to_pattern(pattern_id: str, pattern_dict: Dict) -> BPMNPattern:
        """
        Convert a dictionary to a BPMNPattern object.
        
        Args:
            pattern_id: Pattern ID
            pattern_dict: Dictionary with pattern data
            
        Returns:
            BPMNPattern object
        """
        # Handle graph structure
        graph_data = pattern_dict.get("graph_structure", {})
        graph_structure = GraphStructure(
            nodes=graph_data.get("nodes", []),
            edges=graph_data.get("edges", []),
            node_types=graph_data.get("node_types", {})
        )
        
        # Create pattern
        pattern = BPMNPattern(
            id=pattern_id,
            name=pattern_dict.get("name", pattern_id),
            description=pattern_dict.get("description", ""),
            domain=DomainType(pattern_dict.get("domain", "generic")),
            category=pattern_dict.get("category", "sequential"),
            complexity=ComplexityLevel(
                pattern_dict.get("complexity", "moderate")
            ),
            graph_structure=graph_structure,
            examples=pattern_dict.get("examples", []),
            validation_rules=pattern_dict.get("validation_rules", []),
            anti_patterns=pattern_dict.get("anti_patterns", []),
            tags=set(pattern_dict.get("tags", [])),
            related_patterns=pattern_dict.get("related_patterns", []),
            confidence=pattern_dict.get("confidence", 0.9),
            version=pattern_dict.get("version", "1.0"),
        )
        
        return pattern
    
    @staticmethod
    def _dict_to_example(example_id: str, example_dict: Dict) -> DomainExample:
        """
        Convert a dictionary to a DomainExample object.
        
        Args:
            example_id: Example ID
            example_dict: Dictionary with example data
            
        Returns:
            DomainExample object
        """
        example = DomainExample(
            id=example_id,
            text=example_dict.get("text", ""),
            domain=DomainType(example_dict.get("domain", "generic")),
            complexity=ComplexityLevel(
                example_dict.get("complexity", "moderate")
            ),
            patterns_used=example_dict.get("patterns_used", []),
            entities_expected=example_dict.get("entities_expected", {}),
            relations_expected=example_dict.get("relations_expected", {}),
            difficulty=example_dict.get("difficulty", "medium"),
            validation_score=example_dict.get("validation_score", 0.8),
            bpmn_structure=example_dict.get("bpmn_structure"),
        )
        
        return example
    
    def ensure_patterns_dir_exists(self) -> Path:
        """
        Ensure patterns directory exists, creating it if needed.
        
        Returns:
            Path to patterns directory
        """
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        return self.patterns_dir
    
    def save_pattern(self, pattern: BPMNPattern, domain: Optional[DomainType] = None) -> None:
        """
        Save a pattern to the patterns directory.
        
        Args:
            pattern: Pattern to save
            domain: Optional domain override. If None, uses pattern's domain.
        """
        if domain is None:
            domain = pattern.domain
        
        filename = f"{domain.value}_patterns.json"
        filepath = self.patterns_dir / filename
        
        # Ensure directory exists
        self.ensure_patterns_dir_exists()
        
        # Load existing patterns
        patterns_data: Dict = {}
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    patterns_data = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load existing patterns: {e}")
        
        # Add/update pattern
        patterns_data[pattern.id] = {
            "name": pattern.name,
            "description": pattern.description,
            "domain": pattern.domain.value,
            "category": pattern.category.value,
            "complexity": pattern.complexity.value,
            "graph_structure": {
                "nodes": pattern.graph_structure.nodes,
                "edges": pattern.graph_structure.edges,
                "node_types": pattern.graph_structure.node_types,
            },
            "examples": pattern.examples,
            "validation_rules": pattern.validation_rules,
            "anti_patterns": pattern.anti_patterns,
            "tags": list(pattern.tags),
            "related_patterns": pattern.related_patterns,
            "confidence": pattern.confidence,
            "version": pattern.version,
        }
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            self.logger.info(f"Saved pattern {pattern.id} to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save pattern: {e}")
    
    def save_example(self, example: DomainExample) -> None:
        """
        Save an example to the examples.json file.
        
        Args:
            example: Example to save
        """
        # Ensure directory exists
        self.ensure_patterns_dir_exists()
        
        examples_file = self.patterns_dir / "examples.json"
        
        # Load existing examples
        examples_data: Dict = {}
        if examples_file.exists():
            try:
                with open(examples_file, 'r') as f:
                    examples_data = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load existing examples: {e}")
        
        # Add/update example
        examples_data[example.id] = {
            "text": example.text,
            "domain": example.domain.value,
            "complexity": example.complexity.value,
            "patterns_used": example.patterns_used,
            "entities_expected": example.entities_expected,
            "relations_expected": example.relations_expected,
            "difficulty": example.difficulty,
            "validation_score": example.validation_score,
            "bpmn_structure": example.bpmn_structure,
        }
        
        # Save to file
        try:
            with open(examples_file, 'w') as f:
                json.dump(examples_data, f, indent=2)
            self.logger.info(f"Saved example {example.id}")
        except Exception as e:
            self.logger.error(f"Failed to save example: {e}")
