import json
import os
from pathlib import Path

# Category mapping: domain-specific categories to valid BPMN categories
CATEGORY_MAPPING = {
    # HR
    "recruitment": "sequential",
    "onboarding": "sequential", 
    "performance": "exclusive_choice",
    "career": "sequential",
    "time-off": "exclusive_choice",
    "offboarding": "sequential",
    "development": "multi_instance",
    "compliance": "data_flow",
    "relations": "event_driven",
    "compensation": "parallel",
    "benefits": "parallel",
    # Finance
    "payment": "sequential",
    "expense": "exclusive_choice",
    "planning": "data_flow",
    "accounting": "sequential",
    "lending": "exclusive_choice",
    "procurement": "sequential",
    "investment": "data_flow",
    "credit_management": "exclusive_choice",
    "control": "data_flow",
    "human_capital": "sequential",
    "liabilities": "data_flow",
    # IT
    "change": "exclusive_choice",
    "incident": "event_driven",
    "access": "exclusive_choice",
    "asset": "data_flow",
    "monitoring": "event_driven",
    "backup": "sequential",
    "security": "parallel",
    "maintenance": "sequential",
    "problem": "event_driven",
    "release": "sequential",
    "vendor": "sequential",
    "project": "sequential",
    "infrastructure": "parallel",
    # Healthcare
    "admission": "sequential",
    "treatment": "parallel",
    "medication": "sequential",
    "discharge": "sequential",
    "diagnostics": "exclusive_choice",
    "billing": "exclusive_choice",
    "quality": "data_flow",
    "records": "data_flow",
    "education": "multi_instance",
    "surgery": "sequential",
    # Manufacturing
    "production": "parallel",
    "quality": "data_flow",
    "inventory": "data_flow",
    "maintenance": "event_driven",
    "logistics": "sequential",
    "development": "sequential",
    "operations": "parallel",
    "tracking": "data_flow",
}

# Complexity mapping: ensure only valid values
COMPLEXITY_MAPPING = {
    "simple": "simple",
    "easy": "simple",
    "low": "simple",
    "moderate": "moderate", 
    "medium": "moderate",
    "complex": "complex",
    "hard": "complex",
    "high": "complex",
}

patterns_dir = Path("bpmn_agent/knowledge/patterns")
files = [
    "generic_patterns.json",
    "hr_patterns.json",
    "finance_patterns.json",
    "it_patterns.json",
    "healthcare_patterns.json",
    "manufacturing_patterns.json",
]

for filename in files:
    filepath = patterns_dir / filename
    if not filepath.exists():
        print(f"Skipping {filename} - not found")
        continue
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    fixed_count = 0
    for pattern_id, pattern in data.items():
        # Fix category
        original_category = pattern.get("category", "sequential")
        if original_category in CATEGORY_MAPPING:
            pattern["category"] = CATEGORY_MAPPING[original_category]
            fixed_count += 1
        elif original_category not in ["sequential", "parallel", "exclusive_choice", "inclusive_choice", "multi_instance", "event_driven", "exception_handling", "synchronization", "message_passing", "data_flow"]:
            # Map to a sensible default based on pattern name
            pattern["category"] = "sequential"  # Default
            fixed_count += 1
        
        # Fix complexity
        original_complexity = pattern.get("complexity", "moderate")
        if original_complexity in COMPLEXITY_MAPPING:
            pattern["complexity"] = COMPLEXITY_MAPPING[original_complexity]
            fixed_count += 1
        elif original_complexity not in ["simple", "moderate", "complex"]:
            pattern["complexity"] = "moderate"  # Default
            fixed_count += 1
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Fixed {filename}: {fixed_count} patterns updated")

print("\nPattern files fixed!")
