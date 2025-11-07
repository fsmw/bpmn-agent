#!/usr/bin/env python3
"""
KB Integration Validation Script - Lightweight Version
Tests all KB components without requiring sentence-transformers
"""

import json
import sys
from pathlib import Path


def test_json_files():
    """Test that all JSON pattern files are valid"""
    print("\n" + "=" * 80)
    print("TEST 1: Pattern JSON Files Validation")
    print("=" * 80)

    patterns_dir = Path(__file__).parent / "patterns"
    json_files = sorted(patterns_dir.glob("*.json"))

    print(f"\n✓ Found {len(json_files)} JSON files:")

    total_patterns = 0
    total_examples = 0

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Count patterns/examples
            if json_file.name == "examples.json":
                count = len(data)
                print(f"  - {json_file.name}: {count} domain examples ✓")
                total_examples += count
            else:
                count = len(data)
                print(f"  - {json_file.name}: {count} patterns ✓")
                total_patterns += count

        except json.JSONDecodeError as e:
            print(f"  - {json_file.name}: ✗ JSON Parse Error: {e}")
            return False
        except Exception as e:
            print(f"  - {json_file.name}: ✗ Error: {e}")
            return False

    print("\n✓ Total statistics:")
    print(f"  - Total patterns: {total_patterns}")
    print(f"  - Total examples: {total_examples}")
    print(f"  - Files validated: {len(json_files)}")

    return True


def test_pattern_structure():
    """Test that patterns have the correct structure"""
    print("\n" + "=" * 80)
    print("TEST 2: Pattern Structure Validation")
    print("=" * 80)

    patterns_dir = Path(__file__).parent / "patterns"
    required_fields = {
        "name",
        "description",
        "domain",
        "category",
        "complexity",
        "graph_structure",
        "examples",
        "validation_rules",
        "anti_patterns",
        "tags",
        "confidence",
        "version",
    }

    pattern_count = 0
    domain_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}

    for json_file in sorted(patterns_dir.glob("*_patterns.json")):
        with open(json_file, "r") as f:
            data = json.load(f)

        for pattern_id, pattern in data.items():
            # Check required fields
            missing_fields = required_fields - set(pattern.keys())
            if missing_fields:
                print(f"✗ Pattern {pattern_id} missing fields: {missing_fields}")
                return False

            # Check domain and category values
            if "domain" in pattern:
                domain_counts[pattern["domain"]] = domain_counts.get(pattern["domain"], 0) + 1
            if "category" in pattern:
                category_counts[pattern["category"]] = (
                    category_counts.get(pattern["category"], 0) + 1
                )

            pattern_count += 1

    print(f"\n✓ Validated {pattern_count} patterns")
    print("  Domains covered:")
    for domain, count in sorted(domain_counts.items()):
        print(f"    - {domain}: {count} patterns")

    print("\n  Pattern categories used:")
    for category, count in sorted(category_counts.items()):
        print(f"    - {category}: {count} patterns")

    return True


def test_example_structure():
    """Test that examples have the correct structure"""
    print("\n" + "=" * 80)
    print("TEST 3: Example Structure Validation")
    print("=" * 80)

    examples_file = Path(__file__).parent / "patterns" / "examples.json"
    required_fields = {
        "text",
        "domain",
        "complexity",
        "pattern_id",
        "entities",
        "relations",
        "confidence",
        "source",
    }

    with open(examples_file, "r") as f:
        data = json.load(f)

    example_count = 0
    domain_examples = {}

    for example_id, example in data.items():
        # Check required fields
        missing_fields = required_fields - set(example.keys())
        if missing_fields:
            print(f"✗ Example {example_id} missing fields: {missing_fields}")
            return False

        # Track domains
        if "domain" in example:
            domain_examples[example["domain"]] = domain_examples.get(example["domain"], 0) + 1

        example_count += 1

    print(f"\n✓ Validated {example_count} examples")
    print("  Domains covered:")
    for domain, count in sorted(domain_examples.items()):
        print(f"    - {domain}: {count} examples")

    return True


def test_cross_references():
    """Test that examples reference valid patterns"""
    print("\n" + "=" * 80)
    print("TEST 4: Cross-Reference Validation")
    print("=" * 80)

    patterns_dir = Path(__file__).parent / "patterns"

    # Load all pattern IDs
    pattern_ids = set()
    for json_file in patterns_dir.glob("*_patterns.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            pattern_ids.update(data.keys())

    print(f"\n✓ Found {len(pattern_ids)} unique patterns")

    # Check examples reference valid patterns
    examples_file = patterns_dir / "examples.json"
    with open(examples_file, "r") as f:
        examples = json.load(f)

    invalid_refs = 0
    for example_id, example in examples.items():
        if "pattern_id" in example:
            if example["pattern_id"] not in pattern_ids:
                print(f"✗ Example {example_id} references invalid pattern: {example['pattern_id']}")
                invalid_refs += 1

    if invalid_refs == 0:
        print(f"✓ All {len(examples)} examples reference valid patterns")
    else:
        print(f"✗ Found {invalid_refs} invalid pattern references")
        return False

    return True


def test_domain_coverage():
    """Test that all domains are covered"""
    print("\n" + "=" * 80)
    print("TEST 5: Domain Coverage Analysis")
    print("=" * 80)

    patterns_dir = Path(__file__).parent / "patterns"

    # Expected domains from DomainType
    expected_domains = {"generic", "hr", "it", "finance", "healthcare", "manufacturing"}

    # Find domains in patterns
    domains_found = set()
    pattern_count_by_domain: dict[str, int] = {}

    for json_file in patterns_dir.glob("*_patterns.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        for pattern in data.values():
            domain = pattern.get("domain")
            if domain:
                domains_found.add(domain)
                pattern_count_by_domain[domain] = pattern_count_by_domain.get(domain, 0) + 1

    missing_domains = expected_domains - domains_found
    extra_domains = domains_found - expected_domains

    if missing_domains:
        print(f"✗ Missing domains: {missing_domains}")
        return False

    if extra_domains:
        print(f"⚠ Extra domains found: {extra_domains}")

    print("\n✓ All expected domains covered:")
    for domain in sorted(expected_domains):
        count = pattern_count_by_domain.get(domain, 0)
        print(f"  - {domain}: {count} patterns")

    return True


def test_pattern_categories():
    """Test that pattern categories are valid"""
    print("\n" + "=" * 80)
    print("TEST 6: Pattern Category Validation")
    print("=" * 80)

    # Expected categories from PatternCategory
    expected_categories = {
        "sequential",
        "parallel",
        "exclusive_choice",
        "inclusive_choice",
        "multi_instance",
        "event_driven",
        "exception_handling",
        "data_flow",
    }

    patterns_dir = Path(__file__).parent / "patterns"
    categories_found = set()
    category_count: dict[str, int] = {}

    for json_file in patterns_dir.glob("*_patterns.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        for pattern in data.values():
            category = pattern.get("category")
            if category:
                categories_found.add(category)
                category_count[category] = category_count.get(category, 0) + 1

    print(f"\n✓ Found {len(categories_found)} pattern categories:")
    for category in sorted(categories_found):
        count = category_count.get(category, 0)
        status = "✓" if category in expected_categories else "⚠ (extra)"
        print(f"  - {category}: {count} patterns {status}")

    missing = expected_categories - categories_found
    if missing:
        print(f"\n⚠ Missing categories: {missing}")

    return True


def main():
    """Run all validation tests"""
    try:
        print("\n" + "=" * 80)
        print("KB INTEGRATION VALIDATION SUITE - LIGHTWEIGHT")
        print("=" * 80)

        tests = [
            ("JSON Files", test_json_files),
            ("Pattern Structure", test_pattern_structure),
            ("Example Structure", test_example_structure),
            ("Cross-References", test_cross_references),
            ("Domain Coverage", test_domain_coverage),
            ("Pattern Categories", test_pattern_categories),
        ]

        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"\n✗ Test failed with exception: {e}")
                import traceback

                traceback.print_exc()
                results.append((test_name, False))

        # Final summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        passed = sum(1 for _, result in results if result)
        total = len(results)

        for test_name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status}: {test_name}")

        print(f"\n{passed}/{total} tests passed")

        if passed == total:
            print("\n✓ All KB validations completed successfully!")
            return 0
        else:
            print(f"\n✗ {total - passed} validation(s) failed")
            return 1

    except Exception as e:
        print(f"\n✗ Validation suite failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
