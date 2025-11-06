#!/bin/bash
# Phase 4 Validation & Quality Assurance Demo
# This script tests the enhanced XSD validation capabilities

echo "🎭 Phase 4 Validation & Quality Assurance Demo"
echo "=" 60

cd /home/fsmw/dev/bpmn-agent

echo "🧪 Testing Phase 4 Enhanced XSD Validation..."

# Test 1: Validate basic BPMN XML
python -c "
import sys
sys.path.insert(0, '/home/fsmw/dev/bpmn-agent/src/bpmn-agent')
from bpmn_agent.validation.enhanced_xsd_validation import EnhancedXSDValidator

# Test enhanced XSD validation
validator = EnhancedXSDValidator()
test_xml = '''<?xml version='1.0' encoding='UTF-8'?>
<definitions xmlns='http://www.omg.org/spec/BPMN/20100524/MODEL'>
    <process id='test_process' name='Test Process' isExecutable='true'>
        <startEvent id='start'/>
        <task id='task_1' name='Process Start'/>
        <endEvent id='end_1'/>
        <sequenceFlow sourceRef='start' targetRef='task_1'/>
        <sequenceFlow sourceRef='task_1' targetRef='end_1'/>
    </process>
</definitions>'''

result = validator.validate_xml_with_phase4_enhancement(test_xml)
echo "✅ Basic XSD Validation Complete:"
echo "  - Valid: $result.is_valid"
echo "  - Errors: $result.total_errors"
echo "  - Warnings: $result.total_warnings"
echo "  - Quality Score: $result.quality_score"

# Test 2: Validate with domain awareness
echo ""
echo "🔧 Testing Domain-Aware Validation..."

domain_xml = '''<?xml version='1.0' encoding='UTF-8'?>
<definitions xmlns='http://www.omg.org/spec/BPMN/20100524/MODEL'>
    <process id='hr_process' name='Employee Onboarding'>
        <startEvent id='start'/>
        <task id='orientation' name='Employee Orientation'/>
        <userTask id='manager_review' name='Manager Review'/>
        <exclusiveGateway id='decision'/>
        <task id='approval' name='Approval'/>
        <endEvent id='end'/>
        <sequenceFlow sourceRef='start' targetRef='orientation'/>
        <sequenceFlow sourceRef='orientation' targetRef='manager_review'/>
        <sequenceFlow sourceRef='manager_review' targetRef='decision'/>
        <sequenceFlow sourceRef='decision' targetRef='approval'/>
        <sequenceFlow sourceRef='approval' targetRef='end'/>
    </process>
</definitions>'''
    
domain_result = validator.validate_xml_with_phase4_enhancement(domain_xml, domain='hr')
echo "✅ Domain-Aware Validation Complete:"
echo "  - Valid: $domain_result.is_valid"
echo "  - Errors: $domain_result.total_errors"
echo "  - Warnings: $domain_result.total_warnings"
echo "  - Quality Score: $domain_result.quality_score"
echo "  - Issues: $domain_result.remediation_plan"

echo ""
echo "✅ Phase 4 Enhanced XSD Validation Demo Completed!"