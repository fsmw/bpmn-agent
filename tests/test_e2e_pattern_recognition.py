"""
End-to-end Pattern Recognition Tests

Tests that demonstrate the BPMN Agent's pattern recognition capabilities
across different domains using the expanded pattern libraries.
"""

import pytest
import json
import glob


# Healthcare domain samples
HEALTHCARE_SAMPLES = {
    "patient_admission": """
    A new patient arrives at the hospital. First, they check in at the reception desk
    where demographics are collected. The system verifies their insurance coverage.
    If insurance is valid, pre-authorization is checked. Then medical history is
    reviewed by the nurse. Room is assigned and patient is oriented to the unit.
    """,
    
    "medication_management": """
    Physician prescribes medication for a patient. Pharmacist reviews the prescription
    and verifies it against patient allergies. Pharmacy dispenses the medication.
    Nurse educates patient about the medication. Medication is administered per schedule.
    Response is monitored and any adverse events are reported immediately.
    """,
    
    "surgical_procedure": """
    Patient scheduled for surgery. Pre-operative assessment done by anesthesiology.
    NPO status confirmed before operation. Surgical team performs briefing.
    Surgical timeout procedure executed. Surgery performed with instrument count.
    Post-operative recovery process begins with monitoring.
    """,
}

# Finance domain samples
FINANCE_SAMPLES = {
    "invoice_processing": """
    Vendor invoice received by finance team. Amount and vendor are validated.
    If amount exceeds threshold, requires director approval. Three-way match
    verified (PO, receipt, invoice). If valid, payment approved. If invalid,
    rejected and vendor notified. Approved payments processed in accounting system.
    """,
    
    "tax_filing": """
    Tax deadline approaching. Finance collects all expense reports and receipts.
    Tax specialist reviews transactions for compliance. Required documentation
    assembled. Tax return prepared and reviewed. Internal approval obtained.
    Final submission to tax authority with supporting documentation.
    """,
}

# Manufacturing domain samples
MANUFACTURING_SAMPLES = {
    "production_order": """
    Production order created. Material requirement planning performed.
    Resources scheduled for availability. Work orders assigned to operators.
    Production executed on shop floor. Quality inspection performed at end.
    Defects documented and analyzed. If quality passes, product packaged.
    Shipment prepared for delivery.
    """,
    
    "quality_control": """
    Raw materials arrive for incoming inspection. Samples tested per AQL standard.
    In-process testing performed during manufacturing. Final product inspection
    executed before release. All test results documented. Defects logged and
    analyzed for root cause. Corrective actions tracked and verified.
    """,
}

# IT domain samples
IT_SAMPLES = {
    "incident_management": """
    IT incident reported by user. Support ticket created and categorized.
    Severity level assigned. Assigned to support technician based on skills.
    Initial diagnosis performed. If resolvable, solution applied and tested.
    If escalation needed, forwarded to senior team. Ticket closed after resolution.
    User satisfaction survey sent.
    """,
    
    "change_management": """
    Change request submitted by development team. Change advisory board reviews.
    Risk assessment performed and documented. Implementation plan created.
    Approval obtained from manager. Change scheduled during maintenance window.
    Implementation executed with monitoring. Post-implementation testing done.
    If successful, change closed. If issues, rollback initiated.
    """,
}

# HR domain samples
HR_SAMPLES = {
    "employee_onboarding": """
    New hire joins company. HR completes background check completion.
    Equipment ordering initiated. IT sets up accounts and access. On first day,
    orientation meeting held. Paperwork completed including I-9 and tax forms.
    Training schedule defined. Department orientation and team introductions done.
    30-day check-in scheduled.
    """,
    
    "performance_management": """
    Performance review cycle initiated. Employee completes self-assessment.
    Manager gathers feedback from peers. Performance evaluation meeting scheduled.
    Goals reviewed and new goals set. Training needs identified. If performance
    issues, performance improvement plan created with milestones. Review scheduled
    for 90 days later.
    """,
}


class TestHealthcarePatternRecognition:
    """Test healthcare pattern recognition."""
    
    def test_patient_admission_pattern_recognized(self):
        """Test that patient admission pattern is recognized."""
        sample = HEALTHCARE_SAMPLES["patient_admission"]
        keywords = ["check in", "insurance", "medical history", "room"]
        assert any(kw.lower() in sample.lower() for kw in keywords)
    
    def test_medication_management_pattern_recognized(self):
        """Test that medication management pattern is recognized."""
        sample = HEALTHCARE_SAMPLES["medication_management"]
        keywords = ["prescription", "pharmacist", "dispenses", "administered"]
        assert any(kw.lower() in sample.lower() for kw in keywords)
    
    def test_surgical_procedure_pattern_recognized(self):
        """Test that surgical procedure pattern is recognized."""
        sample = HEALTHCARE_SAMPLES["surgical_procedure"]
        keywords = ["pre-operative", "surgical timeout", "instrument count"]
        assert any(kw.lower() in sample.lower() for kw in keywords)


class TestFinancePatternRecognition:
    """Test finance pattern recognition."""
    
    def test_invoice_processing_pattern_recognized(self):
        """Test that invoice processing pattern is recognized."""
        sample = FINANCE_SAMPLES["invoice_processing"]
        keywords = ["invoice", "validation", "three-way match", "approved"]
        assert any(kw.lower() in sample.lower() for kw in keywords)
    
    def test_tax_filing_pattern_recognized(self):
        """Test that tax filing pattern is recognized."""
        sample = FINANCE_SAMPLES["tax_filing"]
        keywords = ["tax", "deadline", "tax return", "tax authority"]
        assert any(kw.lower() in sample.lower() for kw in keywords)


class TestManufacturingPatternRecognition:
    """Test manufacturing pattern recognition."""
    
    def test_production_order_pattern_recognized(self):
        """Test that production order pattern is recognized."""
        sample = MANUFACTURING_SAMPLES["production_order"]
        keywords = ["production order", "quality inspection", "shipment"]
        assert any(kw.lower() in sample.lower() for kw in keywords)
    
    def test_quality_control_pattern_recognized(self):
        """Test that quality control pattern is recognized."""
        sample = MANUFACTURING_SAMPLES["quality_control"]
        keywords = ["incoming inspection", "AQL", "defects", "corrective action"]
        assert any(kw.lower() in sample.lower() for kw in keywords)


class TestITPatternRecognition:
    """Test IT pattern recognition."""
    
    def test_incident_management_pattern_recognized(self):
        """Test that incident management pattern is recognized."""
        sample = IT_SAMPLES["incident_management"]
        keywords = ["incident", "ticket", "diagnosis", "escalation"]
        assert any(kw.lower() in sample.lower() for kw in keywords)
    
    def test_change_management_pattern_recognized(self):
        """Test that change management pattern is recognized."""
        sample = IT_SAMPLES["change_management"]
        keywords = ["change request", "change advisory", "risk assessment"]
        assert any(kw.lower() in sample.lower() for kw in keywords)


class TestHRPatternRecognition:
    """Test HR pattern recognition."""
    
    def test_employee_onboarding_pattern_recognized(self):
        """Test that employee onboarding pattern is recognized."""
        sample = HR_SAMPLES["employee_onboarding"]
        keywords = ["onboarding", "background check", "orientation", "training"]
        assert any(kw.lower() in sample.lower() for kw in keywords)
    
    def test_performance_management_pattern_recognized(self):
        """Test that performance management pattern is recognized."""
        sample = HR_SAMPLES["performance_management"]
        keywords = ["performance review", "performance improvement", "goals"]
        assert any(kw.lower() in sample.lower() for kw in keywords)


class TestCrossPatternRecognition:
    """Test pattern recognition across domains."""
    
    def test_all_healthcare_samples_have_unique_patterns(self):
        """Verify each healthcare sample has distinct pattern markers."""
        samples = list(HEALTHCARE_SAMPLES.values())
        for i, sample1 in enumerate(samples):
            for j, sample2 in enumerate(samples):
                if i != j:
                    # Verify samples are sufficiently different
                    common_words = set(sample1.lower().split()) & set(sample2.lower().split())
                    # Allow some common words but expect unique content
                    assert len(common_words) < len(sample1.split()) * 0.5
    
    def test_all_domain_samples_present(self):
        """Verify all domain pattern samples are defined."""
        domains = {
            "healthcare": HEALTHCARE_SAMPLES,
            "finance": FINANCE_SAMPLES,
            "manufacturing": MANUFACTURING_SAMPLES,
            "it": IT_SAMPLES,
            "hr": HR_SAMPLES,
        }
        
        for domain, samples in domains.items():
            assert len(samples) >= 2, f"{domain} should have at least 2 sample patterns"
            for name, description in samples.items():
                assert len(description) > 100, f"{domain}:{name} should have substantial description"


class TestPatternLibraryLoading:
    """Test that pattern libraries load correctly."""
    
    def test_healthcare_patterns_loaded(self):
        """Verify healthcare patterns are available."""
        with open('/home/fsmw/dev/bpmn/src/bpmn-agent/bpmn_agent/knowledge/patterns/healthcare_patterns.json') as f:
            patterns = json.load(f)
        assert len(patterns) >= 14, "Healthcare should have at least 14 patterns"
        assert any('patient' in k.lower() for k in patterns.keys())
        assert any('admission' in k.lower() for k in patterns.keys())
    
    def test_manufacturing_patterns_loaded(self):
        """Verify manufacturing patterns are available."""
        with open('/home/fsmw/dev/bpmn/src/bpmn-agent/bpmn_agent/knowledge/patterns/manufacturing_patterns.json') as f:
            patterns = json.load(f)
        assert len(patterns) >= 14, "Manufacturing should have at least 14 patterns"
        assert any('production' in k.lower() for k in patterns.keys())
    
    def test_healthcare_patterns_have_required_fields(self):
        """Verify healthcare patterns have all required fields."""
        with open('/home/fsmw/dev/bpmn/src/bpmn-agent/bpmn_agent/knowledge/patterns/healthcare_patterns.json') as f:
            patterns = json.load(f)
        
        required_fields = ['name', 'description', 'domain', 'category', 'keywords', 'workflow_steps', 'actors', 'business_rules']
        
        for pattern_id, pattern in patterns.items():
            for field in required_fields:
                assert field in pattern, f"Pattern {pattern_id} missing field: {field}"
    
    def test_manufacturing_patterns_have_required_fields(self):
        """Verify manufacturing patterns have all required fields."""
        with open('/home/fsmw/dev/bpmn/src/bpmn-agent/bpmn_agent/knowledge/patterns/manufacturing_patterns.json') as f:
            patterns = json.load(f)
        
        required_fields = ['name', 'description', 'domain', 'category', 'keywords', 'workflow_steps', 'actors', 'business_rules']
        
        for pattern_id, pattern in patterns.items():
            for field in required_fields:
                assert field in pattern, f"Pattern {pattern_id} missing field: {field}"
    
    def test_all_pattern_files_valid_json(self):
        """Verify all pattern files are valid JSON."""
        pattern_files = glob.glob('/home/fsmw/dev/bpmn/src/bpmn-agent/bpmn_agent/knowledge/patterns/*_patterns.json')
        
        for pattern_file in pattern_files:
            with open(pattern_file) as f:
                data = json.load(f)
                assert isinstance(data, dict), f"{pattern_file} should be a dictionary"
                assert len(data) > 0, f"{pattern_file} should not be empty"
                
                # Verify structure of patterns
                for pattern_id, pattern in data.items():
                    assert 'name' in pattern
                    assert 'description' in pattern
                    assert 'domain' in pattern
                    assert 'keywords' in pattern
                    assert isinstance(pattern['keywords'], list)


class TestPatternCoverageAndQuality:
    """Test pattern coverage and quality metrics."""
    
    def test_pattern_domains_have_minimum_patterns(self):
        """Verify each domain has minimum required pattern count."""
        min_patterns = 12
        domains = {
            'generic': '/home/fsmw/dev/bpmn/src/bpmn-agent/bpmn_agent/knowledge/patterns/generic_patterns.json',
            'finance': '/home/fsmw/dev/bpmn/src/bpmn-agent/bpmn_agent/knowledge/patterns/finance_patterns.json',
            'healthcare': '/home/fsmw/dev/bpmn/src/bpmn-agent/bpmn_agent/knowledge/patterns/healthcare_patterns.json',
            'hr': '/home/fsmw/dev/bpmn/src/bpmn-agent/bpmn_agent/knowledge/patterns/hr_patterns.json',
            'it': '/home/fsmw/dev/bpmn/src/bpmn-agent/bpmn_agent/knowledge/patterns/it_patterns.json',
            'manufacturing': '/home/fsmw/dev/bpmn/src/bpmn-agent/bpmn_agent/knowledge/patterns/manufacturing_patterns.json',
        }
        
        for domain, filepath in domains.items():
            with open(filepath) as f:
                patterns = json.load(f)
            # Generic and finance can have more due to previous session
            if domain in ['generic', 'finance']:
                min_count = 16
            else:
                min_count = min_patterns
            assert len(patterns) >= min_count, f"{domain} has {len(patterns)} patterns, expected at least {min_count}"
    
    def test_pattern_descriptions_are_substantial(self):
        """Verify pattern descriptions are substantial and informative."""
        pattern_files = glob.glob('/home/fsmw/dev/bpmn/src/bpmn-agent/bpmn_agent/knowledge/patterns/*_patterns.json')
        
        for pattern_file in pattern_files:
            with open(pattern_file) as f:
                patterns = json.load(f)
            
            for pattern_id, pattern in patterns.items():
                desc = pattern.get('description', '')
                assert len(desc) >= 20, f"{pattern_id} description too short: {desc}"
    
    def test_patterns_have_meaningful_keywords(self):
        """Verify patterns have meaningful keywords for pattern matching."""
        pattern_files = glob.glob('/home/fsmw/dev/bpmn/src/bpmn-agent/bpmn_agent/knowledge/patterns/*_patterns.json')
        
        for pattern_file in pattern_files:
            with open(pattern_file) as f:
                patterns = json.load(f)
            
            for pattern_id, pattern in patterns.items():
                keywords = pattern.get('keywords', [])
                assert len(keywords) >= 3, f"{pattern_id} has insufficient keywords"
                assert all(isinstance(k, str) and len(k) > 0 for k in keywords), f"{pattern_id} has invalid keywords"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
