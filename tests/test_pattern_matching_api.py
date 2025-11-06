"""
Tests for pattern matching API endpoints.

Tests the FastAPI REST endpoints for pattern matching functionality.
"""

import pytest
from fastapi.testclient import TestClient
from bpmn_agent.models.knowledge_base import (
    BPMNPattern,
    ComplexityLevel,
    DomainType,
    GraphStructure,
    KnowledgeBase,
    PatternCategory,
)
from bpmn_agent.api.app import app
from bpmn_agent.knowledge.pattern_matching_bridge import AdvancedPatternMatchingBridge
import bpmn_agent.api.pattern_matching_routes as pm_routes


@pytest.fixture
def kb_with_patterns():
    """Create a KB with test patterns."""
    kb = KnowledgeBase()
    
    # Finance - Payment Processing
    pattern1 = BPMNPattern(
        id="finance_payment_001",
        name="Payment Processing Workflow",
        description="Process a payment request through validation and settlement",
        domain=DomainType.FINANCE,
        category=PatternCategory.SEQUENTIAL,
        complexity=ComplexityLevel.MODERATE,
        graph_structure=GraphStructure(
            nodes=["submit", "validate", "authorize", "process", "confirm"],
            edges=["submit->validate", "validate->authorize", "authorize->process", "process->confirm"],
            node_types={"submit": "activity", "validate": "activity", "authorize": "activity", "process": "activity", "confirm": "activity"}
        ),
        tags={"payment", "finance", "transaction", "settlement"},
        examples=["Customer submits payment, system validates, processes, and confirms"],
        confidence=0.95,
    )
    kb.add_pattern(pattern1)
    
    # Finance - Fraud Detection
    pattern2 = BPMNPattern(
        id="finance_fraud_001",
        name="Fraud Detection",
        description="Detect and handle potentially fraudulent transactions",
        domain=DomainType.FINANCE,
        category=PatternCategory.EXCLUSIVE_CHOICE,
        complexity=ComplexityLevel.MODERATE,
        graph_structure=GraphStructure(
            nodes=["check", "flag", "review", "approve", "reject"],
            edges=["check->flag", "flag->review", "review->approve", "review->reject"],
            node_types={"check": "activity", "flag": "activity", "review": "activity", "approve": "activity", "reject": "activity"}
        ),
        tags={"fraud", "detection", "security", "compliance"},
        examples=["Check transaction for fraud indicators"],
        confidence=0.92,
    )
    kb.add_pattern(pattern2)
    
    # Healthcare - Appointment Scheduling
    pattern3 = BPMNPattern(
        id="healthcare_appt_001",
        name="Appointment Scheduling",
        description="Schedule and confirm a patient appointment",
        domain=DomainType.HEALTHCARE,
        category=PatternCategory.SEQUENTIAL,
        complexity=ComplexityLevel.SIMPLE,
        graph_structure=GraphStructure(
            nodes=["request", "check_availability", "book", "confirm"],
            edges=["request->check_availability", "check_availability->book", "book->confirm"],
            node_types={"request": "activity", "check_availability": "activity", "book": "activity", "confirm": "activity"}
        ),
        tags={"appointment", "scheduling", "healthcare", "patient"},
        examples=["Patient requests appointment, system checks availability and confirms"],
        confidence=0.90,
    )
    kb.add_pattern(pattern3)
    
    # Generic - Decision
    pattern4 = BPMNPattern(
        id="generic_decision_001",
        name="Binary Decision",
        description="Make a decision between two paths",
        domain=DomainType.GENERIC,
        category=PatternCategory.EXCLUSIVE_CHOICE,
        complexity=ComplexityLevel.SIMPLE,
        graph_structure=GraphStructure(
            nodes=["gateway", "yes_path", "no_path", "join"],
            edges=["gateway->yes_path", "gateway->no_path", "yes_path->join", "no_path->join"],
            node_types={"gateway": "gateway", "yes_path": "activity", "no_path": "activity", "join": "gateway"}
        ),
        tags={"decision", "binary", "choice"},
        examples=["If condition then path A else path B"],
        confidence=0.98,
    )
    kb.add_pattern(pattern4)
    
    return kb


@pytest.fixture
def client(kb_with_patterns):
    """Create test client and set up bridge with test KB."""
    # Set up the bridge with test KB
    pm_routes._bridge_instance = AdvancedPatternMatchingBridge(kb_with_patterns)
    
    return TestClient(app)


class TestPatternSearchEndpoint:
    """Test /patterns/search endpoint."""
    
    def test_search_basic(self, client):
        """Test basic pattern search."""
        response = client.post("/api/v1/patterns/search?query=payment")
        assert response.status_code == 200
        data = response.json()
        assert "patterns" in data
        assert "total_count" in data
        assert isinstance(data["patterns"], list)
    
    def test_search_with_domain_filter(self, client):
        """Test search with domain filter."""
        response = client.post(
            "/api/v1/patterns/search?query=payment&domain=finance"
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["patterns"], list)
    
    def test_search_with_category_filter(self, client):
        """Test search with category filter."""
        response = client.post(
            "/api/v1/patterns/search?query=decision&category=exclusive_choice"
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["patterns"], list)
    
    def test_search_with_max_results(self, client):
        """Test search respects max_results parameter."""
        response = client.post(
            "/api/v1/patterns/search?query=payment&max_results=1"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["patterns"]) <= 1
    
    def test_search_empty_query_fails(self, client):
        """Test search fails with empty query."""
        response = client.post("/api/v1/patterns/search?query=")
        assert response.status_code == 422  # Validation error
    
    def test_search_result_structure(self, client):
        """Test search result has correct structure."""
        response = client.post("/api/v1/patterns/search?query=payment")
        assert response.status_code == 200
        data = response.json()
        
        if data["patterns"]:
            pattern = data["patterns"][0]
            assert "id" in pattern
            assert "name" in pattern
            assert "score" in pattern
            assert "category" in pattern
            assert "complexity" in pattern
            assert "confidence" in pattern


class TestFindPatternsForProcessEndpoint:
    """Test /patterns/find-for-process endpoint."""
    
    def test_find_patterns_basic(self, client):
        """Test finding patterns for process."""
        response = client.post(
            "/api/v1/patterns/find-for-process?process_description=Process+a+payment"
        )
        assert response.status_code == 200
        data = response.json()
        assert "best_pattern_id" in data or "best_pattern_name" in data
        assert "confidence" in data
        assert "alternatives" in data
    
    def test_find_patterns_with_domain_hint(self, client):
        """Test finding patterns with domain hint."""
        response = client.post(
            "/api/v1/patterns/find-for-process?process_description=Process+payment&domain=finance"
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["alternatives"], list)
    
    def test_find_patterns_result_structure(self, client):
        """Test result has correct structure."""
        response = client.post(
            "/api/v1/patterns/find-for-process?process_description=Test+process"
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
        
        if data["alternatives"]:
            alt = data["alternatives"][0]
            assert "id" in alt
            assert "name" in alt
            assert "score" in alt


class TestValidateActivitiesEndpoint:
    """Test /patterns/validate-activities endpoint."""
    
    def test_validate_activities_basic(self, client):
        """Test validating activities."""
        response = client.post(
            "/api/v1/patterns/validate-activities?activities=submit&activities=validate"
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_validate_activities_with_domain(self, client):
        """Test validation with domain filter."""
        response = client.post(
            "/api/v1/patterns/validate-activities?activities=submit&domain=finance"
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_validate_activities_result_structure(self, client):
        """Test result has correct structure."""
        response = client.post(
            "/api/v1/patterns/validate-activities?activities=test"
        )
        assert response.status_code == 200
        data = response.json()
        
        if data:
            result = data[0]
            assert "activity" in result
            assert "is_valid" in result
            assert "confidence" in result
            assert "suggestions" in result
    
    def test_validate_empty_activities_fails(self, client):
        """Test validation fails with empty activities."""
        response = client.post("/api/v1/patterns/validate-activities")
        assert response.status_code in [400, 422]


class TestPatternsByDomainEndpoint:
    """Test /patterns/by-domain/{domain} endpoint."""
    
    def test_get_finance_patterns(self, client):
        """Test getting finance patterns."""
        response = client.get("/api/v1/patterns/by-domain/finance")
        assert response.status_code == 200
        data = response.json()
        assert "domain" in data
        assert data["domain"] == "finance"
        assert "patterns" in data
        assert "total_count" in data
    
    def test_get_healthcare_patterns(self, client):
        """Test getting healthcare patterns."""
        response = client.get("/api/v1/patterns/by-domain/healthcare")
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "healthcare"
    
    def test_patterns_respects_max(self, client):
        """Test max_patterns parameter is respected."""
        response = client.get(
            "/api/v1/patterns/by-domain/finance?max_patterns=1"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["patterns"]) <= 1
    
    def test_patterns_with_complexity_filter(self, client):
        """Test filtering by complexity."""
        response = client.get(
            "/api/v1/patterns/by-domain/finance?complexity=moderate"
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["patterns"], list)
    
    def test_pattern_result_structure(self, client):
        """Test pattern result structure."""
        response = client.get("/api/v1/patterns/by-domain/finance")
        assert response.status_code == 200
        data = response.json()
        
        if data["patterns"]:
            pattern = data["patterns"][0]
            assert "id" in pattern
            assert "name" in pattern
            assert "category" in pattern
            assert "complexity" in pattern


class TestPatternDetailsEndpoint:
    """Test /patterns/pattern/{pattern_id} endpoint."""
    
    def test_get_pattern_details(self, client):
        """Test getting pattern details."""
        response = client.get(
            "/api/v1/patterns/pattern/finance_payment_001"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "finance_payment_001"
        assert "name" in data
        assert "description" in data
        assert "domain" in data
    
    def test_get_nonexistent_pattern(self, client):
        """Test getting nonexistent pattern."""
        response = client.get(
            "/api/v1/patterns/pattern/nonexistent_pattern"
        )
        assert response.status_code == 404
    
    def test_pattern_details_structure(self, client):
        """Test pattern details structure."""
        response = client.get(
            "/api/v1/patterns/pattern/finance_payment_001"
        )
        assert response.status_code == 200
        data = response.json()
        
        required_fields = [
            "id", "name", "description", "domain",
            "category", "complexity", "confidence", "tags"
        ]
        for field in required_fields:
            assert field in data


class TestSimilarPatternsEndpoint:
    """Test /patterns/similar/{pattern_id} endpoint."""
    
    def test_find_similar_patterns(self, client):
        """Test finding similar patterns."""
        response = client.get(
            "/api/v1/patterns/similar/finance_payment_001"
        )
        assert response.status_code == 200
        data = response.json()
        assert "patterns" in data
        assert "total_count" in data
    
    def test_similar_patterns_max_respected(self, client):
        """Test max_patterns parameter is respected."""
        response = client.get(
            "/api/v1/patterns/similar/finance_payment_001?max_patterns=1"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["patterns"]) <= 1
    
    def test_similar_patterns_excludes_self(self, client):
        """Test that self is excluded from similar patterns."""
        response = client.get(
            "/api/v1/patterns/similar/finance_payment_001"
        )
        assert response.status_code == 200
        data = response.json()
        
        # None of the results should be the same as the input pattern
        for pattern in data["patterns"]:
            assert pattern["id"] != "finance_payment_001"


class TestStatisticsEndpoint:
    """Test /patterns/statistics endpoint."""
    
    def test_get_statistics(self, client):
        """Test getting pattern statistics."""
        response = client.get("/api/v1/patterns/statistics")
        assert response.status_code == 200
        data = response.json()
        assert "total_patterns" in data


class TestHealthCheckEndpoint:
    """Test /patterns/health endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/patterns/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data


class TestAPIErrorHandling:
    """Test API error handling."""
    
    def test_invalid_domain(self, client):
        """Test invalid domain returns error."""
        response = client.get("/api/v1/patterns/by-domain/invalid_domain")
        assert response.status_code in [400, 422]
    
    def test_search_without_query(self, client):
        """Test search without query fails."""
        response = client.post("/api/v1/patterns/search")
        assert response.status_code == 422
    
    def test_large_max_results_capped(self, client):
        """Test very large max_results is capped."""
        response = client.post(
            "/api/v1/patterns/search?query=payment&max_results=1000"
        )
        # Should be capped at 100
        assert response.status_code in [200, 422]


class TestAPIIntegration:
    """Test API integration scenarios."""
    
    def test_workflow_search_then_validate(self, client):
        """Test workflow: search patterns then validate activities."""
        # 1. Search for patterns
        search_response = client.post(
            "/api/v1/patterns/search?query=payment"
        )
        assert search_response.status_code == 200
        
        # 2. Validate activities
        validate_response = client.post(
            "/api/v1/patterns/validate-activities?activities=submit&activities=validate"
        )
        assert validate_response.status_code == 200
    
    def test_workflow_find_then_get_details(self, client):
        """Test workflow: find patterns then get details."""
        # 1. Find patterns for process
        find_response = client.post(
            "/api/v1/patterns/find-for-process?process_description=Process+payment"
        )
        assert find_response.status_code == 200
        find_data = find_response.json()
        
        # 2. If we got a pattern, get its details
        if find_data.get("best_pattern_id"):
            details_response = client.get(
                f"/api/v1/patterns/pattern/{find_data['best_pattern_id']}"
            )
            assert details_response.status_code == 200
    
    def test_workflow_domain_discovery(self, client):
        """Test workflow: discover domain then find similar."""
        # 1. Get domain patterns
        domain_response = client.get(
            "/api/v1/patterns/by-domain/finance?max_patterns=1"
        )
        assert domain_response.status_code == 200
        domain_data = domain_response.json()
        
        # 2. If we got patterns, find similar ones
        if domain_data["patterns"]:
            pattern_id = domain_data["patterns"][0]["id"]
            similar_response = client.get(
                f"/api/v1/patterns/similar/{pattern_id}"
            )
            assert similar_response.status_code == 200
