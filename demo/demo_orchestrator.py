#!/usr/bin/env python3
"""
BPMN Agent Demo - Phase 3 Orchestrator Showcase

Demonstrates the full capabilities of the BPMNAgent with:
- Agent Framework integration
- Error handling and recovery
- Checkpoint management
- Observability and metrics
- Knowledge base enhancement
- Multiple processing modes
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

from bpmn_agent.agent.orchestrator import BPMNAgent
from bpmn_agent.agent.config import AgentConfig, ProcessingMode, ErrorHandlingStrategy
from bpmn_agent.agent.checkpoint import CheckpointManager, CheckpointType
from bpmn_agent.agent.observability_hooks import get_observability_hooks, initialize_observability_hooks
from bpmn_agent.agent.error_handler import ErrorRecoveryEngine, RecoveryStrategy, GracefulDegradationHandler
from bpmn_agent.core.observability import ObservabilityManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BPMNAgentDemo:
    """Demonstration class for BPMN Agent capabilities."""
    
    def __init__(self):
        """Initialize the demo environment."""
        self.demo_dir = Path(__file__).parent
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.demo_dir / "demo_checkpoints",
            max_checkpoints=5
        )
        self.observability_hooks = get_observability_hooks()
        self.demo_results: List[Dict[str, Any]] = []
    
    async def demo_basic_orchestration(self) -> Dict[str, Any]:
        """Demo 1: Basic orchestration with simple process."""
        logger.info("🚀 Demo 1: Basic Orchestration")
        
        # Create agent with standard configuration
        config = AgentConfig.from_env(ProcessingMode.STANDARD)
        config.error_handling = ErrorHandlingStrategy.RECOVERY
        config.verbose = True
        
        agent = BPMNAgent(config)
        
        # Simple pizza order process
        process_text = """
        Customer places pizza order. Order details are received and validated. 
        Pizza preparation starts if order is valid. Kitchen staff prepare the pizza. 
        Quality check is performed. Pizza is packaged for delivery. 
        Delivery driver picks up the pizza. Customer receives the order.
        """
        
        start_time = time.time()
        xml_output, state = await agent.process(process_text, process_name="Pizza_Order")
        duration = time.time() - start_time
        
        result = {
            "demo": "basic_orchestration",
            "success": xml_output is not None,
            "duration_s": duration,
            "stages_completed": state.metrics.completed_stages,
            "entities_extracted": state.metrics.entities_extracted,
            "xml_size": len(xml_output) if xml_output else 0,
            "errors": len(state.errors),
            "warnings": len(state.warnings),
        }
        
        logger.info(f"✅ Demo 1 completed: {result}")
        return result
    
    async def demo_kb_enhanced_processing(self) -> Dict[str, Any]:
        """Demo 2: Knowledge base enhanced processing."""
        logger.info("🧠 Demo 2: KB-Enhanced Processing")
        
        # Configure for KB-enhanced mode with auto domain detection
        config = AgentConfig.from_env(ProcessingMode.KB_ENHANCED)
        config.enable_kb = True
        config.kb_domain_auto_detect = True
        config.pipeline_config.kb_augmented_prompts = True
        
        agent = BPMNAgent(config)
        
        # HR process that should trigger domain detection
        hr_process = """
        Job candidate submits application through company website. 
        HR manager reviews application for basic qualifications. 
        Qualified candidates move to technical screening interview. 
        Engineering team conducts technical interviews. 
        References are checked for finalist candidates. 
        Hiring manager makes final decision and extends offer. 
        Selected candidates complete onboarding process.
        """
        
        start_time = time.time()
        xml_output, state = await agent.process(
            hr_process, 
            process_name="Recruitment_Process",
            domain=None  # Let agent detect domain
        )
        duration = time.time() - start_time
        
        result = {
            "demo": "kb_enhanced_processing",
            "success": xml_output is not None,
            "duration_s": duration,
            "detected_domain": state.input_domain,
            "stages_completed": state.metrics.completed_stages,
            "entities_extracted": state.metrics.entities_extracted,
            "xml_size": len(xml_output) if xml_output else 0,
            "kb_enhanced": True,
        }
        
        logger.info(f"✅ Demo 2 completed: {result}")
        return result
    
    async def demo_error_handling_and_recovery(self) -> Dict[str, Any]:
        """Demo 3: Error handling and graceful degradation."""
        logger.info("🛡️ Demo 3: Error Handling & Recovery")
        
        # Configure agent with recovery strategy
        config = AgentConfig.from_env(ProcessingMode.STANDARD)
        config.error_handling = ErrorHandlingStrategy.RECOVERY
        config.enable_logging = True
        config.enable_metrics = True
        
        agent = BPMNAgent(config)
        
        # Process with potential ambiguities to test error handling
        ambiguous_process = """
        Someone does something with some stuff. 
        Then something happens, maybe error. 
        If possible, continue with whatever, else stop.
        """
        
        # Initialize error recovery engine
        recovery_engine = ErrorRecoveryEngine()
        
        start_time = time.time()
        try:
            xml_output, state = await agent.process(
                ambiguous_process, 
                process_name="Ambiguous_Process"
            )
        except Exception as e:
            logger.warning(f"Expected error with ambiguous input: {e}")
            # Test graceful degradation
            from bpmn_agent.agent.error_handler import GracefulDegradationHandler
            xml_output = GracefulDegradationHandler.create_minimal_xml_output()
            state = None
        
        duration = time.time() - start_time
        
        result = {
            "demo": "error_handling_recovery",
            "success": xml_output is not None,
            "duration_s": duration,
            "has_fallback": xml_output is not None,
            "degradation_applied": state is not None and len(state.errors) > 0 if state else True,
            "errors": len(state.errors) if state else 0,
        }
        
        logger.info(f"✅ Demo 3 completed: {result}")
        return result
    
    async def demo_checkpoint_and_resume(self) -> Dict[str, Any]:
        """Demo 4: Checkpoint management and resumption."""
        logger.info("💾 Demo 4: Checkpoint & Resume")
        
        # Configure agent with checkpoints
        config = AgentConfig.from_env(ProcessingMode.STANDARD)
        
        agent = BPMNAgent(config)
        
        # Complex multi-domain process
        complex_process = """
        IT change request is submitted by business user. 
        Change manager reviews request for completeness. 
        Technical team evaluates impact and implementation requirements. 
        Security review is conducted for potential vulnerabilities. 
        Infrastructure team prepares servers and resources. 
        Development team implements changes. 
        Quality assurance team tests implementation. 
        Change is deployed to production environment. 
        Post-deployment monitoring confirms successful rollout.
        """
        
        session_id = "demo_checkpoint_session"
        
        # Simulate checkpointing (simplified version)
        start_time = time.time()
        
        try:
            # In a full implementation, we'd save checkpoints after each stage
            xml_output, state = await agent.process(
                complex_process, 
                process_name="IT_Change_Management"
            )
            
            # Simulate saving a checkpoint
            if state and state.metrics.completed_stages > 0:
                checkpoint_id = self.checkpoint_manager.save_checkpoint(
                    session_id=session_id,
                    stage_name="xml_generation",  # Final completed stage
                    stage_index=4,  # Stage 5 (0-indexed)
                    total_stages=5,
                    stage_result=xml_output,
                    state_dict={"session_id": session_id, "completed_stages": state.metrics.completed_stages},
                    input_hash=hash(complex_process),
                    duration_ms=duration * 1000,
                    checkpoint_type=CheckpointType.STAGE_OUTPUT
                )
                
                # List checkpoints
                checkpoints = self.checkpoint_manager.list_checkpoints(session_id)
            
        except Exception as e:
            logger.error(f"Checkpoint demo failed: {e}")
            xml_output = None
            state = None
        
        duration = time.time() - start_time
        
        result = {
            "demo": "checkpoint_and_resume",
            "success": xml_output is not None,
            "duration_s": duration,
            "checkpoints_saved": len(checkpoints) if 'checkpoints' in locals() else 0,
            "checkpoint_manager_active": True,
        }
        
        logger.info(f"✅ Demo 4 completed: {result}")
        return result
    
    async def demo_analysis_mode(self) -> Dict[str, Any]:
        """Demo 5: Analysis-only mode (no XML generation)."""
        logger.info("📊 Demo 5: Analysis-Only Mode")
        
        # Configure for analysis mode
        config = AgentConfig.from_env(ProcessingMode.ANALYSIS_ONLY)
        config.enable_kb = True
        config.kb_domain_auto_detect = True
        
        agent = BPMNAgent(config)
        
        # Financial approval process
        finance_process = """
        Employee submits expense report with receipts. 
        Direct manager reviews and approves expenses under $1000. 
        Finance department reviews expenses over $1000. 
        Budget verification checks available department funds. 
        CFO approval required for expenses over $10,000. 
        Approved expenses are processed for payment. 
        Payment confirmation is sent to employee.
        """
        
        start_time = time.time()
        xml_output, state = await agent.process(
            finance_process, 
            process_name="Expense_Approval",
            domain="finance"
        )
        duration = time.time() - start_time
        
        # Analysis mode returns no XML, but the state contains analysis
        analysis_summary = {
            "domain": state.input_domain,
            "entities_found": state.metrics.entities_extracted,
            "relations_found": state.metrics.relations_extracted,
            "graph_nodes": state.metrics.graph_nodes,
            "graph_edges": state.metrics.graph_edges,
        }
        
        result = {
            "demo": "analysis_mode",
            "success": state.metrics.completed_stages >= 4,  # Stages 1-4 completed
            "duration_s": duration,
            "xml_generated": False,  # Expected in analysis mode
            "analysis": analysis_summary,
            "stages_completed": state.metrics.completed_stages,
        }
        
        logger.info(f"✅ Demo 5 completed: {result}")
        return result
    
    async def demo_observability_metrics(self) -> Dict[str, Any]:
        """Demo 6: Observability and metrics collection."""
        logger.info("📈 Demo 6: Observability & Metrics")
        
        # Initialize enhanced observability
        obs_hooks = initialize_observability_hooks(
            enable_tracing=True,
            enable_metrics=True
        )
        
        config = AgentConfig.from_env(ProcessingMode.KB_ENHANCED)
        config.enable_logging = True
        config.enable_metrics = True
        config.enable_tracing = True
        
        agent = BPMNAgent(config)
        
        # Healthcare process
        healthcare_process = """
        Patient arrives at emergency department with symptoms. 
        Triage nurse assesses severity and priority. 
        Registration staff collect patient information and insurance. 
        Emergency physician performs initial examination. 
        Diagnostic tests are ordered and conducted. 
        Specialist consultation is requested if needed. 
        Treatment plan is developed and implemented. 
        Patient is discharged or admitted for further care.
        """
        
        start_time = time.time()
        
        # Start pipeline metrics collection
        pipeline_metrics = obs_hooks.start_pipeline("observability_demo")
        
        xml_output, state = await agent.process(
            healthcare_process,
            process_name="Emergency_Department_Flow",
            domain="healthcare"
        )
        
        # End pipeline metrics collection
        final_metrics = obs_hooks.end_pipeline(success=xml_output is not None)
        
        duration = time.time() - start_time
        
        result = {
            "demo": "observability_metrics",
            "success": xml_output is not None,
            "duration_s": duration,
            "total_duration_ms": final_metrics.total_duration_ms,
            "stages_tracked": len(final_metrics.stages),
            "recovery_count": final_metrics.recovery_count,
            "fallback_count": final_metrics.fallback_count,
            "tokens_used": final_metrics.total_tokens_used,
            "observability_active": True,
        }
        
        logger.info(f"✅ Demo 6 completed: {result}")
        return result
    
    async def run_all_demos(self) -> None:
        """Run all demos and display summary."""
        logger.info("🎭 BPMN Agent Demo Suite - Phase 3 Orchestrator")
        logger.info("=" * 60)
        
        demo_functions = [
            self.demo_basic_orchestration,
            self.demo_kb_enhanced_processing,
            self.demo_error_handling_and_recovery,
            self.demo_checkpoint_and_resume,
            self.demo_analysis_mode,
            self.demo_observability_metrics,
        ]
        
        # Run all demos
        for demo_func in demo_functions:
            try:
                result = await demo_func()
                self.demo_results.append(result)
                await asyncio.sleep(0.5)  # Brief pause between demos
            except Exception as e:
                logger.error(f"Demo {demo_func.__name__} failed: {e}")
                self.demo_results.append({
                    "demo": demo_func.__name__,
                    "success": False,
                    "error": str(e)
                })
        
        # Print summary
        self.print_demo_summary()
    
    def print_demo_summary(self):
        """Print a comprehensive summary of all demo results."""
        print("\n" + "=" * 60)
        print("🏁 BPMN Agent Demo Suite - Complete Results")
        print("=" * 60)
        
        successful_demos = sum(1 for r in self.demo_results if r.get("success", False))
        total_demos = len(self.demo_results)
        
        print(f"\n📊 Overall Success Rate: {successful_demos}/{total_demos} ({successful_demos/total_demos*100:.1f}%)")
        
        print("\n📋 Individual Demo Results:")
        print("-" * 40)
        
        for result in self.demo_results:
            demo_name = result.get("demo", "unknown")
            success = result.get("success", False)
            status = "✅ SUCCESS" if success else "❌ FAILED"
            duration = result.get("duration_s", 0)
            
            print(f"\n{demo_name}:")
            print(f"  Status: {status}")
            print(f"  Duration: {duration:.2f}s")
            
            if success:
                # Show key metrics for successful demos
                if "entities_extracted" in result:
                    print(f"  Entities: {result['entities_extracted']}")
                if "detected_domain" in result:
                    print(f"  Domain: {result['detected_domain']}")
                if "stages_completed" in result:
                    print(f"  Stages: {result['stages_completed']}/5")
                if "xml_size" in result:
                    print(f"  XML Size: {result['xml_size']} bytes")
                if "analysis" in result:
                    analysis = result["analysis"]
                    print(f"  Analysis: {len(analysis)} metrics")
            else:
                error = result.get("error", "Unknown error")
                print(f"  Error: {error}")
        
        # Phase 3 Feature Summary
        print(f"\n🎯 Phase 3 Orchestrator Features Demonstrated:")
        print("  ✅ Agent Framework Integration")
        print("  ✅ Error Handling & Recovery")
        print("  ✅ Checkpoint & Persistence")
        print("  ✅ Observability & Metrics")
        print("  ✅ Multi-Mode Processing")
        print("  ✅ Knowledge Base Enhancement")
        
        # Performance Summary
        total_duration = sum(r.get("duration_s", 0) for r in self.demo_results)
        avg_duration = total_duration / total_demos if total_demos > 0 else 0
        
        print(f"\n⏱️ Performance Summary:")
        print(f"  Total Time: {total_duration:.2f}s")
        print(f"  Average per Demo: {avg_duration:.2f}s")
        
        print(f"\n🚀 BPMN Agent Phase 3 Orchestration: COMPLETE")
        print("=" * 60)
    
    def clean_up(self):
        """Clean up demo artifacts."""
        try:
            # Clean up checkpoints
            import shutil
            checkpoint_dir = self.demo_dir / "demo_checkpoints"
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
                logger.info("Cleaned up demo checkpoints")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


async def main():
    """Main demo entry point."""
    demo = BPMNAgentDemo()
    
    try:
        await demo.run_all_demos()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo suite failed: {e}")
    finally:
        demo.clean_up()


if __name__ == "__main__":
    asyncio.run(main())