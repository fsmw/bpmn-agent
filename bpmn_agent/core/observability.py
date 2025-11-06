"""
Observability Infrastructure

Provides structured logging, tracing, and metrics collection for the BPMN agent.
Integrates with OpenTelemetry for cloud-native observability.
"""

import asyncio
import contextlib
import functools
import json
import sys
import time
import traceback
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from loguru import logger
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

F = TypeVar("F", bound=Callable[..., Any])


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ObservabilityConfig:
    """Configuration for observability."""
    
    def __init__(
        self,
        service_name: str = "bpmn-agent",
        log_level: Union[str, LogLevel] = LogLevel.INFO,
        json_logs: bool = False,
        enable_tracing: bool = True,
        jaeger_enabled: bool = False,
        jaeger_agent_host: str = "localhost",
        jaeger_agent_port: int = 6831,
        enable_metrics: bool = True,
    ):
        """Initialize observability configuration."""
        self.service_name = service_name
        self.log_level = log_level if isinstance(log_level, str) else log_level.value
        self.json_logs = json_logs
        self.enable_tracing = enable_tracing
        self.jaeger_enabled = jaeger_enabled
        self.jaeger_agent_host = jaeger_agent_host
        self.jaeger_agent_port = jaeger_agent_port
        self.enable_metrics = enable_metrics


class JSONFormatter:
    """Custom JSON formatter for loguru."""
    
    def __call__(self, record: Dict[str, Any]) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "module": record["module"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
        }
        
        # Add extra fields if present
        if record["extra"]:
            log_data["extra"] = record["extra"]
        
        # Add exception info if present
        if record["exception"]:
            log_data["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": "".join(
                    traceback.format_exception(
                        record["exception"].type,
                        record["exception"].value,
                        record["exception"].tb,
                    )
                ),
            }
        
        return json.dumps(log_data)


class ObservabilityManager:
    """Centralized observability management."""
    
    _instance: Optional["ObservabilityManager"] = None
    
    def __init__(self, config: ObservabilityConfig):
        """Initialize observability manager."""
        self.config = config
        self._setup_logging()
        
        if config.enable_tracing:
            self._setup_tracing()
        
        if config.enable_metrics:
            self._setup_metrics()
        
        logger.info(
            f"Observability initialized: service={config.service_name}, "
            f"log_level={config.log_level}"
        )
    
    def _setup_logging(self) -> None:
        """Set up structured logging with loguru."""
        # Remove default handler
        logger.remove()
        
        # Configure loguru format
        log_format = (
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        
        if self.config.json_logs:
            formatter = JSONFormatter()
            logger.add(
                sys.stdout,
                format="{message}",
                level=self.config.log_level,
                serialize=False,
                colorize=False,
                filter=lambda record: self._custom_filter(record),
            )
            # Use custom serialization
            logger.add(
                sys.stdout,
                format=formatter,
                level=self.config.log_level,
                colorize=False,
            )
        else:
            logger.add(
                sys.stdout,
                format=log_format,
                level=self.config.log_level,
                colorize=True,
                backtrace=True,
                diagnose=True,
            )
    
    def _custom_filter(self, record: Dict[str, Any]) -> bool:
        """Custom filter for log records."""
        return True
    
    def _setup_tracing(self) -> None:
        """Set up OpenTelemetry tracing."""
        resource = Resource(
            attributes={
                SERVICE_NAME: self.config.service_name,
            }
        )
        
        tracer_provider = TracerProvider(resource=resource)
        
        if self.config.jaeger_enabled:
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config.jaeger_agent_host,
                agent_port=self.config.jaeger_agent_port,
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
        
        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)
        logger.info("OpenTelemetry tracing initialized")
    
    def _setup_metrics(self) -> None:
        """Set up OpenTelemetry metrics."""
        # Use in-memory reader for collecting metrics
        metric_reader = InMemoryMetricReader()
        
        resource = Resource(
            attributes={
                SERVICE_NAME: self.config.service_name,
            }
        )
        
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        self.meter = metrics.get_meter(__name__)
        
        # Create specific metric instruments
        self.counter = self.meter.create_counter(
            "function_calls_total",
            description="Total number of function calls",
            unit="1",
        )
        self.histogram = self.meter.create_histogram(
            "function_duration_ms",
            description="Function execution duration in milliseconds",
            unit="ms",
        )
        
        logger.info("OpenTelemetry metrics initialized")
    
    @classmethod
    def initialize(cls, config: Optional[ObservabilityConfig] = None) -> "ObservabilityManager":
        """Initialize or get singleton instance."""
        if cls._instance is None:
            if config is None:
                config = ObservabilityConfig()
            cls._instance = cls(config)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "ObservabilityManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(ObservabilityConfig())
        return cls._instance


@contextlib.contextmanager
def span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager for creating spans."""
    manager = ObservabilityManager.get_instance()
    
    if hasattr(manager, "tracer"):
        with manager.tracer.start_as_current_span(name) as span_obj:
            if attributes:
                for key, value in attributes.items():
                    span_obj.set_attribute(key, value)
            yield span_obj
    else:
        yield None


def log_execution(
    level: Union[str, LogLevel] = LogLevel.INFO,
    include_args: bool = True,
    include_result: bool = True,
    include_duration: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for logging function execution.
    
    Works with both sync and async functions.
    
    Args:
        level: Logging level
        include_args: Whether to log function arguments
        include_result: Whether to log function result
        include_duration: Whether to log execution duration
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            log_level = level if isinstance(level, str) else level.value
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            log_data: Dict[str, Any] = {"function": func_name}
            
            if include_args:
                # Format args and kwargs for logging
                arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
                log_data["args"] = dict(zip(arg_names, args))
                log_data["kwargs"] = kwargs
            
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                if include_result:
                    log_data["result"] = str(result)[:200]  # Truncate large results
                
                if include_duration:
                    duration_ms = (time.time() - start_time) * 1000
                    log_data["duration_ms"] = duration_ms
                    # Record metrics
                    record_metric(f"{func.__name__}_duration", duration_ms)
                
                logger.log(log_level, f"Function executed: {func_name}", extra=log_data)
                return result
            
            except Exception as e:
                if include_duration:
                    log_data["duration_ms"] = (time.time() - start_time) * 1000
                log_data["error"] = str(e)
                
                logger.error(f"Function failed: {func_name}", extra=log_data, exc_info=True)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            log_level = level if isinstance(level, str) else level.value
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            log_data: Dict[str, Any] = {"function": func_name}
            
            if include_args:
                arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
                log_data["args"] = dict(zip(arg_names, args))
                log_data["kwargs"] = kwargs
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    log_data["result"] = str(result)[:200]
                
                if include_duration:
                    duration_ms = (time.time() - start_time) * 1000
                    log_data["duration_ms"] = duration_ms
                    # Record metrics
                    record_metric(f"{func.__name__}_duration", duration_ms)
                
                logger.log(log_level, f"Function executed: {func_name}", extra=log_data)
                return result
            
            except Exception as e:
                if include_duration:
                    log_data["duration_ms"] = (time.time() - start_time) * 1000
                log_data["error"] = str(e)
                
                logger.error(f"Function failed: {func_name}", extra=log_data, exc_info=True)
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore
    
    return decorator


def record_metric(
    metric_name: str,
    value: Union[int, float],
    attributes: Optional[Dict[str, str]] = None,
) -> None:
    """
    Record a metric value with OpenTelemetry.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        attributes: Optional attributes for the metric
    """
    manager = ObservabilityManager.get_instance()
    
    if hasattr(manager, "counter") and hasattr(manager, "histogram"):
        # Record both as counter (for counts) and histogram (for distributions)
        if metric_name.endswith("_total") or isinstance(value, int):
            manager.counter.add(value, attributes=attributes or {})
        else:
            manager.histogram.record(value, attributes=attributes or {})
        
        logger.debug(f"Metric recorded: {metric_name}={value}", extra={
            "metric": metric_name,
            "value": value,
            "attributes": attributes,
        })
    else:
        # Fallback to logging if metrics not initialized
        log_data = {
            "metric": metric_name,
            "value": value,
        }
        if attributes:
            log_data["attributes"] = attributes
        logger.debug(f"Metric recorded: {metric_name}", extra=log_data)


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str, log: bool = True):
        """Initialize timer."""
        self.name = name
        self.log = log
        self.start_time: float = 0
        self.elapsed: float = 0
    
    def __enter__(self) -> "Timer":
        """Enter context."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Exit context."""
        self.elapsed = time.time() - self.start_time
        if self.log:
            logger.debug(f"Timer '{self.name}': {self.elapsed:.3f}s")
            # Record metric
            record_metric(f"{self.name}_duration", self.elapsed * 1000)


__all__ = [
    "LogLevel",
    "ObservabilityConfig",
    "ObservabilityManager",
    "span",
    "log_execution",
    "record_metric",
    "Timer",
]
