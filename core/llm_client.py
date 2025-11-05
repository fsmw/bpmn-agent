"""
LLM Provider Abstraction Layer

Supports multiple LLM providers (Ollama, OpenAI-compatible APIs) with:
- Configuration management via environment variables and explicit parameters
- Connection validation and retry logic
- Streaming and non-streaming responses
- Accurate token counting for different models
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from .tokenizer import TokenCounter


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    AZURE_OPENAI = "azure_openai"


class LLMMessage(BaseModel):
    """Structured LLM message format."""
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name for role disambiguation")
    
    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate role is one of the allowed values."""
        valid_roles = {"system", "user", "assistant"}
        if v not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}, got '{v}'")
        return v
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is not empty."""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate name if provided."""
        if v is not None and not v.strip():
            raise ValueError("Name cannot be empty if provided")
        return v.strip() if v else None


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""
    
    provider: LLMProviderType = Field(
        default=LLMProviderType.OLLAMA,
        description="LLM provider type"
    )
    
    # Connection settings
    base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for LLM provider (Ollama or OpenAI-compatible)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication (OpenAI, Azure, etc.)"
    )
    
    # Model settings
    model: str = Field(
        default="mistral",
        description="Model name or ID"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 to 2.0)"
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens in response"
    )
    
    # Retry and timeout settings
    timeout: float = Field(
        default=60.0,
        gt=0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retry attempts"
    )
    retry_backoff: float = Field(
        default=1.0,
        gt=0,
        description="Exponential backoff factor for retries"
    )
    
    # Context window settings
    context_window: int = Field(
        default=4096,
        gt=0,
        description="LLM context window size (tokens)"
    )
    
    # Advanced settings
    system_prompt: Optional[str] = Field(
        default=None,
        description="Default system prompt for all requests"
    )
    
    # Azure-specific settings
    azure_deployment: Optional[str] = Field(
        default=None,
        description="Azure deployment name (for Azure OpenAI only)"
    )
    azure_api_version: str = Field(
        default="2024-02-15-preview",
        description="Azure API version (for Azure OpenAI only)"
    )
    
    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Ensure base_url is properly formatted."""
        return v.rstrip("/")
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables."""
        provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        
        config_data = {
            "provider": provider,
            "base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434"),
            "api_key": os.getenv("LLM_API_KEY"),
            "model": os.getenv("LLM_MODEL", "mistral"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "0")) or None,
            "timeout": float(os.getenv("LLM_TIMEOUT", "60")),
            "max_retries": int(os.getenv("LLM_MAX_RETRIES", "3")),
            "context_window": int(os.getenv("LLM_CONTEXT_WINDOW", "4096")),
        }
        
        # Add Azure-specific configuration if using Azure OpenAI
        if provider == "azure_openai":
            config_data["azure_deployment"] = os.getenv("AZURE_DEPLOYMENT")
            config_data["azure_api_version"] = os.getenv(
                "AZURE_API_VERSION", "2024-02-15-preview"
            )
        
        return cls(**config_data)


class LLMResponse(BaseModel):
    """Structured LLM response."""
    content: str = Field(..., description="Response content")
    model: str = Field(..., description="Model that generated the response")
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage: prompt_tokens, completion_tokens, total_tokens"
    )
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason for completion: 'stop', 'length', 'error', etc."
    )
    raw_response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw API response for debugging"
    )


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: LLMConfig):
        """Initialize LLM client."""
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info(
            f"Initializing {self.__class__.__name__} with provider={config.provider}, "
            f"model={config.model}"
        )
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close_session(self) -> None:
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate connection to LLM provider."""
        pass
    
    @abstractmethod
    async def call(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, AsyncIterator[str]]:
        """Call LLM with messages."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        pass


class OllamaClient(BaseLLMClient):
    """Ollama LLM client with async support."""
    
    async def validate_connection(self) -> bool:
        """Validate Ollama connection and model availability."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.config.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to connect to Ollama: HTTP {response.status}")
                    return False
                
                data = await response.json()
                models = data.get("models", [])
                model_names = [m["name"] for m in models]
                
                if self.config.model not in model_names:
                    logger.warning(
                        f"Model {self.config.model} not found in Ollama. "
                        f"Available: {model_names}"
                    )
                    return False
                
                logger.info(f"Successfully connected to Ollama at {self.config.base_url}")
                return True
        except asyncio.TimeoutError:
            logger.error("Timeout connecting to Ollama")
            return False
        except aiohttp.ClientError as e:
            logger.error(f"Failed to validate Ollama connection: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating Ollama connection: {e}")
            return False
    
    async def call(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, AsyncIterator[str]]:
        """Call Ollama API."""
        # Validate inputs
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list")
        
        for msg in messages:
            if not isinstance(msg, LLMMessage):
                raise ValueError(f"All messages must be LLMMessage instances, got {type(msg)}")
        
        # Validate temperature
        if temperature is not None:
            if not isinstance(temperature, (int, float)):
                raise ValueError(f"Temperature must be numeric, got {type(temperature)}")
            if not 0.0 <= temperature <= 2.0:
                raise ValueError(f"Temperature must be between 0.0 and 2.0, got {temperature}")
        
        # Validate max_tokens
        if max_tokens is not None:
            if not isinstance(max_tokens, int):
                raise ValueError(f"max_tokens must be an integer, got {type(max_tokens)}")
            if max_tokens <= 0:
                raise ValueError(f"max_tokens must be positive, got {max_tokens}")
            if max_tokens > self.config.context_window:
                raise ValueError(
                    f"max_tokens ({max_tokens}) exceeds context window ({self.config.context_window})"
                )
        
        temp = temperature or self.config.temperature
        
        # Format messages for Ollama
        formatted_messages = [m.model_dump() for m in messages]
        
        payload = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": temp,
            "stream": stream,
        }
        
        if max_tokens:
            payload["num_predict"] = max_tokens
        
        try:
            session = await self._get_session()
            
            if stream:
                return self._stream_response_async(session, payload)
            else:
                async with session.post(
                    f"{self.config.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: HTTP {response.status} - {error_text}")
                        raise RuntimeError(f"Ollama API error: {response.status}")
                    
                    data = await response.json()
                    return LLMResponse(
                        content=data.get("message", {}).get("content", ""),
                        model=self.config.model,
                        usage={
                            "prompt_tokens": data.get("prompt_eval_count", 0),
                            "completion_tokens": data.get("eval_count", 0),
                            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                        },
                        finish_reason="stop",
                        raw_response=data,
                    )
        except asyncio.TimeoutError:
            logger.error("Timeout calling Ollama API")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Ollama API call failed: {e}")
            raise
    
    async def _stream_response_async(
        self, session: aiohttp.ClientSession, payload: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """Stream response from Ollama asynchronously."""
        async with session.post(
            f"{self.config.base_url}/api/chat",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Ollama streaming error: HTTP {response.status} - {error_text}")
                raise RuntimeError(f"Ollama streaming error: {response.status}")
            
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse response line: {line}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens for Ollama models.
        
        Uses model-aware tokenization from TokenCounter.
        """
        return TokenCounter.count_tokens(text, self.config.model)


class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI-compatible API client (OpenAI, Azure, LiteLLM, etc.) with async support."""
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests.
        
        Returns appropriate authentication headers based on the provider:
        - OpenAI/standard: Bearer token in Authorization header
        - Azure: API key in api-key header
        
        Returns:
            Dictionary of HTTP headers for authentication
        """
        headers = {
            "Content-Type": "application/json",
        }
        
        # Handle Azure-specific authentication
        if self.config.provider == LLMProviderType.AZURE_OPENAI:
            headers["api-key"] = self.config.api_key or ""
        else:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        return headers
    
    def _get_endpoint_url(self, endpoint_path: str) -> str:
        """
        Construct endpoint URL based on provider.
        
        Handles different URL formats for different providers:
        - Standard OpenAI-compatible: {base_url}/{endpoint_path}
        - Azure OpenAI: {base_url}/openai/deployments/{deployment}/{endpoint_path}
        
        Args:
            endpoint_path: The API endpoint path (e.g., 'chat/completions', 'models')
        
        Returns:
            Complete URL for the API endpoint
        
        Raises:
            ValueError: If Azure deployment name is not configured for Azure provider
        """
        if self.config.provider == LLMProviderType.AZURE_OPENAI:
            # Azure uses deployment-based URLs
            if not self.config.azure_deployment:
                raise ValueError("azure_deployment must be set for Azure OpenAI")
            
            base = self.config.base_url.rstrip("/")
            deployment = self.config.azure_deployment
            api_version = self.config.azure_api_version
            
            # Azure endpoint format: {base}/openai/deployments/{deployment}/chat/completions?api-version={version}
            if endpoint_path == "chat/completions":
                return f"{base}/openai/deployments/{deployment}/chat/completions"
            elif endpoint_path == "models":
                return f"{base}/openai/deployments/{deployment}/models"
            else:
                return f"{base}/openai/deployments/{deployment}/{endpoint_path}"
        else:
            # Standard OpenAI-compatible format
            return f"{self.config.base_url}/{endpoint_path}"
    
    def _get_query_params(self) -> Dict[str, str]:
        """
        Get query parameters for the request.
        
        Returns provider-specific query parameters:
        - Azure OpenAI: {'api-version': configured api_version}
        - Standard OpenAI-compatible: empty dict
        
        Returns:
            Dictionary of query parameters to append to the request URL
        """
        if self.config.provider == LLMProviderType.AZURE_OPENAI:
            return {"api-version": self.config.azure_api_version}
        return {}
    
    async def validate_connection(self) -> bool:
        """Validate OpenAI-compatible API connection."""
        try:
            headers = self._get_headers()
            query_params = self._get_query_params()
            session = await self._get_session()
            
            endpoint = self._get_endpoint_url("models")
            
            async with session.get(
                endpoint,
                headers=headers,
                params=query_params,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to validate API connection: HTTP {response.status}")
                    return False
                
                logger.info(
                    f"Successfully connected to OpenAI-compatible API at {self.config.base_url}"
                )
                return True
        except asyncio.TimeoutError:
            logger.error("Timeout validating API connection")
            return False
        except aiohttp.ClientError as e:
            logger.error(f"Failed to validate API connection: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating API connection: {e}")
            return False
    
    async def call(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, AsyncIterator[str]]:
        """Call OpenAI-compatible API."""
        # Validate inputs
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list")
        
        for msg in messages:
            if not isinstance(msg, LLMMessage):
                raise ValueError(f"All messages must be LLMMessage instances, got {type(msg)}")
        
        # Validate temperature
        if temperature is not None:
            if not isinstance(temperature, (int, float)):
                raise ValueError(f"Temperature must be numeric, got {type(temperature)}")
            if not 0.0 <= temperature <= 2.0:
                raise ValueError(f"Temperature must be between 0.0 and 2.0, got {temperature}")
        
        # Validate max_tokens
        if max_tokens is not None:
            if not isinstance(max_tokens, int):
                raise ValueError(f"max_tokens must be an integer, got {type(max_tokens)}")
            if max_tokens <= 0:
                raise ValueError(f"max_tokens must be positive, got {max_tokens}")
            if max_tokens > self.config.context_window:
                raise ValueError(
                    f"max_tokens ({max_tokens}) exceeds context window ({self.config.context_window})"
                )
        
        temp = temperature or self.config.temperature
        
        formatted_messages = [m.model_dump() for m in messages]
        
        payload = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": temp,
            "stream": stream,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        headers = self._get_headers()
        query_params = self._get_query_params()
        
        try:
            session = await self._get_session()
            
            if stream:
                return self._stream_response_async(session, payload, headers, query_params)
            else:
                endpoint = self._get_endpoint_url("chat/completions")
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    params=query_params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API error: HTTP {response.status} - {error_text}")
                        raise RuntimeError(f"API error: {response.status}")
                    
                    data = await response.json()
                    choice = data["choices"][0]
                    return LLMResponse(
                        content=choice["message"]["content"],
                        model=self.config.model,
                        usage={
                            "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                            "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                            "total_tokens": data.get("usage", {}).get("total_tokens", 0),
                        },
                        finish_reason=choice.get("finish_reason", "stop"),
                        raw_response=data,
                    )
        except asyncio.TimeoutError:
            logger.error("Timeout calling API")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"API call failed: {e}")
            raise
    
    async def _stream_response_async(
        self,
        session: aiohttp.ClientSession,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        query_params: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[str]:
         """
         Stream response from OpenAI-compatible API asynchronously.
         
         Handles server-sent events (SSE) format used by OpenAI and compatible APIs.
         Processes each line of the response, decoding JSON data and yielding content chunks.
         
         Args:
             session: Active aiohttp client session for making the request
             payload: Request payload containing messages and parameters
             headers: HTTP headers including authentication
             query_params: Optional query parameters (e.g., api-version for Azure)
         
         Yields:
             Content chunks as they arrive from the API
         
         Raises:
             RuntimeError: If API returns non-200 status code
             json.JSONDecodeError: If response contains invalid JSON (logs warning, continues)
         """
         endpoint = self._get_endpoint_url("chat/completions")
         
         async with session.post(
             endpoint,
             json=payload,
             headers=headers,
             params=query_params,
             timeout=aiohttp.ClientTimeout(total=self.config.timeout),
         ) as response:
             if response.status != 200:
                 error_text = await response.text()
                 logger.error(f"API streaming error: HTTP {response.status} - {error_text}")
                 raise RuntimeError(f"API streaming error: {response.status}")
             
             async for line in response.content:
                 if line:
                     line_str = line.decode("utf-8").strip()
                     if line_str.startswith("data:"):
                         line_str = line_str[5:].strip()
                         if line_str == "[DONE]":
                             break
                         try:
                             data = json.loads(line_str)
                             if "choices" in data and len(data["choices"]) > 0:
                                 delta = data["choices"][0].get("delta", {})
                                 if "content" in delta:
                                     yield delta["content"]
                         except json.JSONDecodeError:
                            logger.warning(f"Failed to parse SSE line: {line_str}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens for OpenAI-compatible models.
        
        Uses model-aware tokenization from TokenCounter.
        """
        return TokenCounter.count_tokens(text, self.config.model)


class LLMClientFactory:
    """Factory for creating LLM clients."""
    
    _clients: Dict[LLMProviderType, type[BaseLLMClient]] = {
        LLMProviderType.OLLAMA: OllamaClient,
        LLMProviderType.OPENAI: OpenAICompatibleClient,
        LLMProviderType.OPENAI_COMPATIBLE: OpenAICompatibleClient,
        LLMProviderType.AZURE_OPENAI: OpenAICompatibleClient,
    }
    
    @classmethod
    def create(cls, config: LLMConfig) -> BaseLLMClient:
        """Create LLM client based on configuration."""
        client_class = cls._clients.get(config.provider)
        if not client_class:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
        
        return client_class(config)
    
    @classmethod
    def create_from_env(cls) -> BaseLLMClient:
        """Create LLM client from environment variables."""
        config = LLMConfig.from_env()
        return cls.create(config)


# Convenience functions
async def validate_llm_config(config: Optional[LLMConfig] = None) -> bool:
    """
    Validate LLM configuration and connection.
    
    Tests the connection to the LLM provider by attempting to connect and retrieve models.
    Uses environment variables for configuration if no config is provided.
    
    Args:
        config: LLM configuration (optional, defaults to environment variables)
    
    Returns:
        True if connection is valid and configuration is working, False otherwise
    """
    if config is None:
        config = LLMConfig.from_env()
    
    client = LLMClientFactory.create(config)
    try:
        return await client.validate_connection()
    finally:
        await client.close_session()


async def call_llm(
    messages: List[LLMMessage],
    config: Optional[LLMConfig] = None,
    **kwargs: Any
) -> LLMResponse:
    """
    Call LLM with messages and collect complete response.
    
    Sends messages to the configured LLM provider and collects the full response.
    Automatically handles streaming internally by collecting all chunks.
    Uses environment variables for configuration if no config is provided.
    
    Args:
        messages: List of LLMMessage objects to send to the LLM
        config: LLM configuration (optional, defaults to environment variables)
        **kwargs: Additional parameters to pass to the LLM client (temperature, max_tokens, etc.)
    
    Returns:
        LLMResponse object containing the complete response text and metadata
    
    Raises:
        ValueError: If messages list is empty or contains non-LLMMessage objects
    """
    # Validate inputs
    if not messages:
        raise ValueError("Messages list cannot be empty")
    
    if not isinstance(messages, list):
        raise ValueError("Messages must be a list")
    
    for msg in messages:
        if not isinstance(msg, LLMMessage):
            raise ValueError(f"All messages must be LLMMessage instances, got {type(msg)}")
    
    if config is None:
        config = LLMConfig.from_env()
    
    client = LLMClientFactory.create(config)
    try:
        response = await client.call(messages, **kwargs)
        
        if isinstance(response, LLMResponse):
            return response
        else:
            # Collect streamed response
            content = ""
            async for chunk in response:
                content += chunk
            return LLMResponse(
                content=content,
                model=config.model,
                finish_reason="stop",
            )
    finally:
        await client.close_session()


async def stream_llm(
    messages: List[LLMMessage],
    config: Optional[LLMConfig] = None,
    **kwargs: Any
) -> AsyncIterator[str]:
    """
    Stream LLM response content asynchronously.
    
    Sends messages to the configured LLM provider and yields response chunks as they arrive.
    Allows for real-time processing of long responses without buffering entire response.
    Uses environment variables for configuration if no config is provided.
    
    Args:
        messages: List of LLMMessage objects to send to the LLM
        config: LLM configuration (optional, defaults to environment variables)
        **kwargs: Additional parameters to pass to the LLM client (temperature, max_tokens, etc.)
    
    Yields:
        Response content chunks as strings, as they arrive from the LLM
    
    Raises:
        ValueError: If messages list is empty or contains non-LLMMessage objects
    """
    # Validate inputs
    if not messages:
        raise ValueError("Messages list cannot be empty")
    
    if not isinstance(messages, list):
        raise ValueError("Messages must be a list")
    
    for msg in messages:
        if not isinstance(msg, LLMMessage):
            raise ValueError(f"All messages must be LLMMessage instances, got {type(msg)}")
    
    if config is None:
        config = LLMConfig.from_env()
    
    client = LLMClientFactory.create(config)
    try:
        response = await client.call(messages, stream=True, **kwargs)
        
        if isinstance(response, LLMResponse):
            yield response.content
        else:
            # Stream response
            async for chunk in response:
                yield chunk
    finally:
        await client.close_session()


__all__ = [
    "LLMProviderType",
    "LLMMessage",
    "LLMConfig",
    "LLMResponse",
    "BaseLLMClient",
    "OllamaClient",
    "OpenAICompatibleClient",
    "LLMClientFactory",
    "validate_llm_config",
    "call_llm",
    "stream_llm",
]
