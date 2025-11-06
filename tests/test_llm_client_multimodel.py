"""Tests for LLM client multi-model support.

Tests the ModelType enum, ModelConfig presets, and LLMConfig integration
for supporting multiple LLM providers and models.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from bpmn_agent.core.llm_client import (
    LLMProviderType,
    ModelType,
    ModelConfig,
    LLMConfig,
)


class TestModelType:
    """Test ModelType enum."""
    
    def test_model_type_enum_values(self):
        """Verify all model types are defined."""
        expected_models = {
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "llama2-70b",
            "llama2-13b",
            "mistral",
            "mixtral",
        }
        
        actual_models = {model.value for model in ModelType}
        assert actual_models == expected_models
    
    def test_model_type_string_values(self):
        """Verify model types can be accessed as strings."""
        assert ModelType.GPT4.value == "gpt-4"
        assert ModelType.CLAUDE_3_OPUS.value == "claude-3-opus"
        assert ModelType.LLAMA2_70B.value == "llama2-70b"


class TestModelConfig:
    """Test ModelConfig presets."""
    
    def test_model_config_presets_exist(self):
        """Verify all presets are defined."""
        expected_presets = {
            ModelType.GPT4,
            ModelType.GPT4_TURBO,
            ModelType.GPT35_TURBO,
            ModelType.CLAUDE_3_OPUS,
            ModelType.CLAUDE_3_SONNET,
            ModelType.CLAUDE_3_HAIKU,
            ModelType.LLAMA2_70B,
            ModelType.LLAMA2_13B,
            ModelType.MISTRAL,
            ModelType.MIXTRAL,
        }
        
        assert set(ModelConfig.PRESETS.keys()) == expected_presets
    
    def test_gpt4_config(self):
        """Test GPT-4 configuration preset."""
        config = ModelConfig.get_config_for_model(ModelType.GPT4)
        
        assert config["provider"] == LLMProviderType.OPENAI
        assert config["model"] == "gpt-4"
        assert config["temperature"] == 0.5
        assert config["base_url"] == "https://api.openai.com/v1"
    
    def test_claude_opus_config(self):
        """Test Claude 3 Opus configuration preset."""
        config = ModelConfig.get_config_for_model(ModelType.CLAUDE_3_OPUS)
        
        assert config["provider"] == LLMProviderType.OPENAI_COMPATIBLE
        assert config["model"] == "claude-3-opus-20240229"
        assert config["temperature"] == 0.3
        assert config["base_url"] == "https://api.anthropic.com/v1"
    
    def test_llama2_config(self):
        """Test Llama2 configuration preset."""
        config = ModelConfig.get_config_for_model(ModelType.LLAMA2_70B)
        
        assert config["provider"] == LLMProviderType.OLLAMA
        assert config["model"] == "llama2:70b"
        assert config["temperature"] == 0.7
        assert config["base_url"] == "http://localhost:11434"
    
    def test_get_config_with_string(self):
        """Test getting config with string model name."""
        config = ModelConfig.get_config_for_model("gpt-4")
        assert config["model"] == "gpt-4"
    
    def test_get_config_with_invalid_string(self):
        """Test getting config with invalid string returns empty dict."""
        config = ModelConfig.get_config_for_model("invalid-model")
        assert config == {}


class TestLLMConfigModelType:
    """Test LLMConfig integration with ModelType."""
    
    def test_llm_config_from_model_type_gpt4(self):
        """Test creating LLMConfig from GPT-4 model type."""
        config = LLMConfig.from_model_type(ModelType.GPT4)
        
        assert config.model_type == ModelType.GPT4
        assert config.model == "gpt-4"
        assert config.provider == LLMProviderType.OPENAI
        assert config.temperature == 0.5
        assert config.base_url == "https://api.openai.com/v1"
    
    def test_llm_config_from_model_type_claude(self):
        """Test creating LLMConfig from Claude model type."""
        config = LLMConfig.from_model_type(ModelType.CLAUDE_3_SONNET)
        
        assert config.model_type == ModelType.CLAUDE_3_SONNET
        assert config.model == "claude-3-sonnet-20240229"
        assert config.provider == LLMProviderType.OPENAI_COMPATIBLE
        assert config.temperature == 0.3
    
    def test_llm_config_from_model_type_ollama(self):
        """Test creating LLMConfig from Ollama model type."""
        config = LLMConfig.from_model_type(ModelType.MISTRAL)
        
        assert config.model_type == ModelType.MISTRAL
        assert config.model == "mistral"
        assert config.provider == LLMProviderType.OLLAMA
        assert config.base_url == "http://localhost:11434"
    
    def test_llm_config_from_model_type_string(self):
        """Test creating LLMConfig from string model name."""
        config = LLMConfig.from_model_type("gpt-4-turbo")
        
        assert config.model_type == ModelType.GPT4_TURBO
        assert config.model == "gpt-4-turbo"
    
    def test_llm_config_from_model_type_invalid(self):
        """Test creating LLMConfig with invalid model type raises error."""
        with pytest.raises(ValueError, match="Unknown model type"):
            LLMConfig.from_model_type("invalid-model-type")
    
    def test_llm_config_model_type_optional(self):
        """Test that model_type field is optional."""
        config = LLMConfig(
            model="custom-model",
            provider=LLMProviderType.OLLAMA,
        )
        
        assert config.model_type is None
        assert config.model == "custom-model"


class TestLLMConfigFromEnv:
    """Test LLMConfig.from_env() with model type support."""
    
    def test_from_env_without_model_type(self):
        """Test from_env without LLM_MODEL_TYPE env var."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "mistral",
            "LLM_TEMPERATURE": "0.7",
        }, clear=True):
            config = LLMConfig.from_env()
            
            assert config.provider == "ollama"
            assert config.model == "mistral"
            assert config.temperature == 0.7
            assert config.model_type is None
    
    def test_from_env_with_model_type_gpt4(self):
        """Test from_env with LLM_MODEL_TYPE=gpt-4."""
        with patch.dict(os.environ, {
            "LLM_MODEL_TYPE": "gpt-4",
            "LLM_API_KEY": "test-key",
        }, clear=True):
            config = LLMConfig.from_env()
            
            assert config.model_type == ModelType.GPT4
            assert config.model == "gpt-4"
            assert config.provider == LLMProviderType.OPENAI
            assert config.api_key == "test-key"
    
    def test_from_env_with_model_type_claude(self):
        """Test from_env with LLM_MODEL_TYPE=claude-3-opus."""
        with patch.dict(os.environ, {
            "LLM_MODEL_TYPE": "claude-3-opus",
            "LLM_API_KEY": "claude-key",
        }, clear=True):
            config = LLMConfig.from_env()
            
            assert config.model_type == ModelType.CLAUDE_3_OPUS
            assert config.provider == LLMProviderType.OPENAI_COMPATIBLE
    
    def test_from_env_with_invalid_model_type_fallback(self):
        """Test from_env with invalid model type falls back to env vars."""
        with patch.dict(os.environ, {
            "LLM_MODEL_TYPE": "invalid-model",
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "fallback-model",
        }, clear=True):
            config = LLMConfig.from_env()
            
            # Should fall back to the standard env vars
            assert config.provider == "ollama"
            assert config.model == "fallback-model"
    
    def test_from_env_with_model_type_override_api_key(self):
        """Test that API key env var overrides model type config."""
        with patch.dict(os.environ, {
            "LLM_MODEL_TYPE": "gpt-4",
            "LLM_API_KEY": "my-custom-key",
        }, clear=True):
            config = LLMConfig.from_env()
            
            assert config.api_key == "my-custom-key"


class TestModelConfigAllModels:
    """Comprehensive tests for all model configurations."""
    
    @pytest.mark.parametrize("model_type", [
        ModelType.GPT4,
        ModelType.GPT4_TURBO,
        ModelType.GPT35_TURBO,
        ModelType.CLAUDE_3_OPUS,
        ModelType.CLAUDE_3_SONNET,
        ModelType.CLAUDE_3_HAIKU,
        ModelType.LLAMA2_70B,
        ModelType.LLAMA2_13B,
        ModelType.MISTRAL,
        ModelType.MIXTRAL,
    ])
    def test_all_models_have_valid_config(self, model_type):
        """Test that all models have valid configurations."""
        config = ModelConfig.get_config_for_model(model_type)
        
        assert config is not None
        assert "provider" in config
        assert "model" in config
        assert "temperature" in config
        assert "base_url" in config
        assert "max_tokens" in config
        assert "timeout" in config
    
    @pytest.mark.parametrize("model_type", [
        ModelType.GPT4,
        ModelType.GPT4_TURBO,
        ModelType.GPT35_TURBO,
        ModelType.CLAUDE_3_OPUS,
        ModelType.CLAUDE_3_SONNET,
        ModelType.CLAUDE_3_HAIKU,
        ModelType.LLAMA2_70B,
        ModelType.LLAMA2_13B,
        ModelType.MISTRAL,
        ModelType.MIXTRAL,
    ])
    def test_all_models_create_valid_llm_config(self, model_type):
        """Test that all models can create valid LLMConfig."""
        config = LLMConfig.from_model_type(model_type)
        
        assert config.model_type == model_type
        assert config.provider is not None
        assert config.model is not None
        assert 0.0 <= config.temperature <= 2.0


class TestModelTemperatureSettings:
    """Test that temperature settings are appropriate for each model."""
    
    def test_gpt_models_lower_temperature(self):
        """GPT models should have lower temperature for consistency."""
        gpt4_config = ModelConfig.get_config_for_model(ModelType.GPT4)
        gpt35_config = ModelConfig.get_config_for_model(ModelType.GPT35_TURBO)
        
        assert gpt4_config["temperature"] == 0.5
        assert gpt35_config["temperature"] == 0.5
    
    def test_claude_models_lower_temperature(self):
        """Claude models should have lower temperature."""
        claude_config = ModelConfig.get_config_for_model(ModelType.CLAUDE_3_OPUS)
        
        assert claude_config["temperature"] == 0.3
    
    def test_llama_models_higher_temperature(self):
        """Llama models can use higher temperature."""
        llama_config = ModelConfig.get_config_for_model(ModelType.LLAMA2_70B)
        
        assert llama_config["temperature"] == 0.7


class TestModelTimeoutSettings:
    """Test that timeout settings are appropriate for each model."""
    
    def test_gpt_models_reasonable_timeout(self):
        """GPT models should have reasonable timeouts."""
        gpt4_config = ModelConfig.get_config_for_model(ModelType.GPT4)
        gpt35_config = ModelConfig.get_config_for_model(ModelType.GPT35_TURBO)
        
        assert gpt4_config["timeout"] == 120.0
        assert gpt35_config["timeout"] == 60.0
    
    def test_claude_models_reasonable_timeout(self):
        """Claude models should have reasonable timeouts."""
        opus_config = ModelConfig.get_config_for_model(ModelType.CLAUDE_3_OPUS)
        sonnet_config = ModelConfig.get_config_for_model(ModelType.CLAUDE_3_SONNET)
        
        assert opus_config["timeout"] == 120.0
        assert sonnet_config["timeout"] == 90.0
    
    def test_local_models_longer_timeout(self):
        """Local models (Ollama) should have longer timeouts."""
        llama_config = ModelConfig.get_config_for_model(ModelType.LLAMA2_70B)
        mistral_config = ModelConfig.get_config_for_model(ModelType.MISTRAL)
        
        # Local inference can be slower
        assert llama_config["timeout"] == 180.0
        assert mistral_config["timeout"] == 90.0
