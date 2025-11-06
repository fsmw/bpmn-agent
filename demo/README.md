# Demos y Ejemplos Ejecutables

Esta carpeta contiene scripts de demostración y ejemplos ejecutables del proyecto BPMN Agent.

## 📋 Demos Disponibles

### Orchestrator Demos

- **`orchestrator.py`** - Demo básico del orchestrator
- **`phase3_orchestrator.py`** - Demo del orchestrator con funcionalidades Phase 3
- **`phase3_tools.py`** - Demo de herramientas Phase 3
- **`phase3_tools_working.py`** - Demo funcional de herramientas Phase 3

### Validation Demos

- **`validation_phase4_demo.sh`** - Script de demostración de validación Phase 4

## 🚀 Ejecución

### Requisitos Previos

1. Asegúrate de tener el entorno virtual activado:
   ```bash
   cd /home/fsmw/dev/bpmn/src/bpmn-agent
   source .venv/bin/activate
   ```

2. Verifica que las dependencias estén instaladas:
   ```bash
   pip install -e ".[dev]"
   ```

### Ejecutar Demos

```bash
# Demo básico del orchestrator
python demo/orchestrator.py

# Demo Phase 3
python demo/phase3_orchestrator.py

# Demo de validación Phase 4
bash demo/validation_phase4_demo.sh
```

## 📝 Notas

- Los demos pueden requerir configuración de variables de entorno (LLM_PROVIDER, etc.)
- Algunos demos pueden requerir servicios externos (Ollama, OpenAI, etc.)
- Revisa los comentarios en cada archivo para más detalles

## 🔗 Ver También

- [Guías de Usuario](../docs/guides/)
- [Ejemplos BPMN](../examples/)
- [Documentación Principal](../README.md)
