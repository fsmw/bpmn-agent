# Guía de Contribución

¡Gracias por tu interés en contribuir a `bpmn-agent`! Este documento describe nuestro proceso de desarrollo y cómo trabajar con Issues, Branches, Pull Requests y CI/CD.

## 📋 Tabla de Contenidos

- [Workflow de Desarrollo](#workflow-de-desarrollo)
- [Crear un Issue](#crear-un-issue)
- [Crear un Branch](#crear-un-branch)
- [Desarrollo Local](#desarrollo-local)
- [Crear un Pull Request](#crear-un-pull-request)
- [CI/CD y Quality Gates](#cicd-y-quality-gates)
- [Estándares de Código](#estándares-de-código)
- [Testing](#testing)
- [Commits y Mensajes](#commits-y-mensajes)

---

## 🔄 Workflow de Desarrollo

Nuestro proceso sigue un flujo estructurado: **Issue → Branch → PR → CI → Merge**

```
1. Crear Issue          → Documenta qué se va a hacer
2. Crear Branch          → Trabaja en una rama específica
3. Desarrollo Local      → Pre-commit hooks aseguran calidad
4. Crear Pull Request    → Vincula PR al Issue
5. CI/CD Automático      → Tests, linting, coverage
6. Code Review           → Revisión y aprobación
7. Merge                 → Cierre automático del Issue
```

### ¿Por qué este workflow?

- ✅ **Trazabilidad completa**: Cada cambio está vinculado a un issue
- ✅ **Calidad garantizada**: CI/CD previene código roto en `main`
- ✅ **Documentación automática**: Issues documentan decisiones y contexto
- ✅ **Colaboración efectiva**: Code review más informado
- ✅ **Métricas**: Visibilidad del progreso del proyecto

---

## 📝 Crear un Issue

Antes de empezar a codificar, **siempre crea un Issue** que describa el trabajo a realizar.

### Cuándo crear un Issue

- ✅ Nueva funcionalidad
- ✅ Corrección de bugs
- ✅ Mejoras de documentación
- ✅ Refactorización
- ✅ Tareas de infraestructura

### Cómo crear un Issue

#### Opción 1: Desde GitHub CLI

```bash
cd src/bpmn-agent
gh issue create --title "Título descriptivo" --body "Descripción detallada" --label "enhancement"
```

#### Opción 2: Desde GitHub Web

1. Ve a https://github.com/fsmw/bpmn-agent/issues/new
2. Usa el template apropiado (si existe)
3. Completa título, descripción, labels

### Template de Issue Recomendado

```markdown
## 🎯 Objetivo
Descripción clara del objetivo y contexto.

## 📋 Tareas
- [ ] Tarea 1
- [ ] Tarea 2
- [ ] Tarea 3

## 📁 Archivos a Modificar
- `ruta/al/archivo.py`
- `tests/test_archivo.py`

## ✅ Criterios de Éxito
- [ ] Criterio 1
- [ ] Criterio 2
- [ ] Tests pasan
- [ ] Coverage > 75%

## 🔗 Relacionado
- Issue: #X
- Plan: `ARCHIVO.md`

## 📝 Notas
Notas adicionales, consideraciones, etc.
```

### Labels Disponibles

**Por Tipo:**
- `bug` - Corrección de errores
- `enhancement` - Nueva funcionalidad
- `documentation` - Cambios en documentación
- `refactoring` - Refactorización

**Por Fase/Componente:**
- `phase1`, `phase2`, `phase3`, `phase4`, `phase5`
- `ci-cd` - Mejoras de CI/CD
- `testing` - Tests
- `validation` - Validación

**Por Prioridad:**
- `priority:high` - Alta prioridad
- `priority:medium` - Prioridad media
- `priority:low` - Baja prioridad

---

## 🌿 Crear un Branch

Una vez que tengas un Issue, crea un branch para trabajar en él.

### Opción 1: Usar GitHub CLI (Recomendado)

```bash
cd src/bpmn-agent
gh issue develop <ISSUE_NUMBER>
```

Esto automáticamente:
- Crea un branch vinculado al issue (ej: `issue-2-integrate-phase4-validation`)
- Hace checkout del branch
- Vincula el branch al issue en GitHub

### Opción 2: Crear Manualmente

```bash
cd src/bpmn-agent
git checkout main
git pull origin main
git checkout -b feature/issue-<NUMBER>-descripcion-corta
```

### Convención de Nombres de Branches

```
feature/issue-<NUMBER>-descripcion-corta
bugfix/issue-<NUMBER>-descripcion-corta
docs/issue-<NUMBER>-descripcion-corta
refactor/issue-<NUMBER>-descripcion-corta
```

Ejemplos:
- `feature/issue-2-integrate-phase4-validation`
- `bugfix/issue-5-fix-xsd-validation-error`
- `docs/issue-8-update-readme`

---

## 💻 Desarrollo Local

### Setup Inicial

```bash
cd src/bpmn-agent

# Setup automatizado (recomendado)
bash scripts/setup-dev.sh

# O manualmente
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type pre-push
```

### Pre-commit Hooks

Los hooks se ejecutan automáticamente en cada commit:

- ✅ **black** - Formateo de código
- ✅ **ruff** - Linting y auto-fix
- ✅ **mypy** - Type checking
- ✅ **pytest-unit** - Tests unitarios rápidos
- ✅ **pytest-integration** - Tests de integración
- ✅ **coverage-check** - Verificación de coverage (warning si < 75%)

### Ejecutar Hooks Manualmente

```bash
# Todos los archivos
pre-commit run --all-files

# Solo archivos staged
pre-commit run

# Hook específico
pre-commit run black --all-files
```

### Desarrollo Iterativo

```bash
# 1. Hacer cambios
vim bpmn_agent/archivo.py

# 2. Agregar cambios
git add bpmn_agent/archivo.py

# 3. Commit (pre-commit se ejecuta automáticamente)
git commit -m "feat: agregar funcionalidad X (refs #<ISSUE_NUMBER>)"

# 4. Si pre-commit falla, corregir y volver a commitear
# 5. Repetir hasta completar el trabajo
```

---

## 🔀 Crear un Pull Request

Cuando tu trabajo esté listo, crea un Pull Request.

### Requisitos Previos

- ✅ Todos los tests pasan localmente
- ✅ Pre-commit hooks pasan
- ✅ Coverage > 75%
- ✅ Código formateado y linted
- ✅ Commits con mensajes descriptivos

### Crear PR desde Terminal

```bash
cd src/bpmn-agent

# Push del branch
git push origin feature/issue-<NUMBER>-descripcion

# Crear PR vinculado al issue
gh pr create --title "feat: Título descriptivo" --body "Fixes #<ISSUE_NUMBER>

## Cambios
- Cambio 1
- Cambio 2
- Cambio 3

## Testing
- [x] Tests unitarios pasan
- [x] Tests de integración pasan
- [x] Coverage > 75%

## Checklist
- [x] Código sigue estilo del proyecto
- [x] Documentación actualizada si es necesario
- [x] Sin breaking changes (o documentados)

Fixes #<ISSUE_NUMBER>"
```

### Crear PR desde GitHub Web

1. Push tu branch: `git push origin feature/issue-<NUMBER>-descripcion`
2. Ve a https://github.com/fsmw/bpmn-agent/compare
3. Selecciona tu branch
4. Completa el formulario del PR
5. **Importante**: Incluye `Fixes #<ISSUE_NUMBER>` en la descripción

### Template de PR

```markdown
## Descripción
Breve descripción de los cambios.

Fixes #<ISSUE_NUMBER>

## Cambios
- Cambio 1
- Cambio 2
- Cambio 3

## Testing
- [x] Tests unitarios pasan
- [x] Tests de integración pasan
- [x] Coverage > 75%
- [x] Probado localmente

## Checklist
- [x] Código sigue estilo del proyecto (black, ruff)
- [x] Type checking pasa (mypy)
- [x] Documentación actualizada si es necesario
- [x] Sin breaking changes (o documentados)
- [x] Pre-commit hooks pasan

## Screenshots (si aplica)
...

## Notas Adicionales
...
```

### Palabras Clave para Cerrar Issues

Incluye una de estas frases en el PR para cerrar automáticamente el issue:

- `Fixes #<NUMBER>` - Cierra el issue cuando se mergea
- `Closes #<NUMBER>` - Igual que Fixes
- `Resolves #<NUMBER>` - Igual que Fixes
- `Related to #<NUMBER>` - Solo vincula, no cierra

---

## ✅ CI/CD y Quality Gates

### ¿Qué pasa cuando creas un PR?

GitHub Actions ejecuta automáticamente:

#### 1. CI Workflow (`ci.yml`)
- ✅ Tests en Python 3.10, 3.11, 3.12
- ✅ Linting (black, ruff)
- ✅ Type checking (mypy)
- ✅ Unit tests
- ✅ Integration tests
- ✅ Coverage check (debe ser > 75%)

#### 2. Quality Workflow (`quality.yml`)
- ✅ Security scanning (Bandit)
- ✅ Dependency vulnerabilities (Safety)
- ✅ Code complexity (xenon, radon)
- ✅ License compliance
- ✅ SBOM generation

#### 3. Release Workflow (`release.yml`)
- Solo se ejecuta cuando se crea un tag `v*`

### Verificar Estado de CI

```bash
# Ver checks del PR actual
gh pr checks

# Ver runs recientes
gh run list

# Ver detalles de un run específico
gh run view <RUN_ID>
```

### ¿Qué hacer si CI falla?

1. **Revisar logs**: `gh run view <RUN_ID>` o desde GitHub web
2. **Reproducir localmente**: Ejecutar el mismo comando que falló
3. **Corregir el problema**
4. **Push nuevo commit**: CI se ejecutará automáticamente

### Quality Gates (REQUERIDOS)

**El PR NO puede mergearse si:**

- ❌ Tests fallan
- ❌ Coverage < 75%
- ❌ Linting falla
- ❌ Type checking falla
- ❌ Security issues críticos

**El PR puede mergearse cuando:**

- ✅ Todos los checks pasan
- ✅ Al menos 1 aprobación de review (si está configurado)
- ✅ Branch está actualizado con `main`

---

## 📏 Estándares de Código

### Formateo

Usamos **black** con configuración estándar:

```bash
black bpmn_agent/
```

### Linting

Usamos **ruff** para linting rápido:

```bash
ruff check bpmn_agent/
ruff check --fix bpmn_agent/  # Auto-fix
```

### Type Checking

Usamos **mypy** para type safety:

```bash
mypy bpmn_agent/ --ignore-missing-imports
```

### Imports

Orden de imports (ruff los ordena automáticamente):

```python
# 1. Standard library
import os
from typing import List

# 2. Third-party
from pydantic import BaseModel
import requests

# 3. Local
from bpmn_agent.core import LLMClient
from bpmn_agent.models import Process
```

---

## 🧪 Testing

### Estructura de Tests

```
tests/
├── unit/              # Tests unitarios rápidos
├── integration/      # Tests de integración
├── test_*.py         # Tests por módulo
└── conftest.py       # Fixtures compartidas
```

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Solo unitarios (rápidos)
pytest -m "unit and not llm and not slow"

# Solo integración
pytest -m "integration and not llm"

# Con coverage
pytest --cov=bpmn_agent --cov-report=term --cov-report=html

# Archivo específico
pytest tests/test_archivo.py

# Test específico
pytest tests/test_archivo.py::test_funcion
```

### Markers Disponibles

- `@pytest.mark.unit` - Test unitario
- `@pytest.mark.integration` - Test de integración
- `@pytest.mark.llm` - Requiere LLM (se salta en CI por defecto)
- `@pytest.mark.slow` - Test lento

### Coverage Requirements

- **Mínimo**: 75% (línea y branch)
- **Objetivo**: 80%+
- **CI falla** si coverage < 75%

### Escribir Tests

```python
import pytest
from bpmn_agent.core import LLMClient

@pytest.mark.unit
def test_funcion_basica():
    """Test unitario simple."""
    result = funcion_a_testear()
    assert result == expected

@pytest.mark.integration
async def test_integracion():
    """Test de integración."""
    client = LLMClient(...)
    result = await client.call(...)
    assert result is not None
```

---

## 📝 Commits y Mensajes

### Convención de Commits

Usamos **Conventional Commits**:

```
<tipo>(<scope>): <descripción>

[body opcional]

[footer opcional]
```

### Tipos de Commits

- `feat`: Nueva funcionalidad
- `fix`: Corrección de bug
- `docs`: Documentación
- `style`: Formateo (sin cambios de código)
- `refactor`: Refactorización
- `test`: Tests
- `chore`: Tareas de mantenimiento
- `ci`: Cambios en CI/CD
- `perf`: Mejoras de rendimiento

### Ejemplos

```bash
# Feature
git commit -m "feat(validation): agregar validación XSD (refs #2)"

# Bug fix
git commit -m "fix(xsd): corregir error de parsing (fixes #5)"

# Documentación
git commit -m "docs(readme): actualizar instrucciones de instalación"

# Refactor
git commit -m "refactor(orchestrator): simplificar lógica de validación"
```

### Vincular Commits a Issues

```bash
# Cierra el issue cuando se mergea el PR
git commit -m "feat: implementar X (fixes #2)"

# Solo referencia el issue
git commit -m "feat: implementar X (refs #2)"
```

---

## 🔍 Code Review

### Proceso de Review

1. **Crear PR** → CI se ejecuta automáticamente
2. **Esperar aprobaciones** → Al menos 1 reviewer (si está configurado)
3. **Abordar comentarios** → Hacer cambios y push nuevos commits
4. **Merge** → Cuando todos los checks pasan y hay aprobación

### Qué Buscar en un Review

- ✅ Código sigue estándares del proyecto
- ✅ Tests cubren los cambios
- ✅ Sin regresiones
- ✅ Documentación actualizada
- ✅ Performance aceptable
- ✅ Seguridad considerada

---

## 🚀 Merge

### Opciones de Merge

```bash
# Squash merge (recomendado)
gh pr merge <PR_NUMBER> --squash --delete-branch

# Merge commit
gh pr merge <PR_NUMBER> --merge --delete-branch

# Rebase merge
gh pr merge <PR_NUMBER> --rebase --delete-branch
```

### Después del Merge

- ✅ Issue se cierra automáticamente (si PR tiene "Fixes #X")
- ✅ Branch se elimina automáticamente (si usas `--delete-branch`)
- ✅ CI se ejecuta en `main` para verificar que todo sigue funcionando

---

## 📚 Recursos Adicionales

- **[DevOps Setup](docs/development/devops-setup.md)** - Setup detallado de CI/CD
- **[GitHub Workflow](docs/development/github-workflow.md)** - Mejores prácticas de workflow
- **README.md** - Documentación general del proyecto
- **CI_CD_IMPROVEMENTS.md** - Detalles de mejoras CI/CD

---

## ❓ Preguntas Frecuentes

### ¿Puedo saltarme los pre-commit hooks?

Sí, con `git commit --no-verify`, pero **NO recomendado**. Los hooks previenen problemas antes de que lleguen a CI.

### ¿Qué pasa si mi PR falla en CI pero pasa localmente?

1. Verifica versiones de herramientas (black, ruff, mypy, pytest)
2. Verifica versión de Python (CI usa 3.10, 3.11, 3.12)
3. Ejecuta `pre-commit run --all-files` localmente
4. Revisa logs de CI para detalles específicos

### ¿Puedo crear un PR sin un Issue?

Técnicamente sí, pero **recomendamos crear un Issue primero** para documentar el contexto y permitir discusión antes de implementar.

### ¿Cómo actualizo mi branch con cambios de main?

```bash
git checkout feature/issue-<NUMBER>-descripcion
git fetch origin
git merge origin/main
# O usar rebase si prefieres
git rebase origin/main
```

---

## 🎯 Resumen Rápido

```bash
# 1. Crear Issue
gh issue create --title "..." --body "..." --label "enhancement"

# 2. Crear Branch
gh issue develop <ISSUE_NUMBER>

# 3. Desarrollo
# ... hacer cambios ...
git add .
git commit -m "feat: ... (refs #<NUMBER>)"
git push origin feature/issue-<NUMBER>-descripcion

# 4. Crear PR
gh pr create --title "..." --body "Fixes #<NUMBER>"

# 5. Esperar CI y Review
gh pr checks

# 6. Merge (cuando esté listo)
gh pr merge <PR_NUMBER> --squash --delete-branch
```

---

**¡Gracias por contribuir a bpmn-agent!** 🚀
