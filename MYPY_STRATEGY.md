# Estrategia para Corregir Todos los Errores de Mypy

## Análisis de Errores Actuales

**Total: 1175 errores**

### Distribución por Tipo:
1. **727 `call-arg`** (62% del total)
   - 21 en código fuente (prioridad alta)
   - 632 en tests (prioridad media)
   - 74 en demo (prioridad baja)

2. **116 `arg-type`** (10%)
3. **73 `union-attr`** (6%)
4. **43 `operator`** (4%)
5. **35 `name-defined`** (3%) - Fácil: imports faltantes
6. **30 `attr-defined`** (3%)
7. **Otros**: ~151 errores diversos

## Estrategia Propuesta

### Fase 1: Configuración y Exclusión Temporal (Rápido)
1. **Configurar mypy para excluir tests y demo temporalmente**
   - Esto reduce errores de 1175 a ~480 (solo código fuente)
   - Nos permite enfocarnos en el código de producción primero

### Fase 2: Correcciones Sistemáticas por Tipo (Prioridad)

#### 2.1 Errores Fáciles (35 errores `name-defined`)
- Agregar imports faltantes
- Tiempo estimado: 30 minutos

#### 2.2 Errores `call-arg` en Código Fuente (21 errores)
- Agregar campos requeridos faltantes
- Tiempo estimado: 1-2 horas

#### 2.3 Errores `union-attr` (73 errores)
- Agregar type guards y verificaciones de None
- Tiempo estimado: 2-3 horas

#### 2.4 Errores `arg-type` (116 errores)
- Corregir tipos de argumentos incompatibles
- Tiempo estimado: 3-4 horas

#### 2.5 Errores `operator` (43 errores)
- Corregir operadores no soportados en tipos
- Tiempo estimado: 1-2 horas

### Fase 3: Tests y Demo (Después del código fuente)
- Crear factories/helpers para reducir repetición
- Corregir errores en tests sistemáticamente

## Plan de Acción Inmediato

1. **Actualizar configuración de mypy** para excluir tests/demo temporalmente
2. **Corregir errores `name-defined`** (35 errores - más rápido)
3. **Corregir errores `call-arg` en código fuente** (21 errores)
4. **Continuar con otros tipos sistemáticamente**

## Opciones de Configuración

### Opción A: Excluir tests/demo temporalmente
```toml
[tool.mypy]
exclude = [
    "tests/.*",
    "demo/.*",
]
```

### Opción B: Usar `# type: ignore` estratégicamente
- Solo para casos donde la corrección es muy compleja
- Documentar por qué se ignora

### Opción C: Crear helpers/factories
- Para reducir repetición en tests
- Ejemplo: `create_test_entity()`, `create_test_relation()`
