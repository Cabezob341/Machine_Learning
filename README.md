# Challenge: Pipeline de Machine Learning — Sueño en los primeros meses de vida

**Curso:** Machine Learning — UPCH | **Caso de examen:** Tipo C, sueño en los recién nacidos.

> La hoja de evaluación adjunta se titula **Challenge 1 (Individual): Pipeline de Machine Learning**, aunque la solicitud del estudiante dice Challenge 2. Se desarrolla íntegramente la hoja adjunta, sin suponer que existe otra rúbrica.

## 1. Presentación del caso y problema (5 puntos)

### Contexto del examen
La infografía **El Sueño en los Primeros Meses de Vida** presenta un estudio científico de 2025 con **152 bebés, 100 padres y más de 35 000 horas de sueño registradas por actigrafía**; refiere observaciones de los 3, 6 y 12 meses. El gráfico sobre ciclos muestra un aumento aproximado desde **50 min (3 meses) a 58 min (6 meses) y 62 min (12 meses)**. La misma infografía cita como orientación divulgativa que los recién nacidos duermen aproximadamente 16 horas diarias en las primeras semanas, de manera fragmentada. **No son filas individuales de datos liberadas por el estudio**.

### Problema
Los patrones del sueño infantil varían con la edad y están influidos por múltiples observaciones, lo que dificulta describir la evolución de la duración del ciclo de sueño y detectar cuándo faltan registros relevantes. Se propone un sistema **académico y demostrativo** para ordenar observaciones y **estimar la duración del ciclo de sueño (minutos)** a partir de la edad, lactancia, actividad motora y despertares nocturnos. **No se pretende diagnosticar ni indicar tratamientos médicos.**

### Pregunta de investigación
¿Es posible estimar la duración promedio del ciclo de sueño en minutos a partir de variables observables del bebé, mediante aprendizaje supervisado con regresión lineal, en un conjunto de datos sintéticos inspirado en la infografía?

### Objetivo general
Diseñar un pipeline reproducible que formalice el caso, explore estados válidos y entrene/evalúe un modelo de regresión para estimar duración de ciclos de sueño.

### Objetivos específicos
1. Identificar variables, estados, acciones y restricciones en el registro de sueño.
2. Implementar búsqueda en anchura (BFS) hasta alcanzar un estado objetivo documentado.
3. Crear y verificar datos sintéticos de ejemplo, distinguiéndolos de fuentes reales.
4. Entrenar regresión lineal con división entrenamiento/prueba, evaluar con MAE, RMSE y R², y visualizar las predicciones.

### Justificación de IA/ML
El aprendizaje supervisado usa entradas conocidas (X) y una variable numérica objetivo (y). Un algoritmo puede ajustar relaciones observadas en ejemplos para estimar valores continuos futuros. En este caso, **regresión** es adecuada porque se predicen **minutos**, no una clase sí/no. Se utiliza regresión lineal como modelo interpretable de referencia, según los temas de las sesiones 5, 6 y 7. Una asociación estadística no demuestra causalidad.

### Variables
| Variable | Tipo | Función |
|---|---|---|
| edad_meses | entero, 0 a 12 | Predictora |
| lactancia | binaria, 0/1 | Predictora ficticia, sin interpretación causal |
| actividad_motora_0_10 | continua, 0 a 10 | Predictora |
| despertares_noche | entero, 1 a 9 | Predictora |
| duracion_ciclo_min | continua, minutos | **Objetivo de regresión (y)** |
| horas_sueno_dia | continua, horas | Variable contextual, no utilizada como predictor |

**Advertencia:** la infografía no contiene una tabla individual con estas observaciones. El CSV de 180 casos es **100 % simulado** y sus parámetros son supuestos didácticos. No puede utilizarse para afirmaciones de precisión clínica.

## 2. Espacio de estados (5 puntos)

**Definición formal:** P=(S,A,T,s0,G,C).

- **S (estados):** tupla `(edad_registrada, sueno_registrado, variables_revisadas, orientacion_documentada)`, cuatro indicadores booleanos de progreso. Son estados del **proceso de información**, no estados clínicos del bebé.
- **Estado inicial s0:** `(False, False, False, False)`.
- **A (operadores):** registrar_edad, registrar_sueno, validar_variables, documentar_orientacion.
- **T (transición):** la acción activa el indicador asociado cuando se satisfacen las precondiciones.
- **G (objetivo):** `(True, True, True, True)`, registro completo, validado y con orientación informativa documentada.
- **C (restricciones):** edad entre 0 y 12 meses; duración del ciclo positiva; horas de sueño en 0–24; no validar antes de contar con edad y sueño; no documentar orientación antes de validar; no cambiar valores ya registrados en esta búsqueda simplificada. **Los límites son reglas computacionales de la demostración, no umbrales médicos.**

### Ejemplo de transición
`(0,0,0,0) --registrar_edad--> (1,0,0,0) --registrar_sueno--> (1,1,0,0) --validar_variables--> (1,1,1,0) --documentar_orientacion--> (1,1,1,1)`.

El algoritmo BFS explora todas las transiciones permitidas por niveles y encuentra un camino de cuatro acciones, ya que cada operación activa como máximo un indicador. Incluye búsqueda, estados visitados, comprobación de meta y reporte del resultado.

![Representación visual](imagenes/diagrama_estados.png)

## 3. Notebook: razonamiento, búsqueda y resultados (5 puntos)

Abrir `notebooks/01_Razonamiento_Busqueda.ipynb`. Contiene problema, estado inicial, operadores, restricciones, función de sucesores, búsqueda BFS, resultado, trayectoria y gráfica de estados recorridos.

## 4. Notebook: predicción y visualización (5 puntos)

Abrir `notebooks/02_Regresion_Sueno.ipynb`. Contiene generación/carga de observaciones sintéticas, exploración, entrenamiento, prueba, métricas, gráfico edad vs duración, gráfico real vs predicho, residuos y una predicción demostrativa.

**Elección del modelo:** regresión lineal múltiple `y = β0 + β1·edad + β2·lactancia + β3·actividad + β4·despertares + ε`. Es un modelo base con coeficientes interpretables. La calidad del ajuste se evalúa sobre datos de prueba no usados en el entrenamiento. Los coeficientes son asociaciones **dentro de datos generados**, no efectos biológicos demostrados. Una futura versión con datos reales debe revisar sesgos y comparar modelos no lineales.

## Bibliografía del trabajo
- Infografía entregada para el parcial: *El Sueño en los Primeros Meses de Vida* (incluye mención de un estudio de 2025 y referencia divulgativa SleepyKids; datos resumidos en la propia foto).
- Material de clase UPCH: *Sesión 5 — Tipos de Aprendizaje*.
- Material de clase UPCH: *Sesión 6 — Clasificación vs Regresión y Clustering vs Reducción*.
- Material de clase UPCH: *Sesión 7 — Regresión Lineal, Regresión No Lineal y Regresión Logística*.
- Hoja de actividad *Challenge 1 (Individual): Pipeline de Machine Learning*, 06/10/2026.

## Reproducibilidad
Python 3, numpy, pandas, matplotlib y scikit-learn. Ejecutar cada notebook de arriba hacia abajo en Google Colab. La semilla aleatoria permite repetir la simulación. El archivo `datos/sueno_sintetico.csv` contiene los mismos datos usados en la preparación del proyecto.
