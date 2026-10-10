# Taller práctico: Reducción de dimensionalidad con PCA
## Dataset: Cáncer de mama (Wisconsin) · Aprendizaje no supervisado

**Asignatura:** Inteligencia Artificial  
**Modalidad:** Taller guiado en laboratorio (libreto del profesor, disponible para repaso del estudiante)  
**Duración sugerida:** 2 horas  
**Herramientas:** Python, **UV** y **VS Code** (sin Jupyter: trabajamos con archivos `.py`)

---

## 0. ¿Para qué sirve este taller?

En la clase magistral vimos que **PCA (Análisis de Componentes Principales)** busca nuevas variables, llamadas *componentes*, que resumen la información de muchas variables originales perdiendo la menor cantidad posible de información.

Hoy lo hacemos con las manos. Al terminar, el estudiante podrá:

1. Explicar por qué es necesario **estandarizar** los datos antes de PCA.
2. Leer la gráfica de **varianza explicada acumulada** y decidir cuántos componentes conservar.
3. Interpretar **valores propios** (varianza de cada componente) y **vectores propios** (pesos de cada variable).
4. Comprobar que, con 6 componentes en lugar de 30 variables, un modelo sencillo sigue clasificando bien.

> **Nota para el profesor (idea didáctica):** PCA es *no supervisado*: nunca mira la etiqueta `y`. Insista en esto. La etiqueta solo se usa al final, como "prueba de fuego" para verificar si la compresión conservó información útil.

**Estructura de la sesión**

| Bloque | Contenido | Tiempo |
|---|---|---|
| Preparación | Entorno con UV y VS Code | 15 min |
| Pasos 1-3 | Datos, estandarización, curva de varianza | 35 min |
| Pasos 4-5 | PCA con 6 componentes, valores y vectores propios | 30 min |
| Pasos 6-7 | Validación con árbol de decisión y visualización | 30 min |
| Cierre | Preguntas y reto | 10 min |

---

## 1. Preparación del entorno con UV

UV crea el entorno virtual y administra las librerías por nosotros. En una terminal (puede ser la integrada de VS Code: `Ctrl + ñ` o `` Ctrl + ` ``):

```bash
# 1. Crear la carpeta del proyecto e iniciarla con UV
uv init taller-pca
cd taller-pca

# 2. Agregar las librerías que vamos a usar
uv add numpy pandas matplotlib scikit-learn

# 3. Abrir la carpeta en VS Code
code .
```

**¿Qué hace cada comando?**

- `uv init taller-pca` crea la carpeta con un `pyproject.toml` (la "hoja de vida" del proyecto) y un `main.py`.
- `uv add ...` instala las librerías dentro de un entorno aislado `.venv` y las registra en `pyproject.toml`. Así cualquier compañero puede reproducir exactamente su entorno.
- `code .` abre VS Code en esa carpeta.

**En VS Code:**

1. Cree un archivo nuevo llamado `taller_pca.py` (lo usaremos en todo el taller).
2. Abajo a la derecha, o con `Ctrl+Shift+P` → *Python: Select Interpreter*, elija el que está en `.venv`.
3. Para ejecutar, en la terminal:

```bash
uv run taller_pca.py
```

> **Cómo trabajar:** vamos **agregando** cada bloque al final de `taller_pca.py` y ejecutando de nuevo. En Jupyter cada celda recordaba el estado; aquí el archivo se ejecuta completo de arriba hacia abajo, y por eso el orden de los bloques importa.

**Punto de control:** `uv run python -c "import sklearn; print(sklearn.__version__)"` debe imprimir un número de versión sin errores.

> **Problema frecuente:** si en Linux la gráfica no se abre, puede faltar `python3-tk`. Alternativa: reemplazar `plt.show()` por `plt.savefig("figura.png")` y abrir la imagen desde VS Code.

---

## 2. Paso 1: Importar las librerías

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

**Explicación**

| Línea | Para qué sirve |
|---|---|
| `numpy` | Cálculo con arreglos numéricos (los datos son matrices). |
| `pandas` | Tablas de datos. Hoy lo usaremos poco, pero es la herramienta estándar para explorar. |
| `matplotlib.pyplot` | Gráficas. |
| `load_breast_cancer` | Carga el dataset que ya viene incluido en scikit-learn (no hay que descargar nada). |
| `PCA` | El algoritmo protagonista. |
| `StandardScaler` | Estandariza las variables (media 0 y desviación 1). |

**Pregunta al aula:** ¿qué pasaría si en lugar de importar `PCA` importáramos `KMeans`? (Para recordar que ambos son técnicas no supervisadas, pero con objetivos distintos: reducir variables vs. agrupar observaciones.)

---

## 3. Paso 2: Cargar y conocer los datos

```python
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
feature_names = cancer.feature_names
target_names = cancer.target_names

# --- Exploración (agregado didáctico) ---
print("Forma de x:", x.shape)
print("Clases:", target_names)
print("Primeras 5 variables:", feature_names[:5])
```

**Explicación**

- `cancer` es un objeto que contiene todo el dataset.
- `x` es la **matriz de características**: 569 pacientes (filas) × 30 variables (columnas) medidas sobre imágenes de células (radio, textura, perímetro, área, suavidad, etc.).
- `y` es la **etiqueta**: 0 = maligno, 1 = benigno.
- `feature_names` y `target_names` son los nombres, para interpretar resultados.

**Qué debe ver:** `Forma de x: (569, 30)` y `['malignant' 'benign']`.

**Pregunta al aula:** ¿30 variables se pueden graficar? ¿Cuántas dimensiones podemos ver los humanos? Este es el problema que PCA ayuda a resolver. Además, muchas de estas variables son redundantes (el radio, el perímetro y el área de una célula están muy relacionados).

> **Convención:** por tradición `X` se escribe en mayúscula (matriz) y `y` en minúscula (vector). Aquí se usa `x` minúscula; es válido, pero mantenga la misma convención en todo el código.

---

## 4. Paso 3: Estandarizar los datos

```python
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
```

**Explicación**

- `StandardScaler()` crea el estandarizador.
- `fit_transform(x)` hace dos cosas: **calcula** la media y desviación de cada columna (`fit`) y luego **transforma** cada valor así: `(valor − media) / desviación`.
- El resultado `x_scaled` tiene cada variable con media 0 y desviación estándar 1.

**¿Por qué es obligatorio?** PCA busca las direcciones de mayor varianza. Si una variable (por ejemplo, el *área*, que va en cientos) tiene una escala mucho mayor que otra (la *suavidad*, que va de 0 a 0.2), PCA creería que el área es "más importante" solo por sus unidades. Estandarizar pone a todas en igualdad de condiciones.

**Verificación (sugerida, 2 minutos):**

```python
print("Medias después de escalar:", x_scaled.mean(axis=0).round(2)[:5])
print("Desv. estándar después de escalar:", x_scaled.std(axis=0).round(2)[:5])
```

Debe ver ceros y unos.

**Experimento opcional (muy didáctico):** repetir el PCA del paso siguiente con `x` sin escalar y comparar el primer componente. Los estudiantes verán que un solo componente "se come" casi toda la varianza solo por la escala del área.

---

## 5. Paso 4: ¿Cuántos componentes necesitamos?

```python
pca = PCA()
pca.fit(x_scaled)

varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = varianza_explicada.cumsum()

plt.plot(range(1, 31), varianza_acumulada, marker="o")
plt.axhline(y=0.85, color="r", linestyle="--")
plt.xlabel("Número de componentes")
plt.ylabel("Varianza acumulada")
plt.grid(True)
plt.show()
```

**Explicación línea por línea**

- `PCA()` sin parámetros crea **todos** los componentes posibles (30, uno por variable original).
- `pca.fit(x_scaled)` aprende los componentes. Observe que se usa `fit` y no `fit_transform`: por ahora solo queremos ver cuánta información aporta cada uno.
- `explained_variance_ratio_` es una lista con la **fracción de la varianza total** que explica cada componente (el primero explica la mayor parte, el segundo la siguiente, y así sucesivamente).
- `.cumsum()` calcula la suma acumulada: con 1 componente, con 2, con 3...
- `plt.plot(range(1, 31), ...)` dibuja la curva; `range(1, 31)` genera 1, 2, ..., 30.
- `plt.axhline(y=0.85, ...)` dibuja una línea roja horizontal en 85 %: nuestro umbral de "información suficiente".
- Las demás líneas rotulan la gráfica y activan la cuadrícula.

**Cómo leer la gráfica:** buscamos el **primer punto donde la curva cruza la línea roja**. Ahí está el número mínimo de componentes para conservar al menos el 85 % de la información.

**Qué debe ver:** la curva cruza la línea en el componente **6** (con 5 componentes está en torno a 84.7 % y con 6 pasa de 88 %). Por eso el siguiente paso usa 6.

**Preguntas al aula**

1. ¿Por qué el 85 %? (Es un criterio práctico; en otros problemas se usa 90 % o 95 %. Es una decisión del analista, no una ley.)
2. Pasamos de 30 variables a 6: ¿qué porcentaje de reducción es? (80 %).
3. ¿Qué compensación (*trade-off*) estamos aceptando? (Perdemos algo de información a cambio de simplicidad.)

---

## 6. Paso 5: Aplicar PCA con 6 componentes

```python
pca = PCA(n_components=6)
x_pca = pca.fit_transform(x_scaled)

# --- Comprobación ---
print("Forma original:", x_scaled.shape)
print("Forma reducida:", x_pca.shape)
```

**Explicación**

- `n_components=6` le dice a PCA que conserve solo los 6 componentes principales.
- `fit_transform` aprende los componentes **y** proyecta los datos sobre ellos.
- `x_pca` es la nueva matriz: **569 × 6**. Cada paciente ahora se describe con 6 números en lugar de 30.

**Idea clave para el estudiante:** los 6 nuevos números ya no son "radio" ni "textura": son *combinaciones* de las 30 variables originales. Perdemos interpretabilidad directa, pero ganamos compactación.

---

## 7. Paso 6: Valores propios y vectores propios

```python
valores_propios = pca.explained_variance_
vectores_propios = pca.components_

print(valores_propios)
print(vectores_propios)
```

**Explicación**

- `explained_variance_` son los **valores propios** (*eigenvalues*): cuánta varianza captura cada componente.
- `components_` son los **vectores propios** (*eigenvectors*): una matriz de **6 filas × 30 columnas**. Cada fila es un componente y cada número es el **peso** de una variable original en ese componente.

**Qué debe ver** (valores propios, aproximados):

```
[13.30  5.70  2.82  1.98  1.65  1.21]
```

**Cómo interpretarlos**

- El primer componente tiene un valor propio de ~13.3. Como los datos estandarizados tienen 30 variables y una varianza total cercana a 30, el primer componente explica aproximadamente **44 %** de la información (13.3 / 30). El segundo, cerca del **19 %**.
- Los valores propios **siempre decrecen**: el componente 1 es el más informativo.
- En la primera fila de `vectores_propios`, casi todos los pesos son positivos y de magnitud parecida (alrededor de 0.2). Eso significa que el primer componente funciona como un **"índice general de tamaño/irregularidad celular"**: sube cuando casi todas las medidas suben.

**Actividad guiada (5 minutos):** pida a los estudiantes que identifiquen, con ayuda de `feature_names`, las 3 variables con mayor peso absoluto en el primer componente:

```python
pesos_pc1 = pd.Series(vectores_propios[0], index=feature_names)
print(pesos_pc1.abs().sort_values(ascending=False).head(3))
```

**Pregunta al aula:** ¿qué significa que un peso sea negativo? (La variable se mueve en sentido contrario al componente.)

---

## 8. Paso 7: ¿La compresión conservó información útil? (validación)

Aquí usamos la etiqueta `y` por primera vez, solo para **evaluar**. Entrenamos un árbol de decisión sencillo sobre los 6 componentes.

Primero agregue la importación a su archivo (puede ir al inicio, junto a las demás):

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
```

Luego el bloque principal:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, test_size=0.3, random_state=42
)

tree = DecisionTreeClassifier(max_depth=2, random_state=42)
tree.fit(x_train, y_train)

pred_tree = tree.predict(x_test)
print("Exactitud:", accuracy_score(y_test, pred_tree))

cm = confusion_matrix(y_test, pred_tree)
disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
disp.plot()
plt.show()
```

**Explicación**

- `train_test_split(..., test_size=0.3)` separa 70 % de los datos para entrenar y 30 % para probar. `random_state=42` fija el azar para que todos obtengan el mismo resultado.
- `DecisionTreeClassifier(max_depth=2)` crea un árbol **muy simple** (solo 2 niveles). Lo hacemos a propósito: si un modelo tan básico funciona bien, es señal de que los 6 componentes contienen información valiosa. Agregamos `random_state=42` para que el árbol también sea reproducible.
- `fit` entrena; `predict` clasifica a los pacientes de prueba.
- `confusion_matrix` cuenta aciertos y errores; `ConfusionMatrixDisplay` la dibuja con los nombres de clase.

**Cómo leer la matriz de confusión**

| | Predicho maligno | Predicho benigno |
|---|---|---|
| **Real maligno** | Aciertos | **Falsos negativos** |
| **Real benigno** | Falsos positivos | Aciertos |

Diagonal principal = aciertos. Pregunta clave de ética y contexto: **¿cuál error es más grave en un diagnóstico de cáncer?** (El falso negativo: decirle "benigno" a una paciente con tumor maligno.)

### Visualizar el árbol

El notebook original usa `graphviz`, que requiere instalar además un programa del sistema operativo. Para evitar problemas en el laboratorio usamos `plot_tree`, que viene con scikit-learn:

```python
plt.figure(figsize=(10, 6))
plot_tree(
    tree,
    class_names=list(target_names),
    feature_names=[f"PC{i}" for i in range(1, 7)],
    filled=True,
)
plt.show()
```

**Qué observar:** los nodos del árbol preguntan por `PC1`, `PC2`, etc. Es decir, el árbol decide con los componentes, no con las 30 variables originales. Cada nodo muestra la condición, el número de muestras y la clase mayoritaria (el color indica la clase y la pureza).

> **Opcional para estudiantes avanzados:** si quieren usar `graphviz` como en el notebook: `uv add graphviz`, instalar el programa Graphviz del sistema, usar `export_graphviz` y generar una imagen con `graphviz.Source(dot).render("arbol", format="png")`.

---

## 9. Cierre: preguntas de comprensión

1. ¿Por qué decimos que PCA es una técnica **no supervisada**? ¿En qué momento del taller usamos la etiqueta?
2. ¿Qué pasaría con los resultados si olvidáramos estandarizar?
3. Si el umbral fuera 95 % en vez de 85 %, ¿cuántos componentes necesitaríamos? Pruébelo en la gráfica.
4. ¿Qué se gana y qué se pierde al pasar de 30 a 6 variables?
5. ¿Por qué elegimos un árbol de profundidad 2? ¿Qué demuestra que funcione bien?

### Reflexión crítica (para discutir)

En este flujo el PCA se ajustó con **todos** los datos *antes* de dividir en entrenamiento y prueba. Esto filtra una pequeña cantidad de información de la prueba hacia el entrenamiento (*data leakage*). La práctica rigurosa es dividir primero y ajustar el escalador y el PCA solo con los datos de entrenamiento. Es un buen tema para la siguiente sesión.

### Retos de extensión

- **Reto 1:** Cambiar `n_components` a 2 y graficar un diagrama de dispersión de `x_pca[:, 0]` vs `x_pca[:, 1]` coloreado por `y`. ¿Se separan visualmente las clases?
- **Reto 2:** Comparar la exactitud del árbol con 2, 6 y 30 variables (sin PCA).
- **Reto 3:** Reemplazar el árbol por `KMeans(n_clusters=2)` sobre `x_pca` y comparar los grupos con las etiquetas reales (enlaza con la siguiente clase de agrupamiento).

---

## 10. Glosario rápido

| Término | Significado sencillo |
|---|---|
| **Aprendizaje no supervisado** | Encontrar estructura en los datos sin usar etiquetas. |
| **Componente principal** | Nueva variable construida combinando las originales. |
| **Varianza explicada** | Qué tanta información captura un componente. |
| **Valor propio** | Cantidad de varianza de un componente. |
| **Vector propio** | Los pesos de cada variable original en un componente. |
| **Estandarizar** | Llevar cada variable a media 0 y desviación 1. |
| **Matriz de confusión** | Tabla de aciertos y errores de un clasificador. |

---

## Lista de verificación para el estudiante

- [ ] El proyecto se creó con `uv init` y las librerías con `uv add`.
- [ ] El archivo se ejecuta con `uv run taller_pca.py`.
- [ ] Obtuve la curva de varianza acumulada y vi que cruza el 85 % en 6 componentes.
- [ ] Imprimí los valores propios y reconozco que decrecen.
- [ ] Obtuve la matriz de confusión y puedo explicar qué es un falso negativo.
- [ ] Respondí las preguntas de cierre.
