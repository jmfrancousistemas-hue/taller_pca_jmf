import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import graphviz
from sklearn.tree import export_graphviz

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, adjusted_rand_score

from sklearn.cluster import KMeans

from ucimlrepo import fetch_ucirepo

# --- Cargar dataset desde UCI Repo ---
cancer = fetch_ucirepo(id=17)

# Extraer matrices
X_df = cancer.data.features
y_df = cancer.data.targets

# Convertir arreglos númericos para scikit-learn
X = X_df.values
feature_names = np.array(X_df.columns)

# M: Maligno=1, B: Benigno=0
target_names = np.array(["Benign","Malignant"])
y = (y_df["Diagnosis"]=="M").astype(int).values

print("Forma de X:", X_df.shape)
print("Clases:", target_names)
print("Primera 5 variables:", feature_names[:5])


#Estandarizar los datos
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)
print("\nMedias después de escalar:", x_scaled.mean(axis=0).round(2)[:5])
print("Desviación estándar después de escalar:", x_scaled.std(axis=0).round(2)[:5])

# Varianza
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


# Aplicar PCA con 6 componentes
pca = PCA(n_components=6)
x_pca = pca.fit_transform(x_scaled)

print("\nForma original:", x_scaled.shape)
print("Forma reducida con 6 componentes:", x_pca.shape)

# Valores y vectores propios
valores_propios = pca.explained_variance_
vectores_propios = pca.components_
print("\nValores Propios:",valores_propios)
print("\nVectores Propios:",vectores_propios)


# 3 Variales con mayor peso absoluto en el primer componente
pesos_pc1 = pd.Series(vectores_propios[0], index=feature_names)
print(pesos_pc1.abs().sort_values(ascending=False).head(3))


# Validación
x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, test_size=0.3, random_state=42
)

tree = DecisionTreeClassifier(max_depth=2, random_state=42)
tree.fit(x_train, y_train)

pred_tree = tree.predict(x_test)
print("\nExactitud:", accuracy_score(y_test, pred_tree))

#Matriz de confusión
cm = confusion_matrix(y_test, pred_tree)
disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
disp.plot()
plt.show()


# Visualizar el árbol
plt.figure(figsize=(10, 6))
plot_tree(
    tree,
    class_names=list(target_names),
    feature_names=[f"PC{i}" for i in range(1, 7)],
    filled=True,
)
plt.title("Árbol de Decisión sobre Componentes Principales")
plt.show()

#Exportar la estructura del árbol a formato DOT
#Indicar a Python dónde está el ejecutable 'dot.exe' de Graphviz
# Primero debe instalar el Graphviz al Windows "https://graphviz.org/download/")
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
dot_data = export_graphviz(
    tree,
    out_file=None,
    feature_names=[f"PC{i}" for i in range(1, 7)],
    class_names=list(target_names),
    filled=True,
    rounded=True,
    special_characters=True
)

#Crear el objeto gráfico
graph = graphviz.Source(dot_data)

#Guardar y abrir la imagen en formato PNG
graph.render("arbol_decision_pca", format="png", cleanup=True, view=True)
print("\nÁrbol renderizado y guardado como 'arbol_decision_pca.png'.")


# Retos de extensión

#Reto 1: Diagrama de dispersión (PC1 VS PC2)
plt.figure(figsize=(9, 6))

# Graficar los puntos: PC1 en X, PC2 en Y
scatter = plt.scatter(
    x_pca[:, 0], 
    x_pca[:, 1], 
    c=y, 
    cmap='coolwarm', 
    alpha=0.7, 
    edgecolors='k'
)

plt.xlabel('Primer Componente Principal (PC1)')
plt.ylabel('Segundo Componente Principal (PC2)')
plt.title('Proyección 2D del Dataset de Cáncer de Mama (PCA)')
plt.grid(True, linestyle='--', alpha=0.5)
# Agregar leyenda con los nombres de las clases
handles, _ = scatter.legend_elements()
plt.legend(handles, target_names, title="Diagnóstico")
plt.show()

# Reto 2: Comparación de exactitud del árbol con 2, 6 y 30 variables
#Modelo con 2 Componentes Principales
x_pca_2 = x_pca[:, :2]
x_tr2, x_te2, y_tr2, y_te2 = train_test_split(x_pca_2, y, test_size=0.3, random_state=42)
tree_2 = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_2.fit(x_tr2, y_tr2)
acc_2 = accuracy_score(y_te2, tree_2.predict(x_te2))

#Modelo con 6 Componentes Principales (ya entrenado)
acc_6 = accuracy_score(y_test, pred_tree)

#Modelo con las 30 Variables Originales (sin PCA)
x_tr30, x_te30, y_tr30, y_te30 = train_test_split(x_scaled, y, test_size=0.3, random_state=42)
tree_30 = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_30.fit(x_tr30, y_tr30)
acc_30 = accuracy_score(y_te30, tree_30.predict(x_te30))

#Imprimir resultados
print("\n" + "="*50)
print(" COMPARACIÓN DE EXACTITUD (Árbol de profundidad = 2)")
print("="*50)
print(f"• Con 2 Componentes Principales : {acc_2:.2%}")
print(f"• Con 6 Componentes Principales : {acc_6:.2%}")
print(f"• Con 30 Variables Originales  : {acc_30:.2%}")
print("="*50)

# Reto 3: Agrupamiento con KMeans
#Aplicar KMeans con 2 clústeres sobre los 6 componentes principales
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(x_pca)

#Ajustar etiquetas (KMeans asigna 0 y 1 arbitrariamente a los clústeres)
#Verificamos si las etiquetas de KMeans están invertidas respecto a 'y'
if accuracy_score(y, clusters) < accuracy_score(y, 1 - clusters):
    clusters = 1 - clusters

#Métricas de evaluación
acc_kmeans = accuracy_score(y, clusters)
ari_kmeans = adjusted_rand_score(y, clusters)

print("\n" + "="*50)
print(" RESULTADOS DE K-MEANS SOBRE PCA (6 COMPONENTES)")
print("="*50)
print(f"• Exactitud de coincidencia con etiquetas reales : {acc_kmeans:.2%}")
print(f"• Índice Rand Ajustado (ARI)                    : {ari_kmeans:.4f}")
print("="*50)

#Matriz de confusión para comparar clústeres vs clases reales
cm_km = confusion_matrix(y, clusters)
disp_km = ConfusionMatrixDisplay(cm_km, display_labels=target_names)
disp_km.plot(cmap="Oranges")
plt.title("Coincidencia de Clústeres KMeans con Diagnóstico Real")
plt.show()

#EXPERIMENTO: PCA con datos sin escalar vs con escalar
#PCA con datos SIN escalar
pca_unscaled = PCA()
pca_unscaled.fit(X)  # X original, sin pasar por StandardScaler

var_unscaled = pca_unscaled.explained_variance_ratio_
val_unscaled = pca_unscaled.explained_variance_

#PCA con datos ESCALADOS StandardScaler
pca_scaled = PCA()
pca_scaled.fit(x_scaled)  # x escalado

var_scaled = pca_scaled.explained_variance_ratio_

print("\n" + "="*60)
print(" COMPARACIÓN: PCA SIN ESCALAR VS. CON ESCALAR")
print("="*60)
print(f"• PC1 SIN escalar explica : {var_unscaled[0]:.2%} de la varianza total")
print(f"• PC1 CON escalar explica : {var_scaled[0]:.2%} de la varianza total")

#Identificar qué variable domina el PC1 sin escalar
pesos_pc1_unscaled = pd.Series(pca_unscaled.components_[0], index=feature_names)
var_dominante = pesos_pc1_unscaled.abs().idxmax()
peso_dominante = pesos_pc1_unscaled.abs().max()

print(f"\n• Variable dominante en PC1 (sin escalar): '{var_dominante}' (peso absoluto: {peso_dominante:.4f})")
print("="*60)

#Graficar la comparación
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#Gráfica SIN escalar
axes[0].bar(range(1, 11), var_unscaled[:10], color='crimson', alpha=0.8)
axes[0].set_title('SIN Escalar: PC1 "se come" casi toda la varianza', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Componente Principal')
axes[0].set_ylabel('Proporción de Varianza Explicada')
axes[0].set_ylim(0, 1.05)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].annotate(f'{var_unscaled[0]:.1%}', xy=(1, var_unscaled[0]), xytext=(1.5, var_unscaled[0] - 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
#Gráfica CON escalar
axes[1].bar(range(1, 11), var_scaled[:10], color='navy', alpha=0.8)
axes[1].set_title('CON Escalar (StandardScaler): Distribución real', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Componente Principal')
axes[1].set_ylabel('Proporción de Varianza Explicada')
axes[1].set_ylim(0, 1.05)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()