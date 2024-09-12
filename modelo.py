#Autor: Gustavo Alejandro Gutiérrez Valdes - A01747869

#Se importan los recursos necesarios del framework de sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler #type: ignore
from sklearn.model_selection import StratifiedKFold, train_test_split #type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #type: ignore
from sklearn.metrics import mean_squared_error #type: ignore
from sklearn.ensemble import RandomForestClassifier #type: ignore
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
import seaborn as sns

#Se lee el dataset de estudiantes que compondrá la etapa de training y validation
df = pd.read_csv('dataset_students.csv')

#Son las columnas que se utilizarán para entrenar el modelo (quitando la columna objetivo)
features = df.drop(columns='Admision')

#Se guarda la columna objetivo para hacer comparaciones posteriormente
objective = df['Admision']

# Dividir en entrenamiento y validación (70% train, 30% validation)
X_train, X_val, y_train, y_val = train_test_split(features, objective, test_size=0.3, random_state=35)

# Se realiza la escalación de los datos de entrenamiento y validación utilizando MinMaxScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_valid_scaled = scaler.transform(X_val)

# Se definen los diferentes conjuntos de hiperparámetros que se probarán entre si para encontrar la mejor combinación en el 
# siguiente paso. Se eligieron la cantidad de arboles a generar, la profundiad máxima de cada uno y la cantidad  de muestras 
# con la que los nodos se dividirán, ya que estos hiperparámetros controlan directamente la capacidad predictiva al 
# reducir la varianza y hacer las predicciones mas robustas. De igual forma, se mantiene dominio acerca del subajuste o 
# sobreajuste del modelo y finalmente se mitiga el impacto del ruido, dandole al modelo una mayor capacidad de generalización.
param_grid = {
    'n_estimators': [150,250,300,500],
    'max_depth': [None, 2,4,6],
    'min_samples_split': [2, 5, 10]
}

# Crear un objeto GridSearchCV. Este objeto se encargará de probar todas las combinaciones posibles de hiperparámetros y elegir 
# la que mejor funcione con el modelo elegido, se probará con validación cruzada de 5 folds y se utilizarán todos los núcleos 
# disponibles para el trabajo
grid_search = GridSearchCV(estimator=RandomForestClassifier(), 
                          param_grid=param_grid, 
                          cv= 5, 
                          n_jobs=-1)

#Se aplican el objeto de GridSearchCV a los datos de entrenamiento
grid_search.fit(x_train_scaled, y_train)

# Se obtiene el mejor modelo dentro de las combinaciones disponibles
best_model = grid_search.best_estimator_

y_train_pred = best_model.predict(x_train_scaled)

accuracy_train = accuracy_score(y_train, y_train_pred)

# Evaluar el modelo en el conjunto de validación
y_val_pred = best_model.predict(x_valid_scaled)

#Se obtiene el error cuadrático medio y la raíz del error cuadrático medio para entender su desempeño con el conjunto de validación
mse_train = mean_squared_error(y_train,y_train_pred)
mse_val = mean_squared_error(y_val, y_val_pred)

print("*"*50)
print("MSE del modelo entrenado:", mse_val)
rmse = np.sqrt(mse_val)
print("\nRaíz del MSE:", rmse)

#Se obtiene el valor de la precisión del modelo en la etapa de validación
accuracy_val = accuracy_score(y_val, y_val_pred)
print(f"\nPrecisión en validación: {accuracy_val:.2f}")

# Se obtiene el porcentaje de error en la etapa de validación
porcentaje_error = (1-accuracy_val) * 100

#Se muestra el porcentaje de error en la etapa de validación
print(f"\nPorcentaje de error para set de validación: {porcentaje_error:.2f}%")
print("*"*50)

#Se obtiene la matriz de confusión del modelo
print("\nMatriz de confusión:")
print(confusion_matrix(y_val, y_val_pred))
print("*"*50)

#Se obtiene el reporte de clasificación del modelo (Métricas como F1Score, Precision, Recall)
print("\nReporte de clasificación:")
print(classification_report(y_val, y_val_pred))
print("*"*50)

#Se lee el dataset de la etapa de test
dataset_test = pd.read_csv('test.csv')

#Se convierte en array para poder hacer un despligue más amigable de las predicciones
entradas = np.array(dataset_test)

#Se escalan los datos de testing para poder hacer predicciones
x_test_scaled = scaler.transform(dataset_test)

# Se hacen las predicciones correspondientes con el modelo entrenado
predicciones = best_model.predict(x_test_scaled)

#Se recorren tanto las entradas como las predicciones para mostrarlas al usuario en la consola
for x,y in zip (entradas,predicciones):
    estado = "Admitido" if y == 1 else "No admitido"
    print(f"Entrada (PuntajeExamen,PromedioAcumulado): {x} -> Predicción: {estado}")
print("*"*50)

cantidad_respuestas = df['Admision'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Distribución de la variable objetivo')
plt.xlabel('Admitido (1) / No admitido (0)')
plt.ylabel('Cantidad de respuestas')
plt.show()


# Crear lista con las cantidades de registros
registros = [len(X_train), len(X_val)]

# Etiquetas para los conjuntos
etiquetas = ['Entrenamiento', 'Validación']

# Crear gráfico de barras
plt.figure(figsize=(6, 4))
plt.bar(etiquetas, registros, color=['blue', 'green'])

# Añadir título y etiquetas
plt.title('Cantidad de Registros en Conjuntos de Entrenamiento y Validación')
plt.ylabel('Número de Registros')
plt.show()


# Obtener la curva de aprendizaje
# Curva de aprendizaje: Muestra cómo varía la precisión del modelo cuando se incrementa el tamaño del conjunto de entrenamiento. Esto es útil para diagnosticar problemas de sobreajuste o subajuste.
# Si la precisión de validación mejora significativamente a medida que aumentas los datos de entrenamiento, el modelo podría estar mejorando con más datos y podrías considerar obtener más ejemplos.
# Si las curvas se estabilizan rápidamente, podría significar que ya tienes suficientes datos y el modelo ha alcanzado su máximo rendimiento.
train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_model,
    X=x_train_scaled,
    y=y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

# Promediar las puntuaciones de precisión
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Graficar la curva de aprendizaje
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Precisión en Entrenamiento', color='blue', marker='o')
plt.plot(train_sizes, val_mean, label='Precisión en Validación', color='green', marker='o')
plt.title('Curva de Aprendizaje')
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('Precisión')
plt.legend(loc='best')
plt.grid(True)
plt.show()

labels = ['Entrenamiento', 'Validación']
mse_values = [mse_train, mse_val]

plt.figure(figsize=(8, 6))
plt.bar(labels, mse_values, color=['blue', 'orange'])
plt.ylabel('Error Cuadrático Medio (MSE)')
plt.title('Bias del Modelo en Entrenamiento y Validación')
plt.show()

# Calcular la media y desviación estándar de las puntuaciones de precisión
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_std = np.std(val_scores, axis=1)



# Crear dataframe para seaborn
data = {
    'Etapa': ['Entrenamiento'] * len(train_scores.flatten()) + ['Validación'] * len(val_scores.flatten()),
    'Precisión': np.concatenate((train_scores.flatten(), val_scores.flatten()))
}

df = pd.DataFrame(data)

# Graficar boxplot para la precisión en entrenamiento y validación
plt.figure(figsize=(8, 6))
sns.boxplot(x='Etapa', y='Precisión', data=df, palette=['blue', 'green'])
plt.title('Distribución de la Precisión (Varianza) en Entrenamiento y Validación')
plt.ylabel('Precisión')
plt.grid(True)
plt.show()


param_range = np.arange(1, 100)  # Valores del hiperparámetro a evaluar
train_scores, val_scores = validation_curve(
    RandomForestClassifier(), x_train_scaled, y_train, 
    param_name="n_estimators", 
    param_range=param_range, 
    cv=5, scoring="accuracy"
)

# Promedio y desviación estándar
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Gráfico de curva de validación
plt.figure(figsize=(8, 6))
plt.plot(param_range, train_mean, label='Entrenamiento', color='blue', marker='o')
plt.plot(param_range, val_mean, label='Validación', color='green', marker='o')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, color='green', alpha=0.2)
plt.title('Curva de Validación para n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('Precisión')
plt.legend(loc='best')
plt.grid(True)
plt.show()

#Grafica de importancia de hiperparámetros
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Importancia de las características")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.tight_layout()
plt.show()





