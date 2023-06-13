import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations

from sklearn.isotonic import spearmanr
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import euclidean

# Identidad


def identidad(xa, xb, ea, eb):
    distancia_x = np.linalg.norm(xa - xb, axis=None)
    print("Distancia de muestras: ", distancia_x)
    if np.any(distancia_x == 0):
        distancia_e = np.linalg.norm(ea - eb, axis=None)
        print("Distancia de explicaciones: ", distancia_e)
        return np.all(distancia_e == 0)
    else:
        return True


# Separabilidad


def separabilidad(xa, xb, ea, eb):
    distancia_x = np.linalg.norm(xa - xb, axis=None)
    print("Distancia de muestras: ", distancia_x)
    if np.any(distancia_x != 0):
        distancia_e = np.linalg.norm(ea - eb, axis=None)
        print("Distancia de explicaciones: ", distancia_e)
        return np.all(distancia_e != 0)
    else:
        return True


# Estabilidad


def estabilidad(x1, muestras, e1, explicaciones):
    distancias_muestras = []
    distancias_explicaciones = []
    for muestra in muestras:
        distancia_muestra = euclidean(x1.flatten(), muestra.flatten())
        distancias_muestras.append(distancia_muestra)

    for explicacion in explicaciones:
        distancia_explicacion = euclidean(e1.flatten(), explicacion.flatten())
        distancias_explicaciones.append(distancia_explicacion)
    correlacion, _ = spearmanr(distancias_muestras, distancias_explicaciones)
    print("Correlación: ", correlacion)
    return correlacion > 0


# Selectividad


# Ordenamos los atributos de mayor a menor importancia en función de los coeficientes de la regresión lineal
# Para cada atributo, vamos estableciendo su valor a cero y calculamos la predicción
# Calculamos el error residual entre la predicción original y la predicción con los atributos a cero
def lime_selectividad(coef, model, x):
    y = model.predict(x.reshape(1, -1))
    class_index = np.argmax(y)  # Obtener el índice de la clase de mayor probabilidad
    coef_class = coef[class_index]  # Seleccionar la fila correspondiente a la clase
    importance = np.abs(coef_class)  # Calcular la importancia de los atributos
    sorted_indices = np.argsort(importance)[
        ::-1
    ]  # Ordenar los atributos de mayor a menor importancia

    x_perturbed = x.copy()
    list_perturbed = []
    for i in range(len(x)):  # CAMBIAR
        x_perturbed_i = x_perturbed.copy()
        x_perturbed_i[sorted_indices[i]] = 0
        x_perturbed_i = x_perturbed_i.reshape(1, -1)
        list_perturbed.append(x_perturbed_i)

    list_predictions = auxiliar_selectividad(list_perturbed, model)
    errores_residuales = []

    for i in range(len(list_perturbed)):
        prediction = list_predictions[i]
        residual = y - prediction
        errores_residuales.append(residual.tolist())

    return errores_residuales

    # Calculamos las predicciones de las muestras perturbadas


def auxiliar_selectividad(list_perturbed, model):
    list_predictions = []
    for n in range(len(list_perturbed)):
        p = model.predict(list_perturbed[n])
        list_predictions.append(p)

    return list_predictions


# Coherencia
# Aquellos atributos con menor importancia (se saca con selectividad) serán eliminados de X_original
# La señal resultante será X_sin_caracteristicas: data.drop(["noImportante1", "noImportante2"..], axis=1)


def coherence(X_original, y_original, X_sin_caracteristicas):
    # Entrenar el modelo con la señal original
    modelo = LinearRegression()
    modelo.fit(X_original, y_original)

    # Calcular el error de predicción con características importantes
    predicciones_originales = modelo.predict(X_original)
    error_prediccion_originales = np.mean((predicciones_originales - y_original) ** 2)

    # Calcular el error de predicción sin características no importantes
    predicciones_sin_caracteristicas = modelo.predict(X_sin_caracteristicas)
    error_prediccion_sin_caracteristicas = np.mean(
        (predicciones_sin_caracteristicas - y_original) ** 2
    )

    # Calcular la métrica de coherencia como la diferencia de errores
    coherencia = abs(error_prediccion_originales - error_prediccion_sin_caracteristicas)

    return coherencia, error_prediccion_originales, error_prediccion_sin_caracteristicas


# Completitud
# error_explicacion es error_prediccion_sin_caracteristicas de coherence
# error_prediccion es error_prediccion_originales de coherence


def completitud(error_explicacion, error_prediccion):
    return error_explicacion / error_prediccion


# Congruencia, desviación estándar de la coherencia


def congruencia(coherencia):
    return np.std(coherencia)
