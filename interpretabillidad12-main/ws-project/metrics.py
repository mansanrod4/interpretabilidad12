import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations

from sklearn.isotonic import spearmanr
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


from sklearn.metrics import roc_auc_score

def lime_selectividad(coef, intercept, model, x):
    auc_scores = []
    y = model.predict(x.reshape(1, -1))
    class_index = np.argmax(y)  # Obtener el índice de la clase de mayor probabilidad
    coef_class = coef[class_index]  # Seleccionar la fila correspondiente a la clase
    importance = np.abs(coef_class) # Calcular la importancia de los atributos
    sorted_indices = np.argsort(importance)[::-1] # Ordenar los atributos de mayor a menor importancia

    x_perturbed = x.copy()
    list_perturbed = []
    for i in range(len(x)):  #CAMBIAR
        # Establecer el atributo i en cero
        x_perturbed_i = x_perturbed
        x_perturbed_i[sorted_indices[i]] = 0
        print("ponemos a 0 =", sorted_indices[i])
        print("x_perturbed =", x_perturbed_i)
        list_perturbed.append(x_perturbed_i.reshape(1, -1))

    list_predictions = auxiliar_selectividad(list_perturbed, model)

    for i in range(len(list_perturbed)):
        prediction = list_predictions[i]
        # Calcular el error residual
        residual = y - prediction
        print("Residual with attribute", sorted_indices[i], "set to 0:", residual)

        # Calcular el AUC utilizando el error residual
        auc = roc_auc_score(y, prediction)
        auc_scores.append(auc)
        print("AUC with attribute", sorted_indices[i], "set to 0:", auc)

    print("auc_scores =", auc_scores)
    return auc_scores

def auxiliar_selectividad(list_perturbed, model):
    list_predictions = []
    for n in range(len(list_perturbed)):
        p = model.predict(list_perturbed[n])
        print("p =", p)
        list_predictions.append(p)
    return list_predictions


# Coherencia

# Completitud

# Congruencia
