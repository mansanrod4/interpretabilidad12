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


def selectivity(importances):
    num_classes = len(importances)
    num_features = len(importances[0])

    selectivities = []

    for feature in range(num_features):
        selectivity = np.mean([importances[c][feature] for c in range(num_classes)])
        selectivities.append(selectivity)

    average_selectivity = np.mean(selectivities)

    return average_selectivity


# Coherencia

# Completitud

# Congruencia
