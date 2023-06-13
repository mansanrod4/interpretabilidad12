import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations

from sklearn.isotonic import spearmanr
from scipy.spatial.distance import euclidean

#Identidad

def identidad(xa, xb, ea, eb):
    distancia_x = np.linalg.norm(xa - xb, axis=None)
    if np.any(distancia_x == 0):
        distancia_e = np.linalg.norm(ea - eb, axis=None)
        return np.all(distancia_e == 0)
    else:
        return True
    
#Separabilidad

def separability(importances, true_labels):
    classes = np.unique(true_labels)
    num_classes = len(classes)

    class_importances = [np.mean([importance for importance, label in zip(importances, true_labels) if label == c]) for c in classes]
    class_stds = [np.std([importance for importance, label in zip(importances, true_labels) if label == c]) for c in classes]

    combinations_list = list(combinations(range(num_classes), 2))
    separabilities = []
    
    for class1, class2 in combinations_list:
        separability = (class_importances[class1] - class_importances[class2]) / (class_stds[class1] + class_stds[class2])
        separabilities.append(separability)

    average_separability = np.mean(separabilities)

    return average_separability

#Estabilidad

def stability(x1, x2, e1, e2):
    distancia_muestras = euclidean(x1.flatten(), x2.flatten())
    distancia_explicaciones = euclidean(e1.flatten(), e2.flatten())

    correlacion, _ = spearmanr(distancia_muestras, distancia_explicaciones)
    return correlacion

#Selectividad

def selectivity(importances):
    num_classes = len(importances)
    num_features = len(importances[0])

    selectivities = []

    for feature in range(num_features):
        selectivity = np.mean([importances[c][feature] for c in range(num_classes)])
        selectivities.append(selectivity)

    average_selectivity = np.mean(selectivities)

    return average_selectivity

#Coherencia

#Completitud

#Congruencia

