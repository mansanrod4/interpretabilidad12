import os
import random
from itertools import combinations

import numpy as np
import pandas
import sklearn
from sklearn import model_selection
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.model_selection import GridSearchCV
from tensorflow import get_logger, keras
from tensorflow import random as tensorflow_random

import lime
import generador_de_modelos as gm
import metrics


def estudio_identidad(N, k, model1, atributos_prueba, min_vals, max_vals):
    correct_count = 0
    total_pairs = len(list(combinations(range(atributos_prueba.shape[0]), 2)))

    for i, j in combinations(range(atributos_prueba.shape[0]), 2):
        dist = np.linalg.norm(atributos_prueba[i] - atributos_prueba[j])
        if dist == 0:
            explainer1 = lime.algoritm_lime(
                N, model1, atributos_prueba[i], k, min_vals, max_vals
            )
            explainer2 = lime.algoritm_lime(
                N, model1, atributos_prueba[j], k, min_vals, max_vals
            )

            if np.allclose(explainer1[0], explainer2[0]):
                correct_count += 1

    accuracy = (correct_count / total_pairs) * 100
    return accuracy

# def estudio_interpretabilidad_occupation():
#     np.set_printoptions(threshold=np.inf)

#     rn, rf, ap1, min_vals, max_vals = generador_de_modelos.occupation_models()
#     muestra_prueba1 = ap1.iloc[0].values
#     N = 100
#     k = 10
#     occupation_rn_explainer = lime.algoritm_lime(N, rn, muestra_prueba1, k, min_vals, max_vals)
#     print(occupation_rn_explainer)
#     occupation_rf_explainer = lime.algoritm_lime(N, rf, muestra_prueba1, k, min_vals, max_vals)
#     print(occupation_rf_explainer)

#     return occupation_rn_explainer, occupation_rf_explainer

def generar_explicaciones(rn_model, rf_model, muestras_prueba, min_vals, max_vals, N, k, indice_prueba):
    np.set_printoptions(threshold=np.inf)

    muestra_prueba = muestras_prueba.iloc[indice_prueba].values
    N = 100
    k = 10
    rn_explainer = lime.algoritm_lime(N, rn_model, muestra_prueba, k, min_vals, max_vals)
    print(rn_explainer)
    rf_explainer = lime.algoritm_lime(N, rf_model, muestra_prueba, k, min_vals, max_vals)
    print(rf_explainer)

    return rn_explainer, rf_explainer

def estudio_metricas():
    N = 100
    k = 10
    indice_prueba1 = 3
    indice_prueba2 = 4

    #Generamos los modelos y las muestras de prueba
    rn_model, rf_model, muestras_prueba, min_vals, max_vals = gm.occupation_models()


    #Generamos las explicaciones
    explicacion_rn1, explicacion_rf1 = generar_explicaciones(rn_model, rf_model, muestras_prueba, min_vals, max_vals, N, k, indice_prueba1)
    explicacion_rn2, explicacion_rf2 = generar_explicaciones(rn_model, rf_model, muestras_prueba, min_vals, max_vals, N, k, indice_prueba2)
    x1 = muestras_prueba.iloc[indice_prueba1].values
    x2 = muestras_prueba.iloc[indice_prueba2].values



    #Calculamos las metricas
    identidad = metrics.identidad(x1, x1, explicacion_rn1[0].reshape(1, -1), explicacion_rn1[0].reshape(1, -1))
    print("Identidad: ", identidad)



    correlacion1 = metrics.stability(x1.reshape(1, -1), x2.reshape(1, -1), explicacion_rn1[0].reshape(1, -1), explicacion_rn2[0].reshape(1, -1))
    correlacion2 = metrics.stability(x1, x2, explicacion_rf1[0], explicacion_rf2[0])

    print("correlacion rn: ",correlacion1)
    print("correlacion rf: ",correlacion2)