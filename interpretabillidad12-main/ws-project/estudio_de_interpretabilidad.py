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


def generar_explicaciones(
    rn_model, rf_model, muestras_prueba, min_vals, max_vals, N, k, indice_prueba
):
    np.set_printoptions(threshold=np.inf)

    muestra_prueba = muestras_prueba.iloc[indice_prueba].values
    N = 100
    k = 10
    rn_explainer = lime.algoritm_lime(
        N, rn_model, muestra_prueba, k, min_vals, max_vals
    )
    rf_explainer = lime.algoritm_lime(
        N, rf_model, muestra_prueba, k, min_vals, max_vals
    )

    return rn_explainer, rf_explainer


def estudio_metricas(occupation1ORdrybean2, indice_prueba1, indice_prueba2):
    N = 100
    k = 10

    # Generamos los modelos y las muestras de prueba
    if occupation1ORdrybean2 == 1:
        print("Está generando modelos del dataset Occupation")
        rn_model, rf_model, muestras_prueba, clases_prueba, min_vals, max_vals = gm.occupation_models()
    else:
        print("Está generando modelos del dataset Dry Bean")
        rn_model, rf_model, muestras_prueba, clases_prueba, min_vals, max_vals = gm.drybean_models()

    # Generamos las explicaciones
    explicacion_rn1, explicacion_rf1= generar_explicaciones(
        rn_model, rf_model, muestras_prueba, min_vals, max_vals, N, k, indice_prueba1
    )
    explicacion_rn2, explicacion_rf2 = generar_explicaciones(
        rn_model, rf_model, muestras_prueba, min_vals, max_vals, N, k, indice_prueba2
    )
    print(clases_prueba)
    x1 = muestras_prueba.iloc[indice_prueba1].values
    y1 = clases_prueba.iloc[indice_prueba1]
    x2 = muestras_prueba.iloc[indice_prueba2].values
    y2 = clases_prueba.iloc[indice_prueba2]

    print("=================DATOS=================")

    print("Muestras")
    print(x1)
    print(x2)
    print("Clases")
    print(y1)
    print(y2)
    # Calculamos las métricas
    print("=================METRICAS=================")
    print("----------------Identidad----------------")
    print("..............Identidad modelo RN..............")
    identidad1_rn = metrics.identidad(
        x1, x1, explicacion_rn1[0].reshape(1, -1), explicacion_rn1[0].reshape(1, -1)
    )
    print("Identidad para muestras idénticas (x1, x1):", identidad1_rn)
    identidad2_rn = metrics.identidad(
        x1, x2, explicacion_rn1[0].reshape(1, -1), explicacion_rn2[0].reshape(1, -1)
    )
    print("Identidad para muestras diferentes (x1, x2):", identidad2_rn)
    print(".............Identidad modelo RF..............")
    identidad1_rf = metrics.identidad(
        x1, x1, explicacion_rf1[0].reshape(1, -1), explicacion_rf1[0].reshape(1, -1)
    )
    print("Identidad para muestras idénticas (x1, x1):", identidad1_rf)
    identidad2_rf = metrics.identidad(
        x1, x2, explicacion_rf1[0].reshape(1, -1), explicacion_rf2[0].reshape(1, -1)
    )
    print("Identidad para muestras diferentes (x1, x2):", identidad2_rf)

    print("--------------Separabilidad--------------")
    print("..............Separabilidad modelo RN..............")
    separabilidad1_rn = metrics.separabilidad(
        x1, x1, explicacion_rn1[0].reshape(1, -1), explicacion_rn1[0].reshape(1, -1)
    )
    print("Separabilidad para muestras idénticas (x1, x1):", separabilidad1_rn)
    separabilidad2_rn = metrics.separabilidad(
        x1, x2, explicacion_rn1[0].reshape(1, -1), explicacion_rn2[0].reshape(1, -1)
    )
    print("Separabilidad para muestras diferentes (x1, x2):", separabilidad2_rn)
    print(".............Separabilidad modelo RF..............")
    separabilidad1_rf = metrics.separabilidad(
        x1, x1, explicacion_rf1[0].reshape(1, -1), explicacion_rf1[0].reshape(1, -1)
    )
    print("Separabilidad para muestras idénticas (x1, x1):", separabilidad1_rf)
    separabilidad2_rf = metrics.separabilidad(
        x1, x2, explicacion_rf1[0].reshape(1, -1), explicacion_rf2[0].reshape(1, -1)
    )
    print("Separabilidad para muestras diferentes (x1, x2):", separabilidad2_rf)

    print("--------------Estabilidad--------------")
    muestras = []
    explicaciones_rn = []
    explicaciones_rf = []
    print("Generando muestras y explicaciones...")
    for i in range(256):
        explicacion_rn_i, explicacion_rf_i = generar_explicaciones(
            rn_model, rf_model, muestras_prueba, min_vals, max_vals, N, k, i
        )
        xi = muestras_prueba.iloc[i].values
        muestras.append(xi.reshape(1, -1))
        explicaciones_rn.append(explicacion_rn_i[0].reshape(1, -1))
        explicaciones_rf.append(explicacion_rf_i[0].reshape(1, -1))
    estabilidad1 = metrics.estabilidad(
        x1.reshape(1, -1), muestras, explicacion_rn1[0].reshape(1, -1), explicaciones_rn
    )
    print("Estabilidad rn: ", estabilidad1)

    estabilidad2 = metrics.estabilidad(
        x1.reshape(1, -1), muestras, explicacion_rf1[0].reshape(1, -1), explicaciones_rf
    )
    print("Estabilidad rf: ", estabilidad2)

    # print("--------------Selectividad--------------")
    # a = metrics.lime_selectividad(explicacion_rn1[0], explicacion_rn1[1], rn_model, x1)
    # print("Selectividad rn: ", a)
