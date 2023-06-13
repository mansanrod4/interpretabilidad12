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


# Metodo para generar explicaciones de una serie de modelos a partir de un dataset
# Parametros:
#  - rn_model: modelo de red neuronal
#  - rf_model: modelo de random forest
#  - muestras_prueba: muestras de prueba
#  - min_vals: valores minimos de las muestras
#  - max_vals: valores maximos de las muestras
#  - N: numero de muestras perturbadas a generar
#  - k: numero de caracteristicas a perturbar por cada muestra
#  - indice_prueba: indice de la muestra de prueba
def generar_explicaciones(
    rn_model, rf_model, muestras_prueba, min_vals, max_vals, N, k, indice_prueba
):
    np.set_printoptions(threshold=np.inf)

    muestra_prueba = muestras_prueba.iloc[indice_prueba].values
    rn_explainer = lime.algoritm_lime(
        N, rn_model, muestra_prueba, k, min_vals, max_vals
    )
    rf_explainer = lime.algoritm_lime(
        N, rf_model, muestra_prueba, k, min_vals, max_vals
    )

    return rn_explainer, rf_explainer


# Metodo para generar metricas de interpretabilidad de una serie de modelos a partir de un dataset
# Parametros:
#   - Elección del datatset: 1 para occupation, 2 para drybean
#   - indice_prueba1: indice de la muestra de prueba 1
#   - indice_prueba2: indice de la muestra de prueba 2
def estudio_metricas(occupation1ORdrybean2, indice_prueba1, indice_prueba2):
    N = 100
    k = 10

    # Generamos los modelos y las muestras de prueba
    if occupation1ORdrybean2 == 1:
        print("Está generando modelos del dataset Occupation")
        (
            rn_model,
            rf_model,
            muestras_prueba,
            clases_prueba,
            min_vals,
            max_vals,
        ) = gm.occupation_models()
    else:
        print("Está generando modelos del dataset Dry Bean")
        (
            rn_model,
            rf_model,
            muestras_prueba,
            clases_prueba,
            min_vals,
            max_vals,
        ) = gm.drybean_models()

    # Generamos las explicaciones
    explicacion_rn1, explicacion_rf1 = generar_explicaciones(
        rn_model, rf_model, muestras_prueba, min_vals, max_vals, N, k, indice_prueba1
    )
    explicacion_rn2, explicacion_rf2 = generar_explicaciones(
        rn_model, rf_model, muestras_prueba, min_vals, max_vals, N, k, indice_prueba2
    )
    x1 = muestras_prueba.iloc[indice_prueba1].values
    y1 = clases_prueba.iloc[indice_prueba1]
    x2 = muestras_prueba.iloc[indice_prueba2].values
    y2 = clases_prueba.iloc[indice_prueba2]

    print("=================DATOS=================")

    print("Muestras")
    print("Muestra x1: ", x1, "Clase: ", y1)
    print("Muestra x2: ", x2, "Clase: ", y2)
    print("=================EXPLICACIONES=================")
    print("..............Explicaciones modelo RN..............")
    print("Explicacion para la muestra de prueba 1:")
    print("Matriz de coeficientes:")
    print(explicacion_rn1[0])
    print("Vector de interceptores:")
    print(explicacion_rn1[1])
    print("Explicacion para la muestra de prueba 2:")
    print("Matriz de coeficientes:")
    print(explicacion_rn2[0])
    print("Vector de interceptores:")
    print(explicacion_rn2[1])

    print(".............Explicaciones modelo RF..............")
    print("Explicacion para la muestra de prueba 1:")
    print("Matriz de coeficientes:")
    print(explicacion_rf1[0])
    print("Vector de interceptores:")
    print(explicacion_rf1[1])
    print("Explicacion para la muestra de prueba 2:")
    print("Matriz de coeficientes:")
    print(explicacion_rf2[0])
    print("Vector de interceptores:")
    print(explicacion_rf2[1])

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

    print("--------------Selectividad--------------")
    selectividad_rn = metrics.lime_selectividad(explicacion_rn1[0], rn_model, x1)
    selectividad_rf = metrics.lime_selectividad(explicacion_rf1[0], rf_model, x1)

    print("Selectividad_rn ERRORES RESIDUALES: ", selectividad_rn)
    print("Selectividad_rf ERRORES RESIDUALES: ", selectividad_rf)

    metricas_rn = {
        "identidad-1": identidad1_rn,
        "identidad-2": identidad2_rn,
        "separabilidad": separabilidad2_rn,
        "estabilidad": estabilidad1,
        "selectividad": selectividad_rn,
    }
    metricas_rf = {
        "identidad-1": identidad1_rf,
        "identidad-2": identidad2_rf,
        "separabilidad": separabilidad2_rf,
        "estabilidad": estabilidad2,
        "selectividad": selectividad_rf,
    }
    return metricas_rn, metricas_rf
