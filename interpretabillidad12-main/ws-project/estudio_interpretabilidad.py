import os
import random

import lime
import numpy
import pandas
import sklearn
from sklearn import model_selection
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tensorflow import get_logger, keras
from tensorflow import random as tensorflow_random
import metrics

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
get_logger().setLevel("ERROR")
random.seed(2398572)
tensorflow_random.set_seed(394867)
numpy.random.seed(43958734)
numpy.set_printoptions(threshold=10)


def occupation_models():
    occupation = pandas.read_csv(
        "../data/Occupancy_Estimation.csv",
        header=None,
        names=[
            "Date",
            "Time",
            "S1_Temp",
            "S2_Temp",
            "S3_Temp",
            "S4_Temp",
            "S1_Light",
            "S2_Light",
            "S3_Light",
            "S4_Light",
            "S1_Sound",
            "S2_Sound",
            "S3_Sound",
            "S4_Sound",
            "S5_CO2",
            "S5_CO2_Slope",
            "S6_PIR",
            "S7_PIR",
            "Room_Occupancy_Count",
        ],
    )
    date_encoder = LabelEncoder()
    time_encoder = LabelEncoder()
    occupation["Date"] = date_encoder.fit_transform(occupation["Date"])
    occupation["Time"] = time_encoder.fit_transform(occupation["Time"])
    # Obtener los valores mínimos y máximos de cada atributo
    min_vals = numpy.min(occupation, axis=0)
    max_vals = numpy.max(occupation, axis=0)
    # Normalizar el dataset utilizando Min-Max Scaling
    occupation = (occupation - min_vals) / (max_vals - min_vals)

    atributos = occupation.loc[:, "Date":"S7_PIR"]
    atributos = atributos.to_numpy()
    objetivo = occupation["Room_Occupancy_Count"]
    objetivo = pandas.get_dummies(objetivo)
    objetivo = objetivo.to_numpy()

    (
        atributos_entrenamiento,
        atributos_prueba,
        objetivo_entrenamiento,
        objetivo_prueba,
    ) = model_selection.train_test_split(atributos, objetivo, test_size=256)

    occupation_rn_model = redneuronal(
        atributos_entrenamiento, objetivo_entrenamiento, 18, 4
    )
    occupation_rf_model = randomforest(
        atributos_entrenamiento, objetivo_entrenamiento, 500
    )

    min_vals = numpy.min(atributos_entrenamiento, axis=0)
    max_vals = numpy.max(atributos_entrenamiento, axis=0)

    return (
        occupation_rn_model,
        occupation_rf_model,
        atributos_prueba,
        min_vals,
        max_vals,
    )

def drybean_models():
    drybean = pandas.read_csv(
        "../data/Dry_Bean.csv",
        header=None,
        names=[
            "Area",
            "Perimeter",
            "MajorAxisLength",
            "MinorAxisLength",
            "AspectRation",
            "Eccentricity",
            "ConvexArea",
            "EquivDiameter",
            "Extent",
            "Solidity",
            "roundness",
            "Compactness",
            "ShapeFactor1",
            "ShapeFactor2",
            "ShapeFactor3",
            "ShapeFactor4",
            "Class",
        ],
    )
    atributos = drybean.loc[:, "Area":"ShapeFactor4"]
    atributos = atributos.to_numpy()
    objetivo = drybean["Class"]
    objetivo = pandas.get_dummies(objetivo)
    objetivo = objetivo.to_numpy()
    # Obtener los valores mínimos y máximos de cada atributo
    min_vals = numpy.min(atributos, axis=0)
    max_vals = numpy.max(atributos, axis=0)

    # Normalizar el dataset utilizando Min-Max Scaling
    atributos = (atributos - min_vals) / (max_vals - min_vals)
    (
        atributos_entrenamiento,
        atributos_prueba,
        objetivo_entrenamiento,
        objetivo_prueba,
    ) = model_selection.train_test_split(atributos, objetivo, test_size=256)

    drybean_rn_model = redneuronal(atributos_entrenamiento, objetivo_entrenamiento, 16, 7)
    drybean_rf_model = randomforest(atributos_entrenamiento, objetivo_entrenamiento, 500)

    return drybean_rn_model, drybean_rf_model, atributos_prueba, min_vals, max_vals

def redneuronal(atributos_entrenamiento, objetivo_entrenamiento, input_shape, output_shape):
    model = keras.Sequential()
    model.add(
        keras.Input(
            shape=input_shape,
        )
    )
    model.add(keras.layers.Dense(70, activation="relu"))
    model.add(keras.layers.Dense(output_shape, activation="softmax"))
    #model.summary()
    model.compile(
        optimizer="SGD", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        atributos_entrenamiento, objetivo_entrenamiento, batch_size=256, epochs=20
    )
    return model

def randomforest(atributos_entrenamiento, objetivo_entrenamiento, n_estimators):
    model = RandomForestClassifier(n_estimators)
    model.fit(atributos_entrenamiento, objetivo_entrenamiento)
    return model

# def estudio_interpretabilidad(
#      N, k, model1, model2, atributos_prueba, min_vals, max_vals
#  ):
#      for i in atributos_prueba.shape[0]:
#         explainer1 = lime.algoritm_lime(N, model1, atributos_prueba[i], k, min_vals, max_vals)
#         for j in atributos_prueba.shape[0]:
#             explainer2 = lime.algoritm_lime(N, model1, atributos_prueba[j], k, min_vals, max_vals)

#             metrics.identidad(atributos_prueba[i], atributos_prueba[j], explainer1[0], explainer2[0])

#         # explainer2 = lime.algoritm_lime(N, model2, atributos, k, min_vals, max_vals)

#          # METRICAS...


# def estudio_identidad(N, k, model1, atributos_prueba, min_vals, max_vals):
#     correct_count = 0
#     total_pairs = atributos_prueba.shape[0] * atributos_prueba.shape[0]

#     for i in range(atributos_prueba.shape[0]):
#         explainer1 = lime.algoritm_lime(N, model1, atributos_prueba[i], k, min_vals, max_vals)
#         for j in range(atributos_prueba.shape[0]):
#             explainer2 = lime.algoritm_lime(N, model1, atributos_prueba[j], k, min_vals, max_vals)

#             if metrics.identidad(atributos_prueba[i], atributos_prueba[j], explainer1[0], explainer2[0]):
#                 correct_count += 1

#     accuracy = (correct_count / total_pairs) * 100
#     return accuracy

from itertools import combinations

def estudio_identidad(N, k, model1, atributos_prueba, min_vals, max_vals):
    correct_count = 0
    total_pairs = len(list(combinations(range(atributos_prueba.shape[0]), 2)))

    for i, j in combinations(range(atributos_prueba.shape[0]), 2):
        dist = numpy.linalg.norm(atributos_prueba[i] - atributos_prueba[j])
        if dist == 0:
            explainer1 = lime.algoritm_lime(N, model1, atributos_prueba[i], k, min_vals, max_vals)
            explainer2 = lime.algoritm_lime(N, model1, atributos_prueba[j], k, min_vals, max_vals)

            distancia_e = numpy.linalg.norm(explainer1[0] - explainer2[0], axis=None)

            if distancia_e == 0:
                correct_count += 1

    accuracy = (correct_count / total_pairs) * 100
    return accuracy
