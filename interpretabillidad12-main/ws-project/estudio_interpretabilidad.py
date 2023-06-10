# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:06:28 2023

@author: LENOVO
"""
import lime
import os
import numpy
import pandas
from  sklearn import model_selection
from tensorflow import get_logger
import random
from tensorflow import random as tensorflow_random
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
get_logger().setLevel('ERROR')
random.seed(2398572)
tensorflow_random.set_seed(394867)
numpy.random.seed(43958734)
numpy.set_printoptions(threshold=10)
ocupation = pandas.read_csv('../data/Occupancy_Estimation.csv', 
                            header=None, names=['Date','Time','S1_Temp','S2_Temp','S3_Temp',
                                                'S4_Temp','S1_Light','S2_Light','S3_Light','S4_Light','S1_Sound',
                                                'S2_Sound','S3_Sound','S4_Sound','S5_CO2','S5_CO2_Slope','S6_PIR',
                                                'S7_PIR','Room_Occupancy_Count'])

ocupation.head()
atributos = ocupation.loc[:, 'S1_Temp':'S7_PIR']
atributos = atributos.to_numpy()
print(atributos)
objetivo = ocupation['Room_Occupancy_Count']
objetivo = pandas.get_dummies(objetivo)
print(objetivo)
objetivo = objetivo.to_numpy()
print(objetivo)
(atributos_entrenamiento, atributos_prueba, objetivo_entrenamiento, objetivo_prueba) = model_selection.train_test_split(atributos, objetivo, test_size=.33)
red_occupancy = keras.Sequential()
red_occupancy.add(keras.Input(shape=16,))
red_occupancy.add(keras.layers.Dense(70,activation='relu'))
red_occupancy.add(keras.layers.Dense(4, activation='softmax'))
red_occupancy.summary()
red_occupancy.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
red_occupancy.fit(atributos_entrenamiento, objetivo_entrenamiento, batch_size=256, epochs=20)
red_occupancy.weights()
normalizador = keras.layers.Normalization()
normalizador.adapt(atributos_entrenamiento)
lime.algoritm_lime(4, red_occupancy.weights)

#Random Forest
model = RandomForestClassifier(n_estimators=500)
model.fit(atributos_entrenamiento, objetivo_entrenamiento)
def f(x):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(atributos_entrenamiento, objetivo_entrenamiento)
    return model.predict(x)
min_vals = numpy.min(atributos_entrenamiento, axis=0)
max_vals = numpy.max(atributos_entrenamiento, axis=0)
N = 100
k = 10
explainer = lime.algoritm_lime(N, f, atributos_prueba[0], k, min_vals, max_vals)