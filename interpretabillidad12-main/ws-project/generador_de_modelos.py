import os
import random
from itertools import combinations

import numpy as np
import pandas
import sklearn
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sb
from sklearn import model_selection
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow import get_logger, keras
from tensorflow import random as tensorflow_random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
get_logger().setLevel("ERROR")
random.seed(2398572)
tensorflow_random.set_seed(394867)
np.random.seed(43958734)
np.set_printoptions(threshold=10)


def exploracion_datos():
    dryBean = pandas.read_csv("../data/Dry_Bean.csv",
                            header=None, names=['A','P','L','l','K','Ec',
                                                'C','Ed','Ex','S','R','CO',
                                                'SF1','SF2','SF3','SF4','Class'])
    occupation = pandas.read_csv("../data/Occupancy_Estimation.csv",
                                  header=None, names=['Date','Time','S1_Temp','S2_Temp','S3_Temp',
                                  'S4_Temp','S1_Light','S2_Light','S3_Light','S4_Light','S1_Sound',
                                  'S2_Sound','S3_Sound','S4_Sound','S5_CO2','S5_CO2_Slope','S6_PIR',
                                  'S7_PIR','Room_Occupancy_Count'])
    
    # Tablas y gráficas de la memoria por orden de aparición
    
    # Primeros 5 elementos de cada conjunto
    print(dryBean.head(5))
    print(occupation.head(5))
    
    # Atributos de cada conjunto
    print(dryBean.info())
    print(occupation.info())

    # Estadísticas de algunos atributos numéricos
    print(occupation['S1_Temp'].describe())
    print(occupation['S1_Light'].describe())
    print(occupation['S1_Sound'].describe())
    print(occupation['S5_CO2'].describe())
    print(occupation['S7_PIR'].describe())

    # Matrices de correlación
    corrDryBean = dryBean.set_index('Class').corr()
    sm.graphics.plot_corr(corrDryBean, xnames=list(corrDryBean.columns))
    plt.show()
    corrOccupation = occupation.set_index('Room_Occupancy_Count').corr()
    sm.graphics.plot_corr(corrOccupation, xnames=list(corrOccupation.columns))
    plt.show()

    # Histogramas de algunas variables numéricas, se ejecutan de uno en uno
    fig,axes=plt.subplots(2,2)
    axes[0,0].hist(x="Ex",data=dryBean,edgecolor="black",linewidth=2)
    axes[0,0].set_title("Extent")
    axes[0,1].hist(x="S",data=dryBean,edgecolor="black",linewidth=2)
    axes[0,1].set_title("Solidity")
    axes[1,0].hist(x="R",data=dryBean,edgecolor="black",linewidth=2)
    axes[1,0].set_title("Roundness")
    axes[1,1].hist(x="CO",data=dryBean,edgecolor="black",linewidth=2)
    axes[1,1].set_title("Compactness")
    fig.set_size_inches(10,10)

    # fig,axes=plt.subplots(2,2)
    # axes[0,0].hist(x="S1_Temp",data=occupation,edgecolor="black",linewidth=2)
    # axes[0,0].set_title("S1_Temp")
    # axes[0,1].hist(x="S1_Light",data=occupation,edgecolor="black",linewidth=2)
    # axes[0,1].set_title("S1_Light")
    # axes[1,0].hist(x="S1_Sound",data=occupation,edgecolor="black",linewidth=2)
    # axes[1,0].set_title("S1_Sound")
    # axes[1,1].hist(x="S5_CO2_Slope",data=occupation,edgecolor="black",linewidth=2)
    # axes[1,1].set_title("S5_CO2_Slope")
    # fig.set_size_inches(10,10)

    # Histogramas atributo - clase objetivo, se ejecutan de uno en uno 
    # con los histogramas de variables numéricas comentados
    # sb.histplot(data=dryBean,x='K',hue="Class",kde=True)
    # sb.histplot(data=occupation,x='S4_Temp',hue="Room_Occupancy_Count",kde=True)


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
    # min_vals = np.min(occupation, axis=0)
    # max_vals = np.max(occupation, axis=0)
    # Normalizar el dataset utilizando Min-Max Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    occupation.iloc[:, 2:-1] = scaler.fit_transform(occupation.iloc[:, 2:-1])

#    occupation = (occupation - min_vals) / (max_vals - min_vals)

    # Guardar las 256 líneas antes de la división en entrenamiento y prueba
    muestras_prueba = occupation.sample(n=256, random_state=42)
    occupation = occupation.drop(muestras_prueba.index)
    clases_prueba = muestras_prueba["Room_Occupancy_Count"]

    #Muestras prueba sin fecha, tiempo y clase objetivo
    muestras_prueba = muestras_prueba.iloc[:, 2:-1]

    atributos = occupation.loc[:, "S1_Temp":"S7_PIR"]
    atributos = atributos.to_numpy()
    objetivo = occupation["Room_Occupancy_Count"]
    objetivo = pandas.get_dummies(objetivo)
    objetivo = objetivo.to_numpy()

    (
        atributos_entrenamiento,
        atributos_prueba,
        objetivo_entrenamiento,
        objetivo_prueba,
    ) = model_selection.train_test_split(atributos, objetivo, test_size=0.2, random_state=42)
    print("==============RED NEURONAL==================")
    occupation_rn_model = redneuronal(
        atributos_entrenamiento, objetivo_entrenamiento, 16, 4
    )
    print("==============RANDOM FOREST==================")
    occupation_rfr_model = randomforest(
        atributos_entrenamiento,
        objetivo_entrenamiento,
        atributos_prueba,
        objetivo_prueba,
    )

    min_vals = np.min(atributos_entrenamiento, axis=0)
    max_vals = np.max(atributos_entrenamiento, axis=0)
    return (
        occupation_rn_model,
        occupation_rfr_model,
        muestras_prueba,
        clases_prueba,
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

    # # Separar la columna de clase
    clase = drybean["Class"]
    drybean = drybean.drop("Class", axis=1)

    # Obtener los valores mínimos y máximos de cada atributo
    min_vals = np.min(drybean, axis=0)
    max_vals = np.max(drybean, axis=0)

    # # Normalizar el dataset utilizando Min-Max Scaling
    drybean = (drybean - min_vals) / (max_vals - min_vals)

    # Volver a unir los datos normalizados con la columna de clase
    drybean = pandas.concat([drybean, clase], axis=1)

    # MUESTRAS PARA PROBAR LA CALIDAD DE LAS EXPLICACIONES
    muestras_prueba = drybean.sample(n=256, random_state=42)
    drybean = drybean.drop(muestras_prueba.index)
    clases_prueba = muestras_prueba["Class"]
    muestras_prueba = muestras_prueba.iloc[:, :-1]

    atributos = drybean.loc[:, "Area":"ShapeFactor4"]
    atributos = atributos.to_numpy()
    objetivo = drybean["Class"]
    objetivo = pandas.get_dummies(objetivo)
    objetivo = objetivo.to_numpy()

    (
        atributos_entrenamiento,
        atributos_prueba,
        objetivo_entrenamiento,
        objetivo_prueba,
    ) = model_selection.train_test_split(atributos, objetivo, test_size=0.3, random_state=42)
    print("==============RED NEURONAL==================")
    drybean_rn_model = redneuronal(
        atributos_entrenamiento, objetivo_entrenamiento, 16, 7
    )
    print("==============RANDOM FOREST==================")
    drybean_rf_model = randomforest(
        atributos_entrenamiento, objetivo_entrenamiento, atributos_prueba, objetivo_prueba
    )
    min_vals = np.min(atributos_entrenamiento, axis=0)
    max_vals = np.max(atributos_entrenamiento, axis=0)
    return drybean_rn_model, drybean_rf_model, muestras_prueba, clases_prueba, min_vals, max_vals


def redneuronal(
    atributos_entrenamiento, objetivo_entrenamiento, input_shape, output_shape
):
    model = keras.Sequential()
    model.add(
        keras.Input(
            shape=input_shape,
        )
    )
    model.add(keras.layers.Dense(70, activation="relu"))
    model.add(keras.layers.Dense(output_shape, activation="softmax"))
    # model.summary()
    model.compile(
        optimizer="SGD", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        atributos_entrenamiento, objetivo_entrenamiento, batch_size=256, epochs=20
    )
    return model

def randomforest(
    atributos_entrenamiento, objetivo_entrenamiento, atributos_test, objetivo_test
):
    parameters_rf = {
        "n_estimators": [100, 150, 200, 250, 300],
        "max_depth": [None, 1, 2, 3, 4],
        #'criterion': ['mse', 'mae'],
    }
    model = RandomForestRegressor(random_state=0)
    grid_rf = GridSearchCV(estimator=model, param_grid=parameters_rf, cv=2)
    grid_rf.fit(atributos_entrenamiento, objetivo_entrenamiento)
    print("Mejores parámetros para Random Forest: ", grid_rf.best_params_)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=None)
    rf_model.fit(atributos_entrenamiento, objetivo_entrenamiento)
    y_pred_rf = rf_model.predict(atributos_test)
    print(
        "RMSE para Random Forest Regressor: ",
        np.sqrt(mean_squared_error(objetivo_test, y_pred_rf)),
    )
    print("R2 para Random Forest Regressor: ", r2_score(objetivo_test, y_pred_rf))
    return rf_model

