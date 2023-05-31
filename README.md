## interpretabilidad12
#TODO List
  1. Implementación de un método XAI -> LIME
  2. Implementación de las métricas(usando numpy y scipy):
      - identidad: objetos idénticos, explicaciones idénticas
      - separabilidad: objetos no idénticos, explicaciones no idénticas
      - estabilidad: objetos similares, explicaciones similares
      - selectividad: la eliminación de las variables relevantes debe afectar negativamente a la predicción
      - coherencia: error de predicción - error de explicación
      - completitud: error de explicación / error de predicción
      - congruencia: desviación estándar de la coherencia
  3. Entrenamiento de cuatro modelos de caja negra: random forest y xgboost para el Dry Bean Dataset, y redes neuronales y svn para       Occupancy Estimation, por ejemplo
  4. Medición de las métricas sobre los 4 modelos entrenados para el método LIME implementado: para las explicaciones se deben seleccionar al menos 256 muestras de los datasets que no hayan sido usadas para entrenar los modelos
