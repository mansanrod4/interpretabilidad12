import numpy
from scipy.spatial.distance import cdist

#Identidad

def identidad(xa, xb, ea, eb, metrica_distancia='euclidean'):
    distancia_x = cdist(xa, xb, metric=metrica_distancia)
    if numpy.any(distancia_x == 0):
        idx_zeros = numpy.where(distancia_x == 0)

        distancia_e = cdist(ea[idx_zeros], eb[idx_zeros], meric=metrica_distancia)
        return numpy.all(distancia_e == 0)
    else:
        return True
    
#Separabilidad

#Estabilidad

#Selectividad

#Coherencia

#Completitud

#Congruencia

