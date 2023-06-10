import numpy
from sklearn.linear_model import Ridge

def algoritm_lime(N, f, x, k, min_vals, max_vals):


    X_perturbed = []
    R = []
    W = []

    for i in range(N):
        selected_features = numpy.random.choice(range(len(x)), size=k, replace=False)
        x_perturbed = x.copy()
        x_perturbed[selected_features] = numpy.random.uniform(min_vals[selected_features], max_vals[selected_features])
        w = numpy.linalg.norm(x - x_perturbed)
        r = numpy.ones(len(x))
        r[selected_features] = 0
        X_perturbed.append(x_perturbed)
        R.append(r)
        W.append(1/w)
    
    Y_perturbed = f(numpy.array(X_perturbed))
    G = Ridge().fit(numpy.array(R), Y_perturbed, sample_weight=W)

    return G.coef_, G.intercept_

