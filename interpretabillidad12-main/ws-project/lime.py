import numpy
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances


def algoritm_lime(N, model, x, k, min_vals, max_vals):
    X_perturbed = []
    R = []
    W = []
    for n in range(N):
        selected_features = numpy.random.choice(range(len(x)), size=k, replace=False)
        x_perturbed = x.copy()
        x_perturbed[selected_features] = numpy.random.uniform(
            min_vals[selected_features], max_vals[selected_features]
        )
        # Calcula la distancia coseno entre x y x_perturbed
        dist_cosine = cosine_distances(x.reshape(1, -1), x_perturbed.reshape(1, -1))
        w = dist_cosine[0, 0]
        # Crea el vector de representación
        r = numpy.ones(len(x))
        # Asigna 0 a las características que se han perturbado
        r[selected_features] = 0

        X_perturbed.append(x_perturbed)
        R.append(r)
        W.append(w)

    Y_perturbed = model.predict(numpy.array(X_perturbed))
    G = Ridge().fit(numpy.array(R), Y_perturbed, sample_weight=W)

    return G.coef_, G.intercept_
