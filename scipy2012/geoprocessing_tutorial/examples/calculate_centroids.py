"""This module provides different approaches to calculation of centroids
"""




def centroid_formula_naive(P):
    """Naive implementation of centroid formula

    Input
        P: Polygon


    NOTE: This is not used for anything. Also it does not normalise
          the input so is prone to rounding errors.
    """

    P = numpy.array(P)

    msg = ('Polygon is assumed to consist of coordinate pairs. '
           'I got second dimension %i instead of 2' % P.shape[1])
    assert P.shape[1] == 2, msg
    N = P.shape[0] - 1

    x = P[:, 0]
    y = P[:, 1]

    # Area: 0.5 sum_{i=0}^{N-1} (x_i y_{i+1} - x_{i+1} y_i)
    A = 0.0
    for i in range(N):
        A += x[i] * y[i + 1] - x[i + 1] * y[i]
    A = A / 2

    # Centroid: sum_{i=0}^{N-1} (x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)/(6A)
    Cx = 0.0
    for i in range(N):
        Cx += (x[i] + x[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
    Cx = Cx / 6 / A

    Cy = 0.0
    for i in range(N):
        Cy += (y[i] + y[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
    Cy = Cy / 6 / A

    return [Cx, Cy]
