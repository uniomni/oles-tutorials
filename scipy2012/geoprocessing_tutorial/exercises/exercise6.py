"""Implement bilinear interpolation
"""

import numpy
import time
from source.interpolation2d import check_inputs
from source.utilities import combine_coordinates, nanallclose


def interpolate2d(x, y, Z, points):
    """Fundamental 2D interpolation routine

    Input
        x: 1D array of x-coordinates of the mesh on which to interpolate
        y: 1D array of y-coordinates of the mesh on which to interpolate
        Z: 2D array of values for each x, y pair
        points: Nx2 array of coordinates where interpolated values are sought

    Output
        1D array with same length as points with interpolated values

    Notes
        Input coordinates x and y are assumed to be monotonically increasing,
        but need not be equidistantly spaced.

        Z is assumed to have dimension M x N, where M = len(x) and N = len(y).
        In other words it is assumed that the x values follow the first
        (vertical) axis downwards and y values the second (horizontal) axis
        from left to right.

        If this routine is to be used for interpolation of raster grids where
        data is typically organised with longitudes (x) going from left to
        right and latitudes (y) from left to right then user
        interpolate_raster in this module

    Derivation

        Bilinear interpolation is based on the standard 1D linear interpolation
        formula:

        Given points (x0, y0) and (x1, x0) and a value of x where x0 <= x <= x1,
        the linearly interpolated value y at x is given as

        alpha*(y1-y0) + y0

        or

        alpha*y1 + (1-alpha)*y0                (1)

        where alpha = (x-x0)/(x1-x0)           (1a)


        2D bilinear interpolation aims at obtaining an interpolated value z at a 
        point (x,y) which lies inside a square formed by points (x0, y0), (x1, y0),
        (x0, y1) and (x1, y1) for which values z00, z10, z01 and z11 are known.

        This obtained be first applying equation (1) twice in in the x-direction
        to obtain interpolated points q0 and q1 for (x, y0) and (x, y1), respectively.

        q0 = alpha*z10 + (1-alpha)*z00         (2)

        and

        q1 = alpha*z11 + (1-alpha)*z01         (3)


        Then using equation (1) in the y-direction on the results from (2) and (3)

        z = beta*q1 + (1-beta)*q0              (4)

        where beta = (y-y0)/(y1-y0)            (4a)


        Substituting (2) and (3) into (4) yields

        z = alpha*beta*z11 + beta*z01 - alpha*beta*z01 +
            alpha*z10 + z00 - alpha*z00 - alpha*beta*z10 - beta*z00 +
            alpha*beta*z00
          = alpha*beta*(z11 - z01 - z10 + z00) +
            alpha*(z10 - z00) + beta*(z01 - z00) + z00

        which can be further simplified to

        z = alpha*beta*(z11 - dx - dy - z00) + alpha*dx + beta*dy + z00  (5)

        where
        dx = z10 - z00
        dy = z01 - z00

        Equation (5) is what is implemented in the function interpolate2d above.


        Piecewise constant interpolation can be implemented using the same coefficients
        (1a) and (4a) that are used for bilinear interpolation as they are a measure of
        the relative distance to the left and lower neigbours. A value of 0 will pick
        the left or lower bound whereas a value of 1 will pick the right or higher
        bound. Hence z can be assigned to its nearest neigbour as follows

            | z00   alpha < 0.5 and beta < 0.5    # lower left corner
            |
            | z10   alpha >= 0.5 and beta < 0.5   # lower right corner
        z = |
            | z01   alpha < 0.5 and beta >= 0.5   # upper left corner
            |
            | z11   alpha >= 0.5 and beta >= 0.5  # upper right corner
                
    """

    # Check inputs and provid xi, eta as x and y coordinates from points vector
    x, y, Z, xi, eta = check_inputs(x, y, Z, points, 'linear', False)

    # Identify elements that are outside interpolation domain or NaN
    outside = (xi < x[0]) + (eta < y[0]) + (xi > x[-1]) + (eta > y[-1])
    outside += numpy.isnan(xi) + numpy.isnan(eta)

    # Restrict interpolation points to those that are inside the grid
    inside = -outside  # Invert boolean array to find elements inside
    xi = xi[inside]
    eta = eta[inside]

    # Find upper neighbours for each interpolation point 
    # ('left' means first occurrence)
    idx = numpy.searchsorted(x, xi, side='left')
    idy = numpy.searchsorted(y, eta, side='left')

    # Get the four neighbours for each interpolation point
    x0 = x[idx - 1]  # Left
    x1 = x[idx]      # Right
    y0 = y[idy - 1]  # Lower
    y1 = y[idy]      # Upper

    # And the corresponding four grid values
    z00 = Z[idx - 1, idy - 1]
    z01 = Z[idx - 1, idy]
    z10 = Z[idx, idy - 1]
    z11 = Z[idx, idy]

    # Coefficients for weighting between lower and upper bounds
    numpy.seterr(invalid='ignore')  # Ignore division by zero
    alpha = (xi - x0) / (x1 - x0)
    beta = (eta - y0) / (y1 - y0)

    # Bilinear interpolation formula as per equation (5) above
    dx = z10 - z00
    dy = z01 - z00
    z = z00 + alpha * dx + beta * dy + alpha * beta * (z11 - dx - dy - z00)

    # Populate result with interpolated values for points inside domain
    # and NaN for values outside
    r = numpy.zeros(len(points))
    r[inside] = z
    r[outside] = numpy.nan

    return r



def linear_function(x, y):
    """Auxiliary function for use with interpolation test
    """

    return x + y / 2.0


if __name__ == '__main__':

    # ------------------------------------------------------------
    # Interpolation library works for linear function - basic test
    # ------------------------------------------------------------

    t0 = time.time()
    # Define pixel centers along each direction
    x = [1.0, 2.0, 4.0]
    y = [5.0, 9.0]

    # Define ny by nx array with corresponding values
    A = numpy.zeros((len(x), len(y)))

    # Define values for each x, y pair as a linear function
    for i in range(len(x)):
        for j in range(len(y)):
            A[i, j] = linear_function(x[i], y[j])

    # Test first that original points are reproduced correctly
    for i, xi in enumerate(x):
        for j, eta in enumerate(y):
            val = interpolate2d(x, y, A, [(xi, eta)])[0]
            ref = linear_function(xi, eta)
            assert numpy.allclose(val, ref, rtol=1e-12, atol=1e-12)

    # Then test that genuinly interpolated points are correct
    xis = numpy.linspace(x[0], x[-1], 10)
    etas = numpy.linspace(y[0], y[-1], 10)
    points = combine_coordinates(xis, etas)

    vals = interpolate2d(x, y, A, points)
    refs = linear_function(points[:, 0], points[:, 1])
    assert numpy.allclose(vals, refs, rtol=1e-12, atol=1e-12)

    #-------------------------------------------------------
    # Interpolation library works with grid points being NaN
    #-------------------------------------------------------

    # Define pixel centers along each direction
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y = [4.0, 5.0, 7.0, 9.0, 11.0, 13.0]

    # Define ny by nx array with corresponding values
    A = numpy.zeros((len(x), len(y)))

    # Define values for each x, y pair as a linear function
    for i in range(len(x)):
        for j in range(len(y)):
            A[i, j] = linear_function(x[i], y[j])
    A[2, 3] = numpy.nan  # (x=2.0, y=9.0): NaN

    # Then test that interpolated points can contain NaN
    xis = numpy.linspace(x[0], x[-1], 12)
    etas = numpy.linspace(y[0], y[-1], 10)
    points = combine_coordinates(xis, etas)

    vals = interpolate2d(x, y, A, points)
    refs = linear_function(points[:, 0], points[:, 1])

    # Set reference result with expected NaNs and compare
    for i, (xi, eta) in enumerate(points):
        if (1.0 < xi <= 3.0) and (7.0 < eta <= 11.0):
            refs[i] = numpy.nan

    assert nanallclose(vals, refs, rtol=1e-12, atol=1e-12)

    #-------------------------------------
    # Interpolation library works with NaN
    #-------------------------------------

    # Define pixel centers along each direction
    x = numpy.arange(20) * 1.0
    y = numpy.arange(25) * 1.0

    # Define ny by nx array with corresponding values
    A = numpy.zeros((len(x), len(y)))

    # Define arbitrary values for each x, y pair
    numpy.random.seed(17)
    A = numpy.random.random((len(x), len(y))) * 10

    # Create islands of NaN
    A[5, 13] = numpy.nan
    A[6, 14] = A[6, 18] = numpy.nan
    A[7, 14:18] = numpy.nan
    A[8, 13:18] = numpy.nan
    A[9, 12:19] = numpy.nan
    A[10, 14:17] = numpy.nan
    A[11, 15] = numpy.nan

    A[15, 5:6] = numpy.nan

    # Create interpolation points
    xis = numpy.linspace(x[0], x[-1], 39)   # Hit all mid points
    etas = numpy.linspace(y[0], y[-1], 73)  # Hit thirds
    points = combine_coordinates(xis, etas)

    vals = interpolate2d(x, y, A, points)

    # Calculate reference result with expected NaNs and compare
    i = j = 0
    for k, (xi, eta) in enumerate(points):

        # Find indices of nearest higher value in x and y
        i = numpy.searchsorted(x, xi)
        j = numpy.searchsorted(y, eta)

        if i > 0 and j > 0:

            # Get four neigbours
            A00 = A[i - 1, j - 1]
            A01 = A[i - 1, j]
            A10 = A[i, j - 1]
            A11 = A[i, j]

            if numpy.allclose(xi, x[i]):
                alpha = 1.0
            else:
                alpha = 0.5

            if numpy.allclose(eta, y[j]):
                beta = 1.0
            else:
                beta = eta - y[j - 1]


            if numpy.any(numpy.isnan([A00, A01, A10, A11])):
                ref = numpy.nan
            else:
                ref = (A00 * (1 - alpha) * (1 - beta) +
                       A01 * (1 - alpha) * beta +
                       A10 * alpha * (1 - beta) +
                       A11 * alpha * beta)

            #print i, j, xi, eta, alpha, beta, vals[k], ref
            assert nanallclose(vals[k], ref, rtol=1e-12, atol=1e-12)

    #-----------------------------------------------------------------
    # Interpolation library sensibly handles values outside the domain
    #-----------------------------------------------------------------

    # Define pixel centers along each direction
    x = [1.0, 2.0, 4.0]
    y = [5.0, 9.0]

    # Define ny by nx array with corresponding values
    A = numpy.zeros((len(x), len(y)))

    # Define values for each x, y pair as a linear function
    for i in range(len(x)):
        for j in range(len(y)):
            A[i, j] = linear_function(x[i], y[j])

    # Simple example first for debugging
    xis = numpy.linspace(0.9, 4.0, 4)
    etas = numpy.linspace(5, 9.1, 3)
    points = combine_coordinates(xis, etas)
    refs = linear_function(points[:, 0], points[:, 1])

    vals = interpolate2d(x, y, A, points)
    msg = ('Length of interpolation points %i differs from length '
           'of interpolated values %i' % (len(points), len(vals)))
    assert len(points) == len(vals), msg
    for i, (xi, eta) in enumerate(points):
        if xi < x[0] or xi > x[-1] or eta < y[0] or eta > y[-1]:
            assert numpy.isnan(vals[i])
        else:
            msg = ('Got %.15f for (%f, %f), expected %.15f'
                   % (vals[i], xi, eta, refs[i]))
            assert numpy.allclose(vals[i], refs[i],
                                  rtol=1.0e-12, atol=1.0e-12), msg

    # Try a range of combinations of points outside domain with
    # error_bounds False
    for lox in [x[0], x[0] - 1, x[0] - 10]:
        for hix in [x[-1], x[-1] + 1, x[-1] + 5]:
            for loy in [y[0], y[0] - 1, y[0] - 10]:
                for hiy in [y[-1], y[-1] + 1, y[-1] + 10]:

                    # Then test that points outside domain can be handled
                    xis = numpy.linspace(lox, hix, 10)
                    etas = numpy.linspace(loy, hiy, 10)
                    points = combine_coordinates(xis, etas)
                    refs = linear_function(points[:, 0], points[:, 1])
                    vals = interpolate2d(x, y, A, points)

                    assert len(points) == len(vals), msg
                    for i, (xi, eta) in enumerate(points):
                        if xi < x[0] or xi > x[-1] or\
                                eta < y[0] or eta > y[-1]:
                            msg = 'Expected NaN for %f, %f' % (xi, eta)
                            assert numpy.isnan(vals[i]), msg
                        else:
                            msg = ('Got %.15f for (%f, %f), expected '
                                   '%.15f' % (vals[i], xi, eta, refs[i]))
                            assert numpy.allclose(vals[i], refs[i],
                                                  rtol=1.0e-12,
                                                  atol=1.0e-12), msg

    #-------------------------------------------------------------
    # Interpolation library returns NaN for incomplete grid points
    #-------------------------------------------------------------

    # Define four pixel centers
    x = [2.0, 4.0]
    y = [5.0, 9.0]

    # Define ny by nx array with corresponding values
    A = numpy.zeros((len(x), len(y)))

    # Define values for each x, y pair as a linear function
    for i in range(len(x)):
        for j in range(len(y)):
            A[i, j] = linear_function(x[i], y[j])

    # Test that interpolated points are correct
    xis = numpy.linspace(x[0], x[-1], 3)
    etas = numpy.linspace(y[0], y[-1], 3)
    points = combine_coordinates(xis, etas)

    # Interpolate to cropped grids
    for xc, yc, Ac in [([x[0]], [y[0]], numpy.array([[A[0, 0]]])),  # 1 x 1
                       ([x[0]], y, numpy.array([A[0, :]])),  # 1 x 2
                       ]:

        vals = interpolate2d(xc, yc, Ac, points)
        msg = 'Expected NaN when grid %s is incomplete' % str(Ac.shape)
        assert numpy.all(numpy.isnan(vals)), msg

    print 'All interpolation tests done in %f seconds' % (time.time() - t0)


