""" Use numpy to speed up calculations.
"""

import time
import numpy
from source.core import read_layer
from source.projection import Projection
from source.utilities import TESTDATA


def calculate_polygon_area(polygon, signed=False):
    """Calculate the signed area of non-self-intersecting polygon

    Input
        polygon: Numeric array of points (longitude, latitude). It is assumed
                 to be closed, i.e. first and last points are identical
        signed: Optional flag deciding whether returned area retains its sign:
                If points are ordered counter clockwise, the signed area
                will be positive.
                If points are ordered clockwise, it will be negative
                Default is False which means that the area is always positive.
    Output
        area: Area of polygon (subject to the value of argument signed)
    """

    # Make sure it is numeric
    P = numpy.array(polygon)

    # Check input
    msg = ('Polygon is assumed to consist of coordinate pairs. '
           'I got second dimension %i instead of 2' % P.shape[1])
    assert P.shape[1] == 2, msg

    msg = ('Polygon is assumed to be closed. '
           'However first and last coordinates are different: '
           '(%f, %f) and (%f, %f)' % (P[0, 0], P[0, 1], P[-1, 0], P[-1, 1]))
    assert numpy.allclose(P[0, :], P[-1, :]), msg

    # Extract x and y coordinates
    x = P[:, 0]
    y = P[:, 1]

    # Exercise: Replace naive loop below with numpy vector operations.
    #           How much faster does the performance test run?
    N = len(P) - 1
    A = 0.0
    for i in range(N):
        A += x[i] * y[i + 1] - x[i + 1] * y[i]
    A = A / 2

    # Return signed or unsigned area
    if signed:
        return A
    else:
        return abs(A)


if __name__ == '__main__':

    # Test the area function with a couple of examples

    #----------------------
    # Test 1 - a super simple test using a "square"
    #----------------------
    P = numpy.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    A = calculate_polygon_area(P)
    msg = 'Calculated area was %f, expected 1.0 deg^2' % A
    assert A == 1, msg

    #----------------------
    # Test 2 - test against real life dataset
    #----------------------
    vectorname = 'test_polygon.shp'
    filename = '%s/%s' % (TESTDATA, vectorname)
    polygons = read_layer(filename)

    # Get coordinates of all polygons in dataset
    geometry = polygons.get_geometry()

    # Pull out first polygon
    P = geometry[0]

    # Call routine to calculate its area
    A = calculate_polygon_area(P)

    # Verify against area reported by ESRI ARC
    esri_area = 2.63924787273461e-3
    msg = 'Calculated area was %.12f, expected %.12f deg^2' % (A, esri_area)
    assert numpy.allclose(A, esri_area, rtol=0, atol=1.0e-10), msg

    #-----------------------------------
    # Test 3 - another real life dataset
    #-----------------------------------
    vectorname = 'kecamatan_geo.shp'
    filename = '%s/%s' % (TESTDATA, vectorname)
    polygons = read_layer(filename)

    # Get coordinates of all polygons in dataset
    geometry = polygons.get_geometry()

    # Pull out first polygon
    P = geometry[0]

    # Call routine to calculate its area
    A = calculate_polygon_area(P)

    # Verify against area reported by ESRI ARC
    known_area = 0.003396270556
    msg = 'Calculated area was %.12f, expected %.12f deg^2' % (A, known_area)
    assert numpy.allclose(A, known_area, rtol=0, atol=1.0e-10), msg

    #------------------------------------------------
    # Test 4 - test that open polygons raise an error
    #------------------------------------------------
    P = P[:-2, :]  # Remove last point to make the polygon open
    try:
        calculate_polygon_area(P)
    except AssertionError:
        pass
    else:
        msg = 'Open polygons should have raised on exception'
        raise Exception(msg)

    #-----------------------------------------
    # Test 5 - test performance of calculation
    #-----------------------------------------
    print 'Running performance test'

    # Run through all polygons a 100 times
    t0 = time.time()
    for i in range(100):
        for P in geometry:
            # Call routine to calculate its area
            calculate_polygon_area(P)

    print 'Areas done in %.12f seconds' % (time.time() - t0)
