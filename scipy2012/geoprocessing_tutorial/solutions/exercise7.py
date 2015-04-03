""" Interpolate from raster grid to point data
"""

import time
import numpy
from source.core import read_layer
from source.vector import Vector
from source.projection import Projection
from source.utilities import TESTDATA
from source.interpolation import assign_hazard_values_to_exposure_data

if __name__ == '__main__':

    #-----------------------------
    # Test 1 - a real life dataset
    #-----------------------------

    # Name file names for hazard level, exposure and expected fatalities
    hazard_filename = ('%s/maumere_aos_depth_20m_land_wgs84.asc'
                       % TESTDATA)
    exposure_filename = ('%s/maumere_pop_prj.shp' % TESTDATA)

    # Read input data
    H = read_layer(hazard_filename)
    A = H.get_data()
    depth_min, depth_max = H.get_extrema()

    # Compare extrema to values read off QGIS for this layer
    assert numpy.allclose([depth_min, depth_max], [0.0, 16.68],
                          rtol=1.0e-6, atol=1.0e-10)

    E = read_layer(exposure_filename)
    coordinates = E.get_geometry()
    attributes = E.get_data()

    # Test riab's interpolation function
    I = assign_hazard_values_to_exposure_data(H, E,
                                              attribute_name='depth')
    Icoordinates = I.get_geometry()
    Iattributes = I.get_data()
    assert numpy.allclose(Icoordinates, coordinates)

    N = len(Icoordinates)
    assert N == 891

    # Verify interpolated values with test result
    for i in range(N):

        interpolated_depth = Iattributes[i]['depth']
        pointid = attributes[i]['POINTID']

        if pointid == 263:

            # Check that location is correct
            assert numpy.allclose(coordinates[i],
                                  [122.20367299, -8.61300358])

            # This is known to be outside inundation area so should
            # near zero
            assert numpy.allclose(interpolated_depth, 0.0,
                                  rtol=1.0e-12, atol=1.0e-12)

        if pointid == 148:
            # Check that location is correct
            assert numpy.allclose(coordinates[i],
                                  [122.2045912, -8.608483265])

            # This is in an inundated area with a surrounding depths of
            # 4.531, 3.911
            # 2.675, 2.583
            assert interpolated_depth < 4.531
            assert interpolated_depth < 3.911
            assert interpolated_depth > 2.583
            assert interpolated_depth > 2.675

            # This is a characterisation test for bilinear interpolation
            msg = ('Interpolated depth was %.15f, expected %.15f'
                   % (interpolated_depth, 3.624772044548650))
            assert numpy.allclose(interpolated_depth, 3.624772044548650,
                                  rtol=1.0e-12, atol=1.0e-12), msg

        # Check that interpolated points are within range
        msg = ('Interpolated depth %f at point %i was outside extrema: '
               '[%f, %f]. ' % (interpolated_depth, i,
                               depth_min, depth_max))

        if not numpy.isnan(interpolated_depth):
            assert depth_min <= interpolated_depth <= depth_max, msg


