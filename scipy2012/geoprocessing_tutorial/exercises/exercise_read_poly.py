""" This script reads in a shapefile and provides the data as a numpy array
"""

import numpy
from source.core import read_layer
from source.projection import Projection, DEFAULT_PROJECTION
from source.utilities import TESTDATA


def read_vector_polygon_data(filename):
    """Read vector polygon shapefile
    """

    layer = read_layer(filename)
    geometry = layer.get_geometry()
    attributes = layer.get_data()

    # Check basic data integrity
    N = len(layer)

    assert len(geometry) == N
    assert len(attributes) == N
    assert len(attributes[0]) == 8

    assert 42 == N
    assert isinstance(layer.get_name(), basestring)

    # Check projection
    wkt = layer.get_projection(proj4=False)
    assert wkt.startswith('GEOGCS')

    assert layer.projection == Projection(DEFAULT_PROJECTION)

    # Check each polygon
    for i in range(N):
        geom = geometry[i]
        n = geom.shape[0]
        assert n > 2
        assert geom.shape[1] == 2

        # Check that polygon is closed
        assert numpy.allclose(geom[0], geom[-1], rtol=0)

        # But that not all points are the same
        max_dist = 0
        for j in range(n):
            d = numpy.sum((geom[j] - geom[0]) ** 2) / n
            if d > max_dist:
                max_dist = d
        assert max_dist > 0

    # Check integrity of each feature
    expected_features = {13: {'AREA': 28760732,
                              'POP_2007': 255383,
                              'KECAMATAN': 'kali deres',
                              'KEPADATAN': 60,
                              'PROPINSI': 'DKI JAKARTA'},
                         21: {'AREA': 13155073,
                              'POP_2007': 247747,
                              'KECAMATAN': 'kramat jati',
                              'KEPADATAN': 150,
                              'PROPINSI': 'DKI JAKARTA'},
                         35: {'AREA': 4346540,
                              'POP_2007': 108274,
                              'KECAMATAN': 'senen',
                              'KEPADATAN': 246,
                              'PROPINSI': 'DKI JAKARTA'}}

    field_names = None
    for i in range(N):
        # Consistency with attributes read manually with qgis

        if i in expected_features:
            att = attributes[i]
            exp = expected_features[i]

            for key in exp:
                msg = ('Expected attribute %s was not found in feature %i'
                       % (key, i))
                assert key in att, msg

                a = att[key]
                e = exp[key]
                msg = 'Got %s: "%s" but expected "%s"' % (key, a, e)
                assert a == e, msg


if __name__ == '__main__':

    # Read and verify test data
    vectorname = 'kecamatan_geo.shp'
    filename = '%s/%s' % (TESTDATA, vectorname)

    read_vector_polygon_data(filename)
