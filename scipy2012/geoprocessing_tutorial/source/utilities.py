"""**Utilities for storage module**
"""

import os
import re
import copy
import numpy
import math
import gettext
import getpass
from ast import literal_eval
from tempfile import mkstemp
from osgeo import ogr
from datetime import date

from geometry import Polygon


class InaSAFEError(RuntimeError):
    """Base class for all user defined execptions"""
    pass


class ReadLayerError(InaSAFEError):
    """When a layer can't be read"""
    pass


class WriteLayerError(InaSAFEError):
    """When a layer can't be written"""
    pass


class BoundingBoxError(InaSAFEError):
    """For errors relating to bboxes"""
    pass


class VerificationError(InaSAFEError):
    """Exception thrown by verify()
    """
    pass


class PolygonInputError(InaSAFEError):
    """For invalid inputs to numeric polygon functions"""
    pass


class BoundsError(InaSAFEError):
    """For points falling outside interpolation grid"""
    pass


class GetDataError(InaSAFEError):
    """When layer data cannot be obtained"""
    pass


class PostProcessorError(Exception):
    """Raised when requested import cannot be performed if QGIS is too old."""
    pass


class WindowsError(Exception):
    """For windows specific errors."""
    pass


# Default attribute to assign to vector layers
DEFAULT_ATTRIBUTE = 'Affected'

# Spatial layer file extensions that are recognised in Risiko
# FIXME: Perhaps add '.gml', '.zip', ...
LAYER_TYPES = ['.shp', '.asc', '.tif', '.tiff', '.geotif', '.geotiff']

# Map between extensions and ORG drivers
DRIVER_MAP = {'.sqlite': 'SQLITE',
              '.shp': 'ESRI Shapefile',
              '.gml': 'GML',
              '.kml': 'KML',
              '.tif': 'GTiff',
              '.asc': 'AAIGrid'}

# Map between Python types and OGR field types
# FIXME (Ole): I can't find a double precision type for OGR
TYPE_MAP = {type(None): ogr.OFTString,  # What else should this be?
            type(''): ogr.OFTString,
            type(True): ogr.OFTInteger,
            type(0): ogr.OFTInteger,
            type(0.0): ogr.OFTReal,
            type(numpy.array([0.0])[0]): ogr.OFTReal,  # numpy.float64
            type(numpy.array([[0.0]])[0]): ogr.OFTReal}  # numpy.ndarray

# Map between verbose types and OGR geometry types
INVERSE_GEOMETRY_TYPE_MAP = {'point': ogr.wkbPoint,
                             'line': ogr.wkbLineString,
                             'polygon': ogr.wkbPolygon}


# Known feature counts in test data
FEATURE_COUNTS = {'test_buildings.shp': 144,
                  'tsunami_building_exposure.shp': 19,
                  'kecamatan_geo.shp': 42,
                  'Padang_WGS84.shp': 3896,
                  'OSM_building_polygons_20110905.shp': 34960,
                  'indonesia_highway_sample.shp': 2,
                  'OSM_subset.shp': 79,
                  'kecamatan_jakarta_osm.shp': 47}

# For testing of storage modules
GEOTRANSFORMS = [(105.3000035, 0.008333, 0.0, -5.5667785, 0.0, -0.008333),
                 (105.29857, 0.0112, 0.0, -5.565233000000001, 0.0, -0.0112),
                 (96.956, 0.03074106, 0.0, 2.2894972560001, 0.0, -0.03074106)]

try:
    import source
except:
    root = '..'
else:
    # Grab path to root of source tree
    root = os.path.split(os.path.split(source.__file__)[0])[0]

TESTDATA = os.path.join('%s/spatial_test_data' % root)



def ensure_numeric(A, typecode=None):
    """Ensure that sequence is a numeric array.

    Args:
        * A: Sequence. If A is already a numeric array it will be returned
                       unaltered
                       If not, an attempt is made to convert it to a numeric
                       array
        * A: Scalar.   Return 0-dimensional array containing that value. Note
                       that a 0-dim array DOES NOT HAVE A LENGTH UNDER numpy.
        * A: String.   Array of ASCII values (numpy can't handle this)

        * typecode:    numeric type. If specified, use this in the conversion.
                     If not, let numeric package decide.
                     typecode will always be one of num.float, num.int, etc.

    Note :
        that numpy.array(A, dtype) will sometimes copy.  Use 'copy=False' to
        copy only when required.

        This function is necessary as array(A) can cause memory overflow.
    """

    if isinstance(A, basestring):
        msg = 'Sorry, cannot handle strings in ensure_numeric()'
        # FIXME (Ole): Change this to whatever is the appropriate exception
        # for wrong input type
        raise Exception(msg)

    if typecode is None:
        if isinstance(A, numpy.ndarray):
            return A
        else:
            return numpy.array(A)
    else:
        return numpy.array(A, dtype=typecode, copy=False)


def nanallclose(x, y, rtol=1.0e-5, atol=1.0e-8):
    """Numpy allclose function which allows NaN

    Args:
        * x, y: Either scalars or numpy arrays

    Returns:
        * True or False

    Note:
        Returns True if all non-nan elements pass.
    """

    xn = numpy.isnan(x)
    yn = numpy.isnan(y)
    if numpy.any(xn != yn):
        # Presence of NaNs is not the same in x and y
        return False

    if numpy.all(xn):
        # Everything is NaN.
        # This will also take care of x and y being NaN scalars
        return True

    # Filter NaN's out
    if numpy.any(xn):
        x = x[-xn]
        y = y[-yn]

    # Compare non NaN's and return
    return numpy.allclose(x, y, rtol=rtol, atol=atol)


def normal_cdf(x, mu=0, sigma=1):
    """Cumulative Normal Distribution Function

    Args:
        * x: scalar or array of real numbers
        * mu: Mean value. Default 0
        * sigma: Standard deviation. Default 1

    Returns:
        * An approximation of the cdf of the normal

    Note:
        CDF of the normal distribution is defined as
        \frac12 [1 + erf(\frac{x - \mu}{\sigma \sqrt{2}})], x \in \R

        Source: http://en.wikipedia.org/wiki/Normal_distribution
    """

    arg = (x - mu) / (sigma * numpy.sqrt(2))
    res = (1 + erf(arg)) / 2

    return res


def lognormal_cdf(x, median=1, sigma=1):
    """Cumulative Log Normal Distribution Function

    Args:
        * x: scalar or array of real numbers
        * mu: Median (exp(mean of log(x)). Default 1
        * sigma: Log normal standard deviation. Default 1

    Returns:
        * An approximation of the cdf of the normal

    Note:
        CDF of the normal distribution is defined as
        \frac12 [1 + erf(\frac{x - \mu}{\sigma \sqrt{2}})], x \in \R

        Source: http://en.wikipedia.org/wiki/Normal_distribution
    """

    return normal_cdf(numpy.log(x), mu=numpy.log(median), sigma=sigma)


def erf(z):
    """Approximation to ERF

    Note:
        from:
        http://www.cs.princeton.edu/introcs/21function/ErrorFunction.java.html
        Implements the Gauss error function.
        erf(z) = 2 / sqrt(pi) * integral(exp(-t*t), t = 0..z)

        Fractional error in math formula less than 1.2 * 10 ^ -7.
        although subject to catastrophic cancellation when z in very close to 0
        from Chebyshev fitting formula for erf(z) from Numerical Recipes, 6.2

        Source:
        http://stackoverflow.com/questions/457408/
        is-there-an-easily-available-implementation-of-erf-for-python
    """

    # Input check
    try:
        len(z)
    except TypeError:
        scalar = True
        z = [z]
    else:
        scalar = False

    z = numpy.array(z)

    # Begin algorithm
    t = 1.0 / (1.0 + 0.5 * numpy.abs(z))

    # Use Horner's method
    ans = 1 - t * numpy.exp(-z * z - 1.26551223 +
                           t * (1.00002368 +
                           t * (0.37409196 +
                           t * (0.09678418 +
                           t * (-0.18628806 +
                           t * (0.27886807 +
                           t * (-1.13520398 +
                           t * (1.48851587 +
                           t * (-0.82215223 +
                           t * (0.17087277))))))))))

    neg = (z < 0.0)  # Mask for negative input values
    ans[neg] = -ans[neg]

    if scalar:
        return ans[0]
    else:
        return ans


def axes2points(x, y):
    """Generate all combinations of grid point coordinates from x and y axes

    Args:
        * x: x coordinates (array)
        * y: y coordinates (array)

    Returns:
        * P: Nx2 array consisting of coordinates for all
             grid points defined by x and y axes. The x coordinate
             will vary the fastest to match the way 2D numpy
             arrays are laid out by default ('C' order). That way,
             the x and y coordinates will match a corresponding
             2D array A when flattened (A.flat[:] or A.reshape(-1))

    Note:
        Example

        x = [1, 2, 3]
        y = [10, 20]

        P = [[1, 10],
             [2, 10],
             [3, 10],
             [1, 20],
             [2, 20],
             [3, 20]]
    """

    # Reverse y coordinates to have them start at bottom of array
    y = numpy.flipud(y)

    # Repeat x coordinates for each y (fastest varying)
    X = numpy.kron(numpy.ones(len(y)), x)

    # Repeat y coordinates for each x (slowest varying)
    Y = numpy.kron(y, numpy.ones(len(x)))

    # Check
    N = len(X)
    verify(len(Y) == N)

    # Create Nx2 array of x and y coordinates
    X = numpy.reshape(X, (N, 1))
    Y = numpy.reshape(Y, (N, 1))
    P = numpy.concatenate((X, Y), axis=1)

    # Return
    return P


def grid2points(A, x, y):
    """Convert grid data to point data

    Args:
        * A: Array of pixel values
        * x: Longitudes corresponding to columns in A (left->right)
        * y: Latitudes corresponding to rows in A (top->bottom)

    Returns:
        * P: Nx2 array of point coordinates
        * V: N array of point values
    """

    # Create Nx2 array of x, y points corresponding to each
    # element in A.
    points = axes2points(x, y)

    # Create flat 1D row-major view of A cast as
    # one column vector of length MxN where M, N = A.shape
    #values = A.reshape((-1, 1))
    values = A.reshape(-1)

    # Concatenate coordinates with their values from the grid
    #P = numpy.concatenate((points, values), axis=1)

    # Return Nx3 array with rows: x, y, value
    return points, values


def geotransform2axes(G, nx, ny):
    """Convert geotransform to coordinate axes

    Args:
        * G: GDAL geotransform (6-tuple).
             (top left x, w-e pixel resolution, rotation,
              top left y, rotation, n-s pixel resolution).
        * nx: Number of cells in the w-e direction
        * ny: Number of cells in the n-s direction


    Returns:
        * Return two vectors (longitudes and latitudes) representing the grid
          defined by the geotransform.

          The values are offset by half a pixel size to correspond to
          pixel registration.

          I.e. If the grid origin (top left corner) is (105, 10) and the
          resolution is 1 degrees in each direction, then the vectors will
          take the form

          longitudes = [100.5, 101.5, ..., 109.5]
          latitudes = [0.5, 1.5, ..., 9.5]
    """

    lon_ul = float(G[0])  # Longitude of upper left corner
    lat_ul = float(G[3])  # Latitude of upper left corner
    dx = float(G[1])      # Longitudinal resolution
    dy = - float(G[5])    # Latitudinal resolution (always(?) negative)

    verify(dx > 0)
    verify(dy > 0)

    # Coordinates of lower left corner
    lon_ll = lon_ul
    lat_ll = lat_ul - ny * dy

    # Coordinates of upper right corner
    lon_ur = lon_ul + nx * dx

    # Define pixel centers along each directions
    # This is to achieve pixel registration rather
    # than gridline registration
    dx2 = dx / 2
    dy2 = dy / 2

    # Define longitudes and latitudes for each axes
    x = numpy.linspace(lon_ll + dx2,
                       lon_ur - dx2, nx)
    y = numpy.linspace(lat_ll + dy2,
                       lat_ul - dy2, ny)

    # Return
    return x, y


# Miscellaneous auxiliary functions
def _keywords_to_string(keywords, sublayer=None):
    """Create a string from a keywords dict.

    Args:
        * keywords: A required dictionary containing the keywords to stringify.
        * sublayer: str optional group marker for a sub layer.

    Returns:
        str: a String containing the rendered keywords list

    Raises:
        Any exceptions are propogated.

    .. note: Only simple keyword dicts should be passed here, not multilayer
       dicts.

    For example you pass a dict like this::

        {'datatype': 'osm',
         'category': 'exposure',
         'title': 'buildings_osm_4326',
         'subcategory': 'building',
         'purpose': 'dki'}

    and the following string would be returned:

        datatype: osm
        category: exposure
        title: buildings_osm_4326
        subcategory: building
        purpose: dki

    If sublayer is provided e.g. _keywords_to_string(keywords, sublayer='foo'),
    the following:

        [foo]
        datatype: osm
        category: exposure
        title: buildings_osm_4326
        subcategory: building
        purpose: dki
    """

    # Write
    result = ''
    if sublayer is not None:
        result = '[%s]\n' % sublayer
    for k, v in keywords.items():
        # Create key
        msg = ('Key in keywords dictionary must be a string. '
               'I got %s with type %s' % (k, str(type(k))[1:-1]))
        verify(isinstance(k, basestring), msg)

        key = k
        msg = ('Key in keywords dictionary must not contain the ":" '
               'character. I got "%s"' % key)
        verify(':' not in key, msg)

        # Create value
        msg = ('Value in keywords dictionary must be convertible to a string. '
               'For key %s, I got %s with type %s'
               % (k, v, str(type(v))[1:-1]))
        try:
            val = str(v)
        except:
            raise Exception(msg)

        # Store
        result += '%s: %s\n' % (key, val)
    return result


def write_keywords(keywords, filename, sublayer=None):
    """Write keywords dictonary to file

    Args:
        * keywords: Dictionary of keyword, value pairs
        * filename: Name of keywords file. Extension expected to be .keywords
        * sublayer: str Optional sublayer applicable only to multilayer formats
             such as sqlite or netcdf which can potentially hold more than
             one layer. The string should map to the layer group as per the
             example below. **If the keywords file contains sublayer
             definitions but no sublayer was defined, keywords file content
             will be removed and replaced with only the keywords provided
             here.**

    Returns: None

    Raises: None

    A keyword file with sublayers may look like this:

        [osm_buildings]
        datatype: osm
        category: exposure
        subcategory: building
        purpose: dki
        title: buildings_osm_4326

        [osm_flood]
        datatype: flood
        category: hazard
        subcategory: building
        title: flood_osm_4326

    Keys must be strings not containing the ":" character
    Values can be anything that can be converted to a string (using
    Python's str function)

    Surrounding whitespace is removed from values, but keys are unmodified
    The reason being that keys must always be valid for the dictionary they
    came from. For values we have decided to be flexible and treat entries like
    'unit:m' the same as 'unit: m', or indeed 'unit: m '.
    Otherwise, unintentional whitespace in values would lead to surprising
    errors in the application.
    """

    # Input checks
    basename, ext = os.path.splitext(filename)

    msg = ('Unknown extension for file %s. '
           'Expected %s.keywords' % (filename, basename))
    verify(ext == '.keywords', msg)

    # First read any keywords out of the file so that we can retain
    # keywords for other sublayers
    existing_keywords = read_keywords(filename, all_blocks=True)

    first_value = None
    if len(existing_keywords) > 0:
        first_value = existing_keywords[existing_keywords.keys()[0]]
    multilayer_flag = type(first_value) == dict

    handle = file(filename, 'wt')

    if multilayer_flag:
        if sublayer is not None and sublayer != '':
            #replace existing keywords / add new for this layer
            existing_keywords[sublayer] = keywords
            for key, value in existing_keywords.iteritems():
                handle.write(_keywords_to_string(value, sublayer=key))
                handle.write('\n')
        else:
            # It is currently a multilayer but we will replace it with
            # a single keyword block since the user passed no sublayer
            handle.write(_keywords_to_string(keywords))
    else:
        #currently a simple layer so replace it with our content
        handle.write(_keywords_to_string(keywords, sublayer=sublayer))

    handle.close()


def read_keywords(filename, sublayer=None, all_blocks=False):
    """Read keywords dictionary from file

    Args:
        * filename: Name of keywords file. Extension expected to be .keywords
             The format of one line is expected to be either
             string: string or string
        * sublayer: str Optional sublayer applicable only to multilayer formats
             such as sqlite or netcdf which can potentially hold more than
             one layer. The string should map to the layer group as per the
             example below. If the keywords file contains sublayer definitions
             but no sublayer was defined, the first layer group will be
             returned.
        * all_blocks: bool Optional, defaults to False. If True will return
            a dict of dicts, where the top level dict entries each represent
            a sublayer, and the values of that dict will be dicts of keyword
            entries.

    Returns:
        keywords: Dictionary of keyword, value pairs

    Raises: None

    A keyword layer with sublayers may look like this:

        [osm_buildings]
        datatype: osm
        category: exposure
        subcategory: building
        purpose: dki
        title: buildings_osm_4326

        [osm_flood]
        datatype: flood
        category: hazard
        subcategory: building
        title: flood_osm_4326

    Wheras a simple keywords file would look like this

        datatype: flood
        category: hazard
        subcategory: building
        title: flood_osm_4326

    If filename does not exist, an empty dictionary is returned
    Blank lines are ignored
    Surrounding whitespace is removed from values, but keys are unmodified
    If there are no ':', then the keyword is treated as a key with no value
    """

    # Input checks
    basename, ext = os.path.splitext(filename)

    msg = ('Unknown extension for file %s. '
           'Expected %s.keywords' % (filename, basename))
    verify(ext == '.keywords', msg)

    if not os.path.isfile(filename):
        return {}

    # Read all entries
    blocks = {}
    keywords = {}
    fid = open(filename, 'r')
    current_block = None
    first_keywords = None
    for line in fid.readlines():
        # Remove trailing (but not preceeding!) whitespace
        # FIXME: Can be removed altogether
        text = line.rstrip()

        # Ignore blank lines
        if text == '':
            continue

        # Check if it is an ini style group header
        block_flag = re.search(r'^\[.*]$', text, re.M | re.I)

        if block_flag:
            # Write the old block if it exists - must have a current
            # block to prevent orphans
            if len(keywords) > 0 and current_block is not None:
                blocks[current_block] = keywords
            if first_keywords is None and len(keywords) > 0:
                first_keywords = keywords
            # Now set up for a new block
            current_block = text[1:-1]
            # Reset the keywords each time we encounter a new block
            # until we know we are on the desired one
            keywords = {}
            continue

        if ':' not in text:
            key = text.strip()
            val = None
        else:
            # Get splitting point
            idx = text.find(':')

            # Take key as everything up to the first ':'
            key = text[:idx]

            # Take value as everything after the first ':'
            textval = text[idx + 1:].strip()
            try:
                # Take care of python structures like
                # booleans, None, lists, dicts etc
                val = literal_eval(textval)
            except (ValueError, SyntaxError):
                val = textval

        # Add entry to dictionary
        keywords[key] = val

    fid.close()

    # Write our any unfinalised block data
    if len(keywords) > 0 and current_block is not None:
        blocks[current_block] = keywords
    if first_keywords is None:
        first_keywords = keywords

    # Ok we have generated a structure that looks like this:
    # blocks = {{ 'foo' : { 'a': 'b', 'c': 'd'},
    #           { 'bar' : { 'd': 'e', 'f': 'g'}}
    # where foo and bar are sublayers and their dicts are the sublayer keywords
    if all_blocks:
        return blocks
    if sublayer is not None:
        if sublayer in blocks:
            return blocks[sublayer]
    else:
        return first_keywords


def check_geotransform(geotransform):
    """Check that geotransform is valid

    Args
        * geotransform: GDAL geotransform (6-tuple).
        (top left x, w-e pixel resolution, rotation,
        top left y, rotation, n-s pixel resolution).
        See e.g. http://www.gdal.org/gdal_tutorial.html

    Note
       This assumes that the spatial reference uses geographic coordinaties,
       so will not work for projected coordinate systems.
    """

    msg = ('Supplied geotransform must be a tuple with '
           '6 numbers. I got %s' % str(geotransform))
    verify(len(geotransform) == 6, msg)

    for x in geotransform:
        try:
            float(x)
        except TypeError:
            raise InaSAFEError(msg)

    # Check longitude
    msg = ('Element in 0 (first) geotransform must be a valid '
           'longitude. I got %s' % geotransform[0])
    verify(-180 <= geotransform[0] <= 180, msg)

    # Check latitude
    msg = ('Element 3 (fourth) in geotransform must be a valid '
           'latitude. I got %s' % geotransform[3])
    verify(-90 <= geotransform[3] <= 90, msg)

    # Check cell size
    msg = ('Element 1 (second) in geotransform must be a positive '
           'number. I got %s' % geotransform[1])
    verify(geotransform[1] > 0, msg)

    msg = ('Element 5 (sixth) in geotransform must be a negative '
           'number. I got %s' % geotransform[1])
    verify(geotransform[5] < 0, msg)


def geotransform2bbox(geotransform, columns, rows):
    """Convert geotransform to bounding box

    Args :
        * geotransform: GDAL geotransform (6-tuple).
                        (top left x, w-e pixel resolution, rotation,
                        top left y, rotation, n-s pixel resolution).
                        See e.g. http://www.gdal.org/gdal_tutorial.html
        * columns: Number of columns in grid
        * rows: Number of rows in grid

    Returns:
        * bbox: Bounding box as a list of geographic coordinates
                [west, south, east, north]

    Note:
        Rows and columns are needed to determine eastern and northern bounds.
        FIXME: Not sure if the pixel vs gridline registration issue is observed
        correctly here. Need to check against gdal > v1.7
    """

    x_origin = geotransform[0]  # top left x
    y_origin = geotransform[3]  # top left y
    x_res = geotransform[1]     # w-e pixel resolution
    y_res = geotransform[5]     # n-s pixel resolution
    x_pix = columns
    y_pix = rows

    minx = x_origin
    maxx = x_origin + (x_pix * x_res)
    miny = y_origin + (y_pix * y_res)
    maxy = y_origin

    return [minx, miny, maxx, maxy]


def geotransform2resolution(geotransform, isotropic=False):
    """Convert geotransform to resolution

    Args:
        * geotransform: GDAL geotransform (6-tuple).
                        (top left x, w-e pixel resolution, rotation,
                        top left y, rotation, n-s pixel resolution).
                        See e.g. http://www.gdal.org/gdal_tutorial.html
        * isotropic: If True, return the average (dx + dy) / 2

    Returns:
        * resolution: grid spacing (resx, resy) in (positive) decimal
                      degrees ordered as longitude first, then latitude.
                      or (resx + resy) / 2 (if isotropic is True)
    """

    resx = geotransform[1]   # w-e pixel resolution
    resy = -geotransform[5]  # n-s pixel resolution (always negative)

    if isotropic:
        return (resx + resy) / 2
    else:
        return resx, resy


def raster_geometry2geotransform(longitudes, latitudes):
    """Convert vectors of longitudes and latitudes to geotransform

    Note:
        This is the inverse operation of Raster.get_geometry().

    Args:
       * longitudes, latitudes: Vectors of geographic coordinates

    Returns:
       * geotransform: 6-tuple (top left x, w-e pixel resolution, rotation,
                                top left y, rotation, n-s pixel resolution)

    """

    nx = len(longitudes)
    ny = len(latitudes)

    msg = ('You must specify more than 1 longitude to make geotransform: '
           'I got %s' % str(longitudes))
    verify(nx > 1, msg)

    msg = ('You must specify more than 1 latitude to make geotransform: '
           'I got %s' % str(latitudes))
    verify(ny > 1, msg)

    dx = float(longitudes[1] - longitudes[0])  # Longitudinal resolution
    dy = float(latitudes[0] - latitudes[1])  # Latitudinal resolution (neg)

    # Define pixel centers along each directions
    # This is to achieve pixel registration rather
    # than gridline registration
    dx2 = dx / 2
    dy2 = dy / 2

    geotransform = (longitudes[0] - dx2,  # Longitude of upper left corner
                    dx,                   # w-e pixel resolution
                    0,                    # rotation
                    latitudes[-1] - dy2,  # Latitude of upper left corner
                    0,                    # rotation
                    dy)                   # n-s pixel resolution

    return geotransform


def bbox_intersection(*args):
    """Compute intersection between two or more bounding boxes

    Args:
        * args: two or more bounding boxes.
              Each is assumed to be a list or a tuple with
              four coordinates (W, S, E, N)

    Returns:
        * result: The minimal common bounding box

    """

    msg = 'Function bbox_intersection must take at least 2 arguments.'
    verify(len(args) > 1, msg)

    result = [-180, -90, 180, 90]
    for a in args:
        if a is None:
            continue

        msg = ('Bounding box expected to be a list of the '
               'form [W, S, E, N]. '
               'Instead i got "%s"' % str(a))

        try:
            box = list(a)
        except:
            raise Exception(msg)

        if not len(box) == 4:
            raise BoundingBoxError(msg)

        msg = ('Western boundary must be less than or equal to eastern. '
               'I got %s' % box)
        if not box[0] <= box[2]:
            raise BoundingBoxError(msg)

        msg = ('Southern boundary must be less than or equal to northern. '
               'I got %s' % box)
        if not box[1] <= box[3]:
            raise BoundingBoxError(msg)

        # Compute intersection

        # West and South
        for i in [0, 1]:
            result[i] = max(result[i], box[i])

        # East and North
        for i in [2, 3]:
            result[i] = min(result[i], box[i])

    # Check validity and return
    if result[0] <= result[2] and result[1] <= result[3]:
        return result
    else:
        return None


def minimal_bounding_box(bbox, min_res, eps=1.0e-6):
    """Grow bounding box to exceed specified resolution if needed

    Args:
        * bbox: Bounding box with format [W, S, E, N]
        * min_res: Minimal acceptable resolution to exceed
        * eps: Optional tolerance that will be applied to 'buffer' result

    Returns:
        * Adjusted bounding box guaranteed to exceed specified resolution
    """

    # FIXME (Ole): Probably obsolete now

    bbox = copy.copy(list(bbox))

    delta_x = bbox[2] - bbox[0]
    delta_y = bbox[3] - bbox[1]

    if delta_x < min_res:
        dx = (min_res - delta_x) / 2 + eps
        bbox[0] -= dx
        bbox[2] += dx

    if delta_y < min_res:
        dy = (min_res - delta_y) / 2 + eps
        bbox[1] -= dy
        bbox[3] += dy

    return bbox


def buffered_bounding_box(bbox, resolution):
    """Grow bounding box with one unit of resolution in each direction

    Note:
        This will ensure there is enough pixels to robustly provide
        interpolated values without having to painstakingly deal with
        all corner cases such as 1 x 1, 1 x 2 and 2 x 1 arrays.

        The border will also make sure that points that would otherwise fall
        outside the domain (as defined by a tight bounding box) get assigned
        values.

    Args:
        * bbox: Bounding box with format [W, S, E, N]
        * resolution: (resx, resy) - Raster resolution in each direction.
                      res - Raster resolution in either direction
                      If resolution is None bbox is returned unchanged.

    Returns:
        * Adjusted bounding box

    Note:
        Case in point: Interpolation point O would fall outside this domain
                       even though there are enough grid points to support it

        --------------
        |            |
        |   *     *  | *    *
        |           O|
        |            |
        |   *     *  | *    *
        --------------
    """

    bbox = copy.copy(list(bbox))

    if resolution is None:
        return bbox

    try:
        resx, resy = resolution
    except TypeError:
        resx = resy = resolution

    bbox[0] -= resx
    bbox[1] -= resy
    bbox[2] += resx
    bbox[3] += resy

    return bbox


def get_geometry_type(geometry, geometry_type):
    """Determine geometry type based on data

    Args:
        * geometry: A list of either point coordinates [lon, lat] or polygons
                    which are assumed to be numpy arrays of coordinates
        * geometry_type: Optional type - 'point', 'line', 'polygon' or None

    Returns:
        * geometry_type: Either ogr.wkbPoint, ogr.wkbLineString or
                         ogr.wkbPolygon

    Note:
        If geometry type cannot be determined an Exception is raised.

        There is no consistency check across all entries of the
        geometry list, only the first element is used in this determination.
    """

    # FIXME (Ole): Perhaps use OGR's own symbols
    msg = ('Argument geometry_type must be either "point", "line", '
           '"polygon" or None')
    verify(geometry_type is None or
           geometry_type in [1, 2, 3] or
           geometry_type.lower() in ['point', 'line', 'polygon'], msg)

    if geometry_type is not None:
        if isinstance(geometry_type, basestring):
            return INVERSE_GEOMETRY_TYPE_MAP[geometry_type.lower()]
        else:
            return geometry_type
        # FIXME (Ole): Should add some additional checks to see if choice
        #              makes sense

    msg = 'Argument geometry must be a sequence. I got %s ' % type(geometry)
    verify(is_sequence(geometry), msg)

    if len(geometry) == 0:
        # Default to point if there is no data
        return ogr.wkbPoint

    msg = ('The first element in geometry must be a sequence of length > 2. '
           'I got %s ' % str(geometry[0]))
    verify(is_sequence(geometry[0]), msg)
    verify(len(geometry[0]) >= 2, msg)

    if len(geometry[0]) == 2:
        try:
            float(geometry[0][0])
            float(geometry[0][1])
        except (ValueError, TypeError, IndexError):
            pass
        else:
            # This geometry appears to be point data
            geometry_type = ogr.wkbPoint
    elif len(geometry[0]) > 2:
        try:
            x = numpy.array(geometry[0])
        except ValueError:
            pass
        else:
            # This geometry appears to be polygon data
            if x.shape[0] > 2 and x.shape[1] == 2:
                geometry_type = ogr.wkbPolygon

    if geometry_type is None:
        msg = 'Could not determine geometry type'
        raise Exception(msg)

    return geometry_type


def is_sequence(x):
    """Determine if x behaves like a true sequence but not a string

    Note:
        This will for example return True for lists, tuples and numpy arrays
        but False for strings and dictionaries.
    """

    if isinstance(x, basestring):
        return False

    try:
        list(x)
    except TypeError:
        return False
    else:
        return True


def array2line(A, geometry_type=ogr.wkbLinearRing):
    """Convert coordinates to linear_ring

    Args:
        * A: Nx2 Array of coordinates representing either a polygon or a line.
             A can be either a numpy array or a list of coordinates.
        * geometry_type: A valid OGR geometry type. Default: ogr.wkbLinearRing
             Other options include ogr.wkbLineString

    Returns:
        * ring: OGR line geometry

    Note:
    Based on http://www.packtpub.com/article/working-geospatial-data-python
    """

    try:
        A = ensure_numeric(A, numpy.float)
    except Exception, e:
        msg = ('Array (%s) could not be converted to numeric array. '
               'I got type %s. Error message: %s'
               % (A, str(type(A)), e))
        raise Exception(msg)

    msg = 'Array must be a 2d array of vertices. I got %s' % (str(A.shape))
    verify(len(A.shape) == 2, msg)

    msg = 'A array must have two columns. I got %s' % (str(A.shape[0]))
    verify(A.shape[1] == 2, msg)

    N = A.shape[0]  # Number of vertices

    line = ogr.Geometry(geometry_type)
    for i in range(N):
        line.AddPoint(A[i, 0], A[i, 1])

    return line


def rings_equal(x, y, rtol=1.0e-6, atol=1.0e-8):
    """Compares to linear rings as numpy arrays

    Args
        * x, y: Nx2 numpy arrays

    Returns:
        * True if x == y or x' == y (up to the specified tolerance)

        where x' is x reversed in the first dimension. This corresponds to
        linear rings being seen as equal irrespective of whether they are
        organised in clock wise or counter clock wise order
    """

    x = ensure_numeric(x, numpy.float)
    y = ensure_numeric(y, numpy.float)

    msg = 'Arrays must a 2d arrays of vertices. I got %s and %s' % (x, y)
    verify(len(x.shape) == 2 and len(y.shape) == 2, msg)

    msg = 'Arrays must have two columns. I got %s and %s' % (x, y)
    verify(x.shape[1] == 2 and y.shape[1] == 2, msg)

    if (numpy.allclose(x, y, rtol=rtol, atol=atol) or
        numpy.allclose(x, y[::-1], rtol=rtol, atol=atol)):
        return True
    else:
        return False


# FIXME (Ole): We can retire this messy function now
#              Positive: Delete it :-)
def array2wkt(A, geom_type='POLYGON'):
    """Convert coordinates to wkt format

    Args:
        * A: Nx2 Array of coordinates representing either a polygon or a line.
             A can be either a numpy array or a list of coordinates.
        * geom_type: Determines output keyword 'POLYGON' or 'LINESTRING'

    Returns:
        * wkt: geometry in the format known to ogr: Examples

    Note:
        POLYGON((1020 1030,1020 1045,1050 1045,1050 1030,1020 1030))
        LINESTRING(1000 1000, 1100 1050)

    """

    try:
        A = ensure_numeric(A, numpy.float)
    except Exception, e:
        msg = ('Array (%s) could not be converted to numeric array. '
               'I got type %s. Error message: %s'
               % (geom_type, str(type(A)), e))
        raise Exception(msg)

    msg = 'Array must be a 2d array of vertices. I got %s' % (str(A.shape))
    verify(len(A.shape) == 2, msg)

    msg = 'A array must have two columns. I got %s' % (str(A.shape[0]))
    verify(A.shape[1] == 2, msg)

    if geom_type == 'LINESTRING':
        # One bracket
        n = 1
    elif geom_type == 'POLYGON':
        # Two brackets (tsk tsk)
        n = 2
    else:
        msg = 'Unknown geom_type: %s' % geom_type
        raise Exception(msg)

    wkt_string = geom_type + '(' * n

    N = len(A)
    for i in range(N):
        # Works for both lists and arrays
        wkt_string += '%f %f, ' % tuple(A[i])

    return wkt_string[:-2] + ')' * n

# Map of ogr numerical geometry types to their textual representation
# FIXME (Ole): Some of them don't exist, even though they show up
# when doing dir(ogr) - Why?:
geometry_type_map = {ogr.wkbPoint: 'Point',
                     ogr.wkbPoint25D: 'Point25D',
                     ogr.wkbPolygon: 'Polygon',
                     ogr.wkbPolygon25D: 'Polygon25D',
                     #ogr.wkbLinePoint: 'LinePoint',  # ??
                     ogr.wkbGeometryCollection: 'GeometryCollection',
                     ogr.wkbGeometryCollection25D: 'GeometryCollection25D',
                     ogr.wkbLineString: 'LineString',
                     ogr.wkbLineString25D: 'LineString25D',
                     ogr.wkbLinearRing: 'LinearRing',
                     ogr.wkbMultiLineString: 'MultiLineString',
                     ogr.wkbMultiLineString25D: 'MultiLineString25D',
                     ogr.wkbMultiPoint: 'MultiPoint',
                     ogr.wkbMultiPoint25D: 'MultiPoint25D',
                     ogr.wkbMultiPolygon: 'MultiPolygon',
                     ogr.wkbMultiPolygon25D: 'MultiPolygon25D',
                     ogr.wkbNDR: 'NDR',
                     ogr.wkbNone: 'None',
                     ogr.wkbUnknown: 'Unknown'}


def geometrytype2string(g_type):
    """Provides string representation of numeric geometry types

    FIXME (Ole): I can't find anything like this in ORG. Why?
    """

    if g_type in geometry_type_map:
        return geometry_type_map[g_type]
    elif g_type is None:
        return 'No geometry type assigned'
    else:
        return 'Unknown geometry type: %s' % str(g_type)


# FIXME: Move to common numerics area along with polygon.py
def calculate_polygon_area(polygon, signed=False):
    """Calculate the signed area of non-self-intersecting polygon

    Args:
        * polygon: Numeric array of points (longitude, latitude). It is assumed
                   to be closed, i.e. first and last points are identical
        * signed: Optional flag deciding whether returned area retains its
                    sign:
                  If points are ordered counter clockwise, the signed area
                  will be positive.
                  If points are ordered clockwise, it will be negative
                  Default is False which means that the area is always
                  positive.

    Returns:
        * area: Area of polygon (subject to the value of argument signed)

    Note:
        Sources
            http://paulbourke.net/geometry/polyarea/
            http://en.wikipedia.org/wiki/Centroid
    """

    # Make sure it is numeric
    P = numpy.array(polygon)

    msg = ('Polygon is assumed to consist of coordinate pairs. '
           'I got second dimension %i instead of 2' % P.shape[1])
    verify(P.shape[1] == 2, msg)

    x = P[:, 0]
    y = P[:, 1]

    # Calculate 0.5 sum_{i=0}^{N-1} (x_i y_{i+1} - x_{i+1} y_i)
    a = x[:-1] * y[1:]
    b = y[:-1] * x[1:]

    A = numpy.sum(a - b) / 2.

    if signed:
        return A
    else:
        return abs(A)


def calculate_polygon_centroid(polygon):
    """Calculate the centroid of non-self-intersecting polygon

    Args:
        * polygon: Numeric array of points (longitude, latitude). It is assumed
                 to be closed, i.e. first and last points are identical

    Note:
        Sources
            http://paulbourke.net/geometry/polyarea/
            http://en.wikipedia.org/wiki/Centroid
    """

    # Make sure it is numeric
    P = numpy.array(polygon)

    # Normalise to ensure numerical accurracy.
    # This requirement in backed by tests in test_io.py and without it
    # centroids at building footprint level may get shifted outside the
    # polygon!
    P_origin = numpy.amin(P, axis=0)
    P = P - P_origin

    # Get area. This calculation could be incorporated to save time
    # if necessary as the two formulas are very similar.
    A = calculate_polygon_area(polygon, signed=True)

    x = P[:, 0]
    y = P[:, 1]

    # Calculate
    # Cx = sum_{i=0}^{N-1} (x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)/(6A)
    # Cy = sum_{i=0}^{N-1} (y_i + y_{i+1})(x_i y_{i+1} - x_{i+1} y_i)/(6A)
    a = x[:-1] * y[1:]
    b = y[:-1] * x[1:]

    cx = x[:-1] + x[1:]
    cy = y[:-1] + y[1:]

    Cx = numpy.sum(cx * (a - b)) / (6. * A)
    Cy = numpy.sum(cy * (a - b)) / (6. * A)

    # Translate back to real location
    C = numpy.array([Cx, Cy]) + P_origin
    return C


def points_between_points(point1, point2, delta):
    """Creates an array of points between two points given a delta

    Note:
       u = (x1-x0, y1-y0)/L, where
       L=sqrt( (x1-x0)^2 + (y1-y0)^2).
       If r is the resolution, then the
       points will be given by
       (x0, y0) + u * n * r for n = 1, 2, ....
       while len(n*u*r) < L
    """
    x0, y0 = point1
    x1, y1 = point2
    L = math.sqrt(math.pow((x1 - x0), 2) + math.pow((y1 - y0), 2))
    pieces = int(L / delta)
    uu = numpy.array([x1 - x0, y1 - y0]) / L
    points = [point1]
    for nn in range(pieces):
        point = point1 + uu * (nn + 1) * delta
        points.append(point)
    return numpy.array(points)


def points_along_line(line, delta):
    """Calculate a list of points along a line with a given delta

    Args:
        * line: Numeric array of points (longitude, latitude).
        * delta: Decimal number to be used as step

    Returns:
        * V: Numeric array of points (longitude, latitude).

    Note:
        Sources
            http://paulbourke.net/geometry/polyarea/
            http://en.wikipedia.org/wiki/Centroid
    """

    # Make sure it is numeric
    P = numpy.array(line)
    points = []
    for i in range(len(P) - 1):
        pts = points_between_points(P[i], P[i + 1], delta)
        # If the first point of this list is the same
        # as the last one recorded, do not use it
        if len(points) > 0:
            if numpy.allclose(points[-1], pts[0]):
                pts = pts[1:]
        points.extend(pts)
    C = numpy.array(points)
    return C


def combine_polygon_and_point_layers(layers):
    """Combine polygon and point layers

    Args:
        layers: List of vector layers of type polygon or point

    Returns:
        One point layer with all input point layers and centroids from all
        input polygon layers.
    Raises:
        IneSAFEError in case a0ttribute names are not the same.
    """

    # This is to implement issue #276
    print layers


def get_ringdata(ring):
    """Extract coordinates from OGR ring object

    Args
        * OGR ring object
    Returns
        * Nx2 numpy array of vertex coordinates (lon, lat)
    """

    N = ring.GetPointCount()
    A = numpy.zeros((N, 2), dtype='d')

    # FIXME (Ole): Is there any way to get the entire data vectors?
    for j in range(N):
        A[j, :] = ring.GetX(j), ring.GetY(j)

    # Return ring as an Nx2 numpy array
    return A


def get_polygondata(G):
    """Extract polygon data from OGR geometry

    :param G: OGR polygon geometry
    :return: List of InaSAFE polygon instances
    """

    # Get outer ring, then inner rings
    # http://osgeo-org.1560.n6.nabble.com/
    # gdal-dev-Polygon-topology-td3745761.html
    number_of_rings = G.GetGeometryCount()

    # Get outer ring
    outer_ring = get_ringdata(G.GetGeometryRef(0))

    # Get inner rings if any
    inner_rings = []
    if number_of_rings > 1:
        for i in range(1, number_of_rings):
            inner_ring = get_ringdata(G.GetGeometryRef(i))
            inner_rings.append(inner_ring)

    # Return Polygon instance
    return Polygon(outer_ring=outer_ring,
                   inner_rings=inner_rings)

# From common.utilities



def verify(statement, message=None):
    """Verification of logical statement similar to assertions
    Input
        statement: expression
        message: error message in case statement evaluates as False

    Output
        None
    Raises
        VerificationError in case statement evaluates to False
    """

    if bool(statement) is False:
        raise VerificationError(message)


def ugettext(s):
    """Translation support
    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..', 'i18n'))
    if 'LANG' not in os.environ:
        return s
    lang = os.environ['LANG']
    filename_prefix = 'inasafe'
    t = gettext.translation(filename_prefix,
                            path, languages=[lang], fallback=True)
    return t.ugettext(s)


def temp_dir(sub_dir='work'):
    """Obtain the temporary working directory for the operating system.

    An inasafe subdirectory will automatically be created under this and
    if specified, a user subdirectory under that.

    .. note:: You can use this together with unique_filename to create
       a file in a temporary directory under the inasafe workspace. e.g.

       tmpdir = temp_dir('testing')
       tmpfile = unique_filename(dir=tmpdir)
       print tmpfile
       /tmp/inasafe/23-08-2012/timlinux/testing/tmpMRpF_C

    If you specify INASAFE_WORK_DIR as an environment var, it will be
    used in preference to the system temp directory.

    Args:
        sub_dir str - optional argument which will cause an additional
                subirectory to be created e.g. /tmp/inasafe/foo/

    Returns:
        Path to the output clipped layer (placed in the system temp dir).

    Raises:
       Any errors from the underlying system calls.
    """
    user = getpass.getuser().replace(' ', '_')
    current_date = date.today()
    date_string = current_date.isoformat()
    if 'INASAFE_WORK_DIR' in os.environ:
        new_directory = os.environ['INASAFE_WORK_DIR']
    else:
        # Following 4 lines are a workaround for tempfile.tempdir()
        # unreliabilty
        handle, filename = mkstemp()
        os.close(handle)
        new_directory = os.path.dirname(filename)
        os.remove(filename)

    path = os.path.join(new_directory, 'inasafe', date_string, user, sub_dir)

    if not os.path.exists(path):
        # Ensure that the dir is world writable
        # Umask sets the new mask and returns the old
        old_mask = os.umask(0000)
        os.makedirs(path, 0777)
        # Resinstate the old mask for tmp
        os.umask(old_mask)
    return path


def unique_filename(**kwargs):
    """Create new filename guaranteed not to exist previously

    Use mkstemp to create the file, then remove it and return the name

    If dir is specified, the tempfile will be created in the path specified
    otherwise the file will be created in a directory following this scheme:

    :file:`/tmp/inasafe/<dd-mm-yyyy>/<user>/impacts'

    See http://docs.python.org/library/tempfile.html for details.

    Example usage:

    tempdir = temp_dir(sub_dir='test')
    filename = unique_filename(suffix='.keywords', dir=tempdir)
    print filename
    /tmp/inasafe/23-08-2012/timlinux/test/tmpyeO5VR.keywords

    Or with no preferred subdir, a default subdir of 'impacts' is used:

    filename = unique_filename(suffix='.shp')
    print filename
    /tmp/inasafe/23-08-2012/timlinux/impacts/tmpoOAmOi.shp

    """

    if 'dir' not in kwargs:
        path = temp_dir('impacts')
        kwargs['dir'] = path
    else:
        path = temp_dir(kwargs['dir'])
        kwargs['dir'] = path
    if not os.path.exists(kwargs['dir']):
        # Ensure that the dir mask won't conflict with the mode
        # Umask sets the new mask and returns the old
        umask = os.umask(0000)
        # Ensure that the dir is world writable by explictly setting mode
        os.makedirs(kwargs['dir'], 0777)
        # Reinstate the old mask for tmp dir
        os.umask(umask)
    # Now we have the working dir set up go on and return the filename
    handle, filename = mkstemp(**kwargs)

    # Need to close it using the filehandle first for windows!
    os.close(handle)
    try:
        os.remove(filename)
    except OSError:
        pass
    return filename

def combine_coordinates(x, y):
    """Make list of all combinations of points for x and y coordinates
    """

    return axes2points(x, y)


