""" Create vector and raster data from Python/numpy structures

"""

import numpy
from source.raster import Raster
from source.projection import Projection, DEFAULT_PROJECTION

# Create a hypothetical grid
def mandelbrot(h, w, maxit=20):
    """Returns an image of the Mandelbrot fractal of size (h,w).
    From http://www.scipy.org/Tentative_NumPy_Tutorial/Mandelbrot_Set_Example
    """

    y, x = numpy.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x+y*1j
    z = c
    divtime = maxit + numpy.zeros(z.shape, dtype=float)

    for i in xrange(maxit):
        z  = z**2 + c
        diverge = z*numpy.conj(z) > 2**2      # who is diverging
        div_now = diverge & (divtime==maxit)  # who is diverging now
        divtime[div_now] = i                  # note when
        z[diverge] = 2                        # avoid diverging too much

    return divtime

A = mandelbrot(400, 400)
A[A == 0] = numpy.nan

# Generate a georeference for the array.
# Note that the n-s pixel must be negative
# Format:
# (top left x, w-e pixel resolution, rotation,
#  top left y, rotation, n-s pixel resolution).
# See also http://www.gdal.org/gdal_tutorial.html
geotransform = (117.0689671410519, 0.0001, 0,
                -20.43131095982087, 0, -0.0001)

# Exercise: Put the mandelbrot set over your home town and control
# the pixel resolution so that it covers it.

# Convert into a vector layer
filename = 'spatially_referenced_grid.tif'
R = Raster(geotransform=geotransform, data=A)
R.write_to_file(filename)
print 'Wrote raster data to file.'
print 'To view and analyse:'
print 'qgis %s' % filename

