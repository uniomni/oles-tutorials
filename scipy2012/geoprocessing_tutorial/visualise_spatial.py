import matplotlib
#matplotlib.use('Cairo')
from osgeo import gdal, ogr
import numpy, pylab
from shapely.wkb import loads

# Raster
r = gdal.Open('Ashload_Gede_VEI4_geographic.asc')
raster_layer = r.ReadAsArray()
pylab.imshow(raster_layer) #, vmin=-0.2, vmax=0.8, interpolation='nearest')
pylab.colorbar()

# Vector
v = ogr.Open('OSM_building_polygons_20110905.shp')
print dir(v)
vector_layer = v.GetLayerByIndex(0)


pylab.show()
#pylab.savefig('map.pdf')
