""" Create vector and raster data from Python/numpy structures
"""

import numpy
from source.vector import Vector
from source.projection import Projection, DEFAULT_PROJECTION

# Create a few point features
points = [[149.158426, -35.343510], # Geoscience Australia,
          [149.122429, -35.276537], # Australian Nat'l University
          [72.905205, 19.119746],   # IIT Mumbay,
          [12.522321, 55.784079],   # Technical University of Denmark
          [106.822085, -6.185568],  # Menara Thamrin
          [117.0776, -20.45]]       # Delambre Island

attributes = [{'Name': 'Geoscience Australia',
               'Country': 'Australia',
               'People': 700},
              {'Name': 'Australian National University',
               'Country': 'Australia',
               'People': 5000},
              {'Name': 'Indian Institute of Technology, Mumbay',
               'Country': 'India',
               'People': 10000},
              {'Name': 'Technical University of Denmark',
               'Country': 'Denmark',
               'People': 8000},
              {'Name': 'Australia-Indonesia Facility for Disaster Reduction',
               'Country': 'Indonesia',
               'People': 20},
              {'Name': 'Delambre Island',
               'Country': 'Australia',
               'People': 7}]

# Convert into a vector layer
filename = 'points_of_interest.kml'
V = Vector(geometry=points, data=attributes)
V.write_to_file(filename)

print 'Wrote point data to file.'
print 'To view and analyse:'
print 'qgis %s' % filename[:-4] + '.shp'



