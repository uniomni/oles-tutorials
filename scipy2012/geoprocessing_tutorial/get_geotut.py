"""Get source code, exercieses, slides and data needed for the 
Scipy2012 India Geoprocessing Tutorial.

Ole Nielsen, 2012
"""

import os

repodir = 'http://oles-tutorials.googlecode.com/svn/trunk/scipy2012/geoprocessing_tutorial' 

rootdir = 'geotut'

cmd = 'mkdir %s' % rootdir
os.system(cmd)
os.chdir(rootdir)

for component in ['source', 'exercises', 'spatial_test_data', 'geoprocessing_tut_2012_ole_nielsen.pdf']:
    cmd = 'svn co %s/%s' % (repodir, component)
    print cmd
    os.system(cmd)

