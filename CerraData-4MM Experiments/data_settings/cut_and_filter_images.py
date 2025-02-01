"""
INSTITUTO NACIONAL DE PESQUISAS ESPACIAIS
COMPUTACAO APLICADA

CODE: DATA MANAGER
AUTHOR: MATEUS DE SOUZA MIRANDA, 2022

"""

# -------- LIBRARY
# Directory
import os
import glob
import shutil

# Geospatial
from osgeo import gdal

# Data
import numpy as np

# -------- CROP THE IMAGE

# Variable
clip_count = 0

# Loop for clipping
for raster in glob.iglob('../datasets/raw/ecorregions_isoladas/mask_parrot_beak_10classes.tif'):
    file = gdal.Open(raster)
    file_gt = file.GetGeoTransform()

    print(raster)
    # Get coordinates of upper left corner
    xmin = file_gt[0]
    ymax = file_gt[3]
    res = file_gt[1]

    # Determine total length of refData
    xlen = res * file.RasterXSize #20
    ylen = res * file.RasterYSize #20

    # Number of tiles in x and y direction
    height = file.RasterXSize
    width = file.RasterYSize

    # Obtaining the exact value for measurement of the patches
    xdiv = int(height / 128.0341463414634)
    ydiv = int(width / 128.24802110817942)

    print('X refData ', file.RasterXSize)
    print('Y refData ', file.RasterYSize)
    print('X clip ', file.RasterXSize / xdiv)
    print('Y clip ', file.RasterYSize / ydiv)
    
    # size of a single tile
    xsize = xlen / xdiv
    ysize = ylen / ydiv

    # create lists of x and y coordinates
    xsteps = [xmin + xsize * i for i in range(xdiv + 1)]
    ysteps = [ymax - ysize * i for i in range(ydiv + 1)]

    # Creating directories
    name_file = os.path.basename(raster)
    if os.path.isdir(name_file) == False:
        os.mkdir('../datasets/raw/ecorregions_isoladas/bico_papagaio_masks/' + name_file + '_mask')

    # loop over mininimo and maximo x and y coordinates
    for i in range(xdiv):
        for j in range(ydiv):
            xmin = xsteps[i]
            xmax = xsteps[i + 1]
            ymax = ysteps[j]
            ymin = ysteps[j + 1]

            print(xmin, xmax, ymin, ymax)

            # use gdal warp
            gdal.Warp('../datasets/raw/ecorregions_isoladas/bico_papagaio_masks/' + name_file + '_mask' + '/' + name_file+'_' + str(i) + str(j) + ".tif",
                      file, outputBounds=(xmin, ymin, xmax, ymax), dstNodata=None)

            # or gdal translate to subset the input refData
            gdal.Translate('../datasets/raw/ecorregions_isoladas/bico_papagaio_masks/' + name_file + '_mask' + '/' + name_file+'_' + str(i) + str(j) + ".tif",
                           file, projWin=(xmin, ymax, xmax, ymin), xRes=res, yRes=-res)

    # close the open dataset!!!
    dem = None
    clip_count = clip_count + 1
    #print('- Raster ', clip_count, 'Ok!')




