import os
import numpy as np
import shutil
import glob
#from osgeo import gdal
#import rasterio
#from rasterio.merge import merge


#def raster_reading(raster_path):
#    """
#    Reads the geo-tiff files and returns the raster objects.
#
#    Args:
#        raster_path (str): The path to the geo-referenced patch file.
#
#    Returns:
#        tuple: A tuple containing:
#            - raster (rasterio object): The rasterio object.
#            - gdal_raster (gdal.Dataset): The GDAL dataset object.
#    """
#    gdal_raster = gdal.Open(raster_path)
#
#    with rasterio.open(raster_path, 'r') as raster:
#        return raster, gdal_raster



def get_patch_id(file_name, ind):
    """
    Extracts the patch ID from the file name.

    Args:
        file_name (str): The name of the file.

    Returns:
        str: The extracted patch ID.
    """
    patch_name = os.path.basename(file_name)
    print(file_name)
    patch_ids = patch_name.split('_')[ind].split('.tif')[0]
    
    return patch_ids


def move_file(src_path, dest_dir, subdir):
    """
    Moves a file to the specified destination directory.

    Args:
        src_path (str): The source file path.
        dest_dir (str): The base destination directory.
        subdir (str): The subdirectory within the destination directory.
    """
    dest_path = os.path.join(dest_dir, subdir)
    #os.makedirs(dest_path, exist_ok=True)
    shutil.move(src_path, dest_path)



def move_missing_patches(source_dir, dest_dir, missing_dir):
    """
    Move patches from the source directory to the missing directory if they do not exist in the destination directory.

    Args:
        source_dir (str): The path to the source directory containing patches.
        dest_dir (str): The path to the destination directory to check for existing patches.
        missing_dir (str): The path to the directory where missing patches will be moved.
    """
    # List all raster files in the source and destination directories
    source_files = glob.glob(os.path.join(source_dir, '*.tif'))
    dest_files = glob.glob(os.path.join(dest_dir, '*.tif'))

    # Extract patch IDs from source files
    source_ids = {get_patch_id(os.path.basename(f), ind=5): f for f in source_files}
    
    # Extract patch IDs from destination files
    dest_ids = {get_patch_id(os.path.basename(f), ind=3) for f in dest_files}
    

    # Check for missing patches and move them to the missing directory
    for patch_id, source_path in source_ids.items():
        if patch_id not in dest_ids:
            #print(f'Patch {patch_id}')
            move_file(source_path, missing_dir, '')
            print(f'Moved missing patch: {patch_id} from {source_dir} to {missing_dir}')

 

destdir = '../datasets/cerradata4mm_exp/train/sar_images/'
sourcedir = '../datasets/cerradata4mm_exp/train/edge_14c/'
movedir = '../datasets/cerradata4mm_exp/train/move_pasta/ed14/'

move_missing_patches(sourcedir, destdir, movedir)


