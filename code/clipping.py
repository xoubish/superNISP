import os
import sys
import gzip
from io import BytesIO
from astropy.io import fits
import matplotlib.pyplot as plt
import dropbox
import time
import glob
import warnings
import numpy as np
import pandas as pd
import re

from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import FITSFixedWarning
from astropy import units as u
from scipy import ndimage
from skimage import restoration

dbx_url = 'https://www.dropbox.com/scl/fo/37ooho4m924wb3d2m1gt8/ABJjd8gNUl0h_rmUP41S3cI?rlkey=nqy0t7p9sgxa3a853bf1ris9l&st=clxv8yui&dl=0'

warnings.simplefilter('ignore', FITSFixedWarning)

###########################################
########### Helper functions ##############
###########################################

def pixel_scale(wcs):
    # Convert one arcsecond to pixels
    start = wcs.pixel_to_world(0,0)
    sep = 1*u.arcsecond
    end = start.directional_offset_by(0*u.arcsecond, sep)
    x0, x1 = wcs.world_to_pixel(start), wcs.world_to_pixel(end)
    x0, x1 = np.array(x0).flatten(), np.array(x1).flatten()
    hyp = x0+x1
    return 1/np.sqrt(np.dot(hyp, hyp)) # arcsec / pixel


def psf_filename(filename):
    tile_id = re.search("TILE([0-9]{9})", filename)[1]
    cat_file = glob.glob(f"../data/*/EUC_MER_CATALOG-PSF-NIR-Y_TILE{tile_id}*.fits")
    if len(cat_file)==0: raise ValueError(f"No matches for TILE ID {tile_id}")
    elif len(cat_file)>1: raise ValueError(f"Too many files match TILE ID {tile_id}")

    return cat_file[0]

def cut_catalog(cat_file, cuts=None):
    # Open cat file
    with fits.open(cat_file) as hdul:
        cat_data = hdul[1].data  # Adjust the HDU index if your data is not in the first extension
    
    # Do cuts on catalog data
    if cuts is None:
        cuts = (cat_data.lp_type==0) & \
                (cat_data.ACS_F814W_MAG < 25) & \
                (cat_data.ez_z_phot > 0.01) & (cat_data.ez_z_phot < 3.0) & \
                (cat_data.FLUX_RADIUS < 104) # TODO: make this a function with J&H bands?
    
    cat_clipped = cat_data[cuts]
    
    my_cat = pd.DataFrame({'id_classic': cat_clipped['ID'].astype(int),
                           'ra': cat_clipped['ALPHA_J2000'].astype(float),
                           'dec': cat_clipped['DELTA_J2000'].astype(float),
                           'jwst_image': "",
                           'nisp_image': "",
                          })
    
    return my_cat

def cut_catalog2(cat_file, cuts=None):

    cat_data = Table.read(cat_file).to_pandas()
    
    cat_data['ACS_F814W_MAG'] = 23.9-2.5*np.log10(cat_data.ACS_F814W_FLUX)
    cat_clipped = cat_data[
            (cat_data.z_best > 0.01) & (cat_data.z_best < 3.0)
            & (cat_data.ACS_F814W_MAG < 25)
            & (cat_data.FLUX_RADIUS_2_F814W < 50) & (cat_data.FLUX_RADIUS_2_F814W > 3) # Can be flexible with this (remove or make much larger)
            & (cat_data.CLASS_STAR < 0.1)
    ]

    my_cat = pd.DataFrame({
        'id': cat_clipped['ID'],
        'ra': cat_clipped['RA_1'],
        'dec': cat_clipped['DEC_1'],
        'jwst_image': '',
        'nisp_image': '',
    })

    my_cat = my_cat.dropna().reset_index(drop=True)
    return my_cat


##########################################
########### Dropbox Functions ############
##########################################


def get_shared_folder_metadata(url, dbx, path=""):
    """Get metadata of the shared folder."""
    shared_link = dropbox.files.SharedLink(url=url)
    try:
        folder_metadata = dbx.files_list_folder(path=f'{path}', shared_link=shared_link)
        return folder_metadata.entries
    except dropbox.exceptions.ApiError as e:
        print(f"Error accessing shared folder: {e}")
        return []

def get_fits_file(url, file_name, dbx, hdu_idx=0):
    tries=5
    while tries >= 0:
        tries -= 1
        try:
            meta, res = dbx.sharing_get_shared_link_file(url, path=file_name)
            break
        except Exception as e:
            print(f"Connection error: {e}")
            time.sleep(10)
            continue

    try: res
    except: raise RuntimeError(f"Could not access file {file_name} at URL {url}.")

    # Unzip file, if necessary
    if '.gz' in file_name:
        with gzip.GzipFile(fileobj=BytesIO(res.content), mode='rb') as f_in:
            decompressed_data = f_in.read()
    else: decompressed_data = res.content

    fits_file = BytesIO(decompressed_data)
    hdul = fits.open(fits_file)

    # Access image data
    image_data = hdul[hdu_idx].data
    image_header = hdul[hdu_idx].header
    hdul.close()
    return image_header, image_data

def match_catalog(file_name, gal_coords, hdu_idx=0, url=None, dbx=None):
    if url is None or dbx is None: # get images locally
        if not os.path.exists(file_name):
            raise ValueError(f"File name '{file_name}' does not exist. " \
                                "Provide URL and dbx instance for dropbox usage.")
        with fits.open(file_name) as hdul:
            image_header = hdul[hdu_idx].header
            image_data = hdul[hdu_idx].data
    else: # Get FITS file from dropbox
        image_header, image_data = get_fits_file(url, file_name, dbx=dbx, 
                                                 hdu_idx=hdu_idx)
    
    wcs = WCS(image_header)
    
    return np.where(gal_coords.contained_by(wcs))[0]

def clip_images(catalog, url=None, dbx=None, size_jwst=69, size_nisp=41, pad=20, 
                rot_jwst=-20.0, limit=None, jwst_hdu=0, nisp_hdu=0, deconvolve=True,
                mirror_euclid=False,
               ):
    ### The catalog must have matched ra, dec, jwst_image, and nisp_image columns
    image_pairs = catalog.groupby(['jwst_image', 'nisp_image'])

    clips = []
    count = 0
    for match, cat in image_pairs:
        jwst_file, nisp_file = match

        # Open JWST and Euclid images
        if dbx is None or url is None: # Open from file
            
            # Check files exist
            if not os.path.exists(jwst_file):
                raise ValueError(f"File name '{jwst_file}' does not exist. " \
                                "Provide URL and dbx instance for dropbox usage.")
            if not os.path.exists(nisp_file):
                raise ValueError(f"File name '{nisp_file}' does not exist. " \
                                "Provide URL and dbx instance for dropbox usage.")

            # Open JWST and NISP files from disk
            with fits.open(jwst_file) as hdul:
                jwst_header = hdul[jwst_hdu].header
                jwst_data = hdul[jwst_hdu].data
            with fits.open(nisp_file) as hdul:
                nisp_header = hdul[nisp_hdu].header
                nisp_data = hdul[nisp_hdu].data
                
        else: # Open from dropbox
            print('Fetching files from dropbox...')
            jwst_header, jwst_data = get_fits_file(url, jwst_file, dbx=dbx, hdu_idx=jwst_hdu)
            nisp_header, nisp_data = get_fits_file(url, nisp_file, dbx=dbx, hdu_idx=nisp_hdu)

        if deconvolve:
            psf_file = psf_filename(nisp_file)
            psf = mpsf.from_file(psf_file)
        
        wcs_jwst, wcs_nisp = WCS(jwst_header), WCS(nisp_header)
        
        # Get the coordinates of matched galaxies
        gal_coords = SkyCoord(cat.ra, cat.dec, unit='deg')
        
        for i in range(len(cat)):
            gal = cat.iloc[i]
            # Get JWST clip (larger than final size)
            clip_jwst = Cutout2D(jwst_data, gal_coords[i], size=size_jwst+pad, wcs=wcs_jwst, mode='trim')
            if sum(clip_jwst.data.shape)!=(size_jwst+pad)*2: continue
            if np.sum((clip_jwst.data==0.0).astype(int))/((size_jwst+pad)**2) > 0.25: continue
            
            # Get NISP clip
            clip_nisp = Cutout2D(nisp_data, gal_coords[i], size=size_nisp, wcs=wcs_nisp, mode='trim')
            if sum(clip_nisp.data.shape)!=(size_nisp)*2: continue
            if np.sum((clip_nisp.data==0.0).astype(int))/((size_nisp)**2) > 0.25: continue
            
            if deconvolve:
                psf_clip = psf.get_closest_stamp_at_radec([gal_coords[i].ra.degree, 
                                                            gal_coords[i].dec.degree])
                psf_clip.normalize()
                psf_data = psf_clip.get_data()
            
            # Rotate JWST image 20 degrees counter-clockwise and crop
            # This loses the WCS
            if rot_jwst != 0:
                clip_jwst = rotate_jwst(clip_jwst, size=size_jwst, angle=rot_jwst)
                # clip_jwst.wcs = rotate_wcs(clip_jwst, angle=rot_jwst)
            # clip_jwst.data = ABmag_jwst(clip_jwst, jwst_header['PIXAR_SR'])
            
            clips.append((gal_coords[i], clip_jwst, clip_nisp))
            count += 1
            if limit is not None and count >= limit: break
        if limit is not None and count >= limit: break

    return clips


############################################
########### Processing pipeline ############
############################################


def process_all(field='cosmos', euclid_type='NISP-Y_MER', save_cat=False, save_clips=False, redo_cat=False,
                redo_clips=False, secret="../../secrets/dropbox_token", meta=meta,
                deconvolve=False):
    params = meta[field][euclid_type]

    # Try to load clip files from disk
    if (os.path.exists(params['matched_jwst']) and os.path.exists(params['matched_nisp']) 
            and not redo_clips and not redo_cat):
        print("Matched clip files exist; not re-running")
        jwst_cutouts = np.load(params['matched_jwst'])
        nisp_cutouts = np.load(params['matched_nisp'])
        return jwst_cutouts, nisp_cutouts
    
    # Instantiate dropbox token
    with open(secret) as token_file:
        token = token_file.read()
        dbx = dropbox.Dropbox(token.strip(), timeout=None)

    # Get JWST files for specific field
    jwst_path = f'/JWST/{field}/'
    jwst_files = get_shared_folder_metadata(dbx_url, dbx=dbx, path=jwst_path)
    jwst_files = [jwst_path+file.name for file in jwst_files]

    # Get NISP files for specific field
    nisp_path = f'/{euclid_type}/{field}/'
    nisp_files = get_shared_folder_metadata(dbx_url, dbx=dbx, path=nisp_path)
    nisp_files = [nisp_path+file.name for file in nisp_files if 'IMAGE' in file.name]

    # Try to load cat file from disk
    if os.path.exists(params['matched_cat']) and not redo_cat:
        print(f"Matched catalog file {params['matched_cat']} exists; not re-running.")
        my_cat = pd.read_csv(params['matched_cat'])
    else:
        # Apply cuts to the catalog
        my_cat = params['cut_func'](params['cat_path'])
        gal_coords = SkyCoord(my_cat.ra, my_cat.dec, unit='deg')
    
        # Match JWST files
        for file in jwst_files:
            found_idxs = match_catalog(file, gal_coords, url=dbx_url, dbx=dbx, 
                                       hdu_idx=params['jwst_hdu'])
            my_cat.loc[found_idxs, 'jwst_image'] = str(file)
    
        # Match NISP files
        for file in nisp_files:
            found_idxs = match_catalog(file, gal_coords, url=dbx_url, dbx=dbx,
                                       hdu_idx=params['nisp_hdu'])
            my_cat.loc[found_idxs, 'nisp_image'] = str(file)
    
        my_cat = my_cat[my_cat.nisp_image!='']
        my_cat = my_cat[my_cat.jwst_image!='']
    
        if save_cat:
            my_cat.to_csv(params['matched_cat'], index=False)
    
    # Clip images
    clips = clip_images(my_cat, url=dbx_url, dbx=dbx, jwst_hdu=params['jwst_hdu'], 
                        size_jwst=params['size_jwst'], rot_jwst=params['rot_jwst'], 
                        size_nisp=params['size_nisp'], pad=params['pad_jwst'],
                        nisp_hdu=params['nisp_hdu'], deconvolve=deconvolve)
    
    # Arrange data
    jwst_cutouts = np.array([clip[1].data for clip in clips])

    # if euclid_type=='NISP-Y':
    #     for i in range(len(clips)):
    #         clips[i][2] = mirror_cutout_along_y(clips[i][2])
    nisp_cutouts = np.array([clip[2].data for clip in clips])
    
    if save_clips:
        np.save(params['matched_jwst'], jwst_cutouts, allow_pickle=False)
        np.save(params['matched_nisp'], nisp_cutouts, allow_pickle=False)
    
    return jwst_cutouts, nisp_cutouts
