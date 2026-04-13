import os
import sys
import gzip
import time
import glob
import re
import warnings
from io import BytesIO
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dropbox

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import FITSFixedWarning
from astropy import units as u
from photutils.psf import resize_psf

from scipy import ndimage
from skimage import restoration

dbx_url = 'https://www.dropbox.com/scl/fo/37ooho4m924wb3d2m1gt8/ABJjd8gNUl0h_rmUP41S3cI?rlkey=nqy0t7p9sgxa3a853bf1ris9l&st=clxv8yui&dl=0'
FRAC_ZERO = 0.25
data_dir = os.path.expandvars("$SCRATCH/data/superNISP")

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


# def psf_filename(filename):
#     tile_id = re.search("TILE([0-9]{9})", filename)[1]
#     cat_file = glob.glob(f"../data/*/EUC_MER_CATALOG-PSF-NIR-Y_TILE{tile_id}*.fits")
#     if len(cat_file)==0: raise ValueError(f"No matches for TILE ID {tile_id}")
#     elif len(cat_file)>1: raise ValueError(f"Too many files match TILE ID {tile_id}")

#     return cat_file[0]

def psf_filename(nisp_filename, psf_files):
    nisp_base = nisp_filename.split('.')[0].replace('IMAGE', 'PSF-I')
    psf_file = [file for file in psf_files if nisp_base in file]
    if len(psf_file)==0:
        raise ValueError(f"Could not find PSF file for {nisp_filename}")
    elif len(psf_file)>1:
        raise ValueError(f"Found too many PSF files for {nisp_filename}")
    return psf_file[0]

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
                           'jwst_image': '',
                           'nisp_image': '',
                           'psf_image': '',
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
        'psf_image': '',
    })

    my_cat = my_cat.dropna().reset_index(drop=True)
    return my_cat

def rotate_jwst(clip, angle=-20, size=69, pad=20):
    # There are different types of interpolation possible for this one, talk about it with Shooby
    wcs = clip.wcs
    new_image = ndimage.rotate(clip.data, angle, reshape=False, order=3, cval=-20)
    new_clip = Cutout2D(new_image, clip.center_cutout, size)
    # new_clip.wcs = wcs
    return new_clip # no WCS info after this


###########################################
############### Metadata ##################
###########################################

meta = {
    'cosmos': {
        'NISP-Y_MER': {
            'cut_func': cut_catalog,
            'jwst_hdu': 0,
            'nisp_hdu': 0,
            'cat_path': '../catalog/COSMOS2020_CLASSIC_R1_v2.2_p3.fits', # COSMOS classic
            'size_jwst': 69, # Change to 41 and 69 px
            'size_nisp': 41,
            'pad_jwst': 20,
            'rot_jwst': -20,
            'psf_hdu': 1,
            'jwst_psf_file': f'{data_dir}/data/PSF_NIRCam_in_flight_opd_filter_F115W.fits',
            'matched_cat': '../catalog/matched_cat_cosmos_1.csv',
            'matched_jwst': f'{data_dir}/data/jwst_cosmos_69px_F115W.npy',
            'matched_nisp': f'{data_dir}/data/euclid_MER_cosmos_41px_Y.npy',
            # 'matched_nisp_psf': '../data/nisp_cosmos_psf.npy',
        },
        'NISP-Y': {
            'cut_func': cut_catalog,
            'jwst_hdu': 0,
            'nisp_hdu': 1,
            'cat_path': '{data_dir}/catalog/COSMOS2020_CLASSIC_R1_v2.2_p3.fits', # COSMOS classic
            'size_jwst': 205,
            'size_nisp': 41,
            'mirror_nisp': True,
            'mask': True,
            'deconvolve': False,
            'pad_jwst': 20,
            'rot_jwst': -3.945,
            'matched_cat': '../catalog/matched_cat_cosmos_2.csv',
            'matched_jwst': f'{data_dir}/data/jwst_cosmos_205px_F115W.npy',
            'matched_nisp': f'{data_dir}/data/euclid_NIR_cosmos_41px_Y.npy',
        },
    },
    'HUDF': {
        'NISP-Y_MER': {
            'cut_func': cut_catalog2,
            'jwst_hdu': 1,
            'nisp_hdu': 0,
            'cat_path': '../catalog/gds.fits', # CANDELS catalog
            'size_jwst': 135, # Change to an odd # of pixels
            'size_nisp': 41,
            'pad_jwst': 0,
            'rot_jwst': 0,
            'psf_hdu': 1,
            'matched_cat': '../catalog/matched_cat_hudf_1.csv',
            'matched_jwst': f'{data_dir}/data/jwst_hudf_135px_F115W.npy',
            'matched_nisp': f'{data_dir}/data/euclid_MER_hudf_41px_Y.npy',
            # 'matched_nisp_psf': '../data/nisp_hudf_psf.npy',
        },
        'NISP-Y': {
            'cut_func': cut_catalog2,
            'jwst_hdu': 1,
            'nisp_hdu': 1,
            'cat_path': '../catalog/gds.fits', # CANDELS catalog
            'size_jwst': 205,
            'size_nisp': 41,
            'pad_jwst': 0,
            'rot_jwst': 0,
            'matched_cat': '../catalog/matched_cat_hudf_2.csv',
            'matched_jwst': f'{data_dir}/data/jwst_hudf_205px_F115W.npy',
            'matched_nisp': f'{data_dir}/data/euclid_NIR_hudf_41px_Y.npy',
        },
    },
}


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
                rot_jwst=-20.0, limit=None, jwst_hdu=0, nisp_hdu=0, deconvolve=False,
                jwst_psf_file=None, mirror_nisp=False, mask=False, **kwargs):
    ### The catalog must have matched ra, dec, jwst_image, and nisp_image columns
    image_pairs = catalog.groupby(['jwst_image', 'nisp_image'])
    ping = True
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
                if mask:
                    mask_data = hdul[nisp_hdu+1].data
                    wcs_mask = WCS(hdul[nisp_hdu+1].header)
            if deconvolve:
                with fits.open(cat.iloc[0].psf_image) as hdul:
                    nisp_psf = hdul[1].data
                    oversamp = hdul[0].header['OVERSAMP']
                    nisp_zoomed = resize_psf(nisp_psf, 1, 6)
                
        else: # Open from dropbox
            print('Fetching files from dropbox...')
            jwst_header, jwst_data = get_fits_file(url, jwst_file, dbx=dbx, hdu_idx=jwst_hdu)
            nisp_header, nisp_data = get_fits_file(url, nisp_file, dbx=dbx, hdu_idx=nisp_hdu)
            if mask:
                mask_header, mask_data = get_fits_file(url, nisp_file, dbx=dbx, hdu_idx=nisp_hdu+1)
                wcs_mask = WCS(mask_header)
            if deconvolve: # Get PSF file at image scale
                _, nisp_psf = get_fits_file(url, cat.iloc[0].psf_image, dbx=dbx, hdu_idx=1)
                hdr, _ = get_fits_file(url, cat.iloc[0].psf_image, dbx=dbx, hdu_idx=0)
                oversamp = hdr['OVERSAMP']
                nisp_zoomed = resize_psf(nisp_psf, 1, float(oversamp))
                nisp_zoomed /= np.sum(nisp_zoomed) # normalize
        # if deconvolve:
        #     with fits.open(jwst_psf_file) as hdul:
        #         jwst_psf = hdul[1].data
        #         cx, cy = np.array(jwst_psf.shape)//2
        #         jwst_zoomed = Cutout2D(jwst_psf, (cx, cy), 20)
        
        wcs_jwst, wcs_nisp = WCS(jwst_header), WCS(nisp_header)
        
        # Get the coordinates of matched galaxies
        gal_coords = SkyCoord(cat.ra, cat.dec, unit='deg')
        
        for i in range(len(cat)):
            gal = cat.iloc[i]
            # Get JWST clip (larger than final size)
            clip_jwst = Cutout2D(jwst_data, gal_coords[i], size=size_jwst+pad, wcs=wcs_jwst, mode='trim')
            if sum(clip_jwst.data.shape)!=(size_jwst+pad)*2: continue
            if np.sum((clip_jwst.data==0.0).astype(int))/((size_jwst+pad)**2) > FRAC_ZERO: continue
            
            # Get NISP clip
            clip_nisp = Cutout2D(nisp_data, gal_coords[i], size=size_nisp, wcs=wcs_nisp, mode='trim')
            if sum(clip_nisp.data.shape)!=(size_nisp)*2: continue
            
            # Check for blank clips
            if np.sum((clip_nisp.data==0.0).astype(int))/((size_nisp)**2) > FRAC_ZERO: continue

            # Mask nisp clip, if requested
            if mask:
                clip_mask = Cutout2D(mask_data, gal_coords[i], size=size_nisp, wcs=wcs_mask, mode='trim')
                clip_mask = (clip_mask.data>=1) # < 1 is good data, greater is bad
                masked_clip = np.ma.array(clip_nisp.data, mask=clip_mask)
                clip_nisp.data = masked_clip.filled(0)

            # Mirror nisp clip, if requested
            if mirror_nisp:
                clip_nisp.data = np.fliplr(clip_nisp.data)
                clip_mask = np.fliplr(clip_mask)

            # Deconvolve nisp clip, if requested
            if deconvolve:
                clip_nisp.data, _ = restoration.unsupervised_wiener(clip_nisp.data, 
                                                    nisp_zoomed, clip=False)
                # clip_jwst.data, _ = restoration.unsupervised_wiener(clip_jwst.data, jwst_zoomed, clip=False)
            
            # Rotate JWST image 20 degrees counter-clockwise and crop
            # This loses the WCS
            if rot_jwst != 0:
                clip_jwst = rotate_jwst(clip_jwst, size=size_jwst, angle=rot_jwst)
                # clip_jwst.wcs = rotate_wcs(clip_jwst, angle=rot_jwst)
            # clip_jwst.data = ABmag_jwst(clip_jwst, jwst_header['PIXAR_SR'])

            if mask: # Mask JWST too
                zoom_factor = size_jwst / size_nisp
                clip_mask = np.fliplr(clip_mask)
                zoomed_mask = ndimage.zoom(clip_mask, zoom_factor, order=0, 
                                           mode='grid-constant', grid_mode=True)
                masked_clip = np.ma.array(clip_jwst.data, mask=zoomed_mask)
                clip_jwst.data = masked_clip.filled(0)
            
            clips.append((gal_coords[i], clip_jwst, clip_nisp))
            count += 1
            if limit is not None and count >= limit: break
        if limit is not None and count >= limit: break

    return clips


############################################
########### Processing pipeline ############
############################################


def process_all(field='cosmos', euclid_type='NISP-Y_MER', save_cat=False, save_clips=False, redo_cat=False,
                redo_clips=False, secret="../../secrets/dropbox_token", meta=meta):
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
    psf_filenames = [nisp_path+file.name for file in nisp_files if 'PSF' in file.name]
    nisp_files = [nisp_path+file.name for file in nisp_files if 'IMAGE' in file.name]
    psf_files = [psf_filename(file, psf_filenames) for file in nisp_files]

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
    
        # Match NISP files and PSFs
        for file,psf_file in zip(nisp_files,psf_files):
            found_idxs = match_catalog(file, gal_coords, url=dbx_url, dbx=dbx,
                                       hdu_idx=params['nisp_hdu'])
            my_cat.loc[found_idxs, 'nisp_image'] = str(file)
            my_cat.loc[found_idxs, 'psf_image'] = str(psf_file)
    
        my_cat = my_cat[my_cat.nisp_image!='']
        my_cat = my_cat[my_cat.jwst_image!='']
    
        if save_cat:
            my_cat.to_csv(params['matched_cat'], index=False)
    
    # Clip images
    clips = clip_images(my_cat, url=dbx_url, dbx=dbx, **params)
    
    # Arrange data
    jwst_cutouts = np.array([clip[1].data for clip in clips])
    nisp_cutouts = np.array([clip[2].data for clip in clips])
    
    if save_clips:
        np.save(params['matched_jwst'], jwst_cutouts, allow_pickle=False)
        np.save(params['matched_nisp'], nisp_cutouts, allow_pickle=False)
    
    return jwst_cutouts, nisp_cutouts


##############################################
########### WCS stuff (not great) ############
##############################################


def plot_on_clip(base_clip, other_clip):
    """ My own plotting function """
    base_size, other_size = base_clip.data.shape[0], other_clip.data.shape[0]
    corners = np.array([[0,0],[0,other_size],[other_size,other_size],[other_size,0],[0,0]])
    corners_sky = other_clip.wcs.pixel_to_world(corners[:,1], corners[:,0])
    corners_base = base_clip.wcs.world_to_pixel(corners_sky)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(base_clip.data, origin='lower')
    axes[0].plot(corners_base[0], corners_base[1], 'r--')
    axes[1].imshow(other_clip.data, origin='lower')
    plt.show()

    base_scale = pixel_scale(base_clip.wcs)
    other_scale = pixel_scale(other_clip.wcs)

    print(f"Base clip scale: {base_scale:.2f} \nOther clip scale: {other_scale:.2f}")
    print(f"Base clip should be {other_scale/base_scale:.4f}x larger than other clip")

def mirror_cutout_along_y(cutout):
    """
    Mirror a Cutout2D object along the y-axis (flip horizontally),
    updating both the data and WCS information.
    
    Parameters:
    -----------
    cutout : Cutout2D
        The input cutout with valid WCS
        
    Returns:
    --------
    Cutout2D
        A new Cutout2D object with mirrored data and corrected WCS
    """
    # Create a deep copy to avoid modifying the original
    new_cutout = deepcopy(cutout)
    
    # Flip the image data horizontally
    new_cutout.data = np.fliplr(cutout.data)
    
    # Get the original WCS
    old_wcs = cutout.wcs
    
    # Create a new WCS object
    new_wcs = WCS(naxis=old_wcs.naxis)
    
    # Copy over the basic WCS parameters
    new_wcs.wcs.crval = old_wcs.wcs.crval.copy()
    new_wcs.wcs.crpix = old_wcs.wcs.crpix.copy()
    new_wcs.wcs.cdelt = old_wcs.wcs.cdelt.copy()
    
    # Get the transformation matrix (either CD or PC)
    if hasattr(old_wcs.wcs, 'cd'):
        old_matrix = old_wcs.wcs.cd.copy()
        # Flip the sign of the first column (RA components)
        new_matrix = old_matrix.copy()
        new_matrix[:, 0] = -old_matrix[:, 0]
        new_wcs.wcs.cd = new_matrix
    else:
        old_matrix = old_wcs.wcs.pc.copy()
        # Flip the sign of the first column (RA components)
        new_matrix = old_matrix.copy()
        new_matrix[:, 0] = -old_matrix[:, 0]
        new_wcs.wcs.pc = new_matrix
        # Also need to flip the sign of the RA scale
        new_wcs.wcs.cdelt[0] = -old_wcs.wcs.cdelt[0]
    
    # Update the reference pixel x-coordinate
    # For a horizontal flip, the new reference pixel is at (width - x)
    width = cutout.data.shape[1]
    new_wcs.wcs.crpix[0] = width + 1 - old_wcs.wcs.crpix[0]
    
    # Copy over any additional WCS attributes
    for attr in ['ctype', 'cunit', 'lonpole', 'latpole', 'radesys', 'equinox']:
        if hasattr(old_wcs.wcs, attr):
            setattr(new_wcs.wcs, attr, getattr(old_wcs.wcs, attr))

    new_cutout.wcs = new_wcs
    
    return new_cutout

def get_relative_rotation_vector(cutout1, cutout2):
    """
    Calculate relative rotation by transforming a vector between the two frames.
    """
    wcs1 = cutout1.wcs
    wcs2 = cutout2.wcs
    
    # Define two points in the first cutout
    center = np.array([cutout1.shape[1]/2, cutout1.shape[0]/2])
    point2 = center + np.array([0, 100])  # 100 pixels in y direction
    
    # Convert to sky coordinates
    center_sky = pixel_to_skycoord(center[0], center[1], wcs1)
    point2_sky = pixel_to_skycoord(point2[0], point2[1], wcs1)
    
    # Convert back to pixel coordinates in the second cutout
    center_pix2 = skycoord_to_pixel(center_sky, wcs2)
    point2_pix2 = skycoord_to_pixel(point2_sky, wcs2)
    
    # Calculate the vector in both coordinate systems
    vec1 = np.array([0, 100])  # Original vector (0, 100)
    vec2 = np.array([point2_pix2[0] - center_pix2[0], 
                     point2_pix2[1] - center_pix2[1]])
    
    # Calculate the angle between the vectors
    angle1 = np.arctan2(vec1[1], vec1[0])
    angle2 = np.arctan2(vec2[1], vec2[0])
    
    # Return the relative rotation in degrees
    rel_angle = np.degrees(angle2 - angle1)
    
    # Normalize to range [-180, 180]
    if rel_angle > 180:
        rel_angle -= 360
    elif rel_angle < -180:
        rel_angle += 360
        
    return rel_angle


def rotate_cutout_with_wcs(cutout, angle_deg, reshape=True, order=1, fill_value=0):
    """
    Rotate a Cutout2D object by a specified angle in degrees,
    updating both the data and WCS information to maintain proper alignment.
    
    Parameters:
    -----------
    cutout : Cutout2D
        The input cutout with valid WCS
    angle_deg : float
        Rotation angle in degrees (positive is counterclockwise)
    reshape : bool, optional
        Whether to reshape the output to contain the entire rotated image
    order : int, optional
        Order of spline interpolation (0-5). Default is 1 (bilinear)
    fill_value : float, optional
        Value used to fill areas outside the input image
        
    Returns:
    --------
    tuple
        (rotated_data, rotated_wcs)
    """
    # Get the original WCS and data
    original_wcs = cutout.wcs
    original_data = cutout.data
    
    # Get the center of the original image in pixel coordinates
    old_center_pix = np.array([(original_data.shape[1] - 1) / 2, 
                              (original_data.shape[0] - 1) / 2])
    
    # Get the center in sky coordinates
    old_center_sky = pixel_to_skycoord(old_center_pix[0], old_center_pix[1], original_wcs)
    
    # Rotate the image data
    rotated_data = rotate(original_data, angle_deg, reshape=reshape, 
                          order=order, mode='constant', cval=fill_value)
    
    # Get the center of the rotated image
    if reshape:
        new_center_pix = np.array([(rotated_data.shape[1] - 1) / 2, 
                                  (rotated_data.shape[0] - 1) / 2])
    else:
        new_center_pix = old_center_pix
    
    # Create a new WCS object
    new_wcs = WCS(naxis=2)
    
    # Copy basic WCS parameters
    new_wcs.wcs.ctype = original_wcs.wcs.ctype
    if hasattr(original_wcs.wcs, 'cunit'):
        new_wcs.wcs.cunit = original_wcs.wcs.cunit
    new_wcs.wcs.crval = old_center_sky.spherical.lon.deg, old_center_sky.spherical.lat.deg
    
    # Set the reference pixel to the center of the rotated image
    new_wcs.wcs.crpix = new_center_pix + 1  # +1 because FITS is 1-indexed
    
    # Create a rotation matrix for the angle
    angle_rad = np.radians(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # Get the original transformation matrix
    if hasattr(original_wcs.wcs, 'cd'):
        # CD matrix case
        cd_matrix = original_wcs.wcs.cd
        # Apply rotation to the CD matrix
        new_cd = np.dot(rot_matrix, cd_matrix)
        new_wcs.wcs.cd = new_cd
    else:
        # PC matrix case
        pc_matrix = original_wcs.wcs.pc
        cdelt = original_wcs.wcs.cdelt
        
        # Apply rotation to the PC matrix
        new_pc = np.dot(rot_matrix, pc_matrix)
        new_wcs.wcs.pc = new_pc
        new_wcs.wcs.cdelt = cdelt
    
    # Copy other WCS attributes
    for attr in ['radesys', 'equinox', 'lonpole', 'latpole']:
        if hasattr(original_wcs.wcs, attr):
            setattr(new_wcs.wcs, attr, getattr(original_wcs.wcs, attr))
    
    return rotated_data, new_wcs


def test_rotation(cutout, angle_deg):
    """Test rotation by plotting original and rotated images with corner mapping"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    
    # Rotate the cutout
    rotated_data, rotated_wcs = rotate_cutout_with_wcs(cutout, angle_deg)
    
    # Create a grid of points to visualize the transformation
    ny, nx = cutout.data.shape
    y, x = np.mgrid[0:ny:10, 0:nx:10]
    points = np.vstack([x.flatten(), y.flatten()]).T
    
    # Transform points from pixel to sky coordinates
    sky_coords = cutout.wcs.pixel_to_world(points[:,0], points[:,1])
    
    # Transform back to pixel coordinates in the rotated frame
    x_rot, y_rot = rotated_wcs.world_to_pixel(sky_coords)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(cutout.data, origin='lower')
    ax1.plot(points[:,0], points[:,1], 'r+')
    ax1.set_title('Original Image')
    
    # Rotated image
    ax2.imshow(rotated_data, origin='lower')
    ax2.plot(x_rot, y_rot, 'r+')
    ax2.set_title(f'Rotated by {angle_deg}°')
    
    plt.tight_layout()
    plt.show()
    
    # Test corners specifically
    corners = np.array([[0, 0], [nx-1, 0], [nx-1, ny-1], [0, ny-1], [0, 0]])
    corners_sky = cutout.wcs.pixel_to_world(corners[:,0], corners[:,1])
    corners_rot = np.array(rotated_wcs.world_to_pixel(corners_sky)).T
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(cutout.data, origin='lower')
    ax1.plot(corners[:,0], corners[:,1], 'r-', linewidth=2)
    ax1.set_title('Original Image')
    
    # Rotated image
    ax2.imshow(rotated_data, origin='lower')
    ax2.plot(corners_rot[:,0], corners_rot[:,1], 'r-', linewidth=2)
    ax2.set_title(f'Rotated by {angle_deg}°')
    
    plt.tight_layout()
    plt.show()
    
    return rotated_data, rotated_wcs

def plot_with_wcs_grid(ax, data, wcs, title):
    """Plot data with WCS grid overlay"""
    ax.imshow(data, origin='lower', cmap='gray')
    ax.set_title(title)
    ax.grid(color='white', ls='solid', alpha=0.5)
    
    # Add coordinate labels
    ra = ax.coords[0]
    dec = ax.coords[1]
    ra.set_axislabel('Right Ascension')
    dec.set_axislabel('Declination')
    ra.set_major_formatter('hh:mm:ss')
    dec.set_major_formatter('dd:mm:ss')

