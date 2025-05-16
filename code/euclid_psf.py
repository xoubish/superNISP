import os
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.table import Table
from reproject import reproject_interp
from scipy import ndimage

class EuclidPSF():

    def __init__(self, filename):
        ### Initialize from a Euclid PSF file
        self.psf_meta = Table.read(filename, hdu=2).to_pandas()

        with fits.open(filename) as hdul:
            hdr = hdul[1].header
            self.wcs = WCS(hdr)
            self.data = hdul[1].data
        
        self.stamp_size = hdr['STMPSIZE']
        y_shape, x_shape = self.data.shape

        self.sky_coords = self.wcs.pixel_to_world(self.psf_meta.y-1, self.psf_meta.x-1) # 1-based index

    def evaluate(self, coord):
        ### Evaluate PSF at coord (SkyCoord object)
        sep = coord.separation(self.sky_coords)
        idx = np.argmin(list(sep.degree))

        # Get x and y location of 
        x_psf, y_psf = (self.psf_meta.loc[idx, ['x', 'y']]-1).astype(int) # 1-based indexing

        psf = Cutout2D(self.data, (y_psf, x_psf), self.stamp_size)

        return psf.data

    def clip(self, coord):

        psf = self.evaluate(coord)

        zoomed = ndimage.zoom(psf, 1/self.oversampling, order=3)

        return zoomed

        
        
        
        
            

        