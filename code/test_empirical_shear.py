"""Physics controls for rendering a supplied effective PSF, without inference."""
import json
import unittest
import galsim
import numpy as np
from empirical_shear_response import (ROOT, effective_profile, draw_effective,
    load_psfs, aggregate, RADII, SNRS, sha)
from pilot_shear_response import draw_sb, observed, SHEARS


class EmpiricalControls(unittest.TestCase):
    def test_effective_pixel_is_applied_once(self):
        scale=.3
        optical=galsim.Gaussian(sigma=.19).shear(e1=.08,e2=-.03)
        effective=galsim.Convolve(optical,galsim.Pixel(scale))
        fine=effective.drawImage(nx=121,ny=121,scale=scale/6,method='no_pixel').array
        tabulated=effective_profile(fine,np.eye(2)*scale/6)
        gal=galsim.Gaussian(sigma=.31,flux=2.5).shear(g1=.08,g2=.04)
        expected=draw_sb(galsim.Convolve(gal,optical),scale,41)
        actual=draw_effective(galsim.Convolve(gal,tabulated),scale)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=expected.max()*3.e-4)
        self.assertAlmostEqual(actual.sum()*scale**2,2.5,places=4)
        np.testing.assert_allclose(observed(actual,scale)[0],observed(expected,scale)[0],atol=1.e-4)
        double=draw_sb(galsim.Convolve(gal,tabulated),scale,41)
        self.assertGreater(np.max(abs(double-expected)),np.max(abs(actual-expected))*20)

    def test_psf_basis_and_recentering(self):
        scale=.04
        original=galsim.Gaussian(sigma=.2).shear(e1=.2,e2=.1)
        array=original.shift(.03,-.02).drawImage(nx=101,ny=101,scale=scale,method='no_pixel').array
        theta=.31
        rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        psf=effective_profile(array,rot*scale)
        actual=draw_effective(psf,.04,101)
        expected=draw_effective(original.rotate(theta*galsim.radians),.04,101)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=expected.max()*3.e-4)
        np.testing.assert_allclose(observed(actual,.04)[0],observed(expected,.04)[0],atol=1.e-4)

    def test_supplied_psfs(self):
        profiles,records=load_psfs()
        self.assertEqual(len(profiles),4)
        for profile,scale in profiles:
            self.assertAlmostEqual(profile.flux,1,places=9)
            image=draw_effective(profile,scale)
            self.assertTrue(np.isfinite(image).all())
            self.assertLess(abs(image.sum()*scale**2-1),.02)
            self.assertTrue(observed(draw_effective(profile,scale/6,121),scale/6)[1])
            gal=galsim.Exponential(half_light_radius=.18).shear(g1=.03,g2=.02)
            default=draw_effective(galsim.Convolve(gal,profile),scale)
            self.assertTrue(observed(default,scale)[1])
            tight=draw_effective(galsim.Convolve(gal,profile,
                gsparams=galsim.GSParams(folding_threshold=1.e-5,maxk_threshold=1.e-5)),scale)
            np.testing.assert_allclose(default,tight,rtol=0,atol=default.max()*5.e-4)
            np.testing.assert_allclose(observed(default,scale)[0],observed(tight,scale)[0],atol=5.e-4)

    def test_common_selection_across_psfs(self):
        arrays={}
        truth=np.array([[[2*g1,2*g2] for g1,g2 in SHEARS]]*3)
        for pi in range(4):
            for ri,_ in enumerate(RADII):
                for si,_ in enumerate(SNRS):
                    e=np.repeat(truth[:,:,None,:],4,axis=2)
                    good=np.ones(e.shape[:-1],bool)
                    # Native correction never selects the observed-method sample.
                    good[:,:,1]=False;e[:,:,1]=np.nan
                    if pi==3:good[2,1,2]=False;e[2,1,2]=np.nan
                    key=f'psf{pi}_radius{ri}_snr{si}'
                    arrays.update({key+'__e':e,key+'__good':good,key+'__truth':truth})
        cells=aggregate(arrays)
        for cell in cells:
            self.assertEqual(cell['common_success'],2)
            self.assertNotIn('native_regauss',cell['methods'])
            for rec in cell['methods'].values():self.assertAlmostEqual(rec['relative_trace_response'],1.)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(EmpiricalControls)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        _,records=load_psfs()
        out=ROOT/'paper/diagnostics/empirical_psf_controls_v01.json'
        assert not out.exists(),out
        out.write_text(json.dumps(dict(passed=True,tests=result.testsRun,psfs=records,
            source_sha256=sha(ROOT/'code/empirical_shear_response.py'),test_sha256=sha(__file__)),indent=2)+'\n')
    else:raise SystemExit(1)
