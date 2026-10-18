"""Scientific controls for the shear pilot, independent of model inference."""
import unittest
import galsim
import numpy as np
from pilot_shear_response import draw_sb, observed, corrected, response, summarize, SHEARS, METHODS


class ShearControls(unittest.TestCase):
    def test_gaussian_correction_and_response(self):
        # Known Gaussian: e=2g/(1+g^2); recovery is an actual physics control.
        psf=galsim.Gaussian(sigma=.22).shear(e1=.03,e2=.02)
        psf_image=draw_sb(psf,.10,81)
        raw, corr=[],[]
        for g1,g2 in SHEARS:
            gal=galsim.Gaussian(sigma=.35).shear(g1=g1,g2=g2)
            image=draw_sb(galsim.Convolve(gal,psf),.10,81)
            r,ok,_=corrected(image,psf_image,.10)
            self.assertTrue(ok)
            np.testing.assert_allclose(r,2*np.array([g1,g2])/(1+g1*g1+g2*g2),atol=4e-4)
            corr.append(r)
            raw.append(observed(image,.10)[0])
        np.testing.assert_allclose(response(np.asarray(corr)),np.eye(2)*2/(1+.01**2),atol=.015)
        self.assertLess(np.trace(response(np.asarray(raw))),np.trace(response(np.asarray(corr))))

    def test_surface_brightness_and_shear_composition(self):
        intrinsic=galsim.Shear(e=.35,beta=.37*galsim.radians)
        lens=galsim.Shear(g1=.01,g2=-.01)
        gal=galsim.Gaussian(sigma=.4,flux=3.).shear(intrinsic).shear(lens)
        image=draw_sb(gal,.02,401)
        self.assertAlmostEqual(image.sum()*.02**2,3.,places=5)
        expected=lens+intrinsic
        np.testing.assert_allclose(observed(image,.02)[0],[expected.e1,expected.e2],atol=2e-4)

    def test_failures_use_matched_objects(self):
        truth=np.array([[[2*g1,2*g2] for g1,g2 in SHEARS]]*2)
        e=np.repeat(truth[:,:,None,:],len(METHODS),axis=2)
        good=np.ones(e.shape[:-1],bool)
        good[1,1,2]=False
        e[1,1,2]=np.nan
        report=summarize(e,good,truth)
        self.assertEqual(report['common_success'],1)
        self.assertEqual(report['methods']['resnet_observed']['success_by_branch'],[2,1,2,2,2])
        for value in report['methods'].values():
            self.assertAlmostEqual(value['relative_trace_response'],1.)


if __name__=='__main__':
    unittest.main()
