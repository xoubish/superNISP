"""Repeated-sample scatter must use the same galaxies and retain failures."""
import unittest
import numpy as np
from diffusion_diagnostics import repeated_shape_report


class RepeatedSampleTests(unittest.TestCase):
    def fixture(self):
        arrays=[]
        for offset in [-.1,0.,.1]:
            data=dict(object_ids=np.array([10,20,30]),rows=np.arange(3),blocks=np.array(['a','b','c']),
                      jwst_e=np.zeros((3,2)),jwst_good=np.ones(3,bool),jwst_flags=np.zeros((3,7),bool),
                      sr_e=np.full((3,2),offset))
            for method in ['native','bilinear','bicubic','sr']: data[method+'_good']=np.ones(3,bool)
            arrays.append(data)
        return arrays

    def test_common_population_and_scatter(self):
        arrays=self.fixture(); arrays[-1]['sr_good'][0]=False
        report=repeated_shape_report(arrays)
        self.assertEqual(report['common_object_ids'],[20,30])
        self.assertEqual(report['sr_failures_per_sample'],[0,0,1])
        np.testing.assert_allclose(report['component_rmse_per_sample'],[.1,0.,.1])
        self.assertAlmostEqual(report['shape_component_scatter_rms'],.1)

    def test_reject_reference_change_and_empty_population(self):
        arrays=self.fixture(); arrays[1]['jwst_e'][0,0]=.5
        with self.assertRaises(AssertionError): repeated_shape_report(arrays)
        arrays=self.fixture(); arrays[0]['sr_good'][:]=False
        report=repeated_shape_report(arrays)
        self.assertEqual(report['common_successes'],0)
        self.assertIsNone(report['shape_component_scatter_rms'])


if __name__=='__main__': unittest.main()
