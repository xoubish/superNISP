"""Checkpoint comparisons must retain identical object identities and baselines."""
import unittest
import numpy as np
from compare_shape_runs import compare
from shape_diagnostics import REASONS


class ComparisonTests(unittest.TestCase):
    def fixture(self):
        data=dict(object_ids=np.array([10,20,30,40]),rows=np.arange(4),blocks=np.array(['a','a','b','b']))
        for m in ['jwst','native','bilinear','bicubic','sr']:
            data[f'{m}_good']=np.ones(4,dtype=bool)
            data[f'{m}_flags']=np.zeros((4,len(REASONS)),dtype=bool)
            data[f'{m}_e']=np.full((4,2),.2 if m=='jwst' else .4)
        return data

    def test_reordering_and_exact_pairing(self):
        before=self.fixture()
        after={k:v[::-1].copy() for k,v in before.items()}
        after['sr_e'][:]=.3
        after['sr_good'][0]=False
        result=compare(before,after)
        self.assertEqual(result['common_object_ids'],[10,20,30])
        self.assertEqual(result['counts']['lost_sr_measurements'],1)
        self.assertAlmostEqual(result['sr_change']['improvement_percent'],50)
        np.testing.assert_allclose(result['sr_change']['ci95_percent'],[50,50])

    def test_reject_changed_reference_or_ids(self):
        before=self.fixture()
        after={k:v.copy() for k,v in before.items()}
        after['jwst_e'][0,0]+=.1
        with self.assertRaisesRegex(ValueError,'diagnostic changed'):
            compare(before,after)
        after=self.fixture()
        after['object_ids'][0]=after['object_ids'][1]
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            compare(before,after)


if __name__=='__main__': unittest.main()
