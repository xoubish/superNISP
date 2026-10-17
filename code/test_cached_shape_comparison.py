import unittest
import numpy as np
from cached_shape_comparison import summarize


class MatchedShapeTests(unittest.TestCase):
    def test_pairing_failures_and_block_weighting(self):
        ref=np.array([[.1,.2],[.3,.1],[.2,.4],[-.1,.3]])
        data={m:dict(e=ref.copy(),good=np.ones(4,bool),flags=np.zeros((4,1),bool)) for m in ['target','bilinear','legacy','direct5','diffusion']}
        data['bilinear']['e']+=.2
        data['diffusion']['e']+=.1
        data['legacy']['good'][3]=False;data['legacy']['flags'][3,0]=True
        data['diffusion']['e'][3]=999
        r=summarize(data,['a','a','b','b'],[1,2,3,4],['bilinear','legacy','direct5','diffusion'],['failed'])
        self.assertEqual(r['object_ids'],[1,2,3])
        self.assertEqual(r['success_counts']['diffusion'],4)
        self.assertEqual(r['failure_reasons']['legacy']['failed'],1)
        self.assertAlmostEqual(r['methods']['diffusion']['component_rmse']['mean'],.1)
        np.testing.assert_allclose(r['paired']['diffusion']['bilinear']['ci95'],[-.1,-.1])
        self.assertEqual(r['methods']['direct5']['component_rmse']['mean'],0)


if __name__=='__main__':unittest.main()
