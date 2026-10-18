"""CPU checks of the new loss, paired masks, schedule and data wrapper."""
import pickle
import random
import unittest
from pathlib import Path
import numpy as np
import torch
from residual_refinement import balanced_l1, fresh_schedule, SourceMaskedSample, restore_parent
from train_paired import paired_augment

ROOT=Path(__file__).resolve().parents[1]

class RefinementTests(unittest.TestCase):
    def test_preflight_cannot_mutate_parent_optimizer(self):
        import copy
        model=torch.nn.Linear(2,1)
        opt=torch.optim.Adam(model.parameters(),lr=1e-4)
        def update():
            opt.zero_grad();model(torch.ones(1,2)).square().sum().backward();opt.step()
        update()
        parent=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=1))
        original=copy.deepcopy(parent)
        restore_parent(model,opt,parent)
        for _ in range(3):update()
        for key,state in parent['optimizer']['state'].items():
            for name,value in state.items():
                torch.testing.assert_close(value,original['optimizer']['state'][key][name],rtol=0,atol=0)
        restore_parent(model,opt,parent)
        self.assertTrue(all(float(v['step'])==1 for v in opt.state.values()))

    def test_equal_regions_and_objects(self):
        pred=torch.tensor([[[[2.,4.,4.,4.]]],[[[6.,0.,0.,0.]]]],requires_grad=True)
        target=torch.zeros_like(pred);valid=torch.ones_like(pred,dtype=torch.bool)
        source=torch.zeros_like(valid);source[:,:,:,0]=True
        self.assertEqual(balanced_l1(pred,target,valid,source).item(),3.)
        balanced_l1(pred,target,valid,source).backward()
        self.assertAlmostEqual(pred.grad[0,0,0,0].item(),.25)
        self.assertAlmostEqual(pred.grad[0,0,0,1].item(),1/12)

    def test_empty_regions_and_invalid_gradients(self):
        pred=torch.ones(2,1,2,2,requires_grad=True);target=torch.zeros_like(pred)
        valid=torch.ones_like(pred,dtype=torch.bool);valid[:,:,0,0]=False
        with torch.no_grad():pred[:,:,0,0]=float('nan')
        for source in [torch.zeros_like(valid),torch.ones_like(valid)]:
            loss=balanced_l1(pred,target,valid,source)
            self.assertEqual(loss.item(),1.)
            pred.grad=None;loss.backward()
            self.assertTrue((pred.grad[:,:,0,0]==0).all())
        with self.assertRaises(ValueError):balanced_l1(pred,target,torch.zeros_like(valid),valid)

    def test_schedule_endpoints_and_resume(self):
        p=torch.tensor([2.],requires_grad=True)
        opt=torch.optim.Adam([p],lr=1e-4)
        opt.param_groups[0].update(initial_lr=4e-6,max_lr=1e-4,min_lr=4e-10)
        sched=fresh_schedule(opt,10)
        self.assertEqual(opt.param_groups[0]['lr'],1e-5)
        for _ in range(4):
            opt.zero_grad();p.square().sum().backward();opt.step();sched.step()
        import copy
        state=(p.detach().clone(),copy.deepcopy(opt.state_dict()),copy.deepcopy(sched.state_dict()))
        for _ in range(6):
            opt.zero_grad();p.square().sum().backward();opt.step();sched.step()
        expected=p.detach().clone()
        self.assertAlmostEqual(opt.param_groups[0]['lr'],1e-6)
        q=state[0].requires_grad_();qo=torch.optim.Adam([q],lr=1e-5)
        qs=fresh_schedule(qo,10);qo.load_state_dict(state[1]);qs.load_state_dict(state[2])
        for _ in range(6):
            qo.zero_grad();q.square().sum().backward();qo.step();qs.step()
        torch.testing.assert_close(q,expected,rtol=0,atol=0)

    def test_augmented_source_mask_and_worker_serialization(self):
        ds=SourceMaskedSample(ROOT.parent/'supernisp_nir_y_f115w_v1',ROOT/'runs/residual-refinement-masks-v01/source_masks.h5')
        item=ds[0];self.assertFalse((item['source_mask'] & ~item['jwst_valid']).any())
        restored=pickle.loads(pickle.dumps(ds))
        self.assertTrue(torch.equal(item['source_mask'],restored[0]['source_mask']))
        x=torch.arange(16).reshape(1,1,4,4).float()
        for seed in range(8):
            random.seed(seed);b=paired_augment(dict(jwst=x.clone(),source_mask=x>5,jwst_valid=x>1))
            self.assertTrue(torch.equal(b['source_mask'],b['jwst']>5))
            self.assertTrue(torch.equal(b['jwst_valid'],b['jwst']>1))
        ds.close();restored.close()

if __name__=='__main__': unittest.main()
