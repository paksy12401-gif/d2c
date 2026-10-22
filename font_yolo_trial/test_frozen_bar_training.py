import unittest
import torch

from train_vertical_bar_frozen import new_class_gradient, restore_protected, preserved
from append_only_bar_head import AppendOnlyBarHead


class FrozenTrainingTests(unittest.TestCase):
    def test_parallel_branch_keeps_legacy_output_exact_after_training_step(self):
        torch.manual_seed(4)
        classifier=torch.nn.Sequential(torch.nn.Conv2d(4,8,3,padding=1),torch.nn.BatchNorm2d(8),torch.nn.SiLU(),torch.nn.Conv2d(8,67,1))
        classifier.eval()
        image=torch.randn(2,4,12,12)
        expected=classifier(image).detach()
        branch=AppendOnlyBarHead(classifier).train()
        optimizer=torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad],lr=.01)
        before=branch(image)
        self.assertTrue(torch.equal(before[:,:67],expected))
        before[:,67:].square().mean().backward();optimizer.step()
        after=branch(image)
        self.assertTrue(torch.equal(after[:,:67],expected))
        self.assertFalse(torch.equal(before[:,67:],after[:,67:]))
        self.assertFalse(branch.legacy.training)

    def test_only_new_class_gradient_survives(self):
        original=torch.ones(68,4)
        masked=new_class_gradient(original)
        self.assertEqual(int(masked[:67].count_nonzero()),0)
        self.assertTrue(torch.equal(masked[67],original[67]))
        self.assertEqual(int(original.count_nonzero()),272)

    def test_restore_preserves_new_row_and_restores_old_rows_and_buffers(self):
        model=torch.nn.Sequential(torch.nn.Conv2d(2,68,1),torch.nn.BatchNorm2d(68))
        protected={k:v.detach().clone() for k,v in model.state_dict().items()}
        # Use a miniature old checkpoint with the same layer sizes except classifier rows.
        protected['0.weight']=protected['0.weight'][:67].clone()
        protected['0.bias']=protected['0.bias'][:67].clone()
        with torch.no_grad():
            for p in model.parameters():p.add_(.25)
            model[1].running_mean.add_(.75)
            newrow=model[0].weight[67].clone()
        self.assertTrue(preserved(model,protected))
        restore_protected(model,protected)
        self.assertEqual(preserved(model,protected),[])
        self.assertTrue(torch.equal(model[0].weight[67],newrow))


if __name__=='__main__':unittest.main()
