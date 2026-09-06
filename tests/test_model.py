"""Portable CPU contracts; no external data, servers or checkpoints required."""

import unittest
from dataclasses import asdict
from types import SimpleNamespace

import torch
from torch import nn

from qtmaster.model import EXPECTED_PARAMETER_COUNT, ModelConfig, build_model
from qtmaster.vendor.tqnet import Model as TQNetCore


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_shapes_parameters_and_pure_arm(self):
        for cycle in (7, 3):
            with self.subTest(cycle=cycle):
                torch.manual_seed(0)
                model = build_model(ModelConfig(cycle=cycle))
                self.assertEqual(sum(p.numel() for p in model.parameters()), EXPECTED_PARAMETER_COUNT)
                self.assertEqual(sum(p.numel() for p in model.lla_block.parameters()), 61_760)
                self.assertFalse(model.core.use_tq)
                self.assertFalse(model.core.channel_aggre)
                self.assertFalse(hasattr(model.core, "temporalQuery"))
                self.assertFalse(hasattr(model.core, "channelAggregator"))
                x = torch.randn(4, 8, 221)
                self.assertEqual(tuple(model(x).shape), (4, 1, 1))
                self.assertTrue(torch.equal(model(x), model(x, torch.arange(4) % cycle)))

    def test_gate_starts_as_identity(self):
        model = build_model()
        market = torch.randn(3, 8, 63)
        self.assertTrue(torch.equal(model.compute_gate(market), torch.ones(3, 8, 158)))
        x = torch.randn(3, 8, 221)
        changed_market = x.clone()
        changed_market[:, :, 158:] = 100.0 * torch.randn(3, 8, 63)
        self.assertTrue(torch.equal(model(x), model(changed_market)))
        self.assertTrue(torch.equal(model.readout.weight, torch.full((1, 158), 1.0 / 158)))

    def test_gradient_and_checkpoint_roundtrip(self):
        torch.manual_seed(42)
        model = build_model()
        x = torch.randn(5, 8, 221)
        y = torch.randn(5, 1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = (model(x) - y).square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
        for parameter in (model.market_gate_weight, model.lla_block.raw_tau, model.lla_block.gate):
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
        optimizer.step()
        clone = build_model()
        clone.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(model(x), clone(x)))

    def test_rng_consumption_matches_complete_original_core_and_readout(self):
        for cycle in (7, 3):
            torch.manual_seed(2)
            TQNetCore(SimpleNamespace(**asdict(ModelConfig(cycle=cycle))))
            nn.Linear(158, 1, bias=False)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(2)
            build_model(ModelConfig(cycle=cycle))
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_rejects_label_column_and_wrong_shapes(self):
        model = build_model()
        for shape in ((2, 8, 222), (2, 8, 158), (2, 7, 221), (8, 221)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(*shape))


if __name__ == "__main__":
    unittest.main()
