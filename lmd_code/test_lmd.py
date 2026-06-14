from __future__ import annotations

import tempfile
import unittest
import sys
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from common import ExponentialMovingAverage
from dataset import sample_checkerboard
from lmd_code.distill_lmd import (
    checkpoint_payload,
    diagonal_probability,
    lmd_outputs,
    load_training_checkpoint,
    maximum_time_gap,
    sample_lmd_times,
    should_checkpoint,
)
from models import FlowMapNet
from sampling import flow_map_sample


def tiny_model() -> FlowMapNet:
    return FlowMapNet(
        hidden_dim=16,
        num_hidden_layers=2,
        time_embedding_dim=8,
        fourier_scale=2.0,
    )


class LmdTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cpu")

    def test_flow_identity_and_diagonal_derivative(self) -> None:
        model = tiny_model()
        x = torch.randn(7, 2)
        time = torch.rand(7)
        self.assertTrue(torch.equal(model.flow(x, time, time), x))
        _, derivative = torch.func.jvp(
            lambda target: model.flow(x, time, target),
            (time,),
            (torch.ones_like(time),),
        )
        self.assertTrue(
            torch.allclose(derivative, model(x, time, time), atol=1e-5, rtol=1e-5)
        )

    def test_analytic_jvp_and_backward(self) -> None:
        weight = torch.nn.Parameter(torch.tensor(3.0))
        target = torch.tensor([0.2, 0.7])
        value, derivative = torch.func.jvp(
            lambda time: weight * time.square(),
            (target,),
            (torch.ones_like(target),),
        )
        self.assertTrue(torch.allclose(value, weight * target.square()))
        self.assertTrue(torch.allclose(derivative, 2.0 * weight * target))
        derivative.square().sum().backward()
        self.assertIsNotNone(weight.grad)
        self.assertGreater(abs(float(weight.grad)), 0.0)

    def test_teacher_and_stop_gradient_target_have_no_grad(self) -> None:
        teacher = tiny_model()
        student = tiny_model()
        teacher.requires_grad_(False)
        x = torch.randn(8, 2)
        s = torch.rand(8)
        t = s * torch.rand(8)
        mapped, derivative, target = lmd_outputs(student, teacher, x, s, t)
        self.assertTrue(mapped.requires_grad)
        self.assertTrue(derivative.requires_grad)
        self.assertFalse(target.requires_grad)
        optimizer = torch.optim.SGD(student.parameters(), lr=1e-3)
        before = {
            name: parameter.detach().clone()
            for name, parameter in student.named_parameters()
        }
        (derivative - target).square().sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        self.assertTrue(
            any(parameter.grad is not None for parameter in student.parameters())
        )
        optimizer.step()
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter)
                for name, parameter in student.named_parameters()
            )
        )

    def test_segmented_flow_map_sampling(self) -> None:
        class ConstantFlow(torch.nn.Module):
            def flow(self, x, s, t):
                delta = torch.as_tensor(t) - torch.as_tensor(s)
                return x + delta * torch.ones_like(x)

        noise = torch.randn(12, 2)
        model = ConstantFlow()
        samples = flow_map_sample(model, noise, nfe=4, batch_size=5)
        self.assertTrue(torch.allclose(samples, noise - 1.0))

    def test_curriculum_and_time_sampling(self) -> None:
        self.assertEqual(diagonal_probability(0), 0.10)
        self.assertAlmostEqual(diagonal_probability(17_500), 0.175)
        self.assertEqual(diagonal_probability(30_000), 0.25)
        self.assertEqual(maximum_time_gap(0), 0.10)
        self.assertAlmostEqual(maximum_time_gap(17_500), 0.55)
        self.assertEqual(maximum_time_gap(30_000), 1.0)

        generator = torch.Generator().manual_seed(4)
        batch_size = 100_000
        s, t, diagonal = sample_lmd_times(
            batch_size,
            step=17_500,
            generator=generator,
            device=self.device,
        )
        self.assertEqual(int(diagonal.sum()), round(batch_size * 0.175))
        self.assertTrue(torch.equal(s[diagonal], t[diagonal]))
        gaps = s[~diagonal] - t[~diagonal]
        self.assertTrue(bool((gaps > 0.0).all()))
        self.assertTrue(bool((gaps <= 0.55 + 1e-6).all()))
        self.assertTrue(bool((t[~diagonal] >= 0.0).all()))
        self.assertTrue(bool((s[~diagonal] <= 1.0 + 1e-6).all()))
        expected_mean_gap = (
            0.55**2 / 2.0 - 0.55**3 / 3.0
        ) / (0.55 - 0.55**2 / 2.0)
        self.assertAlmostEqual(
            float(gaps.mean()),
            expected_mean_gap,
            delta=0.003,
        )

    def test_configurable_checkpoint_interval(self) -> None:
        self.assertTrue(should_checkpoint(41_000, 55_000, 1_000))
        self.assertFalse(should_checkpoint(41_500, 55_000, 1_000))
        self.assertTrue(should_checkpoint(55_000, 55_000, 10_000))

    def test_checkpoint_restores_next_random_batch(self) -> None:
        model = tiny_model()
        teacher = tiny_model()
        teacher.load_state_dict(model.state_dict())
        teacher.requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ema = ExponentialMovingAverage(model)
        generator = torch.Generator().manual_seed(9)
        args = Namespace(steps=90_000)
        teacher_metadata = {"sha256": "test-sha"}
        payload = checkpoint_payload(
            student=model,
            optimizer=optimizer,
            ema=ema,
            generator=generator,
            step=123,
            elapsed_seconds=4.0,
            args=args,
            teacher_metadata=teacher_metadata,
            monitor_state={"best_sw2": 1.0},
        )

        def next_batch_loss(
            current_model: FlowMapNet,
            current_generator: torch.Generator,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            x0 = sample_checkerboard(
                20,
                generator=current_generator,
                device=self.device,
            )
            noise = torch.randn(20, 2, generator=current_generator)
            s, t, _ = sample_lmd_times(
                20,
                step=123,
                generator=current_generator,
                device=self.device,
            )
            x_s = (1.0 - s[:, None]) * x0 + s[:, None] * noise
            _, derivative, target = lmd_outputs(
                current_model,
                teacher,
                x_s,
                s,
                t,
            )
            loss = (derivative - target).square().sum(dim=-1).mean()
            return x0, s, t, loss.detach()

        expected = next_batch_loss(model, generator)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(payload, path)
            restored_model = tiny_model()
            restored_optimizer = torch.optim.Adam(
                restored_model.parameters(),
                lr=1e-3,
            )
            restored_ema = ExponentialMovingAverage(restored_model)
            restored_generator = torch.Generator().manual_seed(100)
            step, elapsed, monitor = load_training_checkpoint(
                path,
                student=restored_model,
                optimizer=restored_optimizer,
                ema=restored_ema,
                generator=restored_generator,
                device=self.device,
                teacher_metadata=teacher_metadata,
            )
            actual = next_batch_loss(restored_model, restored_generator)
        self.assertEqual(step, 123)
        self.assertEqual(elapsed, 4.0)
        self.assertEqual(monitor["best_sw2"], 1.0)
        for expected_value, actual_value in zip(expected, actual):
            self.assertTrue(torch.equal(expected_value, actual_value))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_checkpoint_restores_next_random_batch(self) -> None:
        device = torch.device("cuda")
        model = tiny_model().to(device)
        teacher = tiny_model().to(device)
        teacher.load_state_dict(model.state_dict())
        teacher.requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ema = ExponentialMovingAverage(model)
        generator = torch.Generator(device=device).manual_seed(17)
        args = Namespace(steps=90_000)
        teacher_metadata = {"sha256": "test-cuda-sha"}
        payload = checkpoint_payload(
            student=model,
            optimizer=optimizer,
            ema=ema,
            generator=generator,
            step=321,
            elapsed_seconds=8.0,
            args=args,
            teacher_metadata=teacher_metadata,
            monitor_state={"best_sw2": 0.5},
        )

        def next_batch_loss(
            current_model: FlowMapNet,
            current_generator: torch.Generator,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            x0 = sample_checkerboard(
                20,
                generator=current_generator,
                device=device,
            )
            noise = torch.randn(
                20,
                2,
                generator=current_generator,
                device=device,
            )
            s, t, _ = sample_lmd_times(
                20,
                step=321,
                generator=current_generator,
                device=device,
            )
            x_s = (1.0 - s[:, None]) * x0 + s[:, None] * noise
            _, derivative, target = lmd_outputs(
                current_model,
                teacher,
                x_s,
                s,
                t,
            )
            loss = (derivative - target).square().sum(dim=-1).mean()
            return (
                x0.cpu(),
                s.cpu(),
                t.cpu(),
                loss.detach().cpu(),
            )

        expected = next_batch_loss(model, generator)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(payload, path)
            restored_model = tiny_model().to(device)
            restored_optimizer = torch.optim.Adam(
                restored_model.parameters(),
                lr=1e-3,
            )
            restored_ema = ExponentialMovingAverage(restored_model)
            restored_generator = torch.Generator(device=device).manual_seed(100)
            step, elapsed, monitor = load_training_checkpoint(
                path,
                student=restored_model,
                optimizer=restored_optimizer,
                ema=restored_ema,
                generator=restored_generator,
                device=device,
                teacher_metadata=teacher_metadata,
            )
            actual = next_batch_loss(restored_model, restored_generator)
        self.assertEqual(step, 321)
        self.assertEqual(elapsed, 8.0)
        self.assertEqual(monitor["best_sw2"], 0.5)
        for expected_value, actual_value in zip(expected, actual):
            self.assertTrue(torch.equal(expected_value, actual_value))


if __name__ == "__main__":
    unittest.main()
