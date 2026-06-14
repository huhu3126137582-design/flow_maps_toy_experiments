from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import Tensor

from common import ExponentialMovingAverage
from lsd_code.distill_lsd import (
    branch_sizes,
    checkpoint_payload,
    load_training_checkpoint,
    lsd_learning_rate,
    lsd_outputs,
    sample_lsd_training_batch,
    sample_off_diagonal_times,
    should_checkpoint,
)
from lsd_code.fine_search_lsd import rank_rows
from models import FlowMapNet


def tiny_model() -> FlowMapNet:
    return FlowMapNet(
        hidden_dim=16,
        num_hidden_layers=2,
        time_embedding_dim=8,
        fourier_scale=2.0,
    )


def batch_loss_and_gradients(
    model: FlowMapNet,
    batch,
    *,
    microbatch_size: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    model.zero_grad(set_to_none=True)
    detached_losses = []
    for start in range(0, batch.num_diagonal, microbatch_size):
        stop = min(start + microbatch_size, batch.num_diagonal)
        prediction = model(
            batch.diagonal_x[start:stop],
            batch.diagonal_time[start:stop],
            batch.diagonal_time[start:stop],
        )
        losses = (
            prediction.sub(batch.diagonal_target[start:stop])
            .square()
            .sum(dim=-1)
        )
        (losses.sum() / batch.batch_size).backward()
        detached_losses.append(losses.detach())
    for start in range(0, batch.num_off_diagonal, microbatch_size):
        stop = min(start + microbatch_size, batch.num_off_diagonal)
        _, derivative, target = lsd_outputs(
            model,
            batch.off_diagonal_x[start:stop],
            batch.start_time[start:stop],
            batch.target_time[start:stop],
        )
        losses = derivative.sub(target).square().sum(dim=-1)
        (losses.sum() / batch.batch_size).backward()
        detached_losses.append(losses.detach())
    total = torch.cat(detached_losses).sum() / batch.batch_size
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return total, gradients


class LsdTests(unittest.TestCase):
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
            torch.allclose(
                derivative,
                model(x, time, time),
                atol=1e-5,
                rtol=1e-5,
            )
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

    def test_exact_branch_split_and_flow_matching_construction(self) -> None:
        self.assertEqual(branch_sizes(100_000, 0.75), (75_000, 25_000))
        generator = torch.Generator().manual_seed(5)
        batch = sample_lsd_training_batch(
            100,
            diagonal_fraction=0.75,
            generator=generator,
            device=self.device,
        )
        self.assertEqual(batch.num_diagonal, 75)
        self.assertEqual(batch.num_off_diagonal, 25)

        recovered_noise = (
            batch.diagonal_x
            + (1.0 - batch.diagonal_time[:, None])
            * batch.diagonal_target
        )
        recovered_x0 = recovered_noise - batch.diagonal_target
        reconstructed = (
            (1.0 - batch.diagonal_time[:, None]) * recovered_x0
            + batch.diagonal_time[:, None] * recovered_noise
        )
        self.assertTrue(torch.allclose(reconstructed, batch.diagonal_x))

    def test_uniform_triangle_time_sampling(self) -> None:
        generator = torch.Generator().manual_seed(11)
        start, target = sample_off_diagonal_times(
            200_000,
            generator=generator,
            device=self.device,
        )
        self.assertTrue(bool((target >= 0.0).all()))
        self.assertTrue(bool((target < start).all()))
        self.assertTrue(bool((start <= 1.0).all()))
        self.assertAlmostEqual(float(start.mean()), 2.0 / 3.0, delta=0.003)
        self.assertAlmostEqual(float(target.mean()), 1.0 / 3.0, delta=0.003)
        self.assertAlmostEqual(
            float((start - target).mean()),
            1.0 / 3.0,
            delta=0.003,
        )

    def test_instantaneous_target_has_no_gradient(self) -> None:
        student = tiny_model()
        x = torch.randn(12, 2)
        start = torch.rand(12)
        target = start * torch.rand(12)
        mapped, derivative, velocity_target = lsd_outputs(
            student,
            x,
            start,
            target,
        )
        self.assertTrue(mapped.requires_grad)
        self.assertTrue(derivative.requires_grad)
        self.assertFalse(velocity_target.requires_grad)

        optimizer = torch.optim.SGD(student.parameters(), lr=1e-3)
        before = {
            name: parameter.detach().clone()
            for name, parameter in student.named_parameters()
        }
        derivative.sub(velocity_target).square().sum().backward()
        optimizer.step()
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter)
                for name, parameter in student.named_parameters()
            )
        )

    def test_target_uses_current_student_not_ema(self) -> None:
        student = tiny_model()
        ema = ExponentialMovingAverage(student)
        for value in ema.shadow.values():
            if torch.is_floating_point(value):
                value.add_(100.0)
        x = torch.randn(10, 2)
        start = torch.rand(10)
        target = start * torch.rand(10)
        mapped, _, velocity_target = lsd_outputs(
            student,
            x,
            start,
            target,
        )
        with torch.no_grad():
            expected = student(mapped.detach(), target, target)
        self.assertTrue(torch.equal(velocity_target, expected))

    def test_total_loss_and_microbatch_gradients_are_equivalent(self) -> None:
        generator = torch.Generator().manual_seed(13)
        batch = sample_lsd_training_batch(
            40,
            diagonal_fraction=0.75,
            generator=generator,
            device=self.device,
        )
        full_model = tiny_model()
        chunked_model = tiny_model()
        chunked_model.load_state_dict(full_model.state_dict())

        full_loss, full_gradients = batch_loss_and_gradients(
            full_model,
            batch,
            microbatch_size=40,
        )
        chunked_loss, chunked_gradients = batch_loss_and_gradients(
            chunked_model,
            batch,
            microbatch_size=7,
        )
        self.assertTrue(torch.allclose(full_loss, chunked_loss, atol=1e-6))
        self.assertEqual(full_gradients.keys(), chunked_gradients.keys())
        for name in full_gradients:
            self.assertTrue(
                torch.allclose(
                    full_gradients[name],
                    chunked_gradients[name],
                    atol=2e-6,
                    rtol=2e-5,
                ),
                msg=name,
            )

    def test_learning_rate_and_checkpoint_schedule(self) -> None:
        self.assertEqual(
            lsd_learning_rate(0, initial_lr=1e-3, decay_start=35_000),
            1e-3,
        )
        self.assertEqual(
            lsd_learning_rate(35_000, initial_lr=1e-3, decay_start=35_000),
            1e-3,
        )
        self.assertAlmostEqual(
            lsd_learning_rate(140_000, initial_lr=1e-3, decay_start=35_000),
            5e-4,
        )
        schedule = {
            "target_steps": 130_000,
            "checkpoint_every": 10_000,
            "late_start": 100_000,
            "late_every": 5_000,
        }
        self.assertTrue(should_checkpoint(90_000, **schedule))
        self.assertTrue(should_checkpoint(105_000, **schedule))
        self.assertFalse(should_checkpoint(103_000, **schedule))
        self.assertTrue(should_checkpoint(130_000, **schedule))

    def test_four_nfe_composite_ranking(self) -> None:
        rows = []
        for step, offset in ((99_000, 0.0), (101_000, 1.0)):
            rows.append(
                {
                    "step": step,
                    "metrics_by_nfe": {
                        str(nfe): {
                            "sw2": 0.01 + offset,
                            "histogram_jsd": 0.02 + offset,
                            "in_support_rate": 0.97 - offset,
                        }
                        for nfe in (1, 2, 4, 8)
                    },
                }
            )
        ranked = rank_rows(rows)
        self.assertEqual(ranked[0]["step"], 99_000)
        self.assertEqual(ranked[0]["composite_score"], 0.0)
        self.assertEqual(ranked[1]["composite_score"], 1.0)

    def test_checkpoint_restores_next_batch_and_loss(self) -> None:
        model = tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ema = ExponentialMovingAverage(model)
        generator = torch.Generator().manual_seed(17)
        args = Namespace(steps=130_000)
        payload = checkpoint_payload(
            student=model,
            optimizer=optimizer,
            ema=ema,
            generator=generator,
            step=321,
            elapsed_seconds=8.0,
            args=args,
            monitor_state={"nonfinite_gradient_count": 0},
        )

        def next_batch_loss(
            current_model: FlowMapNet,
            current_generator: torch.Generator,
        ) -> tuple[Tensor, Tensor, Tensor]:
            batch = sample_lsd_training_batch(
                20,
                diagonal_fraction=0.75,
                generator=current_generator,
                device=self.device,
            )
            loss, _ = batch_loss_and_gradients(
                current_model,
                batch,
                microbatch_size=6,
            )
            return (
                batch.diagonal_x.detach().clone(),
                batch.start_time.detach().clone(),
                loss.detach().clone(),
            )

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
            )
            actual = next_batch_loss(restored_model, restored_generator)

        self.assertEqual(step, 321)
        self.assertEqual(elapsed, 8.0)
        self.assertEqual(monitor["nonfinite_gradient_count"], 0)
        for expected_value, actual_value in zip(expected, actual):
            self.assertTrue(torch.equal(expected_value, actual_value))


if __name__ == "__main__":
    unittest.main()
