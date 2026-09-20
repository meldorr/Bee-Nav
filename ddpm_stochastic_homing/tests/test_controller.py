import unittest

import numpy as np

from ddpm_stochastic_homing.controller import (
    DDPMHomingController,
    DDPMHomingControllerConfig,
    DDPMSchedule,
    FlowDDPMHomingController,
    FlowDDPMHomingControllerConfig,
    FlowMatchingHomingController,
    FlowMatchingHomingControllerConfig,
    MeanFlowProxyController,
    MeanFlowProxyControllerConfig,
    NoisyHomeVectorSensor,
    SensorNoiseConfig,
    run_trial,
    run_trials,
)


class DDPMScheduleTests(unittest.TestCase):
    def test_alpha_bar_decreases(self) -> None:
        schedule = DDPMSchedule(num_steps=20)
        self.assertTrue(np.all(np.diff(schedule.alpha_bars) < 0.0))

    def test_last_reverse_step_has_no_noise_and_selects_x0(self) -> None:
        schedule = DDPMSchedule(num_steps=20)
        coefficient_x0, coefficient_xt, variance = schedule.posterior(1)
        self.assertAlmostEqual(coefficient_x0, 1.0)
        self.assertAlmostEqual(coefficient_xt, 0.0)
        self.assertAlmostEqual(variance, 0.0)


class DDPMControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = DDPMSchedule(num_steps=30)

    def test_zero_stochasticity_converges_with_perfect_predictions(self) -> None:
        controller = DDPMHomingController(
            self.schedule,
            DDPMHomingControllerConfig(stochasticity=0.0, max_step_m=2.0),
        )
        sensor = NoisyHomeVectorSensor(
            SensorNoiseConfig(
                angle_std_deg=0.0,
                distance_relative_std=0.0,
                distance_absolute_std_m=0.0,
            )
        )
        result = run_trial(
            start=np.array([10.0, 3.0]),
            home=np.zeros(2),
            sensor=sensor,
            controller=controller,
            rng=np.random.default_rng(1),
        )
        self.assertTrue(result.success)
        self.assertLessEqual(result.final_error_m, 0.5)

    def test_physical_step_limit_is_respected(self) -> None:
        controller = DDPMHomingController(
            self.schedule,
            DDPMHomingControllerConfig(
                stochasticity=2.0,
                max_step_m=0.7,
                noise_mode="isotropic",
            ),
        )
        rng = np.random.default_rng(2)
        position = np.array([10.0, 0.0])
        prediction = np.array([-10.0, 0.0])
        next_position = controller.step(position, prediction, 30, rng)
        self.assertLessEqual(np.linalg.norm(next_position - position), 0.7 + 1.0e-12)

    def test_tangential_noise_preserves_forward_progress(self) -> None:
        controller = DDPMHomingController(
            self.schedule,
            DDPMHomingControllerConfig(
                stochasticity=2.0,
                max_step_m=2.0,
                noise_mode="tangential",
            ),
        )
        position = np.array([8.0, 0.0])
        prediction = np.array([-8.0, 0.0])
        direction = prediction / np.linalg.norm(prediction)
        for seed in range(50):
            next_position = controller.step(
                position, prediction, 30, np.random.default_rng(seed)
            )
            self.assertGreaterEqual(
                float(np.dot(next_position - position, direction)), -1.0e-12
            )

    def test_nonzero_stochasticity_changes_the_reverse_sample(self) -> None:
        controller = DDPMHomingController(
            self.schedule,
            DDPMHomingControllerConfig(
                stochasticity=0.5,
                max_step_m=10.0,
                noise_mode="tangential",
            ),
        )
        position = np.array([5.0, 0.0])
        prediction = np.array([-5.0, 0.0])
        sample_a = controller.step(
            position, prediction, 30, np.random.default_rng(10)
        )
        sample_b = controller.step(
            position, prediction, 30, np.random.default_rng(11)
        )
        self.assertFalse(np.allclose(sample_a, sample_b))

    def test_step_is_translation_invariant(self) -> None:
        controller = DDPMHomingController(self.schedule)
        prediction = np.array([-5.0, 2.0])
        position_a = np.array([5.0, -2.0])
        position_b = np.array([105.0, 48.0])
        next_a = controller.step(
            position_a, prediction, 20, np.random.default_rng(19)
        )
        next_b = controller.step(
            position_b, prediction, 20, np.random.default_rng(19)
        )
        np.testing.assert_allclose(next_a - position_a, next_b - position_b)

    def test_seeded_trials_are_reproducible(self) -> None:
        controller = DDPMHomingController(self.schedule)
        sensor = NoisyHomeVectorSensor()
        summary_a, results_a = run_trials(20, 7, sensor, controller)
        summary_b, results_b = run_trials(20, 7, sensor, controller)
        self.assertEqual(summary_a, summary_b)
        np.testing.assert_allclose(results_a[0].trajectory, results_b[0].trajectory)


class FlowMatchingControllerTests(unittest.TestCase):
    def test_linear_flow_fractions_end_at_endpoint(self) -> None:
        controller = FlowMatchingHomingController(
            FlowMatchingHomingControllerConfig(num_steps=4, max_step_m=100.0)
        )
        fractions = [controller.interpolation_fraction(i) for i in range(4)]
        np.testing.assert_allclose(fractions, [0.25, 1.0 / 3.0, 0.5, 1.0])

    def test_perfect_linear_flow_converges(self) -> None:
        controller = FlowMatchingHomingController(
            FlowMatchingHomingControllerConfig(num_steps=20, max_step_m=100.0)
        )
        sensor = NoisyHomeVectorSensor(
            SensorNoiseConfig(
                angle_std_deg=0.0,
                distance_relative_std=0.0,
                distance_absolute_std_m=0.0,
            )
        )
        result = run_trial(
            start=np.array([12.0, -4.0]),
            home=np.zeros(2),
            sensor=sensor,
            controller=controller,
            rng=np.random.default_rng(5),
            success_radius_m=1.0e-9,
        )
        self.assertTrue(result.success)
        self.assertLessEqual(result.final_error_m, 1.0e-9)

    def test_flow_step_is_translation_invariant(self) -> None:
        controller = FlowMatchingHomingController(
            FlowMatchingHomingControllerConfig(num_steps=20)
        )
        prediction = np.array([-4.0, 1.0])
        position_a = np.array([4.0, -1.0])
        position_b = np.array([54.0, 19.0])
        delta_a = controller.step(position_a, prediction, 7) - position_a
        delta_b = controller.step(position_b, prediction, 7) - position_b
        np.testing.assert_allclose(delta_a, delta_b)


class FlowDDPMControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = DDPMSchedule(num_steps=20)

    def test_zero_stochasticity_matches_straight_flow(self) -> None:
        hybrid = FlowDDPMHomingController(
            self.schedule,
            FlowDDPMHomingControllerConfig(
                stochasticity=0.0,
                max_step_m=100.0,
            ),
        )
        flow = FlowMatchingHomingController(
            FlowMatchingHomingControllerConfig(
                num_steps=20,
                max_step_m=100.0,
            )
        )
        position = np.array([8.0, -3.0])
        prediction = np.array([-8.0, 3.0])
        for iteration in (0, 7, 19):
            hybrid_next = hybrid.step(
                position, prediction, iteration, np.random.default_rng(3)
            )
            flow_next = flow.step(position, prediction, iteration)
            np.testing.assert_allclose(hybrid_next, flow_next)

    def test_noise_fades_to_zero_at_final_flow_step(self) -> None:
        hybrid = FlowDDPMHomingController(
            self.schedule,
            FlowDDPMHomingControllerConfig(
                stochasticity=2.0,
                max_step_m=100.0,
            ),
        )
        position = np.array([3.0, 2.0])
        prediction = np.array([-3.0, -2.0])
        result_a = hybrid.step(position, prediction, 19, np.random.default_rng(1))
        result_b = hybrid.step(position, prediction, 19, np.random.default_rng(2))
        np.testing.assert_allclose(result_a, result_b)
        np.testing.assert_allclose(result_a, np.zeros(2), atol=1.0e-12)

    def test_hybrid_respects_physical_step_limit(self) -> None:
        hybrid = FlowDDPMHomingController(
            self.schedule,
            FlowDDPMHomingControllerConfig(
                stochasticity=3.0,
                max_step_m=0.6,
                noise_mode="isotropic",
            ),
        )
        position = np.array([10.0, 0.0])
        prediction = np.array([-10.0, 0.0])
        next_position = hybrid.step(
            position, prediction, 0, np.random.default_rng(8)
        )
        self.assertLessEqual(
            float(np.linalg.norm(next_position - position)), 0.6 + 1.0e-12
        )


class MeanFlowProxyControllerTests(unittest.TestCase):
    def test_full_interval_reaches_predicted_endpoint(self) -> None:
        controller = MeanFlowProxyController(
            MeanFlowProxyControllerConfig(max_step_m=100.0)
        )
        position = np.array([5.0, -2.0])
        prediction = np.array([-4.0, 3.0])
        np.testing.assert_allclose(
            controller.step(position, prediction), position + prediction
        )

    def test_fractional_interval_uses_average_velocity_chord(self) -> None:
        controller = MeanFlowProxyController(
            MeanFlowProxyControllerConfig(
                interval_fraction=0.25,
                max_step_m=100.0,
            )
        )
        position = np.array([5.0, -2.0])
        prediction = np.array([-4.0, 3.0])
        np.testing.assert_allclose(
            controller.step(position, prediction), position + 0.25 * prediction
        )

    def test_average_velocity_step_respects_physical_limit(self) -> None:
        controller = MeanFlowProxyController(
            MeanFlowProxyControllerConfig(max_step_m=0.75)
        )
        position = np.array([8.0, 0.0])
        next_position = controller.step(position, np.array([-8.0, 0.0]))
        self.assertLessEqual(
            float(np.linalg.norm(next_position - position)), 0.75 + 1.0e-12
        )


if __name__ == "__main__":
    unittest.main()
