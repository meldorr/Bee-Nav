"""DDPM- and flow-matching-inspired controllers for 2D homing.

The controller treats the home displacement predicted from the current
observation as the denoised sample ``x_0`` in a local frame. It applies the
DDPM posterior endpoint coefficient for one reverse step and adds scheduled
Gaussian noise. Physical safety constraints are applied after sampling because
an unconstrained DDPM transition is not a safe robot command.

This module deliberately contains no visual model.  ``NoisyHomeVectorSensor`` is
a reproducible proxy for Bee-Nav's image-to-home-vector network.  A later adapter
can replace it with the real network without changing the controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


Array = np.ndarray


class HomeVectorSensor(Protocol):
    """Interface expected by the simulator and future visual-model adapters."""

    def predict(self, position: Array, home: Array, rng: np.random.Generator) -> Array:
        """Return the predicted world-frame vector from ``position`` to ``home``."""


@dataclass(frozen=True)
class SensorNoiseConfig:
    """Noise model for a visual home-vector estimate."""

    angle_std_deg: float = 12.0
    distance_relative_std: float = 0.15
    distance_absolute_std_m: float = 0.05
    outlier_probability: float = 0.0
    outlier_angle_std_deg: float = 90.0

    def __post_init__(self) -> None:
        if self.angle_std_deg < 0:
            raise ValueError("angle_std_deg must be non-negative")
        if self.distance_relative_std < 0 or self.distance_absolute_std_m < 0:
            raise ValueError("distance noise standard deviations must be non-negative")
        if not 0.0 <= self.outlier_probability <= 1.0:
            raise ValueError("outlier_probability must be between zero and one")
        if self.outlier_angle_std_deg < 0:
            raise ValueError("outlier_angle_std_deg must be non-negative")


class NoisyHomeVectorSensor:
    """Generate configurable noisy predictions of the full home displacement."""

    def __init__(self, config: SensorNoiseConfig | None = None) -> None:
        self.config = config or SensorNoiseConfig()

    def predict(self, position: Array, home: Array, rng: np.random.Generator) -> Array:
        true_vector = np.asarray(home, dtype=float) - np.asarray(position, dtype=float)
        true_distance = float(np.linalg.norm(true_vector))
        if true_distance == 0.0:
            return np.zeros(2, dtype=float)

        true_angle = float(np.arctan2(true_vector[1], true_vector[0]))
        angle_std = self.config.angle_std_deg
        if rng.random() < self.config.outlier_probability:
            angle_std = self.config.outlier_angle_std_deg
        predicted_angle = true_angle + rng.normal(0.0, np.deg2rad(angle_std))

        relative_error = rng.normal(0.0, self.config.distance_relative_std)
        absolute_error = rng.normal(0.0, self.config.distance_absolute_std_m)
        predicted_distance = max(0.0, true_distance * (1.0 + relative_error) + absolute_error)
        return predicted_distance * np.array(
            [np.cos(predicted_angle), np.sin(predicted_angle)], dtype=float
        )


class DDPMSchedule:
    """Linear beta schedule and standard DDPM posterior coefficients."""

    def __init__(
        self,
        num_steps: int = 40,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
    ) -> None:
        if num_steps < 1:
            raise ValueError("num_steps must be at least one")
        if not 0.0 < beta_start <= beta_end < 1.0:
            raise ValueError("betas must satisfy 0 < beta_start <= beta_end < 1")

        self.num_steps = num_steps
        self.betas = np.zeros(num_steps + 1, dtype=float)
        self.betas[1:] = np.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.ones(num_steps + 1, dtype=float)
        self.alpha_bars[1:] = np.cumprod(self.alphas[1:])

    def posterior(self, step: int) -> tuple[float, float, float]:
        """Return ``x0 coefficient``, ``xt coefficient`` and posterior variance."""
        if not 1 <= step <= self.num_steps:
            raise ValueError(f"step must be in [1, {self.num_steps}]")

        beta_t = self.betas[step]
        alpha_t = self.alphas[step]
        alpha_bar_t = self.alpha_bars[step]
        alpha_bar_previous = self.alpha_bars[step - 1]
        denominator = 1.0 - alpha_bar_t

        coefficient_x0 = beta_t * np.sqrt(alpha_bar_previous) / denominator
        coefficient_xt = (
            (1.0 - alpha_bar_previous) * np.sqrt(alpha_t) / denominator
        )
        posterior_variance = beta_t * (1.0 - alpha_bar_previous) / denominator
        return float(coefficient_x0), float(coefficient_xt), float(posterior_variance)


@dataclass(frozen=True)
class DDPMHomingControllerConfig:
    """Physical scaling and safety constraints for reverse diffusion steps."""

    state_scale_m: float = 10.0
    stochasticity: float = 0.35
    max_step_m: float = 1.5
    min_mean_progress_fraction: float = 0.1
    noise_mode: str = "tangential"
    cap_at_predicted_home: bool = True

    def __post_init__(self) -> None:
        if self.state_scale_m <= 0 or self.max_step_m <= 0:
            raise ValueError("state_scale_m and max_step_m must be positive")
        if self.stochasticity < 0:
            raise ValueError("stochasticity must be non-negative")
        if not 0.0 <= self.min_mean_progress_fraction <= 1.0:
            raise ValueError("min_mean_progress_fraction must be in [0, 1]")
        if self.noise_mode not in {"isotropic", "tangential"}:
            raise ValueError("noise_mode must be 'isotropic' or 'tangential'")


class DDPMHomingController:
    """Produce bounded physical movements from DDPM posterior transitions."""

    def __init__(
        self,
        schedule: DDPMSchedule | None = None,
        config: DDPMHomingControllerConfig | None = None,
    ) -> None:
        self.schedule = schedule or DDPMSchedule()
        self.config = config or DDPMHomingControllerConfig()

    def step(
        self,
        position: Array,
        predicted_home_vector: Array,
        reverse_step: int,
        rng: np.random.Generator,
    ) -> Array:
        """Return the next position for one scheduled reverse step.

        ``reverse_step`` follows DDPM indexing: it starts at ``T`` and decreases
        towards one.  At step one, the posterior variance is zero.
        """
        position = np.asarray(position, dtype=float)
        predicted_home_vector = np.asarray(predicted_home_vector, dtype=float)
        predicted_distance = float(np.linalg.norm(predicted_home_vector))
        if predicted_distance == 0.0:
            return position.copy()

        coefficient_x0, _, posterior_variance = self.schedule.posterior(
            reverse_step
        )

        # Apply the DDPM posterior in a local displacement frame whose origin is
        # the drone's current position. In this frame x_t is zero and the
        # predicted x_0 is the home vector. Using absolute world coordinates
        # would introduce an invalid attraction to the arbitrary map origin.
        scale = self.config.state_scale_m
        mean_delta = coefficient_x0 * predicted_home_vector

        if self.config.noise_mode == "tangential":
            direction = predicted_home_vector / predicted_distance
            tangent = np.array([-direction[1], direction[0]], dtype=float)
            gaussian = tangent * rng.normal()
        else:
            gaussian = rng.normal(size=2)

        noise = (
            self.config.stochasticity
            * np.sqrt(max(posterior_variance, 0.0))
            * scale
            * gaussian
        )
        proposed_delta = mean_delta + noise

        # Stochastic exploration may not erase the progress supplied by the
        # posterior mean.  This is a physical safety constraint, not part of DDPM.
        direction = predicted_home_vector / predicted_distance
        mean_forward_progress = max(0.0, float(np.dot(mean_delta, direction)))
        required_progress = (
            self.config.min_mean_progress_fraction * mean_forward_progress
        )
        actual_progress = float(np.dot(proposed_delta, direction))
        if actual_progress < required_progress:
            proposed_delta += (required_progress - actual_progress) * direction

        step_limit = self.config.max_step_m
        if self.config.cap_at_predicted_home:
            step_limit = min(step_limit, predicted_distance)
        proposed_length = float(np.linalg.norm(proposed_delta))
        if proposed_length > step_limit:
            proposed_delta *= step_limit / proposed_length

        return position + proposed_delta


@dataclass(frozen=True)
class FlowMatchingHomingControllerConfig:
    """Integration controls for a conditional endpoint flow."""

    num_steps: int = 40
    max_step_m: float = 1.5
    time_warp_power: float = 1.0
    cap_at_predicted_home: bool = True

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be at least one")
        if self.max_step_m <= 0:
            raise ValueError("max_step_m must be positive")
        if self.time_warp_power <= 0:
            raise ValueError("time_warp_power must be positive")


class FlowMatchingHomingController:
    """Closed-loop integration of a straight conditional flow to home.

    For a linear conditional path from a current state to an endpoint, the
    velocity at flow time ``t`` is ``(x_home - x_t) / (1 - t)``. The controller
    integrates that field for one interval and then re-observes the environment.
    A monotone time warp permits slower or faster early progress without changing
    the endpoint at flow time one.

    This is a flow-matching-inspired controller baseline. The velocity field is
    derived from the existing endpoint predictor; it is not yet a separately
    trained flow network.
    """

    def __init__(
        self, config: FlowMatchingHomingControllerConfig | None = None
    ) -> None:
        self.config = config or FlowMatchingHomingControllerConfig()

    def interpolation_fraction(self, iteration: int) -> float:
        """Return the endpoint fraction applied during this integration step."""
        if not 0 <= iteration < self.config.num_steps:
            raise ValueError(
                f"iteration must be in [0, {self.config.num_steps - 1}]"
            )
        time_now = iteration / self.config.num_steps
        time_next = (iteration + 1) / self.config.num_steps
        progress_now = 1.0 - (1.0 - time_now) ** self.config.time_warp_power
        progress_next = 1.0 - (1.0 - time_next) ** self.config.time_warp_power
        remaining_progress = 1.0 - progress_now
        return float((progress_next - progress_now) / remaining_progress)

    def step(
        self,
        position: Array,
        predicted_home_vector: Array,
        iteration: int,
    ) -> Array:
        position = np.asarray(position, dtype=float)
        predicted_home_vector = np.asarray(predicted_home_vector, dtype=float)
        predicted_distance = float(np.linalg.norm(predicted_home_vector))
        if predicted_distance == 0.0:
            return position.copy()

        fraction = self.interpolation_fraction(iteration)
        proposed_delta = fraction * predicted_home_vector
        step_limit = self.config.max_step_m
        if self.config.cap_at_predicted_home:
            step_limit = min(step_limit, predicted_distance)
        proposed_length = float(np.linalg.norm(proposed_delta))
        if proposed_length > step_limit:
            proposed_delta *= step_limit / proposed_length
        return position + proposed_delta


@dataclass(frozen=True)
class FlowDDPMHomingControllerConfig:
    """Controls for flow drift with scheduled DDPM exploration."""

    state_scale_m: float = 10.0
    stochasticity: float = 0.075
    max_step_m: float = 1.5
    time_warp_power: float = 1.0
    min_flow_progress_fraction: float = 0.1
    noise_mode: str = "tangential"
    cap_at_predicted_home: bool = True

    def __post_init__(self) -> None:
        if self.state_scale_m <= 0 or self.max_step_m <= 0:
            raise ValueError("state_scale_m and max_step_m must be positive")
        if self.stochasticity < 0:
            raise ValueError("stochasticity must be non-negative")
        if self.time_warp_power <= 0:
            raise ValueError("time_warp_power must be positive")
        if not 0.0 <= self.min_flow_progress_fraction <= 1.0:
            raise ValueError("min_flow_progress_fraction must be in [0, 1]")
        if self.noise_mode not in {"isotropic", "tangential"}:
            raise ValueError("noise_mode must be 'isotropic' or 'tangential'")


class FlowDDPMHomingController:
    """Combine flow-matching drift with DDPM-scheduled exploration.

    The deterministic part integrates the straight conditional flow to the
    currently predicted home. The stochastic part uses the variance of the
    corresponding DDPM reverse step. Thus exploration fades to zero as flow
    time approaches one, when the controller takes a final denoised endpoint
    step. This is a controller-level hybrid, not a trained stochastic
    interpolant or score/velocity network.
    """

    def __init__(
        self,
        schedule: DDPMSchedule | None = None,
        config: FlowDDPMHomingControllerConfig | None = None,
    ) -> None:
        self.schedule = schedule or DDPMSchedule()
        self.config = config or FlowDDPMHomingControllerConfig()

    def interpolation_fraction(self, iteration: int) -> float:
        """Return the straight-flow endpoint fraction for this interval."""
        if not 0 <= iteration < self.schedule.num_steps:
            raise ValueError(
                f"iteration must be in [0, {self.schedule.num_steps - 1}]"
            )
        time_now = iteration / self.schedule.num_steps
        time_next = (iteration + 1) / self.schedule.num_steps
        progress_now = 1.0 - (1.0 - time_now) ** self.config.time_warp_power
        progress_next = 1.0 - (1.0 - time_next) ** self.config.time_warp_power
        return float((progress_next - progress_now) / (1.0 - progress_now))

    def step(
        self,
        position: Array,
        predicted_home_vector: Array,
        iteration: int,
        rng: np.random.Generator,
    ) -> Array:
        """Take one bounded stochastic-flow integration step."""
        position = np.asarray(position, dtype=float)
        predicted_home_vector = np.asarray(predicted_home_vector, dtype=float)
        predicted_distance = float(np.linalg.norm(predicted_home_vector))
        if predicted_distance == 0.0:
            return position.copy()

        fraction = self.interpolation_fraction(iteration)
        flow_delta = fraction * predicted_home_vector
        reverse_step = self.schedule.num_steps - iteration
        _, _, posterior_variance = self.schedule.posterior(reverse_step)

        direction = predicted_home_vector / predicted_distance
        if self.config.noise_mode == "tangential":
            tangent = np.array([-direction[1], direction[0]], dtype=float)
            gaussian = tangent * rng.normal()
        else:
            gaussian = rng.normal(size=2)
        noise = (
            self.config.stochasticity
            * np.sqrt(max(posterior_variance, 0.0))
            * self.config.state_scale_m
            * gaussian
        )
        proposed_delta = flow_delta + noise

        required_progress = (
            self.config.min_flow_progress_fraction
            * float(np.dot(flow_delta, direction))
        )
        actual_progress = float(np.dot(proposed_delta, direction))
        if actual_progress < required_progress:
            proposed_delta += (required_progress - actual_progress) * direction

        step_limit = self.config.max_step_m
        if self.config.cap_at_predicted_home:
            step_limit = min(step_limit, predicted_distance)
        proposed_length = float(np.linalg.norm(proposed_delta))
        if proposed_length > step_limit:
            proposed_delta *= step_limit / proposed_length
        return position + proposed_delta


@dataclass(frozen=True)
class MeanFlowProxyControllerConfig:
    """Controls for a finite-interval average-velocity proxy."""

    interval_fraction: float = 1.0
    max_step_m: float = 1.5
    cap_at_predicted_home: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.interval_fraction <= 1.0:
            raise ValueError("interval_fraction must be in (0, 1]")
        if self.max_step_m <= 0:
            raise ValueError("max_step_m must be positive")


class MeanFlowProxyController:
    """Treat the predicted endpoint displacement as an average velocity.

    For a unit flow interval, a straight conditional path has average velocity
    equal to its endpoint displacement. This permits one finite chord step,
    bounded by the physical step limit, instead of numerically integrating many
    instantaneous velocities. It is an inference-time proxy for the MeanFlow
    idea; a genuine MeanFlow model would train a network conditioned on both
    interval endpoints ``(r, t)`` using the MeanFlow identity.
    """

    def __init__(
        self, config: MeanFlowProxyControllerConfig | None = None
    ) -> None:
        self.config = config or MeanFlowProxyControllerConfig()

    def step(self, position: Array, predicted_home_vector: Array) -> Array:
        position = np.asarray(position, dtype=float)
        predicted_home_vector = np.asarray(predicted_home_vector, dtype=float)
        predicted_distance = float(np.linalg.norm(predicted_home_vector))
        if predicted_distance == 0.0:
            return position.copy()

        proposed_delta = self.config.interval_fraction * predicted_home_vector
        step_limit = self.config.max_step_m
        if self.config.cap_at_predicted_home:
            step_limit = min(step_limit, predicted_distance)
        proposed_length = float(np.linalg.norm(proposed_delta))
        if proposed_length > step_limit:
            proposed_delta *= step_limit / proposed_length
        return position + proposed_delta


@dataclass(frozen=True)
class BeeNavController:
    """Original distance-scaled Bee-Nav movement rule used as a baseline."""

    minimum_step_m: float = 0.1
    distance_gain: float = 0.13
    max_step_m: float = 1.5

    def step(self, position: Array, predicted_home_vector: Array) -> Array:
        position = np.asarray(position, dtype=float)
        predicted_home_vector = np.asarray(predicted_home_vector, dtype=float)
        distance = float(np.linalg.norm(predicted_home_vector))
        if distance == 0.0:
            return position.copy()
        step_length = min(self.minimum_step_m + self.distance_gain * distance, self.max_step_m)
        step_length = min(step_length, distance)
        return position + step_length * predicted_home_vector / distance


@dataclass(frozen=True)
class TrialResult:
    success: bool
    steps: int
    path_length_m: float
    final_error_m: float
    trajectory: Array


def run_trial(
    start: Array,
    home: Array,
    sensor: HomeVectorSensor,
    controller: (
        DDPMHomingController
        | FlowMatchingHomingController
        | FlowDDPMHomingController
        | MeanFlowProxyController
        | BeeNavController
    ),
    rng: np.random.Generator,
    controller_rng: np.random.Generator | None = None,
    success_radius_m: float = 0.5,
    max_steps: int | None = None,
) -> TrialResult:
    """Run one closed-loop homing trial."""
    if success_radius_m <= 0:
        raise ValueError("success_radius_m must be positive")
    if max_steps is None:
        if isinstance(controller, (DDPMHomingController, FlowDDPMHomingController)):
            max_steps = controller.schedule.num_steps
        elif isinstance(controller, FlowMatchingHomingController):
            max_steps = controller.config.num_steps
        else:
            max_steps = 40
    if max_steps < 1:
        raise ValueError("max_steps must be at least one")

    position = np.asarray(start, dtype=float).copy()
    home = np.asarray(home, dtype=float)
    trajectory = [position.copy()]
    if controller_rng is None:
        controller_rng = rng

    for iteration in range(max_steps):
        if np.linalg.norm(home - position) <= success_radius_m:
            break
        prediction = sensor.predict(position, home, rng)
        if isinstance(controller, DDPMHomingController):
            reverse_step = max(controller.schedule.num_steps - iteration, 1)
            position = controller.step(position, prediction, reverse_step, controller_rng)
        elif isinstance(controller, FlowDDPMHomingController):
            flow_iteration = min(iteration, controller.schedule.num_steps - 1)
            position = controller.step(
                position, prediction, flow_iteration, controller_rng
            )
        elif isinstance(controller, FlowMatchingHomingController):
            flow_iteration = min(iteration, controller.config.num_steps - 1)
            position = controller.step(position, prediction, flow_iteration)
        else:
            position = controller.step(position, prediction)
        trajectory.append(position.copy())

    trajectory_array = np.asarray(trajectory)
    segment_lengths = np.linalg.norm(np.diff(trajectory_array, axis=0), axis=1)
    final_error = float(np.linalg.norm(home - position))
    return TrialResult(
        success=final_error <= success_radius_m,
        steps=len(trajectory_array) - 1,
        path_length_m=float(np.sum(segment_lengths)),
        final_error_m=final_error,
        trajectory=trajectory_array,
    )


def run_trials(
    num_trials: int,
    seed: int,
    sensor: HomeVectorSensor,
    controller: (
        DDPMHomingController
        | FlowMatchingHomingController
        | FlowDDPMHomingController
        | MeanFlowProxyController
        | BeeNavController
    ),
    start_radius_range_m: tuple[float, float] = (5.0, 15.0),
    success_radius_m: float = 0.5,
    max_steps: int | None = None,
) -> tuple[dict[str, float | int], list[TrialResult]]:
    """Run reproducible trials from random positions around a home at the origin."""
    if num_trials < 1:
        raise ValueError("num_trials must be at least one")
    radius_min, radius_max = start_radius_range_m
    if not 0.0 < radius_min <= radius_max:
        raise ValueError("start radius range must satisfy 0 < minimum <= maximum")

    # Keep start positions, perception noise, and controller noise on separate
    # streams. This makes comparisons paired: adding DDPM noise cannot change
    # the random starts or the sequence of sensor-error samples.
    start_rng = np.random.default_rng(seed)
    sensor_seeds = np.random.SeedSequence(seed + 1).spawn(num_trials)
    controller_seeds = np.random.SeedSequence(seed + 2).spawn(num_trials)
    results: list[TrialResult] = []
    home = np.zeros(2, dtype=float)
    for trial_index in range(num_trials):
        radius = start_rng.uniform(radius_min, radius_max)
        angle = start_rng.uniform(-np.pi, np.pi)
        start = radius * np.array([np.cos(angle), np.sin(angle)])
        sensor_rng = np.random.default_rng(sensor_seeds[trial_index])
        controller_rng = np.random.default_rng(controller_seeds[trial_index])
        results.append(
            run_trial(
                start=start,
                home=home,
                sensor=sensor,
                controller=controller,
                rng=sensor_rng,
                controller_rng=controller_rng,
                success_radius_m=success_radius_m,
                max_steps=max_steps,
            )
        )

    success_count = sum(result.success for result in results)
    summary = {
        "trials": num_trials,
        "success_rate": success_count / num_trials,
        "mean_final_error_m": float(np.mean([r.final_error_m for r in results])),
        "median_final_error_m": float(np.median([r.final_error_m for r in results])),
        "mean_path_length_m": float(np.mean([r.path_length_m for r in results])),
        "mean_steps": float(np.mean([r.steps for r in results])),
        "mean_sensor_evaluations": float(np.mean([r.steps for r in results])),
    }
    return summary, results
