"""Command-line experiment for the DDPM-inspired stochastic homing controller."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ddpm_stochastic_homing.controller import (
    BeeNavController,
    DDPMHomingController,
    DDPMHomingControllerConfig,
    DDPMSchedule,
    NoisyHomeVectorSensor,
    SensorNoiseConfig,
    run_trials,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Bee-Nav stepping with a DDPM-inspired stochastic controller."
    )
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--radius-min", type=float, default=5.0)
    parser.add_argument("--radius-max", type=float, default=15.0)
    parser.add_argument("--success-radius", type=float, default=0.5)
    parser.add_argument("--angle-std-deg", type=float, default=12.0)
    parser.add_argument("--distance-relative-std", type=float, default=0.15)
    parser.add_argument("--outlier-probability", type=float, default=0.02)
    parser.add_argument("--outlier-angle-std-deg", type=float, default=90.0)
    parser.add_argument("--beta-start", type=float, default=1.0e-4)
    parser.add_argument("--beta-end", type=float, default=2.0e-2)
    parser.add_argument("--stochasticity", type=float, default=0.35)
    parser.add_argument("--state-scale-m", type=float, default=10.0)
    parser.add_argument("--max-step-m", type=float, default=1.5)
    parser.add_argument(
        "--noise-mode", choices=("tangential", "isotropic"), default="tangential"
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensor_config = SensorNoiseConfig(
        angle_std_deg=args.angle_std_deg,
        distance_relative_std=args.distance_relative_std,
        outlier_probability=args.outlier_probability,
        outlier_angle_std_deg=args.outlier_angle_std_deg,
    )
    sensor = NoisyHomeVectorSensor(sensor_config)
    schedule = DDPMSchedule(
        num_steps=args.steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    controller_config = DDPMHomingControllerConfig(
        state_scale_m=args.state_scale_m,
        stochasticity=args.stochasticity,
        max_step_m=args.max_step_m,
        noise_mode=args.noise_mode,
    )
    ddpm_controller = DDPMHomingController(schedule, controller_config)
    baseline_controller = BeeNavController(max_step_m=args.max_step_m)

    common = {
        "num_trials": args.trials,
        "seed": args.seed,
        "sensor": sensor,
        "start_radius_range_m": (args.radius_min, args.radius_max),
        "success_radius_m": args.success_radius,
        "max_steps": args.steps,
    }
    ddpm_summary, ddpm_results = run_trials(controller=ddpm_controller, **common)
    baseline_summary, _ = run_trials(controller=baseline_controller, **common)

    report = {
        "configuration": {
            "seed": args.seed,
            "start_radius_range_m": [args.radius_min, args.radius_max],
            "success_radius_m": args.success_radius,
            "sensor": asdict(sensor_config),
            "schedule": {
                "num_steps": args.steps,
                "beta_start": args.beta_start,
                "beta_end": args.beta_end,
            },
            "controller": asdict(controller_config),
        },
        "ddpm_stochastic": ddpm_summary,
        "bee_nav_baseline": baseline_summary,
        "example_trajectory": ddpm_results[0].trajectory.tolist(),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
