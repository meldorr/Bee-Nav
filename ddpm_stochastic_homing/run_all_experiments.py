"""Run every implemented homing controller on one paired benchmark."""

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
    FlowDDPMHomingController,
    FlowDDPMHomingControllerConfig,
    FlowMatchingHomingController,
    FlowMatchingHomingControllerConfig,
    MeanFlowProxyController,
    MeanFlowProxyControllerConfig,
    NoisyHomeVectorSensor,
    SensorNoiseConfig,
    run_trials,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare all implemented Bee-Nav stepping controllers."
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
    parser.add_argument("--ddpm-stochasticity", type=float, default=0.35)
    parser.add_argument("--hybrid-stochasticity", type=float, default=0.075)
    parser.add_argument("--state-scale-m", type=float, default=10.0)
    parser.add_argument("--max-step-m", type=float, default=1.5)
    parser.add_argument("--time-warp-power", type=float, default=1.0)
    parser.add_argument("--meanflow-interval-fraction", type=float, default=1.0)
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
    schedule = DDPMSchedule(args.steps, args.beta_start, args.beta_end)

    ddpm_config = DDPMHomingControllerConfig(
        state_scale_m=args.state_scale_m,
        stochasticity=args.ddpm_stochasticity,
        max_step_m=args.max_step_m,
    )
    flow_config = FlowMatchingHomingControllerConfig(
        num_steps=args.steps,
        max_step_m=args.max_step_m,
        time_warp_power=args.time_warp_power,
    )
    hybrid_config = FlowDDPMHomingControllerConfig(
        state_scale_m=args.state_scale_m,
        stochasticity=args.hybrid_stochasticity,
        max_step_m=args.max_step_m,
        time_warp_power=args.time_warp_power,
    )
    meanflow_config = MeanFlowProxyControllerConfig(
        interval_fraction=args.meanflow_interval_fraction,
        max_step_m=args.max_step_m,
    )
    controllers = {
        "bee_nav_baseline": BeeNavController(max_step_m=args.max_step_m),
        "ddpm_stochastic": DDPMHomingController(schedule, ddpm_config),
        "flow_matching_inspired": FlowMatchingHomingController(flow_config),
        "flow_ddpm_hybrid": FlowDDPMHomingController(schedule, hybrid_config),
        "meanflow_average_velocity_proxy": MeanFlowProxyController(meanflow_config),
    }
    common = {
        "num_trials": args.trials,
        "seed": args.seed,
        "sensor": sensor,
        "start_radius_range_m": (args.radius_min, args.radius_max),
        "success_radius_m": args.success_radius,
        "max_steps": args.steps,
    }
    results = {
        name: run_trials(controller=controller, **common)[0]
        for name, controller in controllers.items()
    }
    report = {
        "configuration": {
            "seed": args.seed,
            "start_radius_range_m": [args.radius_min, args.radius_max],
            "success_radius_m": args.success_radius,
            "max_steps": args.steps,
            "sensor": asdict(sensor_config),
            "schedule": {
                "num_steps": args.steps,
                "beta_start": args.beta_start,
                "beta_end": args.beta_end,
            },
            "ddpm_controller": asdict(ddpm_config),
            "flow_controller": asdict(flow_config),
            "hybrid_controller": asdict(hybrid_config),
            "meanflow_proxy_controller": asdict(meanflow_config),
        },
        "results": results,
        "interpretation_note": (
            "The MeanFlow entry is an inference-time straight-path proxy using "
            "Bee-Nav endpoint predictions; it is not a trained interval-conditioned "
            "MeanFlow network."
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
