# DDPM-inspired stochastic homing

This experiment replaces Bee-Nav's fixed distance-scaled movement with a
scheduled stochastic reverse-diffusion step. It is a controller experiment,
not a diffusion image generator.

For each observation, a home-vector predictor supplies a full displacement
estimate. The controller treats the implied home position as `x_0`, calculates
the DDPM posterior endpoint coefficient for `x_{t-1}`, adds scheduled Gaussian
noise, and converts the sample into a bounded physical movement. The posterior
is evaluated in the drone's local displacement frame so the command is not
affected by the arbitrary origin of a world map.

The default uses **tangential noise**: stochasticity is applied perpendicular
to the predicted home direction. This permits exploratory viewpoints while
preserving expected progress. Isotropic DDPM noise is also implemented for
ablation experiments, but it should not be used on hardware without an
external safety controller.

## Run locally

Only NumPy is required:

```bash
python3 -m ddpm_stochastic_homing.run_experiment --trials 1000
```

The command prints a JSON report comparing the stochastic controller with the
original Bee-Nav movement rule. Save it with:

```bash
python3 -m ddpm_stochastic_homing.run_experiment \
  --trials 1000 \
  --output output/ddpm_stochastic_homing.json
```

Important controls:

- `--stochasticity`: DDPM posterior noise multiplier; zero gives a deterministic
  posterior-mean controller.
- `--noise-mode tangential|isotropic`: safe exploratory noise versus literal
  two-dimensional Gaussian noise.
- `--beta-start`, `--beta-end`, `--steps`: reverse schedule.
- `--angle-std-deg`, `--distance-relative-std`: visual prediction noise.
- `--outlier-probability`: frequency of large angular prediction errors.

## Run tests

```bash
python3 -m unittest discover -s ddpm_stochastic_homing/tests -v
```

The tests verify the DDPM posterior coefficients, deterministic convergence,
physical step bounds, progress under tangential stochasticity, and seeded
reproducibility.

## Scope and limitations

`NoisyHomeVectorSensor` is a controlled proxy for the Bee-Nav visual network.
It lets us test the stochastic controller separately from perception. It does
not generate images after movement and does not model obstacles. Its default
noise values are illustrative rather than a calibrated fit of the released
flight data. The next
integration step is to replace this proxy with a visual-network/Isaac adapter;
hardware testing should happen only after closed-loop simulation and shadow
mode validation.

## Flow-matching-inspired comparison

The package also contains a deterministic conditional endpoint-flow baseline.
It integrates the straight flow

```text
v(x_t, t) = (x_home - x_t) / (1 - t)
```

and obtains a new visual endpoint estimate after every physical step. Run it
with:

```bash
python3 -m ddpm_stochastic_homing.run_flow_experiment --trials 1000
```

This controller derives the flow from the current home-vector prediction. It
does not yet train a separate time-conditioned neural velocity field.

## Flow plus DDPM hybrid

The hybrid uses the flow velocity for directed progress and the DDPM posterior
variance for scheduled exploration. Noise is strongest early and fades to zero
at the final endpoint step:

```bash
python3 -m ddpm_stochastic_homing.run_hybrid_experiment --trials 1000
```

The report compares the hybrid, pure DDPM, pure flow, and original Bee-Nav rule
using identical starting points and perception-noise streams. This is a
controller-level combination, not yet a learned stochastic-interpolant model.

## Average-velocity / MeanFlow proxy and complete report

The average-velocity proxy treats Bee-Nav's predicted endpoint displacement as
the chord over a finite flow interval. It tests the inference-time benefit of a
large average-velocity step, but it is not a trained interval-conditioned
MeanFlow network. Run every implemented controller on identical random trials:

```bash
python3 -m ddpm_stochastic_homing.run_all_experiments --trials 1000
```

Each physical step makes one sensor/model evaluation, so the reported mean
sensor evaluations also measures inference cost.
