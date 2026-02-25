# Calibrated Logistic Regression Sensor (Home Assistant)

Custom integration that exposes a calibrated logistic regression output as a Home Assistant sensor.

## Features

- One sensor per config entry
- Logistic regression from source entity states
- Platt-style calibration (`calibration_slope`, `calibration_intercept`)
- Availability handling when required features are missing/non-numeric
- Diagnostic attributes (`raw_probability`, `linear_score`, feature values, missing features)

## Installation

1. Copy `custom_components/calibrated_logistic_regression` into your HA `custom_components` folder.
2. Restart Home Assistant.
3. Add integration from **Settings -> Devices & Services -> Add Integration**.

## Configuration Inputs

- `name`: Sensor name
- `intercept`: Model intercept
- `coefficients`: JSON map of feature entity IDs to weights
- `required_features`: Comma-separated list of feature entity IDs
- `calibration_slope`: Calibration slope (default `1.0`)
- `calibration_intercept`: Calibration intercept (default `0.0`)

## Sensor Behavior

- State is calibrated probability in percent (`0-100`).
- Sensor becomes unavailable if any required feature is missing or non-numeric.
- Updates when any required feature state changes.

## Best-Practice Notes

- Keep coefficients and required features aligned.
- Use stable numeric source entities (template sensors can help normalize values).
- Start with `calibration_slope=1.0` and `calibration_intercept=0.0` if calibration is unknown.
