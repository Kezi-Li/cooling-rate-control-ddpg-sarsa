"""Finite-difference thermal model for layer-wise cooling-rate simulation.

The default conduction update uses symmetric harmonic cell-face conductivity
to discretize ``div(k(T) grad(T))`` conservatively. The module also supports a
centred-diffusivity scheme selected through the environment configuration. All
physics calculations use float64.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .config import (
    CONSERVATIVE_CONDUCTION,
    DIRECTIONAL_FIXED_BOTTOM_BOUNDARY,
    EnvironmentConfig,
    FIRST_SOLIDUS_CROSSING,
    LEGACY_CONDUCTION,
    LEGACY_LAST_SOLIDUS_CROSSING,
    LEGACY_UNIFORM_BOUNDARY,
)


_DEFAULT_CONFIG = EnvironmentConfig()

# Convenience aliases for commonly used defaults. The finite-difference kernel
# receives the complete set of values from EnvironmentConfig.
AMBIENT_K = _DEFAULT_CONFIG.ambient_temperature_k
T_SOLIDUS_K = _DEFAULT_CONFIG.solidus_temperature_k
T_LIQUIDUS_K = _DEFAULT_CONFIG.liquidus_temperature_k
GRID_MM = _DEFAULT_CONFIG.cell_size_mm
DT_S = _DEFAULT_CONFIG.time_step_s
LAYER_HEIGHT_MM = _DEFAULT_CONFIG.layer_height_mm
LASER_RADIUS_MM = _DEFAULT_CONFIG.laser_radius_mm


def _interp_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Linear interpolation with linear extrapolation at both ends."""
    x = np.asarray(x, dtype=np.float64)
    y = np.interp(x, xp, fp)
    below = x < xp[0]
    above = x > xp[-1]
    y[below] = fp[0] + (x[below] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    y[above] = fp[-1] + (x[above] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y


def material_property_tables(
    config: EnvironmentConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build FD property tables from the supplied environment configuration."""
    cfg = config or EnvironmentConfig()
    t_k = np.asarray(cfg.conductivity_temperature_k, dtype=np.float64)
    k_values = np.asarray(cfg.conductivity_w_mm_k, dtype=np.float64)
    t_cp = np.asarray(cfg.heat_capacity_temperature_k, dtype=np.float64)
    cp_values = np.asarray(cfg.heat_capacity_j_g_k, dtype=np.float64)
    if t_k.size != k_values.size or t_k.size < 2:
        raise ValueError("conductivity temperatures and values must have equal lengths >= 2")
    if t_cp.size != cp_values.size or t_cp.size < 2:
        raise ValueError("heat-capacity temperatures and values must have equal lengths >= 2")
    if np.any(np.diff(t_k) <= 0.0) or np.any(np.diff(t_cp) <= 0.0):
        raise ValueError("material-property temperatures must be strictly increasing")
    if np.any(k_values <= 0.0) or np.any(cp_values <= 0.0):
        raise ValueError("conductivity and heat capacity must be positive")
    if cfg.density_g_mm3 <= 0.0:
        raise ValueError("density must be positive")
    if cfg.latent_heat_j_g < 0.0:
        raise ValueError("latent heat cannot be negative")

    latent_heat = cfg.latent_heat_j_g
    solidus = cfg.solidus_temperature_k
    liquidus = cfg.liquidus_temperature_k
    if liquidus <= solidus:
        raise ValueError("liquidus temperature must exceed solidus temperature")
    t_lh = np.array([
        solidus - 1.0, solidus - 0.1, solidus,
        solidus + 1.0, liquidus, liquidus + 0.1,
        liquidus + 0.2,
    ], dtype=np.float64)
    latent_cp = latent_heat / (liquidus - solidus)
    cp_lh_values = np.array([0.0, 0.0, latent_cp, latent_cp, latent_cp, 0.0, 0.0], dtype=np.float64)

    if cfg.property_table_step_k <= 0.0:
        raise ValueError("property table step must be positive")
    if cfg.property_table_max_k <= cfg.property_table_min_k:
        raise ValueError("property table maximum must exceed its minimum")
    t_range = np.arange(
        cfg.property_table_min_k,
        cfg.property_table_max_k + 0.5 * cfg.property_table_step_k,
        cfg.property_table_step_k,
        dtype=np.float64,
    )
    if t_range.size < 2:
        raise ValueError("property table must contain at least two temperatures")
    cp_eff = _interp_extrap(t_range, t_cp, cp_values) + _interp_extrap(t_range, t_lh, cp_lh_values)
    diffusivity = _interp_extrap(t_range, t_k, k_values) / (
        cfg.density_g_mm3 * cp_eff
    )
    return cp_eff, diffusivity, t_k, k_values, t_range


@njit(cache=True)
def _uniform_table_interp(
    x: float,
    values: np.ndarray,
    table_minimum: float,
    table_step: float,
) -> float:
    table_maximum = table_minimum + table_step * (values.size - 1)
    if x <= table_minimum:
        return values[0] + (x - table_minimum) * (
            values[1] - values[0]
        ) / table_step
    if x >= table_maximum:
        n = values.size
        return values[n - 1] + (x - table_maximum) * (
            values[n - 1] - values[n - 2]
        ) / table_step
    position = (x - table_minimum) / table_step
    idx = int(np.floor(position))
    fraction = position - idx
    return values[idx] + fraction * (values[idx + 1] - values[idx])


@njit(cache=True)
def _knotted_interp(x: float, knots: np.ndarray, values: np.ndarray) -> float:
    if x <= knots[0]:
        idx = 0
    elif x >= knots[knots.size - 1]:
        idx = knots.size - 2
    else:
        idx = 0
        while x > knots[idx + 1]:
            idx += 1
    return values[idx] + (x - knots[idx]) * (values[idx + 1] - values[idx]) / (knots[idx + 1] - knots[idx])


@njit(cache=True)
def _round_half_away_2(x: float) -> float:
    """Round to two decimals with ties away from zero."""
    if x >= 0.0:
        return np.floor(x * 100.0 + 0.5) / 100.0
    return -np.floor(-x * 100.0 + 0.5) / 100.0


@njit(cache=True)
def _harmonic_face_conductivity(left: float, right: float) -> float:
    """Symmetric cell-face conductivity used by the conservative scheme."""
    if left <= 0.0 or right <= 0.0:
        raise ValueError("thermal conductivity must be positive")
    return 2.0 * left * right / (left + right)


@njit(cache=True)
def _should_record_solidus_crossing(
    recorded_time: float,
    old_temperature: float,
    new_temperature: float,
    solidus_temperature: float,
    first_crossing_only: bool,
) -> bool:
    """Return whether this downward solidus crossing should be retained."""
    crossed_downward = (
        old_temperature > solidus_temperature
        and new_temperature <= solidus_temperature
    )
    return crossed_downward and (
        not first_crossing_only or recorded_time == 0.0
    )


@njit(cache=True)
def _update_solidus_record(
    maximum_temperature: float,
    peak_time: float,
    solidus_time: float,
    below_solidus_temperature: float,
    old_temperature: float,
    new_temperature: float,
    time_now: float,
    solidus_temperature: float,
    first_crossing_only: bool,
) -> tuple[float, float, float, float]:
    """Update peak and crossing data for one cell at one time step."""
    if new_temperature > maximum_temperature:
        maximum_temperature = new_temperature
        peak_time = time_now
        if first_crossing_only:
            solidus_time = 0.0
            below_solidus_temperature = 0.0
    if _should_record_solidus_crossing(
        solidus_time,
        old_temperature,
        new_temperature,
        solidus_temperature,
        first_crossing_only,
    ):
        solidus_time = time_now
        below_solidus_temperature = new_temperature
    return (
        maximum_temperature,
        peak_time,
        solidus_time,
        below_solidus_temperature,
    )


@njit(cache=True)
def _cooling_rate_kernel(
    initial_temperature: np.ndarray,
    initial_cell_state: np.ndarray,
    laser_position: np.ndarray,
    laser_power: np.ndarray,
    traverse_speed: np.ndarray,
    powder_feeder: np.ndarray,
    cp_eff_table: np.ndarray,
    diffusivity_table: np.ndarray,
    k_knots: np.ndarray,
    k_values: np.ndarray,
    property_table_minimum: float,
    property_table_step: float,
    grid_mm: float,
    time_step_s: float,
    layer_height_mm: float,
    laser_radius_mm: float,
    ambient_temperature: float,
    solidus_temperature: float,
    density: float,
    emissivity: float,
    legacy_convection: float,
    top_convection: float,
    side_convection: float,
    bottom_fixed_temperature: float,
    stefan_boltzmann: float,
    powder_absorption_fraction: float,
    deposited_material_absorption_fraction: float,
    laser_absorption_efficiency: float,
    beam_distribution_factor: float,
    conservative_conduction: bool,
    first_solidus_crossing: bool,
    directional_fixed_bottom: bool,
    probe_row: int,
    probe_col: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    temperature = initial_temperature.copy()
    cell_state = initial_cell_state.copy()
    rows, cols = cell_state.shape
    if directional_fixed_bottom:
        for col in range(cols):
            if cell_state[rows - 1, col] == 1.0:
                temperature[rows - 1, col] = bottom_fixed_temperature
    old_temperature = np.empty_like(temperature)
    temporary = np.empty_like(temperature)
    conductivity_field = np.empty_like(temperature)
    maximum_temperature = temperature.copy()
    peak_time = np.zeros_like(temperature)
    solidus_time = np.zeros_like(temperature)
    below_solidus_temperature = np.zeros_like(temperature)
    solidus_crossing_count = np.zeros(temperature.shape, dtype=np.int64)
    post_peak_solidus_crossing_count = np.zeros(
        temperature.shape,
        dtype=np.int64,
    )
    track_probe = probe_row >= 0 and probe_col >= 0
    probe_temperature_history = np.empty(
        powder_feeder.size + 1 if track_probe else 0,
        dtype=np.float64,
    )
    if track_probe:
        probe_temperature_history[0] = temperature[probe_row, probe_col]

    x_positions = np.arange(cols, dtype=np.float64) * grid_mm + grid_mm / 2.0
    z_positions = np.empty(rows, dtype=np.float64)
    for row in range(rows):
        z_positions[row] = (rows - row - 1) * grid_mm + grid_mm / 2.0

    laser_x = laser_position[0]
    laser_z = laser_position[1]
    source_scale = beam_distribution_factor / (
        np.pi * laser_radius_mm ** 2 * layer_height_mm
    )
    source_scale *= (
        powder_absorption_fraction
        + (1.0 - powder_absorption_fraction)
        * deposited_material_absorption_fraction
    ) * laser_absorption_efficiency

    for step in range(powder_feeder.size):
        time_now = step * time_step_s
        old_temperature[:, :] = temperature
        laser_x += time_step_s * traverse_speed[step]

        if laser_power[step] > 0.0:
            for row in range(rows):
                rounded_rz = _round_half_away_2(z_positions[row] - laser_z)
                below_laser = z_positions[row] <= laser_z
                for col in range(cols):
                    in_stripe = (
                        x_positions[col] >= laser_x - laser_radius_mm
                        and x_positions[col] <= laser_x + laser_radius_mm
                    )
                    heated = below_laser and in_stripe
                    if heated and powder_feeder[step] == 1.0:
                        cell_state[row, col] = 1.0
                    if heated:
                        rx = x_positions[col] - laser_x
                        radius_sq = rx * rx + rounded_rz * rounded_rz
                        source = laser_power[step] * source_scale * np.exp(
                            -beam_distribution_factor
                            * radius_sq
                            / (laser_radius_mm ** 2)
                        )
                        cp_eff = _uniform_table_interp(
                            temperature[row, col],
                            cp_eff_table,
                            property_table_minimum,
                            property_table_step,
                        )
                        temperature[row, col] += (
                            source
                            * time_step_s
                            / (density * cp_eff)
                            * cell_state[row, col]
                        )

        if conservative_conduction:
            for row in range(rows):
                for col in range(cols):
                    if cell_state[row, col] == 1.0:
                        conductivity_field[row, col] = _knotted_interp(
                            temperature[row, col], k_knots, k_values
                        )

        # Conservative face fluxes are the default. The alternate branch uses
        # a centred-cell diffusivity update.
        for row in range(rows):
            for col in range(cols):
                if cell_state[row, col] != 1.0:
                    temporary[row, col] = 0.0
                    continue

                horizontal_free_faces = 0.0
                side_free_faces = 0.0
                air_sides = 0.0
                neighbour_sum = 0.0
                face_conductivity_sum = 0.0
                conductive_flux_sum = 0.0
                current = temperature[row, col]
                current_conductivity = (
                    conductivity_field[row, col]
                    if conservative_conduction
                    else _knotted_interp(current, k_knots, k_values)
                )
                if row == 0:
                    horizontal_free_faces += 1.0
                    air_sides += 1.0
                elif cell_state[row - 1, col] == 1.0:
                    neighbour = temperature[row - 1, col]
                    neighbour_sum += neighbour
                    if conservative_conduction:
                        neighbour_conductivity = conductivity_field[row - 1, col]
                        face_conductivity = _harmonic_face_conductivity(
                            current_conductivity, neighbour_conductivity
                        )
                        face_conductivity_sum += face_conductivity
                        conductive_flux_sum += face_conductivity * (
                            neighbour - current
                        )
                else:
                    horizontal_free_faces += 1.0
                    air_sides += 1.0
                if row == rows - 1:
                    if not directional_fixed_bottom:
                        horizontal_free_faces += 1.0
                        air_sides += 1.0
                elif cell_state[row + 1, col] == 1.0:
                    neighbour = temperature[row + 1, col]
                    neighbour_sum += neighbour
                    if conservative_conduction:
                        neighbour_conductivity = conductivity_field[row + 1, col]
                        face_conductivity = _harmonic_face_conductivity(
                            current_conductivity, neighbour_conductivity
                        )
                        face_conductivity_sum += face_conductivity
                        conductive_flux_sum += face_conductivity * (
                            neighbour - current
                        )
                else:
                    horizontal_free_faces += 1.0
                    air_sides += 1.0
                if col == 0:
                    side_free_faces += 1.0
                    air_sides += 1.0
                elif cell_state[row, col - 1] == 1.0:
                    neighbour = temperature[row, col - 1]
                    neighbour_sum += neighbour
                    if conservative_conduction:
                        neighbour_conductivity = conductivity_field[row, col - 1]
                        face_conductivity = _harmonic_face_conductivity(
                            current_conductivity, neighbour_conductivity
                        )
                        face_conductivity_sum += face_conductivity
                        conductive_flux_sum += face_conductivity * (
                            neighbour - current
                        )
                else:
                    side_free_faces += 1.0
                    air_sides += 1.0
                if col == cols - 1:
                    side_free_faces += 1.0
                    air_sides += 1.0
                elif cell_state[row, col + 1] == 1.0:
                    neighbour = temperature[row, col + 1]
                    neighbour_sum += neighbour
                    if conservative_conduction:
                        neighbour_conductivity = conductivity_field[row, col + 1]
                        face_conductivity = _harmonic_face_conductivity(
                            current_conductivity, neighbour_conductivity
                        )
                        face_conductivity_sum += face_conductivity
                        conductive_flux_sum += face_conductivity * (
                            neighbour - current
                        )
                else:
                    side_free_faces += 1.0
                    air_sides += 1.0

                cp_eff = _uniform_table_interp(
                    current,
                    cp_eff_table,
                    property_table_minimum,
                    property_table_step,
                )
                if conservative_conduction:
                    coefficient_scale = time_step_s / (
                        density * cp_eff * grid_mm ** 2
                    )
                    convergence = (
                        1.0 - coefficient_scale * face_conductivity_sum
                    )
                    updated = current + coefficient_scale * conductive_flux_sum
                else:
                    diffusivity = _uniform_table_interp(
                        current,
                        diffusivity_table,
                        property_table_minimum,
                        property_table_step,
                    )
                    coefficient = diffusivity * time_step_s / (grid_mm ** 2)
                    convergence = 1.0 - (4.0 - air_sides) * coefficient
                    updated = (
                        coefficient * neighbour_sum + convergence * current
                    )
                if convergence < 0.0 or np.isnan(convergence):
                    raise ValueError("Time step too large for finite-difference stability")

                if directional_fixed_bottom:
                    combined_top_h = 1.0 / (
                        1.0 / top_convection
                        + grid_mm / (2.0 * current_conductivity)
                    )
                    combined_side_h = 1.0 / (
                        1.0 / side_convection
                        + grid_mm / (2.0 * current_conductivity)
                    )
                    q_convection = (current - ambient_temperature) * (
                        combined_top_h * horizontal_free_faces
                        + combined_side_h * side_free_faces
                    )
                    radiating_faces = horizontal_free_faces + side_free_faces
                else:
                    combined_h = 1.0 / (
                        1.0 / legacy_convection
                        + grid_mm / (2.0 * current_conductivity)
                    )
                    q_convection = combined_h * (
                        current - ambient_temperature
                    ) * air_sides
                    radiating_faces = air_sides
                updated -= q_convection * time_step_s / (
                    density * cp_eff * grid_mm
                )
                q_radiation = emissivity * stefan_boltzmann * (
                    current ** 4 - ambient_temperature ** 4
                ) * radiating_faces
                updated -= q_radiation * time_step_s / (
                    density * cp_eff * grid_mm
                )
                temporary[row, col] = updated

        for row in range(rows):
            for col in range(cols):
                if cell_state[row, col] == 1.0:
                    if directional_fixed_bottom and row == rows - 1:
                        temperature[row, col] = bottom_fixed_temperature
                    else:
                        temperature[row, col] = temporary[row, col]
                new_global_peak = (
                    temperature[row, col] > maximum_temperature[row, col]
                )
                if new_global_peak:
                    post_peak_solidus_crossing_count[row, col] = 0
                crossed_downward = (
                    old_temperature[row, col] > solidus_temperature
                    and temperature[row, col] <= solidus_temperature
                )
                if crossed_downward:
                    solidus_crossing_count[row, col] += 1
                    post_peak_solidus_crossing_count[row, col] += 1
                (
                    maximum_temperature[row, col],
                    peak_time[row, col],
                    solidus_time[row, col],
                    below_solidus_temperature[row, col],
                ) = _update_solidus_record(
                    maximum_temperature[row, col],
                    peak_time[row, col],
                    solidus_time[row, col],
                    below_solidus_temperature[row, col],
                    old_temperature[row, col],
                    temperature[row, col],
                    time_now + time_step_s,
                    solidus_temperature,
                    first_solidus_crossing,
                )
        if track_probe:
            probe_temperature_history[step + 1] = temperature[
                probe_row,
                probe_col,
            ]

    cooling_rate = np.zeros(cols, dtype=np.float64)
    for col in range(cols):
        if (
            solidus_time[0, col] > peak_time[0, col]
            and maximum_temperature[0, col] > solidus_temperature
        ):
            cooling_time = solidus_time[0, col] - peak_time[0, col]
            if cooling_time > 0.0:
                cooling_rate[col] = (
                    maximum_temperature[0, col] - below_solidus_temperature[0, col]
                ) / cooling_time
    return (
        temperature,
        cell_state,
        cooling_rate,
        solidus_crossing_count,
        post_peak_solidus_crossing_count,
        probe_temperature_history,
    )


def cooling_rate_simulation(
    initial_temperature: np.ndarray,
    cell_state: np.ndarray,
    laser_position: np.ndarray,
    laser_power: np.ndarray,
    traverse_speed: np.ndarray,
    powder_feeder: np.ndarray,
    *,
    config: EnvironmentConfig | None = None,
    conduction_scheme: str | None = None,
    solidus_crossing_scheme: str | None = None,
    return_diagnostics: bool = False,
    probe_index: tuple[int, int] | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]
):
    """Run the configured thermal model and optionally return diagnostics."""
    cfg = config or EnvironmentConfig()
    active_conduction_scheme = conduction_scheme or cfg.conduction_scheme
    active_crossing_scheme = (
        solidus_crossing_scheme or cfg.solidus_crossing_scheme
    )
    active_boundary_scheme = cfg.boundary_condition_scheme
    temperature = np.ascontiguousarray(initial_temperature, dtype=np.float64)
    state = np.ascontiguousarray(cell_state, dtype=np.float64)
    position = np.ascontiguousarray(laser_position, dtype=np.float64)
    power = np.ascontiguousarray(laser_power, dtype=np.float64).reshape(-1)
    speed = np.ascontiguousarray(traverse_speed, dtype=np.float64).reshape(-1)
    powder = np.ascontiguousarray(powder_feeder, dtype=np.float64).reshape(-1)
    if temperature.shape != state.shape:
        raise ValueError("initial_temperature and cell_state must have identical shapes")
    if not (power.size == speed.size == powder.size):
        raise ValueError("laser_power, traverse_speed, and powder_feeder lengths must match")
    if probe_index is not None and not return_diagnostics:
        raise ValueError("probe_index requires return_diagnostics=True")
    if probe_index is None:
        probe_row = -1
        probe_col = -1
    else:
        probe_row, probe_col = map(int, probe_index)
        if not (0 <= probe_row < state.shape[0] and 0 <= probe_col < state.shape[1]):
            raise ValueError("probe_index is outside the temperature grid")
    if active_conduction_scheme not in (
        CONSERVATIVE_CONDUCTION,
        LEGACY_CONDUCTION,
    ):
        raise ValueError(
            f"unknown conduction scheme: {active_conduction_scheme}"
        )
    if active_crossing_scheme not in (
        FIRST_SOLIDUS_CROSSING,
        LEGACY_LAST_SOLIDUS_CROSSING,
    ):
        raise ValueError(
            f"unknown solidus crossing scheme: {active_crossing_scheme}"
        )
    if active_boundary_scheme not in (
        DIRECTIONAL_FIXED_BOTTOM_BOUNDARY,
        LEGACY_UNIFORM_BOUNDARY,
    ):
        raise ValueError(f"unknown boundary condition scheme: {active_boundary_scheme}")
    if cfg.cell_size_mm <= 0.0 or cfg.time_step_s <= 0.0:
        raise ValueError("cell size and time step must be positive")
    if cfg.layer_height_mm <= 0.0 or cfg.laser_radius_mm <= 0.0:
        raise ValueError("layer height and laser radius must be positive")
    if cfg.density_g_mm3 <= 0.0:
        raise ValueError("density must be positive")
    convection_coefficients = (
        cfg.convection_coefficient_w_mm2_k,
        cfg.top_convection_coefficient_w_mm2_k,
        cfg.side_convection_coefficient_w_mm2_k,
    )
    if any(value <= 0.0 for value in convection_coefficients):
        raise ValueError("convection coefficients must be positive")
    if cfg.bottom_fixed_temperature_k <= 0.0:
        raise ValueError("bottom fixed temperature must be positive")
    if cfg.stefan_boltzmann_w_mm2_k4 <= 0.0:
        raise ValueError("Stefan-Boltzmann constant must be positive")
    if not 0.0 <= cfg.emissivity <= 1.0:
        raise ValueError("emissivity must be between zero and one")
    if cfg.beam_distribution_factor <= 0.0:
        raise ValueError("beam distribution factor must be positive")
    source_fractions = (
        cfg.powder_absorption_fraction,
        cfg.deposited_material_absorption_fraction,
        cfg.laser_absorption_efficiency,
    )
    if any(value < 0.0 or value > 1.0 for value in source_fractions):
        raise ValueError("laser-source absorption fractions must be in [0, 1]")
    cp_eff, diffusivity, k_knots, k_values, _ = material_property_tables(cfg)
    (
        final_temperature,
        _,
        cooling_rate,
        crossing_count,
        post_peak_crossing_count,
        probe_temperature_history,
    ) = _cooling_rate_kernel(
        temperature, state, position, power, speed, powder,
        cp_eff, diffusivity, k_knots, k_values,
        cfg.property_table_min_k,
        cfg.property_table_step_k,
        cfg.cell_size_mm,
        cfg.time_step_s,
        cfg.layer_height_mm,
        cfg.laser_radius_mm,
        cfg.ambient_temperature_k,
        cfg.solidus_temperature_k,
        cfg.density_g_mm3,
        cfg.emissivity,
        cfg.convection_coefficient_w_mm2_k,
        cfg.top_convection_coefficient_w_mm2_k,
        cfg.side_convection_coefficient_w_mm2_k,
        cfg.bottom_fixed_temperature_k,
        cfg.stefan_boltzmann_w_mm2_k4,
        cfg.powder_absorption_fraction,
        cfg.deposited_material_absorption_fraction,
        cfg.laser_absorption_efficiency,
        cfg.beam_distribution_factor,
        active_conduction_scheme == CONSERVATIVE_CONDUCTION,
        active_crossing_scheme == FIRST_SOLIDUS_CROSSING,
        active_boundary_scheme == DIRECTIONAL_FIXED_BOTTOM_BOUNDARY,
        probe_row,
        probe_col,
    )
    # Return a fully active state mask for the next layer simulation.
    final_state = np.ones_like(state)
    if return_diagnostics:
        surface_counts = crossing_count[0]
        surface_post_peak_counts = post_peak_crossing_count[0]
        diagnostics: dict[str, object] = {
            "solidus_crossing_count": crossing_count,
            "cells_with_downward_crossing": int(np.count_nonzero(crossing_count)),
            "cells_with_repeated_downward_crossing": int(
                np.count_nonzero(crossing_count > 1)
            ),
            "extra_downward_crossings": int(
                np.sum(np.maximum(crossing_count - 1, 0))
            ),
            "surface_cells_with_downward_crossing": int(
                np.count_nonzero(surface_counts)
            ),
            "surface_cells_with_repeated_downward_crossing": int(
                np.count_nonzero(surface_counts > 1)
            ),
            "cells_with_repeated_post_peak_downward_crossing": int(
                np.count_nonzero(post_peak_crossing_count > 1)
            ),
            "surface_cells_with_repeated_post_peak_downward_crossing": int(
                np.count_nonzero(surface_post_peak_counts > 1)
            ),
        }
        if probe_index is not None:
            diagnostics["probe_index"] = (probe_row, probe_col)
            diagnostics["probe_temperature_history"] = probe_temperature_history
        return final_temperature, final_state, cooling_rate, diagnostics
    return final_temperature, final_state, cooling_rate
