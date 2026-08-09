"""Configuration for cooling-rate control and thermal simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CONSERVATIVE_CONDUCTION = "conservative_harmonic"
LEGACY_CONDUCTION = "legacy_centered_diffusivity"
FIRST_SOLIDUS_CROSSING = "first_downward_crossing"
LEGACY_LAST_SOLIDUS_CROSSING = "legacy_last_downward_crossing"
DIRECTIONAL_FIXED_BOTTOM_BOUNDARY = "directional_fixed_bottom"
LEGACY_UNIFORM_BOUNDARY = "legacy_uniform"


@dataclass(frozen=True)
class EnvironmentConfig:
    # Geometry and discretization.
    num_layers: int = 5
    layer_height_mm: float = 0.9
    substrate_length_mm: float = 150.0
    substrate_height_mm: float = 9.9
    cell_size_mm: float = 0.3
    time_step_s: float = 0.0008

    # Thermal and phase properties for Ti-6Al-4V. Conductivity is stored in
    # W/(mm K), heat capacity in J/(g K), and density in g/mm^3 so the FD
    # calculation can remain consistently in millimetres, grams, and seconds.
    ambient_temperature_k: float = 296.15
    solidus_temperature_k: float = 1873.15
    liquidus_temperature_k: float = 1943.15
    density_g_mm3: float = 4430.0e-6
    latent_heat_j_g: float = 365.0
    conductivity_temperature_k: tuple[float, ...] = (
        273.15 + 20.0,
        273.15 + 93.0,
        273.15 + 205.0,
        273.15 + 250.0,
        273.15 + 315.0,
        273.15 + 425.0,
        273.15 + 500.0,
        273.15 + 540.0,
        273.15 + 650.0,
        273.15 + 760.0,
        273.15 + 800.0,
        273.15 + 870.0,
    )
    conductivity_w_mm_k: tuple[float, ...] = (
        6.6 * 1.0e-3,
        7.3 * 1.0e-3,
        9.1 * 1.0e-3,
        9.7 * 1.0e-3,
        10.6 * 1.0e-3,
        12.6 * 1.0e-3,
        13.9 * 1.0e-3,
        14.6 * 1.0e-3,
        17.5 * 1.0e-3,
        17.5 * 1.0e-3,
        17.5 * 1.0e-3,
        17.5 * 1.0e-3,
    )
    heat_capacity_temperature_k: tuple[float, ...] = (
        273.15 + 20.0,
        273.15 + 93.0,
        273.15 + 205.0,
        273.15 + 250.0,
        273.15 + 315.0,
        273.15 + 425.0,
        273.15 + 500.0,
        273.15 + 540.0,
        273.15 + 650.0,
        273.15 + 760.0,
        273.15 + 800.0,
        273.15 + 870.0,
        273.15 + 880.0,
    )
    heat_capacity_j_g_k: tuple[float, ...] = (
        565.0 * 1.0e-3,
        565.0 * 1.0e-3,
        574.0 * 1.0e-3,
        586.0 * 1.0e-3,
        603.0 * 1.0e-3,
        649.0 * 1.0e-3,
        682.0 * 1.0e-3,
        699.0 * 1.0e-3,
        770.0 * 1.0e-3,
        858.0 * 1.0e-3,
        895.0 * 1.0e-3,
        959.0 * 1.0e-3,
        959.0 * 1.0e-3,
    )
    property_table_min_k: float = 270.0
    property_table_max_k: float = 5000.0
    property_table_step_k: float = 1.0

    # Surface losses. Newly trained models use direction-specific DED boundary
    # conditions: forced convection and radiation on upward/free horizontal
    # faces, weaker convection and radiation on vertical side faces, and a
    # fixed-temperature bottom row representing a massive build platform.
    emissivity: float = 0.54
    top_convection_coefficient_w_mm2_k: float = 1000.0e-6
    side_convection_coefficient_w_mm2_k: float = 200.0e-6
    bottom_fixed_temperature_k: float = 296.15

    # Uniform-convection coefficient used by the uniform boundary scheme.
    convection_coefficient_w_mm2_k: float = 1000.0e-6
    stefan_boltzmann_w_mm2_k4: float = 5.67e-14

    # Moving volumetric laser source.
    laser_radius_mm: float = 1.5
    powder_absorption_fraction: float = 0.7
    deposited_material_absorption_fraction: float = 0.3
    laser_absorption_efficiency: float = 0.35
    beam_distribution_factor: float = 1.0

    # Numerical and measurement scheme selections.
    conduction_scheme: str = CONSERVATIVE_CONDUCTION
    solidus_crossing_scheme: str = FIRST_SOLIDUS_CROSSING
    boundary_condition_scheme: str = DIRECTIONAL_FIXED_BOTTOM_BOUNDARY

    ref_min: float = 1000.0
    ref_max: float = 1400.0
    state_min: float = 296.15
    state_max: float = 1296.15

    # Normalization bounds stored with the environment configuration.
    error_min: float = 0.0
    error_max: float = 600.0
    time_min: float = 15.0
    time_max: float = 50.0

    power_min: float = 1700.0
    power_max: float = 2700.0
    speed_min: float = 5.0
    speed_max: float = 15.0
    pre_dwell_min: float = 2.0
    pre_dwell_max: float = 15.0

    look_ahead: int = 2
    pad_ref: float = 1000.0
    randomize_per_layer: bool = True

    @property
    def num_rows_substrate(self) -> int:
        return round(self.substrate_height_mm / self.cell_size_mm)

    @property
    def num_cols_substrate(self) -> int:
        return round(self.substrate_length_mm / self.cell_size_mm)

    @property
    def num_rows_per_layer(self) -> int:
        return round(self.layer_height_mm / self.cell_size_mm)

    @property
    def observation_size(self) -> int:
        return 6 + 1 + 2 * self.look_ahead

    @property
    def action_size(self) -> int:
        return 3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    max_episodes: int = 200
    gamma: float = 0.99
    batch_size: int = 60
    replay_capacity: int = 10_000
    tau: float = 1.0e-3
    actor_learning_rate: float = 1.0e-4
    critic_learning_rate: float = 1.0e-3
    noise_std: float = 0.1
    noise_final_std: float | None = None
    noise_decay_start: int | None = None
    noise_decay_end: int | None = None
    noise_decay_type: str = "constant"
    moving_average_window: int = 10

    def to_dict(self) -> dict:
        return asdict(self)
